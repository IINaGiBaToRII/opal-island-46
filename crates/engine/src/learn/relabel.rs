
use super::select::{with_eval, EvalRun, EvalSpec};
use crate::egdb::Egdb;
use crate::eval::Evaluator;
use crate::facade::{GameConfig, VariantId};
use crate::position::Position;
use crate::rules::{American, Brazilian, Czech, International, Italian, Rules, Russian, Spanish, Thai, Turkish};
use crate::search::{search_tt_db, SearchLimits};
use crate::selfplay::extract::{legal, replay, MATE_SKIP};
use crate::selfplay::format::{assemble_chunk, ChunkHeader};
use crate::selfplay::{dispatch_cfg, Chunk, Game};
use crate::tt::Tt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct RelabelParams {
    pub from: PathBuf,
    pub out: PathBuf,
    pub chunks: Option<(u32, u32)>,
    pub eval: EvalSpec,
    pub nodes: u64,
    pub threads: usize,
    pub egdb: Option<PathBuf>,
    pub no_egdb: bool,
    pub any_eval: bool,
    pub samples_only: bool,
    pub verbose: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub variant: String,
    pub forced: bool,
    pub eval_id: u32,
    pub nodes: u64,
    pub samples_only: bool,
    pub no_egdb: bool,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Pairs {
    pub n: u64,
    pub sx: f64,
    pub sy: f64,
    pub sxx: f64,
    pub syy: f64,
    pub sxy: f64,
    pub sabs: f64,
    pub mate_cross: u64,
}

impl Pairs {
    fn add(&mut self, old: i16, new: i16) {
        let (a, b) = (old as i32, new as i32);
        let (ma, mb) = (a.abs() > MATE_SKIP, b.abs() > MATE_SKIP);
        if ma != mb {
            self.mate_cross += 1;
        }
        if ma || mb {
            return;
        }
        let (x, y) = (a as f64, b as f64);
        self.n += 1;
        self.sx += x;
        self.sy += y;
        self.sxx += x * x;
        self.syy += y * y;
        self.sxy += x * y;
        self.sabs += (x - y).abs();
    }
    fn merge(&mut self, o: &Pairs) {
        self.n += o.n;
        self.sx += o.sx;
        self.sy += o.sy;
        self.sxx += o.sxx;
        self.syy += o.syy;
        self.sxy += o.sxy;
        self.sabs += o.sabs;
        self.mate_cross += o.mate_cross;
    }
    pub fn mean_abs(&self) -> f64 {
        self.sabs / self.n.max(1) as f64
    }
    pub fn corr(&self) -> f64 {
        let n = self.n.max(1) as f64;
        let cov = self.sxy / n - self.sx / n * self.sy / n;
        let (mx, my) = (self.sx / n, self.sy / n);
        let vx = self.sxx / n - mx * mx;
        let vy = self.syy / n - my * my;
        cov / (vx * vy).sqrt().max(1e-12)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RelabelReport {
    pub chunks: usize,
    pub chunks_done: usize,
    pub chunks_skipped: usize,
    pub games: u64,
    pub searched: u64,
    pub thread_secs: f64,
    pub wall_secs: f64,
    pub samples: u64,
    pub same_exact: u64,
    pub pairs: Pairs,
    pub nodes: u64,
    pub eval_id: u32,
}

impl RelabelReport {
    pub fn summary(&self) -> String {
        let pps = self.searched as f64 / self.thread_secs.max(1e-9);
        format!(
            "переоценка: кусков {} (сделано {}, уже были {}), партий {}, позиций обучения {}
  поисков {} на {} узлов: {:.1} позиций/с на поток ({:.1}с работы потоков, {:.1}с по часам); тех же оценки и глубины {:.2}%
  старая → новая (без mate-подобных, {} пар): средняя |разница| {:.1} сантиединици, корреляция {:.4}; переходов через порог mate {}
",
            self.chunks,
            self.chunks_done,
            self.chunks_skipped,
            self.games,
            self.samples,
            self.searched,
            self.nodes,
            pps,
            self.thread_secs,
            self.wall_secs,
            100.0 * self.same_exact as f64 / self.searched.max(1) as f64,
            self.pairs.n,
            self.pairs.mean_abs(),
            self.pairs.corr(),
            self.pairs.mate_cross
        )
    }
}

#[derive(Default)]
struct GameAcc {
    searched: u64,
    same: u64,
    samples: u64,
    secs: f64,
    pairs: Pairs,
}

fn positions<R: Rules>(g: &Game) -> Result<Vec<Position<R>>, String> {
    let mut v = Vec::with_capacity(g.plies.len() + 1);
    replay::<R>(g, |_, pos, _, _| v.push(*pos))?;
    Ok(v)
}

fn is_sample<R: Rules>(pos: &Position<R>) -> bool {
    !(R::FORCED && legal::<R>(pos).iter().any(|m| m.is_capture()))
}

#[allow(clippy::too_many_arguments)]
pub fn relabel_game<R: Rules, E: Evaluator<R>>(g: &Game, eval: &E, db: Option<&Egdb<R>>, nodes: u64, hash_mb: usize, samples_only: bool, acc: &mut GameAccPub) -> Result<Game, String> {
    let ps = positions::<R>(g)?;
    let mut tt = Tt::new(hash_mb);
    let lim = SearchLimits { nodes: Some(nodes), hash_mb, threads: 1, ..Default::default() };
    let stop = AtomicBool::new(false);
    let mut out = g.clone();
    for (i, pr) in g.plies.iter().enumerate() {
        if pr.random {
            continue;
        }
        let pos = &ps[i];
        let sample = is_sample(pos);
        if samples_only && !sample {
            continue;
        }
        let r = search_tt_db(pos, &ps[..i], eval, &lim, &stop, &mut |_| {}, &mut tt, db.filter(|_| !pr.in_db));
        let score = r.score.clamp(-32000, 32000) as i16;
        let depth = r.depth.min(255) as u8;
        acc.0.searched += 1;
        if score == pr.score && depth == pr.depth {
            acc.0.same += 1;
        }
        if sample {
            acc.0.samples += 1;
            acc.0.pairs.add(pr.score, score);
        }
        out.plies[i].score = score;
        out.plies[i].depth = depth;
    }
    Ok(out)
}

struct ReplayRun<'a> {
    g: &'a Game,
    nodes: u64,
    hash_mb: usize,
}

impl<'a, R: Rules> EvalRun<R> for ReplayRun<'a> {
    type Out = Result<(Vec<u8>, u32), String>;
    fn run<E: Evaluator<R>>(self, eval: &E, eval_id: u32) -> Self::Out {
        let mut acc = GameAccPub::default();
        let g = relabel_game::<R, E>(self.g, eval, None, self.nodes, self.hash_mb, false, &mut acc)?;
        let mut b = Vec::new();
        g.encode(&mut b);
        Ok((b, eval_id))
    }
}

fn replay_cfg<R: Rules>(eval: &EvalSpec, g: &Game, nodes: u64, hash_mb: usize) -> Result<(Vec<u8>, u32), String> {
    with_eval::<R, _>(eval, ReplayRun { g, nodes, hash_mb })?
}

pub fn relabel_game_bytes(cfg: &GameConfig, eval: &EvalSpec, g: &Game, nodes: u64, hash_mb: usize) -> Result<(Vec<u8>, u32), String> {
    dispatch_cfg!(cfg, replay_cfg(eval, g, nodes, hash_mb.max(1)))
}

#[derive(Default)]
pub struct GameAccPub(GameAcc);

pub fn same_structure(a: &Game, b: &Game) -> bool {
    a.start == b.start
        && a.random_plies == b.random_plies
        && a.result == b.result
        && a.reason == b.reason
        && a.egdb == b.egdb
        && a.plies.len() == b.plies.len()
        && a.plies.iter().zip(&b.plies).all(|(x, y)| x.from == y.from && x.to == y.to && x.idx == y.idx && x.random == y.random && x.in_db == y.in_db)
}

fn chunk_no(p: &Path) -> Option<u32> {
    p.file_name()?.to_str()?.strip_prefix("games-")?.strip_suffix(".dgz")?.parse().ok()
}

fn read_all(path: &Path) -> Result<(ChunkHeader, Vec<Game>), String> {
    let ch = Chunk::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let games: Vec<Game> = ch.games().collect::<Result<_, _>>().map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((ch.header, games))
}

fn compare<R: Rules>(src: &[Game], dst: &[Game]) -> Result<(Pairs, u64), String> {
    if src.len() != dst.len() {
        return Err("другое число партий".into());
    }
    let (mut pairs, mut samples) = (Pairs::default(), 0u64);
    for (a, b) in src.iter().zip(dst) {
        if !same_structure(a, b) {
            return Err("партии не совпадают по ходам или исходу".into());
        }
        let ps = positions::<R>(a)?;
        for (i, (x, y)) in a.plies.iter().zip(&b.plies).enumerate() {
            if !x.random && is_sample(&ps[i]) {
                samples += 1;
                pairs.add(x.score, y.score);
            }
        }
    }
    Ok((pairs, samples))
}

struct Run<'a> {
    p: &'a RelabelParams,
    files: Vec<PathBuf>,
}

impl<'a, R: Rules> EvalRun<R> for Run<'a> {
    type Out = Result<RelabelReport, String>;
    fn run<E: Evaluator<R>>(self, eval: &E, eval_id: u32) -> Self::Out {
        run_r::<R, E>(self.p, &self.files, eval, eval_id)
    }
}

fn run_r<R: Rules, E: Evaluator<R>>(p: &RelabelParams, files: &[PathBuf], eval: &E, eval_id: u32) -> Result<RelabelReport, String> {
    let t0 = std::time::Instant::now();
    let man = Manifest { variant: R::NAME.to_string(), forced: R::FORCED, eval_id, nodes: p.nodes, samples_only: p.samples_only, no_egdb: p.no_egdb };
    let mpath = p.out.join("relabel.json");
    std::fs::create_dir_all(&p.out).map_err(|e| format!("{}: {e}", p.out.display()))?;
    if mpath.exists() {
        let old: Manifest = serde_json::from_slice(&std::fs::read(&mpath).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", mpath.display()))?;
        if old != man {
            return Err(format!("{}: каталог переоценён с другими параметрами ({old:?}); нужен другой --out", mpath.display()));
        }
    } else {
        let js = serde_json::to_vec_pretty(&man).map_err(|e| e.to_string())?;
        crate::datadir::write_atomic(&mpath, &js).map_err(|e| format!("{}: {e}", mpath.display()))?;
    }
    let mut rep = RelabelReport { nodes: p.nodes, eval_id, ..Default::default() };
    let mut dbs: HashMap<u32, Arc<Egdb<R>>> = HashMap::new();
    for src in files {
        let (h, games) = read_all(src)?;
        if h.variant != R::NAME || h.forced != R::FORCED {
            return Err(format!("{}: кусок другой конфигурации ({} forced={})", src.display(), h.variant, h.forced));
        }
        if h.eval_id != eval_id && !p.any_eval {
            return Err(format!("{}: кусок сыгран оценкой {:#010x}, а --eval даёт {:#010x} (другой оценкой — только с --any-eval)", src.display(), h.eval_id, eval_id));
        }
        rep.chunks += 1;
        rep.games += games.len() as u64;
        let dst = p.out.join(src.file_name().ok_or("имя куска")?);
        if dst.exists() {
            let (dh, dg) = read_all(&dst).map_err(|e| format!("готовый кусок не читается ({e}); уберите его вручную"))?;
            if dh.chunk != h.chunk || dh.games != h.games || dh.body_len != h.body_len {
                return Err(format!("{}: заголовок не совпадает с исходным куском", dst.display()));
            }
            let (pairs, samples) = compare::<R>(&games, &dg).map_err(|e| format!("{}: {e}", dst.display()))?;
            rep.pairs.merge(&pairs);
            rep.samples += samples;
            rep.chunks_skipped += 1;
            if p.verbose {
                eprintln!("  {}: уже переоценён — пропуск", dst.display());
            }
            continue;
        }
        let db: Option<Arc<Egdb<R>>> = if h.egdb_max_pieces == 0 || p.no_egdb {
            None
        } else if let Some(d) = dbs.get(&h.egdb_max_pieces) {
            Some(d.clone())
        } else {
            let dir = p.egdb.as_ref().ok_or_else(|| format!("{}: партии играны с базами до {} фигур — нужен --egdb <каталог баз> (или --no-egdb)", src.display(), h.egdb_max_pieces))?;
            let d = Egdb::<R>::open_ro_capped(dir, h.egdb_max_pieces)?;
            if d.max_pieces() != h.egdb_max_pieces {
                return Err(format!("{}: в базах есть только до {} фигур, а партии играны с базами до {}", dir.display(), d.max_pieces(), h.egdb_max_pieces));
            }
            let d = Arc::new(d);
            dbs.insert(h.egdb_max_pieces, d.clone());
            Some(d)
        };
        let hash_mb = (h.hash_mb as usize).max(1);
        let n = games.len();
        let threads = if p.threads == 0 { crate::search::default_threads() } else { p.threads }.clamp(1, n.max(1));
        let next = AtomicUsize::new(0);
        type Slot = Option<Result<(Game, GameAcc), String>>;
        let out: Mutex<Vec<Slot>> = Mutex::new((0..n).map(|_| None).collect());
        let (gr, dbr) = (&games, db.as_deref());
        std::thread::scope(|sc| -> Result<(), String> {
            let mut hs = Vec::new();
            for _ in 0..threads {
                let h = std::thread::Builder::new()
                    .stack_size(128 << 20)
                    .spawn_scoped(sc, || loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            return;
                        }
                        let t = std::time::Instant::now();
                        let mut acc = GameAccPub::default();
                        let r = relabel_game::<R, E>(&gr[i], eval, dbr, p.nodes, hash_mb, p.samples_only, &mut acc).map(|g| {
                            acc.0.secs = t.elapsed().as_secs_f64();
                            (g, acc.0)
                        });
                        out.lock().unwrap()[i] = Some(r);
                    })
                    .map_err(|e| e.to_string())?;
                hs.push(h);
            }
            for h in hs {
                h.join().map_err(|_| "поток переоценки завершился аварийно".to_string())?;
            }
            Ok(())
        })?;
        let mut body = Vec::new();
        let mut new_games = Vec::with_capacity(n);
        for (i, slot) in out.into_inner().unwrap().into_iter().enumerate() {
            let (g, a) = slot.ok_or("партия не обработана")?.map_err(|e| format!("{}: партия {i}: {e}", src.display()))?;
            g.encode(&mut body);
            rep.searched += a.searched;
            rep.same_exact += a.same;
            rep.samples += a.samples;
            rep.thread_secs += a.secs;
            rep.pairs.merge(&a.pairs);
            new_games.push(g);
        }
        if body.len() as u64 != h.body_len {
            return Err(format!("{}: длина тела после переоценки {} != {}", src.display(), body.len(), h.body_len));
        }
        let mut nh = h.clone();
        let bytes = assemble_chunk(&mut nh, &body)?;
        let back = Chunk::from_bytes(&bytes)?;
        let bg: Vec<Game> = back.games().collect::<Result<_, _>>()?;
        if back.header.games != h.games || bg.len() != games.len() || !games.iter().zip(&bg).all(|(a, b)| same_structure(a, b)) || bg != new_games {
            return Err(format!("{}: проверка целостности не прошла", src.display()));
        }
        crate::datadir::write_atomic(&dst, &bytes).map_err(|e| format!("{}: {e}", dst.display()))?;
        rep.chunks_done += 1;
        if p.verbose {
            eprintln!("  {}: готово ({} партий, всего поисков {}, {:.1}с)", dst.display(), n, rep.searched, t0.elapsed().as_secs_f64());
        }
    }
    rep.wall_secs = t0.elapsed().as_secs_f64();
    Ok(rep)
}

fn relabel_cfg<R: Rules>(p: &RelabelParams, files: Vec<PathBuf>) -> Result<RelabelReport, String> {
    with_eval::<R, _>(&p.eval, Run { p, files })?
}

pub fn relabel(cfg: &GameConfig, p: &RelabelParams) -> Result<RelabelReport, String> {
    if p.nodes == 0 {
        return Err("--nodes должно быть > 0".into());
    }
    if p.from == p.out || (p.out.exists() && p.from.canonicalize().ok() == p.out.canonicalize().ok()) {
        return Err("--out должен отличаться от --from".into());
    }
    let mut files: Vec<PathBuf> = super::dataset::list_chunks(&p.from)?;
    if let Some((a, b)) = p.chunks {
        files.retain(|f| chunk_no(f).is_some_and(|c| c >= a && c < b));
    }
    if files.is_empty() {
        return Err("нет кусков в заданном диапазоне".into());
    }
    dispatch_cfg!(cfg, relabel_cfg(p, files))
}

pub fn parse_range(s: &str) -> Result<(u32, u32), String> {
    let bad = || format!("неверный диапазон кусков '{s}' (нужно A..B или A)");
    match s.split_once("..") {
        Some((a, b)) => {
            let (a, b): (u32, u32) = (a.trim().parse().map_err(|_| bad())?, b.trim().parse().map_err(|_| bad())?);
            if b <= a {
                return Err(bad());
            }
            Ok((a, b))
        }
        None => {
            let a: u32 = s.trim().parse().map_err(|_| bad())?;
            Ok((a, a + 1))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LabelsSource {
    pub dir: String,
    pub nodes: u64,
    pub eval_id: u32,
    pub samples_only: bool,
    pub no_egdb: bool,
    pub chunks: usize,
}

pub fn labels_files(files: &[PathBuf], dir: &Path, variant: &str, forced: bool) -> Result<(Vec<PathBuf>, LabelsSource), String> {
    let mpath = dir.join("relabel.json");
    let man: Manifest = serde_json::from_slice(&std::fs::read(&mpath).map_err(|e| format!("{}: {e} (каталог меток — вывод runner learn relabel)", mpath.display()))?)
        .map_err(|e| format!("{}: {e}", mpath.display()))?;
    if man.variant != variant || man.forced != forced {
        return Err(format!("{}: переоценка другой конфигурации ({} forced={})", mpath.display(), man.variant, man.forced));
    }
    let mut out = Vec::with_capacity(files.len());
    let mut missing = Vec::new();
    for f in files {
        let name = f.file_name().ok_or_else(|| format!("{}: нет имени куска", f.display()))?;
        let dst = dir.join(name);
        if !dst.is_file() {
            missing.push(name.to_string_lossy().to_string());
            continue;
        }
        let mut a = crate::selfplay::format::read_header(f).map_err(|e| format!("{}: {e}", f.display()))?;
        let b = crate::selfplay::format::read_header(&dst).map_err(|e| format!("{}: {e}", dst.display()))?;
        a.body_crc = b.body_crc;
        if a != b {
            return Err(format!("{}: заголовок переоценённого куска не совпадает с {} (кусок другого набора)", dst.display(), f.display()));
        }
        out.push(dst);
    }
    if !missing.is_empty() {
        let head: Vec<&str> = missing.iter().take(5).map(|s| s.as_str()).collect();
        return Err(format!("{}: нет переоценки для {} из {} кусков набора (например, {}); переоцените их или сузьте набор", dir.display(), missing.len(), files.len(), head.join(", ")));
    }
    let src = LabelsSource { dir: dir.display().to_string(), nodes: man.nodes, eval_id: man.eval_id, samples_only: man.samples_only, no_egdb: man.no_egdb, chunks: out.len() };
    Ok((out, src))
}
