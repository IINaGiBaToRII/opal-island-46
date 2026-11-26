
pub mod extract;
pub mod format;
pub mod play;
pub mod stats;

pub use extract::{for_each_sample, replay, Sample};
pub use format::{Chunk, ChunkHeader, EndReason, Game, PlyRec};
pub use stats::{stats_for_paths, stats_for_paths_with, Stats};

use crate::datadir::{self, Kind};
use crate::egdb::Egdb;
use crate::eval::Evaluator;
use crate::learn::select::{with_eval, EvalRun, EvalSpec};
use crate::facade::{GameConfig, VariantId};
use crate::rules::{American, Brazilian, Czech, International, Italian, Rules, Russian, Spanish, Thai, Turkish};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

#[derive(Clone, Debug)]
pub struct SelfplayParams {
    pub games_per_chunk: u32,
    pub nodes: u64,
    pub seed: u64,
    pub book_min: u32,
    pub book_max: u32,
    pub adj_score: i32,
    pub adj_plies: u32,
    pub max_plies: u32,
    pub hash_mb: usize,
    pub threads: usize,
    pub use_egdb: bool,
    pub created_unix: Option<u64>,
    pub eval: EvalSpec,
    pub db_play_draws: bool,
    pub rnd_permille: u32,
    pub rnd_min: u32,
    pub rnd_max: u32,
}

pub fn prob_permille(s: &str) -> Result<u32, String> {
    let bad = || format!("вероятность случайного хода: число 0..1, не больше трёх знаков после точки, а не «{s}»");
    let (int, frac) = s.split_once('.').unwrap_or((s, ""));
    if (int.is_empty() && frac.is_empty()) || int.len() > 1 || frac.len() > 3 || !int.bytes().chain(frac.bytes()).all(|b| b.is_ascii_digit()) {
        return Err(bad());
    }
    let i: u32 = if int.is_empty() { 0 } else { int.parse().map_err(|_| bad())? };
    let f: u32 = if frac.is_empty() { 0 } else { format!("{frac:0<3}").parse().map_err(|_| bad())? };
    let v = i * 1000 + f;
    if v > 1000 {
        return Err(bad());
    }
    Ok(v)
}

impl Default for SelfplayParams {
    fn default() -> Self {
        SelfplayParams {
            games_per_chunk: 1000,
            nodes: 10_000,
            seed: 1,
            book_min: 4,
            book_max: 8,
            adj_score: 700,
            adj_plies: 6,
            max_plies: 300,
            hash_mb: 16,
            threads: 0,
            use_egdb: true,
            created_unix: None,
            eval: EvalSpec::Simple,
            db_play_draws: false,
            rnd_permille: 0,
            rnd_min: 0,
            rnd_max: 0,
        }
    }
}

pub struct SelfplayCtl {
    pub cancel: AtomicBool,
    pub progress: AtomicU64,
    pub duty: AtomicU32,
    pub qos: AtomicU32,
    pub active: AtomicUsize,
}

impl Default for SelfplayCtl {
    fn default() -> Self {
        SelfplayCtl { cancel: AtomicBool::new(false), progress: AtomicU64::new(0), duty: AtomicU32::new(100), qos: AtomicU32::new(0), active: AtomicUsize::new(0) }
    }
}

struct LiveGuard<'a>(&'a AtomicUsize);
impl<'a> LiveGuard<'a> {
    fn new(c: &'a AtomicUsize) -> Self {
        c.fetch_add(1, Ordering::Relaxed);
        LiveGuard(c)
    }
}
impl Drop for LiveGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

impl SelfplayCtl {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChunkReport {
    pub path: PathBuf,
    pub chunk: u32,
    pub games: u32,
    pub plies: u64,
    pub white_wins: u32,
    pub draws: u32,
    pub black_wins: u32,
    pub bytes: u64,
    pub secs: f64,
    pub skipped: bool,
    pub cancelled: bool,
}

pub fn train_dir(root: &Path, cfg: &GameConfig) -> PathBuf {
    datadir::dir_in(root, cfg.variant.name(), cfg.forced_capture, Kind::Train)
}

pub fn chunk_path(dir: &Path, chunk: u32) -> PathBuf {
    dir.join(format!("games-{chunk:06}.dgz"))
}

pub fn chunk_no(name: &str) -> Option<u32> {
    let s = name.strip_prefix("games-")?.strip_suffix(".dgz")?;
    if s.len() < 6 || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n: u32 = s.parse().ok()?;
    (format!("{n:06}") == s).then_some(n)
}

pub fn sort_chunk_paths(v: &mut [PathBuf]) {
    v.sort_by_cached_key(|p| (p.file_name().and_then(|n| n.to_str()).and_then(chunk_no).map_or(u64::MAX, u64::from), p.clone()));
}

pub fn generate_chunk_with<R: Rules, E: Evaluator<R>>(
    dir: &Path,
    chunk: u32,
    p: &SelfplayParams,
    ctl: &SelfplayCtl,
    eval: &E,
    eval_id: u32,
    egdb: Option<&Egdb<R>>,
) -> Result<ChunkReport, String> {
    let path = chunk_path(dir, chunk);
    let t0 = std::time::Instant::now();
    if path.exists() {
        match Chunk::read(&path) {
            Ok(c) if c.header.variant == R::NAME && c.header.forced == R::FORCED => {
                ctl.progress.fetch_add(c.header.games as u64, Ordering::Relaxed);
                return Ok(ChunkReport { path, chunk, games: c.header.games, bytes: c.file_bytes, skipped: true, ..Default::default() });
            }
            Ok(_) => return Err(format!("{}: кусок другой конфигурации", path.display())),
            Err(_) => {
                datadir::mark_stale(&path).map_err(|e| e.to_string())?;
            }
        }
    }
    let egdb = egdb.filter(|_| p.use_egdb);
    let n = p.games_per_chunk as usize;
    let threads = if p.threads == 0 { crate::search::default_threads() } else { p.threads }.clamp(1, n.max(1));
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<Option<(Vec<u8>, i8, u32)>>> = Mutex::new(vec![None; n]);
    let mut p = p.clone();
    p.book_max = p.book_max.min(255);
    p.book_min = p.book_min.min(p.book_max);
    p.rnd_permille = p.rnd_permille.min(1000);
    if p.rnd_permille == 0 {
        (p.rnd_min, p.rnd_max) = (0, 0);
    }
    p.rnd_max = p.rnd_max.max(p.rnd_min);
    let pr = &p;
    std::thread::scope(|sc| -> Result<(), String> {
        let mut hs = Vec::new();
        for _ in 0..threads {
            let h = std::thread::Builder::new()
                .stack_size(128 << 20)
                .spawn_scoped(sc, || {
                    let _live = LiveGuard::new(&ctl.active);
                    crate::qos::apply(ctl.qos.load(Ordering::Relaxed).min(255) as u8);
                    let mut thr = crate::matchplay::Throttle::new();
                    loop {
                        if ctl.cancel.load(Ordering::Relaxed) {
                            return;
                        }
                        thr.tick(ctl.duty.load(Ordering::Relaxed), &ctl.cancel);
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            return;
                        }
                        let Some(g) = play::play_game::<R, E>(eval, egdb, pr, chunk, i as u32, &ctl.cancel) else { return };
                        let mut b = Vec::with_capacity(g.plies.len() * 5 + 16);
                        g.encode(&mut b);
                        out.lock().unwrap()[i] = Some((b, g.result, g.plies.len() as u32));
                        ctl.progress.fetch_add(1, Ordering::Relaxed);
                    }
                })
                .map_err(|e| e.to_string())?;
            hs.push(h);
        }
        for h in hs {
            h.join().map_err(|_| "поток самоигры завершился аварийно".to_string())?;
        }
        Ok(())
    })?;
    let games = out.into_inner().unwrap();
    if ctl.cancel.load(Ordering::Relaxed) || games.iter().any(|g| g.is_none()) {
        return Ok(ChunkReport { path, chunk, secs: t0.elapsed().as_secs_f64(), cancelled: true, ..Default::default() });
    }
    let mut rep = ChunkReport { path: path.clone(), chunk, games: n as u32, ..Default::default() };
    let mut body = Vec::new();
    for (b, r, np) in games.into_iter().flatten() {
        body.extend_from_slice(&b);
        rep.plies += np as u64;
        match r {
            1 => rep.white_wins += 1,
            0 => rep.draws += 1,
            _ => rep.black_wins += 1,
        }
    }
    let mut h = ChunkHeader {
        variant: R::NAME.to_string(),
        forced: R::FORCED,
        eval_id,
        nodes: p.nodes,
        seed: p.seed,
        chunk,
        games: n as u32,
        created_unix: p.created_unix.unwrap_or_else(datadir::now_secs),
        book_min: p.book_min,
        book_max: p.book_max,
        adj_score: p.adj_score,
        adj_plies: p.adj_plies,
        max_plies: p.max_plies,
        hash_mb: p.hash_mb as u32,
        egdb_max_pieces: egdb.map(|d| d.max_pieces()).unwrap_or(0),
        db_play_draws: p.db_play_draws && egdb.is_some(),
        rnd_permille: p.rnd_permille,
        rnd_min: p.rnd_min,
        rnd_max: p.rnd_max,
        ..Default::default()
    };
    let bytes = format::assemble_chunk(&mut h, &body)?;
    datadir::write_atomic(&path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    rep.bytes = bytes.len() as u64;
    rep.secs = t0.elapsed().as_secs_f64();
    Ok(rep)
}

struct GenRun<'a, R: Rules> {
    dir: &'a Path,
    chunk: u32,
    p: &'a SelfplayParams,
    ctl: &'a SelfplayCtl,
    db: Option<&'a Egdb<R>>,
}

impl<'a, R: Rules> EvalRun<R> for GenRun<'a, R> {
    type Out = Result<ChunkReport, String>;
    fn run<E: Evaluator<R>>(self, eval: &E, eval_id: u32) -> Self::Out {
        generate_chunk_with::<R, _>(self.dir, self.chunk, self.p, self.ctl, eval, eval_id, self.db)
    }
}

fn gen_spec<R: Rules>(dir: &Path, chunk: u32, p: &SelfplayParams, ctl: &SelfplayCtl) -> Result<ChunkReport, String> {
    let db = if p.use_egdb { crate::egdb::shared::<R>() } else { None };
    with_eval::<R, _>(&p.eval, GenRun { dir, chunk, p, ctl, db: db.as_deref() })?
}

macro_rules! dispatch_cfg {
    ($cfg:expr, $f:ident ( $($a:expr),* )) => {
        match ($cfg.variant, $cfg.forced_capture) {
            (VariantId::Russian, true) => $f::<Russian<true>>($($a),*),
            (VariantId::Russian, false) => $f::<Russian<false>>($($a),*),
            (VariantId::Brazilian, true) => $f::<Brazilian<true>>($($a),*),
            (VariantId::Brazilian, false) => $f::<Brazilian<false>>($($a),*),
            (VariantId::International, true) => $f::<International<true>>($($a),*),
            (VariantId::International, false) => $f::<International<false>>($($a),*),
            (VariantId::American, true) => $f::<American<true>>($($a),*),
            (VariantId::American, false) => $f::<American<false>>($($a),*),
            (VariantId::Italian, true) => $f::<Italian<true>>($($a),*),
            (VariantId::Italian, false) => $f::<Italian<false>>($($a),*),
            (VariantId::Spanish, true) => $f::<Spanish<true>>($($a),*),
            (VariantId::Spanish, false) => $f::<Spanish<false>>($($a),*),
            (VariantId::Czech, true) => $f::<Czech<true>>($($a),*),
            (VariantId::Czech, false) => $f::<Czech<false>>($($a),*),
            (VariantId::Thai, true) => $f::<Thai<true>>($($a),*),
            (VariantId::Thai, false) => $f::<Thai<false>>($($a),*),
            (VariantId::Turkish, true) => $f::<Turkish<true>>($($a),*),
            (VariantId::Turkish, false) => $f::<Turkish<false>>($($a),*),
        }
    };
}
pub(crate) use dispatch_cfg;

pub fn params_of_header(h: &ChunkHeader, eval: EvalSpec) -> Result<SelfplayParams, String> {
    if h.egdb_max_pieces != 0 {
        return Err(format!("кусок {} сыгран с базами до {} фигур — повтор без баз невозможен", h.chunk, h.egdb_max_pieces));
    }
    Ok(SelfplayParams {
        games_per_chunk: h.games,
        nodes: h.nodes,
        seed: h.seed,
        book_min: h.book_min,
        book_max: h.book_max,
        adj_score: h.adj_score,
        adj_plies: h.adj_plies,
        max_plies: h.max_plies,
        hash_mb: h.hash_mb as usize,
        threads: 1,
        use_egdb: false,
        created_unix: Some(h.created_unix),
        eval,
        db_play_draws: h.db_play_draws,
        rnd_permille: h.rnd_permille,
        rnd_min: h.rnd_min,
        rnd_max: h.rnd_max,
    })
}

struct ReplayRun<'a> {
    p: &'a SelfplayParams,
    chunk: u32,
    game: u32,
}

impl<'a, R: Rules> EvalRun<R> for ReplayRun<'a> {
    type Out = Result<(Vec<u8>, u32), String>;
    fn run<E: Evaluator<R>>(self, eval: &E, eval_id: u32) -> Self::Out {
        let stop = AtomicBool::new(false);
        let g = play::play_game::<R, E>(eval, None, self.p, self.chunk, self.game, &stop).ok_or("повтор партии прерван")?;
        let mut b = Vec::new();
        g.encode(&mut b);
        Ok((b, eval_id))
    }
}

fn replay_cfg<R: Rules>(p: &SelfplayParams, chunk: u32, game: u32) -> Result<(Vec<u8>, u32), String> {
    with_eval::<R, _>(&p.eval, ReplayRun { p, chunk, game })?
}

pub fn replay_game_bytes(cfg: &GameConfig, h: &ChunkHeader, eval: EvalSpec, game: u32) -> Result<(Vec<u8>, u32), String> {
    let p = params_of_header(h, eval)?;
    if game >= h.games {
        return Err(format!("в куске {} нет партии {game}", h.chunk));
    }
    dispatch_cfg!(cfg, replay_cfg(&p, h.chunk, game))
}

pub fn generate_chunk(cfg: &GameConfig, dir: &Path, chunk: u32, p: &SelfplayParams, ctl: &SelfplayCtl) -> Result<ChunkReport, String> {
    dispatch_cfg!(cfg, gen_spec(dir, chunk, p, ctl))
}
