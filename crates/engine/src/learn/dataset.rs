
use super::arch::{Arch, PHASE_ONE};
use super::refeval::{Features, RefEval};
use crate::egdb;
use crate::eval::{Evaluator, SimpleEval};
use crate::position::Side;
use crate::rules::Rules;
use crate::selfplay::{self, Chunk, Game};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const SCORE_CLAMP: i32 = 20000;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn is_val_game(chunk: u32, game: u32, seed: u64, frac: f64) -> bool {
    let h = splitmix64(splitmix64(seed ^ (chunk as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) ^ game as u64);
    ((h >> 11) as f64) / ((1u64 << 53) as f64) < frac
}

pub fn list_chunks(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut v: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).map(|n| n.starts_with("games-") && n.ends_with(".dgz")).unwrap_or(false))
        .collect();
    crate::selfplay::sort_chunk_paths(&mut v);
    if v.is_empty() {
        return Err(format!("{}: нет файлов games-*.dgz", dir.display()));
    }
    Ok(v)
}

#[derive(Clone, Copy, Debug)]
pub struct Labels {
    pub t: f32,
    pub r: f32,
    pub sp: f32,
    pub mat: i16,
    pub simple: i16,
    pub egdb: bool,
}

#[derive(Default, Clone)]
pub struct Columns {
    pub np: usize,
    pub n: usize,
    pub idx: Vec<u32>,
    pub st0: Vec<u8>,
    pub w0: Vec<u8>,
    pub gptr: Vec<u32>,
    pub gidx: Vec<u16>,
    pub gval: Vec<i16>,
    pub t: Vec<f32>,
    pub r: Vec<f32>,
    pub sp: Vec<f32>,
    pub mat: Vec<i16>,
    pub simple: Vec<i16>,
    pub w: Vec<f32>,
    key: Vec<u64>,
    cnt: Vec<u32>,
    flag: Vec<u8>,
}

impl Columns {
    pub fn new(np: usize) -> Columns {
        Columns { np, gptr: vec![0], ..Default::default() }
    }

    pub(crate) fn push(&mut self, key: u64, f: &Features, l: &Labels) {
        for view in &f.table_indices {
            for &(_, i) in view {
                self.idx.push(i as u32);
            }
        }
        debug_assert_eq!(self.idx.len(), (self.n + 1) * self.np);
        self.st0.push(f.stages[0].0 as u8);
        self.w0.push(f.stages[0].1 as u8);
        if f.stages.len() == 2 {
            debug_assert!(f.stages[1].0 == f.stages[0].0 + 1 && f.stages[1].1 == PHASE_ONE - f.stages[0].1);
        }
        for (j, &g) in f.globals.iter().enumerate() {
            if g != 0 {
                self.gidx.push(j as u16);
                self.gval.push(g.clamp(-32768, 32767) as i16);
            }
        }
        self.gptr.push(self.gidx.len() as u32);
        self.t.push(l.t);
        self.r.push(l.r);
        self.sp.push(l.sp);
        self.mat.push(l.mat);
        self.simple.push(l.simple);
        self.key.push(key);
        self.cnt.push(1);
        self.flag.push(l.egdb as u8);
        self.n += 1;
    }

    fn copy_row(&self, i: usize, dst: &mut Columns) {
        dst.idx.extend_from_slice(&self.idx[i * self.np..(i + 1) * self.np]);
        dst.st0.push(self.st0[i]);
        dst.w0.push(self.w0[i]);
        let (a, b) = (self.gptr[i] as usize, self.gptr[i + 1] as usize);
        dst.gidx.extend_from_slice(&self.gidx[a..b]);
        dst.gval.extend_from_slice(&self.gval[a..b]);
        dst.gptr.push(dst.gidx.len() as u32);
        dst.t.push(self.t[i]);
        dst.r.push(self.r[i]);
        dst.sp.push(self.sp[i]);
        dst.mat.push(self.mat[i]);
        dst.simple.push(self.simple[i]);
        dst.key.push(self.key[i]);
        dst.cnt.push(self.cnt[i]);
        dst.flag.push(self.flag[i]);
        dst.n += 1;
    }

    pub(crate) fn clear(&mut self) {
        self.n = 0;
        self.idx.clear();
        self.st0.clear();
        self.w0.clear();
        self.gptr.clear();
        self.gptr.push(0);
        self.gidx.clear();
        self.gval.clear();
        self.t.clear();
        self.r.clear();
        self.sp.clear();
        self.mat.clear();
        self.simple.clear();
        self.w.clear();
        self.key.clear();
        self.cnt.clear();
        self.flag.clear();
    }

    pub(crate) fn reserve(&mut self, rows: usize, g: usize) {
        self.idx.reserve(rows * self.np);
        for v in [&mut self.st0, &mut self.w0, &mut self.flag] {
            v.reserve(rows);
        }
        self.gptr.reserve(rows);
        self.gidx.reserve(g);
        self.gval.reserve(g);
        for v in [&mut self.t, &mut self.r, &mut self.sp, &mut self.w] {
            v.reserve(rows);
        }
        self.mat.reserve(rows);
        self.simple.reserve(rows);
        self.key.reserve(rows);
        self.cnt.reserve(rows);
    }

    pub(crate) fn set_unit_weights(&mut self, egdb_weight: f32) {
        self.w.clear();
        let f = &self.flag;
        self.w.extend((0..self.n).map(|i| if f[i] != 0 { egdb_weight } else { 1.0 }));
    }

    pub(crate) fn nnz_globals(&self) -> usize {
        self.gidx.len()
    }

    pub(crate) fn append(&mut self, src: &Columns) {
        for i in 0..src.n {
            src.copy_row(i, self);
        }
    }

    pub(crate) fn finalize(&mut self, cap: u32, egdb_weight: f32) {
        let cap = cap.max(1);
        self.w = (0..self.n).map(|i| self.cnt[i].min(cap) as f32 * if self.flag[i] != 0 { egdb_weight } else { 1.0 }).collect();
        self.key = Vec::new();
        self.cnt = Vec::new();
        self.flag = Vec::new();
        macro_rules! shrink {
            ($($f:ident),*) => { $( self.$f.shrink_to_fit(); )* };
        }
        shrink!(idx, st0, w0, gptr, gidx, gval, t, r, sp, mat, simple, w);
    }

    pub fn mem_bytes(&self) -> usize {
        self.idx.capacity() * 4
            + self.st0.capacity()
            + self.w0.capacity()
            + self.gptr.capacity() * 4
            + self.gidx.capacity() * 2
            + self.gval.capacity() * 2
            + (self.t.capacity() + self.r.capacity() + self.sp.capacity() + self.w.capacity()) * 4
            + (self.mat.capacity() + self.simple.capacity()) * 2
    }

    pub fn sum_w(&self) -> f64 {
        self.w.iter().map(|&x| x as f64).sum()
    }

    pub fn features(&self, i: usize, pat_table: &[usize], g_row: usize, stages: usize) -> Features {
        let p = pat_table.len();
        let row = &self.idx[i * self.np..(i + 1) * self.np];
        let table_indices = row.chunks(p.max(1)).map(|c| c.iter().zip(pat_table).map(|(&x, &t)| (t, x as usize)).collect()).collect();
        let (st, w0) = (self.st0[i] as usize, self.w0[i] as i32);
        let mut st_list = vec![(st, w0)];
        if w0 < PHASE_ONE && st + 1 < stages {
            st_list.push((st + 1, PHASE_ONE - w0));
        }
        let mut globals = vec![0i32; g_row];
        for q in self.gptr[i] as usize..self.gptr[i + 1] as usize {
            globals[self.gidx[q] as usize] = self.gval[q] as i32;
        }
        Features { phase_value: 0, stages: st_list, table_indices, globals }
    }

    pub fn checksum(&self) -> u64 {
        let mut h = 0xCBF2_9CE4_8422_2325u64 ^ self.n as u64;
        for &x in &self.idx {
            h = (h ^ x as u64).wrapping_mul(0x0000_0100_0000_01B3);
        }
        for i in 0..self.n {
            h = (h ^ self.t[i].to_bits() as u64).wrapping_mul(0x0000_0100_0000_01B3);
            h = (h ^ self.w[i].to_bits() as u64).wrapping_mul(0x0000_0100_0000_01B3);
        }
        h
    }
}

#[derive(Default)]
pub(crate) struct Builder {
    pub(crate) cols: Columns,
    map: HashMap<u64, u32>,
    pub(crate) raw: u64,
}

impl Builder {
    pub(crate) fn new(np: usize) -> Builder {
        Builder { cols: Columns::new(np), ..Default::default() }
    }
    pub(crate) fn peak_bytes(&self) -> u64 {
        let c = &self.cols;
        let cols = c.idx.len() * 4 + c.st0.len() + c.w0.len() + c.gptr.len() * 4 + (c.gidx.len() + c.gval.len()) * 2 + (c.t.len() + c.r.len() + c.sp.len() + c.w.len()) * 4 + (c.mat.len() + c.simple.len()) * 2;
        (cols + c.key.len() * 8 + c.cnt.len() * 4 + c.flag.len() + self.map.len() * 16) as u64
    }
    pub(crate) fn add(&mut self, src: &Columns) {
        for i in 0..src.n {
            self.raw += 1;
            match self.map.get(&src.key[i]) {
                Some(&j) => {
                    let j = j as usize;
                    let c = self.cols.cnt[j] as f32;
                    let a = 1.0 / (c + 1.0);
                    self.cols.t[j] += (src.t[i] - self.cols.t[j]) * a;
                    self.cols.r[j] += (src.r[i] - self.cols.r[j]) * a;
                    self.cols.sp[j] += (src.sp[i] - self.cols.sp[j]) * a;
                    self.cols.cnt[j] += 1;
                }
                None => {
                    self.map.insert(src.key[i], self.cols.n as u32);
                    src.copy_row(i, &mut self.cols);
                }
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct Busy {
    pub now: std::sync::atomic::AtomicUsize,
    pub ns: std::sync::atomic::AtomicU64,
    pub peak: std::sync::atomic::AtomicUsize,
}

pub(crate) struct BusyGuard<'a>(&'a Busy, std::time::Instant);
impl<'a> BusyGuard<'a> {
    pub(crate) fn new(c: &'a Busy) -> Self {
        let n = c.now.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        c.peak.fetch_max(n, std::sync::atomic::Ordering::Relaxed);
        BusyGuard(c, std::time::Instant::now())
    }
}
impl Drop for BusyGuard<'_> {
    fn drop(&mut self) {
        self.0.ns.fetch_add(self.1.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        self.0.now.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

#[derive(Clone, Debug)]
pub struct BuildParams {
    pub val_frac: f64,
    pub split_seed: u64,
    pub dup_cap: u32,
    pub lambda: f64,
    pub k: f64,
    pub egdb_weight: f64,
    pub use_egdb: bool,
    pub verbose: bool,
    pub busy: Option<std::sync::Arc<Busy>>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct BuildStats {
    pub chunks: usize,
    pub games_train: u64,
    pub games_val: u64,
    pub draws_train: u64,
    pub draws_val: u64,
    pub raw_train: u64,
    pub raw_val: u64,
    pub unique_train: u64,
    pub unique_val: u64,
    pub egdb_unique: u64,
    pub mem_bytes: u64,
    pub secs: f64,
}

pub struct Built {
    pub train: Columns,
    pub val: Columns,
    pub stats: BuildStats,
    pub pat_table: Vec<usize>,
}

#[derive(Default)]
struct Group {
    train: Columns,
    val: Columns,
    gt: u64,
    gv: u64,
    dt: u64,
    dv: u64,
}

pub(crate) fn pos_key<R: Rules>(pos: &crate::position::Position<R>) -> u64 {
    let mut h = splitmix64(pos.white ^ 0x1234_5678_9ABC_DEF0);
    h = splitmix64(h ^ pos.black);
    h = splitmix64(h ^ pos.kings.rotate_left(17));
    splitmix64(h ^ (pos.side == Side::Black) as u64)
}

pub(crate) fn draws_practical() -> bool {
    std::env::var("RUNNER_EGDB_DRAWS").map(|v| v == "practical").unwrap_or(false)
}

pub(crate) fn labels_of<R: Rules>(pos: &crate::position::Position<R>, s: &selfplay::Sample, lambda: f64, k: f64, draws_practical: bool, se: &SimpleEval) -> Labels {
    let r = (s.result_stm as f32 + 1.0) * 0.5;
    let sc = (s.score as i32).clamp(-SCORE_CLAMP, SCORE_CLAMP);
    let sp = sigmoid(sc as f64 / k) as f32;
    let (t, eg) = match s.egdb {
        Some(e) if !(draws_practical && e == 0) => (if e > 0 { 1.0 } else if e == 0 { 0.5 } else { 0.0 }, true),
        _ => ((lambda * r as f64 + (1.0 - lambda) * sp as f64) as f32, false),
    };
    let (own, opp) = if pos.side == Side::White { (pos.white, pos.black) } else { (pos.black, pos.white) };
    let men = (own & !pos.kings).count_ones() as i32 - (opp & !pos.kings).count_ones() as i32;
    let kg = (own & pos.kings).count_ones() as i32 - (opp & pos.kings).count_ones() as i32;
    let mat = (men * 100 + kg * R::KING_VALUE).clamp(-30000, 30000) as i16;
    let simple = se.eval_fresh(pos).clamp(-SCORE_CLAMP, SCORE_CLAMP) as i16;
    Labels { t, r, sp, mat, simple, egdb: eg }
}

pub fn build_dataset<R: Rules>(dir: &Path, arch: &Arch, bp: &BuildParams) -> Result<Built, String> {
    build_dataset_files::<R>(&list_chunks(dir)?, arch, bp, 0)
}

pub fn build_dataset_files<R: Rules>(files: &[PathBuf], arch: &Arch, bp: &BuildParams, mem_limit: u64) -> Result<Built, String> {
    let t0 = std::time::Instant::now();
    if files.is_empty() {
        return Err("нет файлов games-*.dgz".into());
    }
    let re = RefEval::<R>::new(super::weights::Model::zeros(arch.clone())?)?;
    let se = SimpleEval::new::<R>();
    let db = if bp.use_egdb { egdb::shared::<R>() } else { None };
    let draws_practical = draws_practical();
    let dbr = db.as_deref();
    let pat_table: Vec<usize> = arch.patterns.iter().map(|p| p.table).collect();
    let np = pat_table.len() * arch.views.len();
    let (mut tr, mut va) = (Builder::new(np), Builder::new(np));
    let mut st = BuildStats::default();
    let busy = bp.busy.clone();
    let parse_one = |path: &PathBuf| -> Result<(Vec<Group>, usize), String> {
        let (games, cno) = {
            let _b = busy.as_deref().map(BusyGuard::new);
            let ch = match Chunk::read(path) {
                Ok(c) => c,
                Err(e) => return Err(format!("{}: {e}", path.display())),
            };
            if ch.header.variant != R::NAME || ch.header.forced != R::FORCED {
                return Err(format!("{}: кусок другой конфигурации ({} forced={})", path.display(), ch.header.variant, ch.header.forced));
            }
            let games: Vec<Game> = ch.games().collect::<Result<_, _>>().map_err(|e| format!("{}: {e}", path.display()))?;
            (games, ch.header.chunk)
        };
        let groups: Vec<Group> = games
            .par_chunks(32)
            .enumerate()
            .map(|(gi, gs)| -> Result<Group, String> {
                let _b = busy.as_deref().map(BusyGuard::new);
                let mut grp = Group { train: Columns::new(np), val: Columns::new(np), ..Default::default() };
                for (j, g) in gs.iter().enumerate() {
                    let val = is_val_game(cno, (gi * 32 + j) as u32, bp.split_seed, bp.val_frac);
                    let draw = g.result == 0;
                    if val {
                        grp.gv += 1;
                        grp.dv += draw as u64;
                    } else {
                        grp.gt += 1;
                        grp.dt += draw as u64;
                    }
                    let tgt = if val { &mut grp.val } else { &mut grp.train };
                    selfplay::for_each_sample::<R>(g, dbr, |pos, s| {
                        let f = re.features(pos);
                        let l = labels_of(pos, s, bp.lambda, bp.k, draws_practical, &se);
                        tgt.push(pos_key(pos), &f, &l);
                    })
                    .map_err(|e| format!("{}: партия {}: {e}", path.display(), gi * 32 + j))?;
                }
                Ok(grp)
            })
            .collect::<Result<_, _>>()?;
        Ok((groups, games.len()))
    };
    let window = rayon::current_num_threads().max(1);
    for (wi, win) in files.chunks(window).enumerate() {
        let parsed: Vec<Result<(Vec<Group>, usize), String>> = win.par_iter().map(parse_one).collect();
        for (j, r) in parsed.into_iter().enumerate() {
        let ci = wi * window + j;
        let (groups, n_games) = r?;
        for g in &groups {
            tr.add(&g.train);
            va.add(&g.val);
            st.games_train += g.gt;
            st.games_val += g.gv;
            st.draws_train += g.dt;
            st.draws_val += g.dv;
        }
        st.chunks += 1;
        if mem_limit > 0 {
            let used = tr.peak_bytes() + va.peak_bytes();
            if used > mem_limit {
                return Err(format!("{}: после {} из {} кусков занято {:.2} ГиБ, предел {:.2} ГиБ", super::memplan::MEM_LIMIT_ERR, ci + 1, files.len(), used as f64 / (1u64 << 30) as f64, mem_limit as f64 / (1u64 << 30) as f64));
            }
        }
        if bp.verbose {
            eprintln!(
                "  кусок {}/{}: {} партий; выборок уникальных: обучение {}, проверка {}",
                ci + 1,
                files.len(),
                n_games,
                tr.cols.n,
                va.cols.n
            );
        }
        }
    }
    st.raw_train = tr.raw;
    st.raw_val = va.raw;
    let (mut train, mut val) = (tr.cols, va.cols);
    let egdb_unique = train.flag.iter().filter(|&&f| f != 0).count() as u64;
    train.finalize(bp.dup_cap, bp.egdb_weight as f32);
    val.finalize(bp.dup_cap, bp.egdb_weight as f32);
    st.egdb_unique = egdb_unique;
    st.unique_train = train.n as u64;
    st.unique_val = val.n as u64;
    st.mem_bytes = (train.mem_bytes() + val.mem_bytes()) as u64;
    st.secs = t0.elapsed().as_secs_f64();
    Ok(Built { train, val, stats: st, pat_table })
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FitKReport {
    pub k: f64,
    pub mse: f64,
    pub mse_const: f64,
    pub samples: u64,
    pub games: u64,
    pub skipped_chunks: usize,
    pub table: Vec<(f64, f64)>,
}

fn mse_at(hist: &[(i32, [u64; 3])], k: f64) -> f64 {
    let (mut s, mut n) = (0.0f64, 0.0f64);
    for &(sc, c) in hist {
        let p = sigmoid(sc as f64 / k);
        let (w, d, l) = (c[0] as f64, c[1] as f64, c[2] as f64);
        s += (w + d + l) * p * p - 2.0 * p * (w + 0.5 * d) + (w + 0.25 * d);
        n += w + d + l;
    }
    s / n.max(1.0)
}

pub fn fit_k_dir<R: Rules>(dir: &Path, any_eval: bool, verbose: bool) -> Result<FitKReport, String> {
    let files = list_chunks(dir)?;
    let mut total = vec![[0u64; 3]; 2 * SCORE_CLAMP as usize + 1];
    let (mut skipped, mut games) = (0usize, 0u64);
    for path in &files {
        let ch = Chunk::read(path)?;
        if ch.header.variant != R::NAME || ch.header.forced != R::FORCED {
            return Err(format!("{}: кусок другой конфигурации", path.display()));
        }
        if ch.header.eval_id != 0 && !any_eval {
            skipped += 1;
            continue;
        }
        let gl: Vec<Game> = ch.games().collect::<Result<_, _>>().map_err(|e| format!("{}: {e}", path.display()))?;
        games += gl.len() as u64;
        let part: Vec<Vec<(u32, u8)>> = gl
            .par_chunks(64)
            .map(|gs| {
                let mut v = Vec::new();
                for g in gs {
                    let _ = selfplay::for_each_sample::<R>(g, None, |_, s| {
                        let sc = (s.score as i32).clamp(-SCORE_CLAMP, SCORE_CLAMP) + SCORE_CLAMP;
                        v.push((sc as u32, (1 - s.result_stm) as u8));
                    });
                }
                v
            })
            .collect();
        for v in part {
            for (sc, o) in v {
                total[sc as usize][o as usize] += 1;
            }
        }
        if verbose {
            eprintln!("  {}: учтено", path.display());
        }
    }
    let hist: Vec<(i32, [u64; 3])> = total.iter().enumerate().filter(|(_, c)| c.iter().sum::<u64>() > 0).map(|(i, c)| (i as i32 - SCORE_CLAMP, *c)).collect();
    let n: u64 = hist.iter().map(|(_, c)| c.iter().sum::<u64>()).sum();
    if n == 0 {
        return Err("нет выборок (все куски пропущены или пусты)".into());
    }
    let (lo, hi) = (10f64.ln(), 10000f64.ln());
    let grid: Vec<f64> = (0..48).map(|i| (lo + (hi - lo) * i as f64 / 47.0).exp()).collect();
    let errs: Vec<f64> = grid.iter().map(|&k| mse_at(&hist, k)).collect();
    let b = (0..grid.len()).min_by(|&a, &c| errs[a].partial_cmp(&errs[c]).unwrap()).unwrap();
    let (mut a, mut c) = (grid[b.saturating_sub(1)].ln(), grid[(b + 1).min(grid.len() - 1)].ln());
    let phi = 0.618_033_988_749_895;
    let (mut x1, mut x2) = (c - phi * (c - a), a + phi * (c - a));
    let (mut f1, mut f2) = (mse_at(&hist, x1.exp()), mse_at(&hist, x2.exp()));
    for _ in 0..60 {
        if f1 < f2 {
            c = x2;
            x2 = x1;
            f2 = f1;
            x1 = c - phi * (c - a);
            f1 = mse_at(&hist, x1.exp());
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + phi * (c - a);
            f2 = mse_at(&hist, x2.exp());
        }
    }
    let k = ((a + c) * 0.5).exp();
    let (w, d): (u64, u64) = hist.iter().fold((0, 0), |s, (_, c)| (s.0 + c[0], s.1 + c[1]));
    let mean = (w as f64 + 0.5 * d as f64) / n as f64;
    let l: u64 = hist.iter().map(|(_, c)| c[2]).sum();
    let var = (w as f64 + 0.25 * d as f64) / n as f64 - mean * mean;
    let _ = l;
    let table = [50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1200.0].iter().map(|&kk| (kk, mse_at(&hist, kk))).collect();
    Ok(FitKReport { k, mse: mse_at(&hist, k), mse_const: var, samples: n, games, skipped_chunks: skipped, table })
}
