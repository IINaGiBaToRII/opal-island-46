
use crate::egdb::Egdb;
use crate::eval::Evaluator;
use crate::geometry::Geometry;
use crate::movegen::{generate, generate_captures, has_quiet, ListSink, Move, MoveList, MAX_MOVES};
use crate::position::Position;
use crate::rules::Rules;
use crate::recall::{RecallCtx, RecallEntry};
use crate::timectl::TimeCtl;
use crate::tt::{Tt, BOUND_EXACT, BOUND_LOWER, BOUND_UPPER};
use serde::{Deserialize, Serialize};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

pub const MATE: i32 = 30000;
pub const INF: i32 = 32000;
pub const MAXPLY: usize = 100;
pub const MATE_BOUND: i32 = MATE - 200;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchLimits {
    pub depth: Option<u32>,
    pub time_ms: Option<u64>,
    pub nodes: Option<u64>,
    pub hash_mb: usize,
    pub threads: usize,
    pub opts: String,
    pub multipv: usize,
    pub start_depth: u32,
    #[serde(skip)]
    pub root_hint: Option<(u8, u8)>,
    pub duty: u32,
    pub qos: u8,
}

impl Default for SearchLimits {
    fn default() -> Self {
        SearchLimits {
            depth: None,
            time_ms: None,
            nodes: None,
            hash_mb: 1024,
            threads: 0,
            opts: String::new(),
            multipv: 1,
            start_depth: 0,
            root_hint: None,
            duty: 100,
            qos: 0,
        }
    }
}

pub fn default_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let p = std::process::Command::new("sysctl")
            .args(["-n", "hw.perflevel0.physicalcpu"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<usize>().ok());
        p.filter(|&n| n > 0).unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
    })
}

#[derive(Clone, Debug)]
pub struct RootLine {
    pub mv: Move,
    pub score: i32,
    pub pv: Vec<Move>,
}

#[derive(Clone, Debug)]
pub struct SearchInfo {
    pub lines: Vec<RootLine>,
    pub depth: u32,
    pub score: i32,
    pub nodes: u64,
    pub nps: u64,
    pub time_ms: u64,
    pub hashfull: u32,
    pub pv: Vec<Move>,
    pub hist_dep: bool,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub lines: Vec<RootLine>,
    pub best: Option<Move>,
    pub score: i32,
    pub pv: Vec<Move>,
    pub depth: u32,
    pub nodes: u64,
    pub time_ms: u64,
    pub early: Option<&'static str>,
    pub hist_dep: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub lmr: i32,
    pub lmr_idx: i32,
    pub lmr_c: i32,
    pub lmp: i32,
    pub lmp_base: i32,
    pub asp: i32,
    pub iir: i32,
    pub sing: i32,
    pub pc: i32,
    pub pc_depth: i32,
    pub pcs: i32,
    pub hist_lmr: i32,
    pub ext1: i32,
    pub hist: i32,
    pub cm: i32,
    pub rfp: i32,
    pub soft: i32,
    pub softfix: i32,
    pub nostart: i32,
    pub nopred: i32,
    pub easy: i32,
    pub easy_cp: i32,
    pub easy_pct: i32,
    pub pf: i32,
    pub dbdraw: i32,
    pub partial: i32,
    pub etc: i32,
    pub keep_hist: i32,
    pub contempt: i32,
    pub qegdb: i32,
    pub dbclk: i32,
    pub rep3: i32,
    pub nodecnt: i32,
    pub recallpv: i32,
    pub tteval: i32,
    pub egdb_d: i32,
    pub dbdraw_div: i32,
    pub corr: i32,
    pub ch24: i32,
    pub caph: i32,
    pub ttq: i32,
    pub zz: i32,
    pub zz_n: i32,
    pub impr: i32,
    pub lmrx: i32,
    pub tdrop: i32,
    pub tnodes: i32,
    pub text: i32,
    pub histdep: i32,
    pub matkeep: i32,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            lmr: 2,
            lmr_idx: 3,
            lmr_c: 150,
            lmp: 0,
            lmp_base: 6,
            asp: 30,
            iir: 4,
            sing: 6,
            pc: 0,
            pc_depth: 6,
            pcs: 0,
            hist_lmr: 0,
            ext1: 0,
            hist: 1,
            cm: 0,
            rfp: 80,
            soft: 100,
            softfix: 0,
            nostart: 0,
            nopred: 0,
            easy: 0,
            easy_cp: 30,
            easy_pct: 50,
            pf: 3,
            dbdraw: 0,
            partial: 0,
            etc: 0,
            keep_hist: 0,
            contempt: 0,
            qegdb: 0,
            dbclk: 1,
            rep3: 0,
            nodecnt: 0,
            recallpv: 0,
            tteval: 0,
            egdb_d: 0,
            dbdraw_div: 8,
            corr: 0,
            ch24: 0,
            caph: 0,
            ttq: 0,
            zz: 0,
            zz_n: 6,
            impr: 0,
            lmrx: 0,
            tdrop: 0,
            tnodes: 0,
            text: 150,
            histdep: 0,
            matkeep: 0,
        }
    }
}

impl Params {
    pub fn parse(s: &str) -> Params {
        let mut p = Params::default();
        for kv in s.split(',') {
            let Some((k, v)) = kv.split_once('=') else { continue };
            let Ok(v) = v.trim().parse::<i32>() else { continue };
            match k.trim() {
                "lmr" => p.lmr = v,
                "lmr_idx" => p.lmr_idx = v,
                "lmr_c" => p.lmr_c = v,
                "lmp" => p.lmp = v,
                "lmp_base" => p.lmp_base = v,
                "asp" => p.asp = v,
                "iir" => p.iir = v,
                "sing" => p.sing = v,
                "pc" => p.pc = v,
                "pc_depth" => p.pc_depth = v,
                "pcs" => p.pcs = v,
                "hist_lmr" => p.hist_lmr = v,
                "ext1" => p.ext1 = v,
                "hist" => p.hist = v,
                "cm" => p.cm = v,
                "rfp" => p.rfp = v,
                "soft" => p.soft = v,
                "softfix" => p.softfix = v,
                "nostart" => p.nostart = v,
                "nopred" => p.nopred = v,
                "easy" => p.easy = v,
                "easy_cp" => p.easy_cp = v,
                "easy_pct" => p.easy_pct = v,
                "pf" => p.pf = v,
                "dbdraw" => p.dbdraw = v.max(0),
                "partial" => p.partial = v,
                "etc" => p.etc = v,
                "keep_hist" => p.keep_hist = v,
                "contempt" => p.contempt = v,
                "qegdb" => p.qegdb = v,
                "dbclk" => p.dbclk = v,
                "rep3" => p.rep3 = v,
                "nodecnt" => p.nodecnt = v,
                "recallpv" => p.recallpv = v,
                "tteval" => p.tteval = v,
                "egdb_d" => p.egdb_d = v.max(0),
                "dbdraw_div" => p.dbdraw_div = v.max(1),
                "corr" => p.corr = v,
                "ch24" => p.ch24 = v,
                "caph" => p.caph = v,
                "ttq" => p.ttq = v,
                "zz" => p.zz = v.clamp(0, 8),
                "zz_n" => p.zz_n = v.max(0),
                "impr" => p.impr = v,
                "lmrx" => p.lmrx = v,
                "tdrop" => p.tdrop = v.max(0),
                "tnodes" => p.tnodes = v.clamp(0, 100),
                "text" => p.text = v.max(100),
                "histdep" => p.histdep = v,
                "matkeep" => p.matkeep = v.max(0),
                _ => {}
            }
        }
        p
    }

    pub fn get(&self, key: &str) -> Option<i32> {
        Some(match key {
            "lmr" => self.lmr,
            "lmr_idx" => self.lmr_idx,
            "lmr_c" => self.lmr_c,
            "lmp" => self.lmp,
            "lmp_base" => self.lmp_base,
            "asp" => self.asp,
            "iir" => self.iir,
            "sing" => self.sing,
            "pc" => self.pc,
            "pc_depth" => self.pc_depth,
            "pcs" => self.pcs,
            "hist_lmr" => self.hist_lmr,
            "ext1" => self.ext1,
            "hist" => self.hist,
            "cm" => self.cm,
            "rfp" => self.rfp,
            "soft" => self.soft,
            "softfix" => self.softfix,
            "nostart" => self.nostart,
            "nopred" => self.nopred,
            "easy" => self.easy,
            "easy_cp" => self.easy_cp,
            "easy_pct" => self.easy_pct,
            "pf" => self.pf,
            "dbdraw" => self.dbdraw,
            "partial" => self.partial,
            "etc" => self.etc,
            "keep_hist" => self.keep_hist,
            "contempt" => self.contempt,
            "qegdb" => self.qegdb,
            "dbclk" => self.dbclk,
            "rep3" => self.rep3,
            "nodecnt" => self.nodecnt,
            "recallpv" => self.recallpv,
            "tteval" => self.tteval,
            "egdb_d" => self.egdb_d,
            "dbdraw_div" => self.dbdraw_div,
            "corr" => self.corr,
            "ch24" => self.ch24,
            "caph" => self.caph,
            "ttq" => self.ttq,
            "zz" => self.zz,
            "zz_n" => self.zz_n,
            "impr" => self.impr,
            "lmrx" => self.lmrx,
            "tdrop" => self.tdrop,
            "tnodes" => self.tnodes,
            "text" => self.text,
            "histdep" => self.histdep,
            "matkeep" => self.matkeep,
            _ => return None,
        })
    }
}

struct Scores([MaybeUninit<i32>; MAX_MOVES]);

impl Scores {
    #[inline(always)]
    fn new() -> Self {
        Scores(unsafe { MaybeUninit::<[MaybeUninit<i32>; MAX_MOVES]>::uninit().assume_init() })
    }
    #[inline(always)]
    fn set(&mut self, i: usize, v: i32) {
        self.0[i] = MaybeUninit::new(v);
    }
    #[inline(always)]
    fn get(&self, i: usize) -> i32 {
        unsafe { self.0[i].assume_init() }
    }
    #[inline(always)]
    fn swap(&mut self, i: usize, j: usize) {
        self.0.swap(i, j);
    }
}

struct Searcher<'a, R: Rules, E: Evaluator<R>> {
    tt: &'a Tt,
    eval: &'a E,
    est: Vec<E::State>,
    stop: &'a AtomicBool,
    halt: &'a AtomicBool,
    total: &'a AtomicU64,
    main: bool,
    start: Instant,
    time_limit: Option<u64>,
    node_limit: Option<u64>,
    nodes: u64,
    flushed: u64,
    aborted: bool,
    keys: Vec<u64>,
    clocks: Vec<u32>,
    root_idx: usize,
    hev: u64,
    hroot: u64,
    hlev: [u64; MAXPLY + 2],
    killers: [[Move; 2]; MAXPLY],
    history: Vec<i32>,
    cont: Vec<i16>,
    counter: Vec<Move>,
    excl: [Move; MAXPLY],
    mv_stack: [Move; MAXPLY + 2],
    pv: Vec<Move>,
    pv_len: [usize; MAXPLY],
    root_best: Option<(u8, u8)>,
    root_excl: Vec<Move>,
    duty: u32,
    work: std::time::Duration,
    last_tick: Instant,
    lmr_tab: Vec<u8>,
    prm: Params,
    egdb: Option<&'a Egdb<R>>,
    db_root_draw: bool,
    skip_probe: bool,
    db_sat_root: bool,
    root_ban: Vec<Move>,
    recall_pv: u64,
    recall_cut: u64,
    root_partial: Option<Move>,
    root_call_nodes: u64,
    root_best_nodes: u64,
    partial_used: bool,
    full_best: Option<Move>,
    in_pc: bool,
    db_clock: bool,
    corr: Vec<i16>,
    cont2: Vec<i16>,
    cont4: Vec<i16>,
    caph: Vec<i16>,
    zz_left: i32,
    ss_eval: [i32; MAXPLY + 2],
    cut: [bool; MAXPLY + 2],
    _r: std::marker::PhantomData<R>,
}

#[inline(always)]
fn score_to_tt(s: i32, ply: usize) -> i32 {
    if s > MATE_BOUND {
        s + ply as i32
    } else if s < -MATE_BOUND {
        s - ply as i32
    } else {
        s
    }
}

#[inline(always)]
fn score_from_tt(s: i32, ply: usize) -> i32 {
    if s > MATE_BOUND {
        s - ply as i32
    } else if s < -MATE_BOUND {
        s + ply as i32
    } else {
        s
    }
}

#[inline(always)]
fn is_irreversible<R: Rules>(pos: &Position<R>, mv: &Move) -> bool {
    mv.captured != 0 || pos.kings & (1u64 << mv.from) == 0
}

fn clock_between<R: Rules>(a: &Position<R>, b: &Position<R>, prev: u32) -> u32 {
    let same_men = (a.white | a.black) & !a.kings == (b.white | b.black) & !b.kings;
    if (a.white | a.black).count_ones() == (b.white | b.black).count_ones() && same_men {
        prev + 1
    } else {
        0
    }
}

#[inline(always)]
fn db_clock_stale<R: Rules>(clk: u32, s: i32, ply: usize) -> bool {
    if R::DRAW_MOVES == 0 || clk == 0 || s.abs() <= MATE_BOUND {
        return false;
    }
    let m = if s > 0 { MATE - ply as i32 - s } else { s + MATE - ply as i32 + 1 };
    clk as i64 + m as i64 >= 2 * R::DRAW_MOVES as i64
}

#[inline]
pub(crate) fn side_has_capture<R: Rules>(pos: &Position<R>) -> bool {
    type Gm<R> = <R as Rules>::Geo;
    let (own, opp) = (pos.own(), pos.opp());
    let free = Gm::<R>::VALID & !pos.occ();
    let kings = own & pos.kings;
    let men = own & !kings;
    let w = pos.side == crate::position::Side::White;
    let fwd = if w { Gm::<R>::FWD_WHITE } else { Gm::<R>::FWD_BLACK };
    let men_targets = if R::MAN_CAPTURES_KING { opp } else { opp & !pos.kings };
    for d in 0..Gm::<R>::NDIRS {
        let rev = (d + Gm::<R>::NDIRS / 2) % Gm::<R>::NDIRS;
        if men != 0 && (R::MEN_CAPTURE_BACKWARD || fwd.contains(&d)) && Gm::<R>::shift(Gm::<R>::shift(free, rev) & men_targets, rev) & men != 0 {
            return true;
        }
        if kings != 0 {
            let mut front = Gm::<R>::shift(kings, d) & opp;
            if R::KING_FLYING {
                let mut y = Gm::<R>::shift(kings, d) & free;
                while y != 0 {
                    let z = Gm::<R>::shift(y, d);
                    front |= z & opp;
                    y = z & free;
                }
            }
            if Gm::<R>::shift(front, d) & free != 0 {
                return true;
            }
        }
    }
    false
}

#[inline]
pub(crate) fn quiet_count_at_most<R: Rules>(pos: &Position<R>, limit: u32) -> bool {
    type Gm<R> = <R as Rules>::Geo;
    let own = pos.own();
    let free = Gm::<R>::VALID & !pos.occ();
    let kings = own & pos.kings;
    let men = own & !kings;
    let fwd = if pos.side == crate::position::Side::White { Gm::<R>::FWD_WHITE } else { Gm::<R>::FWD_BLACK };
    let mut n = 0u32;
    for &d in fwd {
        n += (Gm::<R>::shift(men, d) & free).count_ones();
    }
    if n > limit {
        return false;
    }
    if kings != 0 {
        for d in 0..Gm::<R>::NDIRS {
            let mut y = Gm::<R>::shift(kings, d) & free;
            while y != 0 {
                n += y.count_ones();
                if n > limit {
                    return false;
                }
                if !R::KING_FLYING {
                    break;
                }
                y = Gm::<R>::shift(y, d) & free;
            }
        }
    }
    n <= limit
}

#[inline(always)]
fn recall_clock_stale<R: Rules>(clk: u32, depth: i32) -> bool {
    R::DRAW_MOVES > 0 && clk > 0 && clk as i64 + depth as i64 >= 2 * R::DRAW_MOVES as i64
}

const HIST_MAX: i32 = 16000;

const HIST_ALL: i32 = i32::MIN / 2;

fn material_loss<R: Rules>(pos: &Position<R>, pv: &[Move]) -> bool {
    let white = pos.side == crate::position::Side::White;
    let mat = |p: &Position<R>, w: bool| {
        let b = if w { p.white } else { p.black };
        (b & !p.kings).count_ones() as i32 + 3 * (b & p.kings).count_ones() as i32
    };
    let Some(first) = pv.first() else { return false };
    let mut cur = pos.make(first);
    let mut i = 1;
    while side_has_capture(&cur) {
        match pv.get(i) {
            Some(m) if m.captured != 0 => {
                cur = cur.make(m);
                i += 1;
            }
            _ => {
                if (cur.side == crate::position::Side::White) != white {
                    return true;
                }
                break;
            }
        }
    }
    (mat(pos, white) - mat(&cur, white)) > (mat(pos, !white) - mat(&cur, !white))
}

const CORR_SIZE: usize = 1 << 14;
const CORR_GRAIN: i32 = 16;
const CORR_LIM: i32 = 512 * CORR_GRAIN;

#[inline(always)]
fn corr_idx<R: Rules>(pos: &Position<R>) -> usize {
    let h = (pos.white & !pos.kings).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (pos.black & !pos.kings).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    pos.side as usize * CORR_SIZE + (h >> 50) as usize
}

impl<'a, R: Rules, E: Evaluator<R>> Searcher<'a, R, E> {
    #[inline(always)]
    fn check_stop(&mut self) {
        let delta = self.nodes - self.flushed;
        self.flushed = self.nodes;
        let tot = self.total.fetch_add(delta, Ordering::Relaxed) + delta;
        if self.duty < 100 {
            self.throttle();
        }
        if self.stop.load(Ordering::Relaxed) || self.halt.load(Ordering::Relaxed) {
            self.aborted = true;
        } else if self.main {
            if let Some(t) = self.time_limit {
                if self.start.elapsed().as_millis() as u64 >= t {
                    self.aborted = true;
                }
            }
            if let Some(n) = self.node_limit {
                if tot >= n {
                    self.aborted = true;
                }
            }
        }
    }

    #[inline(never)]
    fn throttle(&mut self) {
        let now = Instant::now();
        self.work += now.duration_since(self.last_tick);
        self.last_tick = now;
        if self.work >= std::time::Duration::from_millis(20) {
            let mut left = self.work * (100 - self.duty) / self.duty.max(1);
            if let (true, Some(t)) = (self.main, self.time_limit) {
                left = left.min(std::time::Duration::from_millis(t).saturating_sub(self.start.elapsed()));
            }
            self.work = std::time::Duration::ZERO;
            while left > std::time::Duration::ZERO {
                if self.stop.load(Ordering::Relaxed) || self.halt.load(Ordering::Relaxed) {
                    break;
                }
                let step = left.min(std::time::Duration::from_millis(5));
                std::thread::sleep(step);
                left -= step;
            }
            self.last_tick = Instant::now();
        }
    }

    fn flush_nodes(&mut self) {
        let delta = self.nodes - self.flushed;
        self.flushed = self.nodes;
        self.total.fetch_add(delta, Ordering::Relaxed);
    }

    #[inline(always)]
    fn child_repeats(&self, hash: u64, clk: u32) -> bool {
        let n = self.keys.len();
        let mut k = 4;
        while k <= clk as usize && k <= n {
            if self.keys[n - k] == hash {
                return true;
            }
            k += 2;
        }
        false
    }

    #[inline(never)]
    fn etc_cut(&mut self, pos: &Position<R>, list: &MoveList, beta: i32, depth: i32, ply: usize) -> Option<i32> {
        let prev_clock = *self.clocks.last().unwrap();
        let (own, opp) = match pos.side {
            crate::position::Side::White => (pos.white, pos.black),
            crate::position::Side::Black => (pos.black, pos.white),
        };
        for m in list.as_slice() {
            if R::ONE_V_ONE_DRAW && own.count_ones() == 1 && (opp & !m.captured).count_ones() == 1 {
                continue;
            }
            if let Some(db) = self.egdb {
                if pos.occ().count_ones() - m.captured.count_ones() <= db.max_pieces() {
                    continue;
                }
            }
            let clk = if is_irreversible(pos, m) { 0 } else { prev_clock + 1 };
            if R::DRAW_MOVES > 0 && clk >= 2 * R::DRAW_MOVES {
                continue;
            }
            let h = pos.hash_after(m);
            if self.child_repeats(h, clk) {
                continue;
            }
            let Some(e) = self.tt.probe(h) else { continue };
            if e.hist || e.recall || e.depth < depth - 1 || e.bound == BOUND_LOWER {
                continue;
            }
            let s = score_from_tt(e.score, ply + 1);
            if s <= -beta {
                let v = -s;
                self.tt.store(pos.hash, score_to_tt(v, ply), depth, BOUND_LOWER, Some((m.from, m.to)), false);
                return Some(v);
            }
        }
        None
    }

    #[inline(always)]
    fn draw_value(&self, pos: &Position<R>, ply: usize) -> i32 {
        let c = self.prm.contempt;
        if c == 0 || self.egdb.is_some_and(|db| pos.occ().count_ones() <= db.max_pieces()) {
            0
        } else if ply % 2 == 0 {
            -c
        } else {
            c
        }
    }

    #[inline(always)]
    fn repetition_at(&self, hash: u64) -> Option<usize> {
        let n = self.keys.len() - 1;
        let clk = self.clocks[n] as usize;
        let mut k = 4;
        while k <= clk && k <= n {
            if self.keys[n - k] == hash {
                return Some(n - k);
            }
            k += 2;
        }
        None
    }

    #[inline(always)]
    fn node_eval(&mut self, pos: &Position<R>, ply: usize, cache: &mut Option<i32>) -> i32 {
        if let Some(v) = *cache {
            return v;
        }
        let v = self.eval.eval_at(pos, &mut self.est, ply);
        *cache = Some(v);
        v
    }

    #[inline(never)]
    fn zz_leaf(&mut self, pos: &Position<R>, mut alpha: i32, beta: i32, ply: usize) -> Option<i32> {
        if !quiet_count_at_most(pos, self.prm.zz_n as u32) {
            return None;
        }
        let mut list = MoveList::new();
        generate(pos, &mut ListSink::<true>(&mut list));
        if !list.as_slice().iter().all(|m| side_has_capture(&pos.make(m))) {
            return None;
        }
        let saved = self.zz_left;
        self.zz_left = saved - 1;
        let mut best = -INF;
        for i in 0..list.len {
            let mv = list[i];
            let child = pos.make(&mv);
            self.eval.child_at(&mut self.est, ply, pos, &mv, &child);
            let s = -self.qsearch(&child, -beta, -alpha, ply + 1);
            if self.aborted {
                break;
            }
            if s > best {
                best = s;
                if s > alpha {
                    alpha = s;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        self.zz_left = saved;
        Some(if self.aborted { 0 } else { best })
    }

    #[inline(never)]
    fn improving(&mut self, pos: &Position<R>, ply: usize, se_corr: Option<i32>, cache: &mut Option<i32>, quiet: bool, has_excl: bool) -> bool {
        let cur = if quiet {
            match se_corr {
                Some(v) => v,
                None => self.node_eval(pos, ply, cache),
            }
        } else {
            i32::MIN
        };
        if !has_excl {
            self.ss_eval[ply] = cur;
        }
        ply >= 2 && cur != i32::MIN && self.ss_eval[ply - 2] != i32::MIN && cur > self.ss_eval[ply - 2]
    }

    #[inline(always)]
    fn tt_refine(&mut self, se: i32, tt_hit: Option<(i32, i32, u8)>, tt_dep: bool, ply: usize) -> i32 {
        if let Some((ts, _, tb)) = tt_hit {
            if ts.abs() < MATE_BOUND && (tb == BOUND_EXACT || (tb == BOUND_LOWER && ts > se) || (tb == BOUND_UPPER && ts < se)) {
                if tt_dep {
                    self.tt_dep_event(ply);
                }
                return ts;
            }
        }
        se
    }

    #[inline(always)]
    fn corrected(&self, pos: &Position<R>, raw: i32) -> i32 {
        (raw + self.corr[corr_idx(pos)] as i32 / CORR_GRAIN).clamp(-MATE_BOUND + 1, MATE_BOUND - 1)
    }

    #[inline(never)]
    fn corr_update(&mut self, pos: &Position<R>, diff: i32, depth: i32) {
        let w = (depth + 1).clamp(1, 16);
        let c = &mut self.corr[corr_idx(pos)];
        let target = (diff * CORR_GRAIN).clamp(-CORR_LIM, CORR_LIM);
        *c = ((*c as i32 * (256 - w) + target * w) / 256).clamp(-CORR_LIM, CORR_LIM) as i16;
    }

    #[inline(never)]
    fn repeats_twice(&self, hash: u64, i: usize) -> bool {
        let n = self.keys.len() - 1;
        let clk = self.clocks[n] as usize;
        let mut k = n - i + 2;
        while k <= clk && k <= n {
            if self.keys[n - k] == hash {
                return true;
            }
            k += 2;
        }
        false
    }

    #[inline(always)]
    fn hist_event(&mut self, from: i32, ply: usize) {
        self.hev += 1;
        if self.prm.histdep != 0 {
            self.hist_event_lv(from, ply);
        }
    }

    #[inline(never)]
    fn hist_event_lv(&mut self, from: i32, ply: usize) {
        if from < 0 {
            self.hroot += 1;
        }
        for k in from.max(0) as usize..=ply.min(MAXPLY + 1) {
            self.hlev[k] += 1;
        }
    }

    #[inline(always)]
    fn hmark(&self, ply: usize) -> u64 {
        if self.prm.histdep != 0 {
            self.hlev[ply]
        } else {
            self.hev
        }
    }

    #[inline(always)]
    fn hmark_root(&self) -> u64 {
        if self.prm.histdep != 0 {
            self.hroot
        } else {
            self.hev
        }
    }

    #[inline(always)]
    fn tt_dep_event(&mut self, ply: usize) {
        let clk = self.clocks[self.keys.len() - 1] as i32;
        self.hist_event(ply as i32 - clk, ply);
    }

    #[inline(never)]
    fn qs_dep_event(&mut self, ply: usize) {
        let top = self.keys.len() - 1;
        let from = if ply + self.root_idx == top { ply as i32 - self.clocks[top] as i32 } else { HIST_ALL };
        self.hist_event(from, ply);
    }

    #[inline(always)]
    fn count_exit(&mut self) {
        if self.prm.nodecnt != 0 {
            self.nodes += 1;
            if self.nodes & 1023 == 0 {
                self.check_stop();
            }
        }
    }

    #[inline(never)]
    fn pv_from_tt(&mut self, pos: &Position<R>, mv: Option<(u8, u8)>, ply: usize) {
        let mut seen = [0u64; 32];
        let mut cur = *pos;
        let mut want = mv;
        let mut len = 0usize;
        let cap = (MAXPLY - ply - 1).min(seen.len());
        while let Some((f, t)) = want {
            if len >= cap {
                break;
            }
            let mut l = MoveList::new();
            generate(&cur, &mut ListSink::<true>(&mut l));
            let Some(m) = l.as_slice().iter().find(|m| (m.from, m.to) == (f, t)).copied() else { break };
            seen[len] = cur.hash;
            self.pv[ply * MAXPLY + len] = m;
            len += 1;
            cur = cur.make(&m);
            if seen[..len].contains(&cur.hash) {
                break;
            }
            want = self.tt.probe(cur.hash).and_then(|e| e.mv);
        }
        self.pv_len[ply] = len;
    }

    fn qsearch(&mut self, pos: &Position<R>, alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;
        if self.nodes & 1023 == 0 {
            self.check_stop();
        }
        if self.aborted {
            return 0;
        }
        if R::ONE_V_ONE_DRAW && pos.white.count_ones() == 1 && pos.black.count_ones() == 1 {
            return 0;
        }
        if ply >= MAXPLY - 2 {
            return self.eval.eval_at(pos, &mut self.est, ply);
        }
        if self.prm.qegdb > 0 && ply > 0 {
            if let Some(s) = self.egdb.and_then(|db| db.probe_score(pos, ply)) {
                if self.db_clock && db_clock_stale::<R>(self.clocks[self.keys.len() - 1], s, ply) {
                    self.hist_event(HIST_ALL, ply);
                } else if s != 0 || self.prm.dbdraw == 0 {
                    return s;
                }
            }
        }
        if self.prm.ttq & 2 != 0 && ply > 0 {
            return self.qsearch_tt(pos, alpha, beta, ply);
        }
        self.qsearch_moves(pos, alpha, beta, ply)
    }

    #[inline(never)]
    fn qsearch_tt(&mut self, pos: &Position<R>, alpha: i32, beta: i32, ply: usize) -> i32 {
        let hev0 = self.hmark(ply);
        let mut deeper = false;
        if let Some(e) = self.tt.probe(pos.hash) {
            deeper = e.depth > 0;
            if !e.recall {
                let s = score_from_tt(e.score, ply);
                let cut = match e.bound {
                    BOUND_EXACT => true,
                    BOUND_LOWER => s >= beta,
                    _ => s <= alpha,
                };
                if cut {
                    if e.hist {
                        self.qs_dep_event(ply);
                    }
                    return s;
                }
            }
        }
        let v = self.qsearch_moves(pos, alpha, beta, ply);
        if !self.aborted && !deeper {
            let bound = if v >= beta {
                BOUND_LOWER
            } else if v > alpha {
                BOUND_EXACT
            } else {
                BOUND_UPPER
            };
            self.tt.store(pos.hash, score_to_tt(v, ply), 0, bound, None, self.hmark(ply) != hev0);
        }
        v
    }

    #[inline(always)]
    fn qsearch_moves(&mut self, pos: &Position<R>, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        let mut best;
        let mut list = MoveList::new();
        if !R::FORCED {
            if !has_quiet(pos) {
                generate_captures(pos, &mut ListSink::<true>(&mut list));
                if list.len == 0 {
                    return -MATE + ply as i32;
                }
                best = -INF;
            } else {
                let mut st = self.eval.eval_at(pos, &mut self.est, ply);
                if self.prm.corr >= 2 {
                    st = self.corrected(pos, st);
                }
                if st >= beta {
                    return st;
                }
                if st > alpha {
                    alpha = st;
                }
                best = st;
                generate_captures(pos, &mut ListSink::<true>(&mut list));
            }
        } else {
            best = -INF;
            generate_captures(pos, &mut ListSink::<true>(&mut list));
        }
        if list.len == 0 {
            if R::FORCED {
                if !has_quiet(pos) {
                    return -MATE + ply as i32;
                }
                let mut st = self.eval.eval_at(pos, &mut self.est, ply);
                if self.prm.corr >= 2 {
                    st = self.corrected(pos, st);
                }
                if self.prm.zz > 0 && self.zz_left > 0 && st > alpha {
                    if let Some(v) = self.zz_leaf(pos, alpha, beta, ply) {
                        return v;
                    }
                }
                return st;
            }
            return best;
        }
        let n = list.len;
        let mut sc = Scores::new();
        for i in 0..n {
            sc.set(i, list[i].captured.count_ones() as i32);
        }
        for i in 0..n {
            let mut bi = i;
            for j in i + 1..n {
                if sc.get(j) > sc.get(bi) {
                    bi = j;
                }
            }
            list.swap(i, bi);
            sc.swap(i, bi);
            let child = pos.make(&list[i]);
            self.eval.child_at(&mut self.est, ply, pos, &list[i], &child);
            let s = -self.qsearch(&child, -beta, -alpha, ply + 1);
            if self.aborted {
                return 0;
            }
            if s > best {
                best = s;
                if s > alpha {
                    alpha = s;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        best
    }

    fn search_node<const PV: bool>(
        &mut self,
        pos: &Position<R>,
        mut alpha: i32,
        mut beta: i32,
        mut depth: i32,
        ply: usize,
    ) -> i32 {
        if PV {
            self.pv_len[ply] = 0;
        }
        let mut db_sat = false;
        if ply > 0 {
            if R::ONE_V_ONE_DRAW && pos.white.count_ones() == 1 && pos.black.count_ones() == 1 {
                self.count_exit();
                return 0;
            }
            if let Some(i) = self.repetition_at(pos.hash) {
                if !(self.prm.rep3 != 0 && i < self.root_idx && !self.repeats_twice(pos.hash, i)) {
                    let v = self.draw_value(pos, ply);
                    if i < self.root_idx || v != 0 {
                        self.hist_event(HIST_ALL, ply);
                    }
                    self.count_exit();
                    return v;
                }
            }
            let clk = self.clocks[self.keys.len() - 1];
            if R::DRAW_MOVES > 0 && clk >= 2 * R::DRAW_MOVES {
                let v = self.draw_value(pos, ply);
                self.hist_event(if v != 0 { HIST_ALL } else { ply as i32 - clk as i32 }, ply);
                self.count_exit();
                return v;
            }
            alpha = alpha.max(-MATE + ply as i32);
            beta = beta.min(MATE - ply as i32 - 1);
            if alpha >= beta {
                self.count_exit();
                return alpha;
            }
            if !(self.prm.dbdraw > 0 && std::mem::take(&mut self.skip_probe)) {
                let gate = self.prm.egdb_d > 0 && depth < self.prm.egdb_d && self.prm.dbdraw == 0 && !self.db_sat_root;
                let db = self.egdb.filter(|db| !gate || pos.occ().count_ones() < db.max_pieces());
                if let Some(s) = db.and_then(|db| db.probe_score(pos, ply)) {
                    let stale = self.db_clock && db_clock_stale::<R>(self.clocks[self.keys.len() - 1], s, ply);
                    if stale {
                        self.hist_event(HIST_ALL, ply);
                    } else if self.db_sat_root && depth > 0 && ply < MAXPLY - 2 && crate::egdb::is_sat_score(s) {
                        db_sat = true;
                    } else if s != 0 || self.prm.dbdraw == 0 {
                        self.count_exit();
                        return s;
                    } else {
                        return self.db_draw_node::<PV>(pos, alpha, beta, depth, ply);
                    }
                }
            }
        }
        if ply >= MAXPLY - 2 {
            self.count_exit();
            return self.eval.eval_at(pos, &mut self.est, ply);
        }
        if depth <= 0 {
            if self.prm.zz > 0 {
                self.zz_left = self.prm.zz;
            }
            return self.qsearch(pos, alpha, beta, ply);
        }
        self.nodes += 1;
        if self.nodes & 1023 == 0 {
            self.check_stop();
        }
        if self.aborted {
            return 0;
        }

        let excl = self.excl[ply];
        let has_excl = excl != Move::NULL;
        let key = pos.hash;
        let orig_alpha = alpha;
        let mut tt_move: Option<(u8, u8)> = None;
        let mut tt_hit: Option<(i32, i32, u8)> = None;
        let mut tt_dep = false;
        let mut se_cache: Option<i32> = None;
        let hev0 = self.hmark(ply);
        if let Some(e) = self.tt.probe(key) {
            tt_move = e.mv;
            if self.prm.tteval != 0 {
                se_cache = e.eval;
            }
            let s = score_from_tt(e.score, ply);
            let stale = e.recall && recall_clock_stale::<R>(self.clocks[self.keys.len() - 1], e.depth);
            if !stale {
                tt_hit = Some((s, e.depth, e.bound));
            }
            tt_dep = e.hist;
            if !stale && !PV && !has_excl && e.depth >= depth {
                let cut = match e.bound {
                    BOUND_EXACT => true,
                    BOUND_LOWER => s >= beta,
                    _ => s <= alpha,
                };
                if cut {
                    if tt_dep {
                        self.tt_dep_event(ply);
                    }
                    self.recall_cut += e.recall as u64;
                    return s;
                }
            } else if !stale && PV && e.recall && ply > 0 && !has_excl && e.bound == BOUND_EXACT && e.depth >= depth {
                self.recall_pv += 1;
                if self.prm.recallpv != 0 {
                    self.pv_from_tt(pos, e.mv, ply);
                }
                return s;
            }
        }
        if ply == 0 && tt_move.is_none() {
            tt_move = self.root_best;
        }

        let mut list = MoveList::new();
        generate(pos, &mut ListSink::<true>(&mut list));
        let n = list.len;
        if n == 0 {
            return -MATE + ply as i32;
        }
        let tactical = if R::FORCED { list[0].captured != 0 } else { list.iter().any(|m| m.captured != 0) };
        let prm = self.prm;
        if prm.etc != 0 && !PV && !has_excl && ply > 0 && !db_sat && depth >= 3 && ply + 1 < MAXPLY - 2 {
            if let Some(v) = self.etc_cut(pos, &list, beta, depth, ply) {
                return v;
            }
        }

        let se_corr = if prm.corr > 0 && !tactical && !has_excl && !db_sat {
            let raw = self.node_eval(pos, ply, &mut se_cache);
            Some(self.corrected(pos, raw))
        } else {
            None
        };
        let improving = prm.impr > 0 && self.improving(pos, ply, se_corr, &mut se_cache, !tactical && !has_excl && !db_sat, has_excl);
        let cut_node = prm.lmrx & 1 != 0 && !PV && self.cut[ply];
        if !PV && !has_excl && !tactical && ply > 0 && !db_sat {
            if prm.rfp > 0 && depth <= 4 && beta.abs() < MATE_BOUND {
                let mut se = match se_corr {
                    Some(v) => v,
                    None => self.node_eval(pos, ply, &mut se_cache),
                };
                if prm.ttq & 1 != 0 {
                    se = self.tt_refine(se, tt_hit, tt_dep, ply);
                }
                if se - prm.rfp * if improving { depth - 1 } else { depth } >= beta {
                    return se;
                }
            }
        }
        if prm.pc > 0
            && !PV
            && !has_excl
            && ply > 0
            && !db_sat
            && !self.in_pc
            && depth >= prm.pc_depth
            && !(tactical && n == 1)
            && beta.abs() < MATE_BOUND
        {
            let rbeta = beta + prm.pc;
            let pd = depth - 4;
            let tt_against = matches!(tt_hit, Some((ts, td, _)) if td >= pd && ts < rbeta);
            let mut se = match se_corr {
                Some(v) => v,
                None => self.node_eval(pos, ply, &mut se_cache),
            };
            if prm.ttq & 1 != 0 {
                se = self.tt_refine(se, tt_hit, tt_dep, ply);
            }
            if rbeta < MATE_BOUND && !tt_against && se >= beta + prm.pcs {
                self.in_pc = true;
                let s = self.search_node::<false>(pos, rbeta - 1, rbeta, pd, ply);
                self.in_pc = false;
                if self.aborted {
                    return 0;
                }
                if s >= rbeta {
                    let ev = if prm.tteval != 0 { se_cache } else { None };
                    self.tt.store_eval(key, score_to_tt(s, ply), pd + 1, BOUND_LOWER, None, self.hmark(ply) != hev0, ev);
                    return if s.abs() < MATE_BOUND { s - prm.pc } else { s };
                }
            }
        }
        if prm.iir > 0 && depth >= prm.iir && tt_move.is_none() && !has_excl && !tactical {
            depth -= 1;
        }

        let side = pos.side as usize;
        let prev = self.mv_stack[ply];
        let have_prev = ply > 0 && prev != Move::NULL;
        let counter = if prm.cm > 0 && have_prev { self.counter[prev.from as usize * 64 + prev.to as usize] } else { Move::NULL };
        let cbase = if prm.cm > 0 && have_prev { prev.to as usize * 4096 } else { usize::MAX };
        let (b2, b4) = if prm.ch24 > 0 { self.ch24_bases(ply) } else { (usize::MAX, usize::MAX) };
        let mut sc = Scores::new();
        for i in 0..n {
            let m = &list[i];
            let s = if tt_move == Some((m.from, m.to)) {
                1 << 24
            } else if m.captured != 0 {
                if prm.caph > 0 {
                    let ch = self.caph[side * 4096 + m.from as usize * 64 + m.to as usize] as i32 / 16;
                    (1 << 20) + (m.captured.count_ones() as i32 * 16 + m.promo as i32) * 2048 + ch
                } else {
                    (1 << 20) + m.captured.count_ones() as i32 * 16 + m.promo as i32
                }
            } else if self.killers[ply][0] == *m {
                1 << 19
            } else if self.killers[ply][1] == *m {
                (1 << 19) - 1
            } else {
                let hi = side * 4096 + m.from as usize * 64 + m.to as usize;
                let mut h = self.history[hi].min((1 << 17) - 1);
                if cbase != usize::MAX {
                    h += self.cont[cbase + m.from as usize * 64 + m.to as usize] as i32;
                    if counter == *m {
                        h += 20000;
                    }
                }
                if b2 != usize::MAX {
                    h += self.cont2[b2 + m.from as usize * 64 + m.to as usize] as i32;
                }
                if b4 != usize::MAX {
                    h += self.cont4[b4 + m.from as usize * 64 + m.to as usize] as i32;
                }
                h + m.promo as i32 * 1000
            };
            sc.set(i, s);
        }

        let mut best = -INF;
        let mut best_mv: Option<Move> = None;
        let prev_clock = *self.clocks.last().unwrap();
        let mut tried_quiets = [Move::NULL; 24];
        let mut nq = 0usize;
        let mut tried_caps = [Move::NULL; 24];
        let mut ncap = 0usize;
        let mut searched = 0usize;
        for i in 0..n {
            if i == 0 {
                let mut bi = 0;
                for j in 1..n {
                    if sc.get(j) > sc.get(bi) {
                        bi = j;
                    }
                }
                list.swap(0, bi);
                sc.swap(0, bi);
                if depth >= 3 && self.prm.pf >= 2 {
                    self.tt.prefetch(pos.hash_after(&list[0]));
                }
            }
            let mv = list[i];
            if i + 1 < n {
                let mut bi = i + 1;
                for j in i + 2..n {
                    if sc.get(j) > sc.get(bi) {
                        bi = j;
                    }
                }
                list.swap(i + 1, bi);
                sc.swap(i + 1, bi);
                if depth >= 3 && self.prm.pf >= 2 {
                    self.tt.prefetch(pos.hash_after(&list[i + 1]));
                }
            }
            if has_excl && mv == excl {
                continue;
            }
            if ply == 0 && !self.root_excl.is_empty() && self.root_excl.contains(&mv) {
                continue;
            }
            if ply == 0 && !self.root_ban.is_empty() && self.root_ban.contains(&mv) {
                continue;
            }
            let mv_nodes0 = self.nodes;
            let quiet = mv.captured == 0;
            if prm.lmp > 0
                && !PV
                && quiet
                && !mv.promo
                && depth <= prm.lmp
                && searched as i32 >= if prm.impr > 0 && !improving { (prm.lmp_base + depth * depth) / 2 } else { prm.lmp_base + depth * depth }
                && best > -MATE_BOUND
                && !tactical
                && !db_sat
            {
                continue;
            }
            let mut ext = 0;
            if n == 1 && prm.ext1 > 0 && !has_excl {
                ext = 1;
            } else if prm.sing > 0
                && searched == 0
                && depth >= prm.sing
                && !has_excl
                && ply > 0
                && n > 1
                && !db_sat
                && tt_move == Some((mv.from, mv.to))
            {
                if let Some((ts, td, tb)) = tt_hit {
                    if tb != BOUND_UPPER && td >= depth - 3 && ts.abs() < MATE_BOUND {
                        if tt_dep {
                            self.tt_dep_event(ply);
                        }
                        let sb = ts - 2 * depth;
                        self.excl[ply] = mv;
                        let s = self.search_node::<false>(pos, sb - 1, sb, (depth - 1) / 2, ply);
                        self.excl[ply] = Move::NULL;
                        if self.aborted {
                            return 0;
                        }
                        if s < sb {
                            ext = 1;
                        } else if sb >= beta {
                            return sb;
                        }
                    }
                }
            }
            let child = pos.make(&mv);
            self.eval.child_at(&mut self.est, ply, pos, &mv, &child);
            if self.prm.pf & 1 != 0 {
                self.tt.prefetch(child.hash);
            }
            self.keys.push(child.hash);
            self.clocks.push(if is_irreversible(pos, &mv) { 0 } else { prev_clock + 1 });
            self.mv_stack[ply + 1] = mv;
            let nd = depth - 1 + ext;
            let mut score;
            if searched == 0 {
                if prm.lmrx & 1 != 0 {
                    self.cut[ply + 1] = !PV && !cut_node;
                }
                score = -self.search_node::<PV>(&child, -beta, -alpha, nd, ply + 1);
            } else {
                let mut r = 0;
                if depth >= 3 && searched as i32 >= prm.lmr_idx && quiet && !mv.promo && ext == 0 {
                    r = match prm.lmr {
                        0 => 0,
                        1 => 1,
                        _ => {
                            let mut r = self.lmr_tab[(depth as usize).min(63) * 64 + searched.min(63)] as i32;
                            if PV {
                                r -= 1;
                            }
                            r
                        }
                    };
                    if prm.hist_lmr > 0 && prm.lmr != 0 {
                        let mut h = self.history[side * 4096 + mv.from as usize * 64 + mv.to as usize];
                        if cbase != usize::MAX {
                            h += self.cont[cbase + mv.from as usize * 64 + mv.to as usize] as i32;
                        }
                        r -= h / prm.hist_lmr;
                    }
                    if prm.lmr != 0 {
                        if prm.impr > 0 && !improving {
                            r += 1;
                        }
                        if cut_node {
                            r += 1;
                        }
                    }
                    r = r.clamp(0, (nd - 1).max(0));
                }
                if prm.lmrx & 1 != 0 {
                    self.cut[ply + 1] = r > 0 || !cut_node;
                }
                score = -self.search_node::<false>(&child, -alpha - 1, -alpha, nd - r, ply + 1);
                if score > alpha && r > 0 {
                    let mut fd = nd;
                    if prm.lmrx & 2 != 0 {
                        fd += (score > best + 40 + 2 * nd) as i32 - (score < best + 9) as i32;
                        fd = fd.max(nd - r);
                    }
                    if prm.lmrx & 1 != 0 {
                        self.cut[ply + 1] = !cut_node;
                    }
                    if fd > nd - r || prm.lmrx & 2 == 0 {
                        score = -self.search_node::<false>(&child, -alpha - 1, -alpha, fd, ply + 1);
                    }
                }
                if PV && score > alpha && score < beta {
                    score = -self.search_node::<true>(&child, -beta, -alpha, nd, ply + 1);
                }
            }
            self.keys.pop();
            self.clocks.pop();
            if self.aborted {
                return 0;
            }
            searched += 1;
            if score > best {
                best = score;
                best_mv = Some(mv);
                if ply == 0 && self.prm.tnodes > 0 {
                    self.root_best_nodes = self.nodes - mv_nodes0;
                }
                if score > alpha {
                    alpha = score;
                    if PV {
                        let cl = self.pv_len[ply + 1];
                        self.pv[ply * MAXPLY] = mv;
                        for k in 0..cl {
                            self.pv[ply * MAXPLY + 1 + k] = self.pv[(ply + 1) * MAXPLY + k];
                        }
                        self.pv_len[ply] = cl + 1;
                    }
                    if ply == 0 && self.prm.partial != 0 && score < beta {
                        self.root_partial = Some(mv);
                    }
                    if alpha >= beta {
                        if quiet {
                            self.reward_quiet(mv, side, depth, ply, prev, have_prev, &tried_quiets[..nq]);
                        }
                        if prm.caph > 0 {
                            self.reward_capture(if quiet { None } else { Some(mv) }, side, depth, &tried_caps[..ncap]);
                        }
                        break;
                    }
                }
            }
            if quiet && nq < tried_quiets.len() {
                tried_quiets[nq] = mv;
                nq += 1;
            }
            if prm.caph > 0 && !quiet && ncap < tried_caps.len() {
                tried_caps[ncap] = mv;
                ncap += 1;
            }
        }
        if has_excl {
            return if best == -INF { alpha } else { best };
        }
        let bound = if best >= beta {
            BOUND_LOWER
        } else if best > orig_alpha {
            BOUND_EXACT
        } else {
            BOUND_UPPER
        };
        if let Some(sc) = se_corr {
            let informative = match bound {
                BOUND_LOWER => best > sc,
                BOUND_UPPER => best < sc,
                _ => true,
            };
            if informative && best.abs() < MATE_BOUND && best_mv.is_none_or(|m| m.captured == 0) {
                self.corr_update(pos, best - sc, depth);
            }
        }
        if ply > 0 || (self.root_excl.is_empty() && self.root_ban.is_empty()) {
            let ev = if prm.tteval != 0 { se_cache } else { None };
            self.tt.store_eval(key, score_to_tt(best, ply), depth, bound, best_mv.map(|m| (m.from, m.to)), self.hmark(ply) != hev0, ev);
        }
        best
    }

    #[inline(never)]
    fn db_draw_node<const PV: bool>(&mut self, pos: &Position<R>, alpha: i32, beta: i32, depth: i32, ply: usize) -> i32 {
        let b = self.prm.dbdraw;
        if !self.db_root_draw {
            let mut caps = MoveList::new();
            generate_captures(pos, &mut ListSink::<true>(&mut caps));
            if caps.len != 0 {
                return 0;
            }
            return (self.eval.eval_at(pos, &mut self.est, ply) / self.prm.dbdraw_div).clamp(-b, b);
        }
        if alpha >= b {
            return b;
        }
        if beta <= -b {
            return -b;
        }
        self.skip_probe = true;
        let s = self.search_node::<PV>(pos, alpha, beta, depth, ply);
        self.skip_probe = false;
        s.clamp(-b, b)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn reward_quiet(&mut self, mv: Move, side: usize, depth: i32, ply: usize, prev: Move, have_prev: bool, tried: &[Move]) {
        if self.killers[ply][0] != mv {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = mv;
        }
        let hi = side * 4096 + mv.from as usize * 64 + mv.to as usize;
        if self.prm.hist == 0 {
            let h = &mut self.history[hi];
            *h = (*h + depth * depth).min(1 << 17);
        } else {
            let bonus = (depth * depth * 4).min(1600);
            let h = &mut self.history[hi];
            *h += bonus - *h * bonus / HIST_MAX;
            for t in tried {
                let ti = side * 4096 + t.from as usize * 64 + t.to as usize;
                let h = &mut self.history[ti];
                *h -= bonus + *h * bonus / HIST_MAX;
            }
        }
        if self.prm.ch24 > 0 {
            let (b2, b4) = self.ch24_bases(ply);
            let bonus = (depth * depth * 4).min(1600);
            for (tab, base) in [(&mut self.cont2, b2), (&mut self.cont4, b4)] {
                if base == usize::MAX {
                    continue;
                }
                let c = &mut tab[base + mv.from as usize * 64 + mv.to as usize];
                *c += (bonus - *c as i32 * bonus / HIST_MAX) as i16;
                for t in tried {
                    let c = &mut tab[base + t.from as usize * 64 + t.to as usize];
                    *c -= (bonus + *c as i32 * bonus / HIST_MAX) as i16;
                }
            }
        }
        if self.prm.cm > 0 && have_prev {
            self.counter[prev.from as usize * 64 + prev.to as usize] = mv;
            let bonus = (depth * depth * 4).min(1600);
            let base = prev.to as usize * 4096;
            let c = &mut self.cont[base + mv.from as usize * 64 + mv.to as usize];
            *c += (bonus - *c as i32 * bonus / HIST_MAX) as i16;
            for t in tried {
                let c = &mut self.cont[base + t.from as usize * 64 + t.to as usize];
                *c -= (bonus + *c as i32 * bonus / HIST_MAX) as i16;
            }
        }
    }

    #[inline(always)]
    fn ch24_bases(&self, ply: usize) -> (usize, usize) {
        let at = |k: usize| {
            let m = self.mv_stack[k];
            if m == Move::NULL { usize::MAX } else { m.to as usize * 4096 }
        };
        (if ply >= 2 { at(ply - 1) } else { usize::MAX }, if ply >= 4 { at(ply - 3) } else { usize::MAX })
    }

    #[inline(never)]
    fn reward_capture(&mut self, mv: Option<Move>, side: usize, depth: i32, tried: &[Move]) {
        let bonus = (depth * depth * 4).min(1600);
        if let Some(m) = mv {
            let c = &mut self.caph[side * 4096 + m.from as usize * 64 + m.to as usize];
            *c += (bonus - *c as i32 * bonus / HIST_MAX) as i16;
        }
        for t in tried {
            let c = &mut self.caph[side * 4096 + t.from as usize * 64 + t.to as usize];
            *c -= (bonus + *c as i32 * bonus / HIST_MAX) as i16;
        }
    }

    fn iterate(
        &mut self,
        pos: &Position<R>,
        root_len: usize,
        limits: &SearchLimits,
        tid: usize,
        first: Move,
        on_info: &mut dyn FnMut(&SearchInfo),
    ) -> SearchResult {
        let max_depth = limits.depth.unwrap_or(MAXPLY as u32 - 4).clamp(1, MAXPLY as u32 - 4);
        let mut res = SearchResult { lines: vec![], best: Some(first), score: 0, pv: vec![], depth: 0, nodes: 0, time_ms: 0, early: None, hist_dep: false };
        let mpv = limits.multipv.clamp(1, 10).min(root_len.max(1));
        let mut prev_scores = vec![0i32; mpv];
        let mut have_scores = vec![false; mpv];
        let mut tc = TimeCtl::new(&self.prm, limits.time_ms);
        let sd = limits.start_depth.min(max_depth);
        let mut depth = if sd > 1 { sd + (tid % 3) as u32 } else if tid > 0 { 1 + (tid % 3) as u32 } else { 1 };
        depth = depth.min(max_depth);
        while depth <= max_depth {
            let it_start = self.start.elapsed().as_millis() as u64;
            let hev_it = self.hmark_root();
            let mut lines: Vec<RootLine> = Vec::with_capacity(mpv);
            self.root_excl.clear();
            for k in 0..mpv {
                let mut score;
                let asp = self.prm.asp;
                if depth >= 5 && asp > 0 && have_scores[k] && prev_scores[k].abs() < MATE_BOUND {
                    let mut delta = asp;
                    let mut a = (prev_scores[k] - delta).max(-INF);
                    let mut b = (prev_scores[k] + delta).min(INF);
                    loop {
                        self.root_partial = None;
                        (self.root_call_nodes, self.root_best_nodes) = (self.nodes, 0);
                        score = self.search_node::<true>(pos, a, b, depth as i32, 0);
                        if self.aborted {
                            break;
                        }
                        if score <= a {
                            b = (a + b) / 2;
                            a = (score - delta).max(-INF);
                        } else if score >= b {
                            b = (score + delta).min(INF);
                        } else {
                            break;
                        }
                        delta *= 2;
                        if delta > 600 {
                            a = -INF;
                            b = INF;
                        }
                    }
                } else {
                    self.root_partial = None;
                    (self.root_call_nodes, self.root_best_nodes) = (self.nodes, 0);
                    score = self.search_node::<true>(pos, -INF, INF, depth as i32, 0);
                }
                if self.aborted {
                    break;
                }
                let lpv: Vec<Move> = self.pv[..self.pv_len[0]].to_vec();
                let Some(first_mv) = lpv.first().copied() else { break };
                prev_scores[k] = score;
                have_scores[k] = true;
                self.root_excl.push(first_mv);
                lines.push(RootLine { mv: first_mv, score, pv: lpv });
            }
            self.root_excl.clear();
            if self.aborted || lines.is_empty() {
                if let (true, 1, true, Some(m)) = (self.aborted, mpv, self.prm.partial != 0, self.root_partial) {
                    self.partial_used = true;
                    self.full_best = res.best;
                    res.best = Some(m);
                    let lpv = &self.pv[..self.pv_len[0]];
                    res.pv = if lpv.first() == Some(&m) { lpv.to_vec() } else { vec![m] };
                }
                break;
            }
            if lines.len() > 1 {
                lines.sort_by(|a, b| b.score.cmp(&a.score));
            }
            let score = lines[0].score;
            let pv: Vec<Move> = lines[0].pv.clone();
            res.lines = lines.clone();
            let changed = pv.first().map(|m| Some((m.from, m.to)) != self.root_best).unwrap_or(false);
            if let Some(m) = pv.first() {
                res.best = Some(*m);
                self.root_best = Some((m.from, m.to));
            }
            tc.note_iteration(changed, score);
            if self.prm.tnodes > 0 && mpv == 1 {
                let total = self.nodes - self.root_call_nodes;
                tc.note_best_share((total > 0 && self.root_best_nodes > 0).then(|| (self.root_best_nodes * 100 / total) as u32));
            }
            res.score = score;
            res.pv = pv.clone();
            res.depth = depth;
            let forced = |s: i32| s.abs() > MATE_BOUND && (MATE - s.abs()) <= depth as i32;
            let only_move = root_len == 1 && limits.time_ms.is_some() && limits.depth.is_none();
            let decided = res.lines.iter().all(|l| forced(l.score));
            let hist_dep = self.hmark_root() != hev_it && !only_move && !decided;
            res.hist_dep = hist_dep;
            if self.main {
                let ms = self.start.elapsed().as_millis() as u64;
                let nodes = self.total.load(Ordering::Relaxed) + (self.nodes - self.flushed);
                res.nodes = nodes;
                res.time_ms = ms;
                on_info(&SearchInfo { lines, depth, score, nodes, nps: nodes * 1000 / ms.max(1), time_ms: ms, hashfull: self.tt.hashfull(), pv, hist_dep });
                if only_move {
                    res.early = Some("only_move");
                    break;
                }
                if decided {
                    if depth < max_depth {
                        res.early = Some("decided");
                    }
                    break;
                }
                if tc.stop_after(&self.prm, changed, depth, ms, it_start) {
                    break;
                }
            }
            depth += 1;
        }
        if self.main && self.prm.matkeep > 0 && mpv == 1 {
            self.mat_keep(pos, limits, &mut res, on_info);
        }
        res
    }

    #[inline(never)]
    fn mat_keep(&mut self, pos: &Position<R>, limits: &SearchLimits, res: &mut SearchResult, on_info: &mut dyn FnMut(&SearchInfo)) {
        let Some(best) = res.best else { return };
        if res.depth == 0 || res.early.is_some() || self.partial_used || res.score.abs() >= MATE_BOUND || self.stop.load(Ordering::Relaxed) {
            return;
        }
        if best.captured != 0 || !is_irreversible(pos, &best) || res.pv.first() != Some(&best) || !material_loss(pos, &res.pv) {
            return;
        }
        let mut root = MoveList::new();
        generate(pos, &mut ListSink::<true>(&mut root));
        let excl: Vec<Move> = root.as_slice().iter().filter(|m| is_irreversible(pos, m)).copied().collect();
        if !root.as_slice().iter().any(|m| !is_irreversible(pos, m) && !self.root_ban.contains(m)) {
            return;
        }
        self.aborted = false;
        if let Some(t) = limits.time_ms {
            self.time_limit = Some(self.start.elapsed().as_millis() as u64 + (t / 4).max(1));
        }
        if let Some(n) = limits.nodes {
            self.node_limit = Some(self.total.load(Ordering::Relaxed) + (self.nodes - self.flushed) + (n / 4).max(1));
        }
        self.root_excl = excl;
        let s = self.search_node::<true>(pos, -INF, INF, res.depth as i32, 0);
        self.root_excl.clear();
        if self.aborted {
            return;
        }
        let lpv: Vec<Move> = self.pv[..self.pv_len[0]].to_vec();
        let Some(&rm) = lpv.first() else { return };
        if res.score - s >= self.prm.matkeep {
            return;
        }
        let line = RootLine { mv: rm, score: s, pv: lpv.clone() };
        res.best = Some(rm);
        res.score = s;
        res.pv = lpv.clone();
        res.lines = vec![line.clone()];
        res.hist_dep = true;
        let ms = self.start.elapsed().as_millis() as u64;
        let nodes = self.total.load(Ordering::Relaxed) + (self.nodes - self.flushed);
        res.nodes = nodes;
        res.time_ms = ms;
        on_info(&SearchInfo { lines: vec![line], depth: res.depth, score: s, nodes, nps: nodes * 1000 / ms.max(1), time_ms: ms, hashfull: self.tt.hashfull(), pv: lpv, hist_dep: true });
    }
}

pub fn search<R: Rules, E: Evaluator<R>>(
    pos: &Position<R>,
    history: &[Position<R>],
    eval: &E,
    limits: &SearchLimits,
    stop: &AtomicBool,
    on_info: &mut dyn FnMut(&SearchInfo),
) -> SearchResult {
    let mut tt = Tt::new(limits.hash_mb);
    search_tt(pos, history, eval, limits, stop, on_info, &mut tt)
}

fn make_lmr_tab(c: i32) -> Vec<u8> {
    let mut t = vec![0u8; 64 * 64];
    let c = (c.max(50) as f64) / 100.0;
    for d in 1..64usize {
        for i in 1..64usize {
            t[d * 64 + i] = (((d as f64).ln() * (i as f64).ln() / c) as i32).clamp(0, 8) as u8;
        }
    }
    t
}

pub struct SearchCtx<S> {
    est: Vec<S>,
    pv: Vec<Move>,
    history: Vec<i32>,
    cont: Vec<i16>,
    counter: Vec<Move>,
    keys: Vec<u64>,
    clocks: Vec<u32>,
    lmr: Option<(i32, Vec<u8>)>,
}

impl<S> Default for SearchCtx<S> {
    fn default() -> Self {
        SearchCtx { est: Vec::new(), pv: Vec::new(), history: Vec::new(), cont: Vec::new(), counter: Vec::new(), keys: Vec::new(), clocks: Vec::new(), lmr: None }
    }
}

struct Bufs<S> {
    est: Vec<S>,
    pv: Vec<Move>,
    history: Vec<i32>,
    cont: Vec<i16>,
    counter: Vec<Move>,
    keys: Vec<u64>,
    clocks: Vec<u32>,
}

fn reuse_bufs<R: Rules, E: Evaluator<R>>(c: &mut SearchCtx<E::State>, eval: &E, pos: &Position<R>, prm: &Params, keys: &[u64], clocks: &[u32]) -> Bufs<E::State> {
    let mut est = std::mem::take(&mut c.est);
    if est.len() == MAXPLY + 2 {
        est[0] = eval.root(pos);
        let (root, rest) = est.split_at_mut(1);
        for s in rest {
            s.clone_from(&root[0]);
        }
    } else {
        est = vec![eval.root(pos); MAXPLY + 2];
    }
    let mut pv = std::mem::take(&mut c.pv);
    if pv.len() != MAXPLY * MAXPLY {
        pv = vec![Move::NULL; MAXPLY * MAXPLY];
    }
    let mut history = std::mem::take(&mut c.history);
    if history.len() == 2 * 4096 {
        if prm.keep_hist != 0 {
            for h in history.iter_mut() {
                *h /= 2;
            }
        } else {
            history.fill(0);
        }
    } else {
        history = vec![0; 2 * 4096];
    }
    let (mut cont, mut counter) = (std::mem::take(&mut c.cont), std::mem::take(&mut c.counter));
    cont.clear();
    counter.clear();
    if prm.cm > 0 {
        cont.resize(64 * 4096, 0);
        counter.resize(4096, Move::NULL);
    }
    let (mut k, mut cl) = (std::mem::take(&mut c.keys), std::mem::take(&mut c.clocks));
    k.clear();
    k.extend_from_slice(keys);
    cl.clear();
    cl.extend_from_slice(clocks);
    Bufs { est, pv, history, cont, counter, keys: k, clocks: cl }
}

pub fn search_tt<R: Rules, E: Evaluator<R>>(
    pos: &Position<R>,
    history: &[Position<R>],
    eval: &E,
    limits: &SearchLimits,
    stop: &AtomicBool,
    on_info: &mut dyn FnMut(&SearchInfo),
    tt_slot: &mut Tt,
) -> SearchResult {
    search_tt_db(pos, history, eval, limits, stop, on_info, tt_slot, None)
}

#[allow(clippy::too_many_arguments)]
pub fn search_tt_db<R: Rules, E: Evaluator<R>>(
    pos: &Position<R>,
    history: &[Position<R>],
    eval: &E,
    limits: &SearchLimits,
    stop: &AtomicBool,
    on_info: &mut dyn FnMut(&SearchInfo),
    tt_slot: &mut Tt,
    egdb: Option<&Egdb<R>>,
) -> SearchResult {
    search_tt_db_recall(pos, history, eval, limits, stop, on_info, tt_slot, egdb, None)
}

fn recall_seed<R: Rules>(pos: &Position<R>, root_clock: u32, rc: &RecallCtx, tt: &Tt) -> (bool, u64, u64) {
    let mut seen = std::collections::HashSet::new();
    seen.insert(pos.hash);
    let mut frontier = vec![(*pos, root_clock)];
    let (mut root_hit, mut placed, mut probed) = (false, 0u64, 0u64);
    let cap = rc.max_positions.max(1);
    let mt = if rc.mirror { crate::recall::color_tables::<R>() } else { None };
    let mut mirrored = 0u64;
    let mut rej_clock = 0u64;
    for ply in 0..=rc.plies {
        let mut next = Vec::new();
        for &(p, clk) in &frontier {
            probed += 1;
            if let Some((e, by_mirror)) = rc.mem.get_or_mirror(&p, rc.eval_id, mt) {
                if recall_clock_stale::<R>(clk, e.depth as i32) {
                    rej_clock += 1;
                } else {
                    let ok = tt.store_recall(p.hash, e.score, e.depth as i32, e.bound, e.mv);
                    if ply == 0 {
                        root_hit = true;
                    } else if ok {
                        placed += 1;
                        mirrored += by_mirror as u64;
                    }
                }
            }
            if ply < rc.plies && seen.len() < cap {
                let mut l = MoveList::new();
                generate(&p, &mut ListSink::<true>(&mut l));
                for m in l.as_slice() {
                    let c = p.make(m);
                    if seen.len() < cap && seen.insert(c.hash) {
                        next.push((c, if is_irreversible(&p, m) { 0 } else { clk + 1 }));
                    }
                }
            }
        }
        frontier = next;
    }
    rc.mem.stats.mirror_placed.fetch_add(mirrored, Ordering::Relaxed);
    rc.mem.stats.rej_clock.fetch_add(rej_clock, Ordering::Relaxed);
    (root_hit, placed, probed)
}

#[allow(clippy::too_many_arguments)]
pub fn search_tt_db_recall<R: Rules, E: Evaluator<R>>(
    pos: &Position<R>,
    history: &[Position<R>],
    eval: &E,
    limits: &SearchLimits,
    stop: &AtomicBool,
    on_info: &mut dyn FnMut(&SearchInfo),
    tt_slot: &mut Tt,
    egdb: Option<&Egdb<R>>,
    recall: Option<&RecallCtx>,
) -> SearchResult {
    search_tt_db_recall_ctx(pos, history, eval, limits, stop, on_info, tt_slot, egdb, recall, None)
}

#[allow(clippy::too_many_arguments)]
pub fn search_tt_db_recall_ctx<R: Rules, E: Evaluator<R>>(
    pos: &Position<R>,
    history: &[Position<R>],
    eval: &E,
    limits: &SearchLimits,
    stop: &AtomicBool,
    on_info: &mut dyn FnMut(&SearchInfo),
    tt_slot: &mut Tt,
    egdb: Option<&Egdb<R>>,
    recall: Option<&RecallCtx>,
    mut ctx: Option<&mut SearchCtx<E::State>>,
) -> SearchResult {
    let start = Instant::now();
    let mut prm = Params::parse(&limits.opts);
    let threads = if limits.threads == 0 { default_threads() } else { limits.threads }.clamp(1, 64);
    tt_slot.new_search();
    let tt: &Tt = tt_slot;
    let mut keys = Vec::with_capacity(history.len() + MAXPLY + 4);
    let mut clocks = Vec::with_capacity(history.len() + MAXPLY + 4);
    let mut clock = 0u32;
    let mut prev: Option<&Position<R>> = None;
    for h in history.iter().chain(std::iter::once(pos)) {
        clock = match prev {
            None => 0,
            Some(p) => clock_between(p, h, clock),
        };
        keys.push(h.hash);
        clocks.push(clock);
        prev = Some(h);
    }
    let mut root = MoveList::new();
    generate(pos, &mut ListSink::<true>(&mut root));
    if root.len == 0 {
        return SearchResult { lines: vec![], best: None, score: -MATE, pv: vec![], depth: 0, nodes: 0, time_ms: 0, early: None, hist_dep: false };
    }
    let want_multi = limits.multipv > 1 && root.len > 1;
    let root_v = egdb.filter(|db| prm.dbdraw > 0 && pos.occ().count_ones() <= db.max_pieces()).and_then(|db| db.probe(pos));
    let db_root_draw = root_v == Some(0);
    let db_sat_root = egdb
        .filter(|db| pos.occ().count_ones() <= db.max_pieces())
        .and_then(|db| db.probe(pos))
        .is_some_and(|v| v == crate::egdb::SAT || v == -crate::egdb::SAT);
    if root_v.is_some_and(|v| v != 0) {
        prm.dbdraw = 0;
    }
    let mut root_ban: Vec<Move> = Vec::new();
    if let (true, false, Some(db)) = (db_root_draw, want_multi, egdb) {
        for mv in root.as_slice() {
            if db.probe(&pos.make(mv)).is_some_and(|v| v > 0) {
                root_ban.push(*mv);
            }
        }
    }
    let root_clock = *clocks.last().unwrap();
    let db_clock = prm.dbclk >= 2 || (prm.dbclk == 1 && root_clock > 0);
    let root_stale = db_clock
        && egdb.filter(|db| pos.occ().count_ones() <= db.max_pieces()).and_then(|db| db.probe(pos)).is_some_and(|v| {
            v != 0 && v != crate::egdb::SAT && v != -crate::egdb::SAT && db_clock_stale::<R>(root_clock, crate::egdb::to_score(v, 0), 0)
        });
    if let Some((mv, score)) = egdb.filter(|_| !want_multi && !db_root_draw && !root_stale).and_then(|db| db.root_pick(pos)) {
        let ms = start.elapsed().as_millis() as u64;
        let line = RootLine { mv, score, pv: vec![mv] };
        on_info(&SearchInfo { lines: vec![line.clone()], depth: 1, score, nodes: 1, nps: 0, time_ms: ms, hashfull: 0, pv: vec![mv], hist_dep: false });
        return SearchResult { lines: vec![line], best: Some(mv), score, pv: vec![mv], depth: 1, nodes: 1, time_ms: ms, early: Some("egdb"), hist_dep: false };
    }
    let first = root.as_slice().iter().find(|m| !root_ban.contains(m)).copied().unwrap_or(root[0]);
    let root_len = root.len - root_ban.len();
    if let Some(rc) = recall {
        let (root_hit, placed, probed) = recall_seed(pos, *clocks.last().unwrap(), rc, tt);
        let st = &rc.mem.stats;
        st.searches.fetch_add(1, Ordering::Relaxed);
        st.searches_hit.fetch_add((placed > 0) as u64, Ordering::Relaxed);
        st.root_hits.fetch_add(root_hit as u64, Ordering::Relaxed);
        st.placed.fetch_add(placed, Ordering::Relaxed);
        st.probed.fetch_add(probed, Ordering::Relaxed);
    }
    let lmr_tab = match ctx.as_mut().and_then(|c| c.lmr.take()) {
        Some((c, t)) if c == prm.lmr_c => t,
        _ => make_lmr_tab(prm.lmr_c),
    };
    let halt = AtomicBool::new(false);
    let total = AtomicU64::new(0);
    let fresh = || Bufs {
        est: vec![eval.root(pos); MAXPLY + 2],
        pv: vec![Move::NULL; MAXPLY * MAXPLY],
        history: vec![0; 2 * 4096],
        cont: vec![0; if prm.cm > 0 { 64 * 4096 } else { 0 }],
        counter: vec![Move::NULL; if prm.cm > 0 { 4096 } else { 0 }],
        keys: keys.clone(),
        clocks: clocks.clone(),
    };
    let mk = |tid: usize, b: Bufs<E::State>| Searcher::<R, E> {
        tt,
        eval,
        est: b.est,
        stop,
        halt: &halt,
        total: &total,
        main: tid == 0,
        start,
        time_limit: limits.time_ms,
        node_limit: limits.nodes,
        nodes: 0,
        flushed: 0,
        aborted: false,
        keys: b.keys,
        clocks: b.clocks,
        root_idx: history.len(),
        hev: 0,
        hroot: 0,
        hlev: [0; MAXPLY + 2],
        killers: [[Move::NULL; 2]; MAXPLY],
        history: b.history,
        cont: b.cont,
        counter: b.counter,
        excl: [Move::NULL; MAXPLY],
        mv_stack: [Move::NULL; MAXPLY + 2],
        pv: b.pv,
        pv_len: [0; MAXPLY],
        root_best: limits.root_hint,
        root_excl: Vec::new(),
        duty: limits.duty.clamp(1, 100),
        work: std::time::Duration::ZERO,
        last_tick: Instant::now(),
        lmr_tab: lmr_tab.clone(),
        prm,
        egdb,
        db_root_draw,
        skip_probe: false,
        db_sat_root,
        root_ban: root_ban.clone(),
        recall_pv: 0,
        recall_cut: 0,
        root_partial: None,
        root_call_nodes: 0,
        root_best_nodes: 0,
        partial_used: false,
        full_best: None,
        in_pc: false,
        db_clock,
        corr: vec![0; if prm.corr > 0 { 2 * CORR_SIZE } else { 0 }],
        cont2: vec![0; if prm.ch24 > 0 { 64 * 4096 } else { 0 }],
        cont4: vec![0; if prm.ch24 > 0 { 64 * 4096 } else { 0 }],
        caph: vec![0; if prm.caph > 0 { 2 * 4096 } else { 0 }],
        zz_left: 0,
        ss_eval: [i32::MIN; MAXPLY + 2],
        cut: [false; MAXPLY + 2],
        _r: std::marker::PhantomData,
    };
    let res = std::thread::scope(|sc| {
        let mut handles = vec![];
        for tid in 1..threads {
            let mut s = mk(tid, fresh());
            let qos = limits.qos;
            let h = std::thread::Builder::new().stack_size(64 << 20).spawn_scoped(sc, move || {
                crate::qos::apply(qos);
                s.iterate(pos, root_len, limits, tid, first, &mut |_| {});
                s.flush_nodes();
            });
            if let Ok(h) = h {
                handles.push(h);
            }
        }
        crate::qos::apply(limits.qos);
        let b = match ctx.as_deref_mut() {
            Some(c) => reuse_bufs(c, eval, pos, &prm, &keys, &clocks),
            None => fresh(),
        };
        let mut s = mk(0, b);
        let r = s.iterate(pos, root_len, limits, 0, first, on_info);
        s.flush_nodes();
        halt.store(true, Ordering::Relaxed);
        for h in handles {
            let _ = h.join();
        }
        if let Some(c) = ctx.as_deref_mut() {
            c.est = std::mem::take(&mut s.est);
            c.pv = std::mem::take(&mut s.pv);
            c.history = std::mem::take(&mut s.history);
            c.cont = std::mem::take(&mut s.cont);
            c.counter = std::mem::take(&mut s.counter);
            c.keys = std::mem::take(&mut s.keys);
            c.clocks = std::mem::take(&mut s.clocks);
            c.lmr = Some((prm.lmr_c, std::mem::take(&mut s.lmr_tab)));
        }
        (r, s.recall_pv, s.recall_cut, s.partial_used.then_some(s.full_best))
    });
    let (mut res, used_pv, used_cut, partial_full) = res;
    res.nodes = total.load(Ordering::Relaxed);
    res.time_ms = start.elapsed().as_millis() as u64;
    if let Some(rc) = recall {
        let st = &rc.mem.stats;
        st.used_pv.fetch_add(used_pv, Ordering::Relaxed);
        st.used_cut.fetch_add(used_cut, Ordering::Relaxed);
        st.searches_used.fetch_add((used_pv + used_cut > 0) as u64, Ordering::Relaxed);
        if rc.record {
            let whole = limits.multipv <= 1 && root_ban.is_empty() && !db_root_draw;
            let early_ok = matches!(res.early, None | Some("decided"));
            let best = partial_full.unwrap_or(res.best);
            let why = if best.is_none() || res.depth < 1 {
                Some(&st.rej_incomplete)
            } else if !early_ok {
                Some(&st.rej_early)
            } else if !whole {
                Some(&st.rej_partial)
            } else if res.hist_dep {
                Some(&st.rej_hist)
            } else {
                None
            };
            if let Some(c) = why {
                c.fetch_add(1, Ordering::Relaxed);
            }
            match best.filter(|_| why.is_none()) {
                Some(b) => {
                    let e = RecallEntry {
                        score: res.score,
                        depth: res.depth.min(255) as u8,
                        bound: BOUND_EXACT,
                        mv: Some((b.from, b.to)),
                        nodes: res.nodes,
                        eval_id: rc.eval_id,
                    };
                    if rc.mem.put(pos, e) {
                        st.recorded.fetch_add(1, Ordering::Relaxed);
                    } else {
                        st.kept_deeper.fetch_add(1, Ordering::Relaxed);
                    }
                }
                None => {
                    st.rejected.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
    res
}

#[cfg(test)]
mod time_mgmt_tests {
    use super::*;

    #[test]
    fn qegdb_off_by_default() {
        assert_eq!(Params::default().qegdb, 0);
        assert_eq!(Params::parse("").get("qegdb"), Some(0));
        let p = Params::parse("qegdb=1");
        assert_eq!((p.qegdb, p.get("qegdb")), (1, Some(1)));
    }

    #[test]
    fn parse_keys() {
        let p = Params::parse("nostart=70,nopred=1,easy=4,easy_cp=10,easy_pct=30");
        assert_eq!((p.nostart, p.nopred, p.easy, p.easy_cp, p.easy_pct), (70, 1, 4, 10, 30));
    }
}

#[cfg(test)]
mod zz_tests {
    use super::*;
    use crate::movegen::CountSink;
    use crate::rules::{Czech, Italian, Russian, Turkish};

    fn check<R: Rules>(games: usize, seed: u64) {
        let mut s = seed;
        for _ in 0..games {
            let mut pos = Position::<R>::start();
            for _ in 0..150 {
                let mut caps = MoveList::new();
                generate_captures(&pos, &mut ListSink::<true>(&mut caps));
                assert_eq!(side_has_capture(&pos), caps.len > 0, "{} {}", R::NAME, pos.to_dfen());
                let mut all = MoveList::new();
                generate(&pos, &mut ListSink::<true>(&mut all));
                if all.len == 0 {
                    break;
                }
                if caps.len == 0 {
                    let mut c = CountSink::default();
                    generate(&pos, &mut c);
                    for lim in [0u32, 3, 6, 10, 40] {
                        assert_eq!(quiet_count_at_most(&pos, lim), c.count <= lim as u64, "{} {} lim {lim}", R::NAME, pos.to_dfen());
                    }
                }
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                pos = pos.make(&all[(s % all.len as u64) as usize]);
            }
        }
    }

    #[test]
    fn masks_match_movegen() {
        check::<Russian<true>>(300, 0x1234_5678_9abc_def1);
        check::<Czech<true>>(100, 0x2234_5678_9abc_def1);
        check::<Italian<true>>(100, 0x3234_5678_9abc_def1);
        check::<Turkish<true>>(100, 0x4234_5678_9abc_def1);
    }
}
