
use crate::egdb::Egdb;
use crate::eval::Evaluator;
use crate::movegen::{generate, generate_captures, has_quiet, ListSink, Move, MoveList, MAX_MOVES};
use crate::position::Position;
use crate::rules::Rules;
use crate::recall::{RecallCtx, RecallEntry};
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
            _ => return None,
        })
    }
}

pub(crate) fn no_new_iteration(p: &Params, ms: u64, hard: u64, last: u64, prev: u64) -> bool {
    if p.nostart > 0 && ms * 100 > hard * p.nostart as u64 {
        return true;
    }
    if p.nopred != 0 && last > 0 {
        let growth = if prev > 0 { (last * 100 / prev).max(150) } else { 150 };
        if ms + last * growth / 100 > hard {
            return true;
        }
    }
    false
}

pub(crate) fn easy_move_stop(p: &Params, ms: u64, hard: u64, stable: u32, score_delta: i32) -> bool {
    p.easy > 0 && stable >= p.easy as u32 && score_delta.abs() <= p.easy_cp && ms * 100 >= hard * p.easy_pct as u64
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
    partial_used: bool,
    full_best: Option<Move>,
    in_pc: bool,
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
fn recall_clock_stale<R: Rules>(clk: u32, depth: i32) -> bool {
    R::DRAW_MOVES > 0 && clk > 0 && clk as i64 + depth as i64 >= 2 * R::DRAW_MOVES as i64
}

const HIST_MAX: i32 = 16000;

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

    fn qsearch(&mut self, pos: &Position<R>, mut alpha: i32, beta: i32, ply: usize) -> i32 {
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
                if s != 0 || self.prm.dbdraw == 0 {
                    return s;
                }
            }
        }
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
                let st = self.eval.eval_at(pos, &mut self.est, ply);
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
                return self.eval.eval_at(pos, &mut self.est, ply);
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
                return 0;
            }
            if let Some(i) = self.repetition_at(pos.hash) {
                let v = self.draw_value(pos, ply);
                if i < self.root_idx || v != 0 {
                    self.hev += 1;
                }
                return v;
            }
            if R::DRAW_MOVES > 0 && self.clocks[self.keys.len() - 1] >= 2 * R::DRAW_MOVES {
                self.hev += 1;
                return self.draw_value(pos, ply);
            }
            alpha = alpha.max(-MATE + ply as i32);
            beta = beta.min(MATE - ply as i32 - 1);
            if alpha >= beta {
                return alpha;
            }
            if !(self.prm.dbdraw > 0 && std::mem::take(&mut self.skip_probe)) {
                if let Some(s) = self.egdb.and_then(|db| db.probe_score(pos, ply)) {
                    if self.db_sat_root && depth > 0 && ply < MAXPLY - 2 && crate::egdb::is_sat_score(s) {
                        db_sat = true;
                    } else if s != 0 || self.prm.dbdraw == 0 {
                        return s;
                    } else {
                        return self.db_draw_node::<PV>(pos, alpha, beta, depth, ply);
                    }
                }
            }
        }
        if ply >= MAXPLY - 2 {
            return self.eval.eval_at(pos, &mut self.est, ply);
        }
        if depth <= 0 {
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
        let hev0 = self.hev;
        if let Some(e) = self.tt.probe(key) {
            tt_move = e.mv;
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
                    self.hev += tt_dep as u64;
                    self.recall_cut += e.recall as u64;
                    return s;
                }
            } else if !stale && PV && e.recall && ply > 0 && !has_excl && e.bound == BOUND_EXACT && e.depth >= depth {
                self.recall_pv += 1;
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

        if !PV && !has_excl && !tactical && ply > 0 && !db_sat {
            if prm.rfp > 0 && depth <= 4 && beta.abs() < MATE_BOUND {
                let se = self.eval.eval_at(pos, &mut self.est, ply);
                if se - prm.rfp * depth >= beta {
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
            if rbeta < MATE_BOUND && !tt_against && self.eval.eval_at(pos, &mut self.est, ply) >= beta + prm.pcs {
                self.in_pc = true;
                let s = self.search_node::<false>(pos, rbeta - 1, rbeta, pd, ply);
                self.in_pc = false;
                if self.aborted {
                    return 0;
                }
                if s >= rbeta {
                    self.tt.store(key, score_to_tt(s, ply), pd + 1, BOUND_LOWER, None, self.hev != hev0);
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
        let mut sc = Scores::new();
        for i in 0..n {
            let m = &list[i];
            let s = if tt_move == Some((m.from, m.to)) {
                1 << 24
            } else if m.captured != 0 {
                (1 << 20) + m.captured.count_ones() as i32 * 16 + m.promo as i32
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
                h + m.promo as i32 * 1000
            };
            sc.set(i, s);
        }

        let mut best = -INF;
        let mut best_mv: Option<Move> = None;
        let prev_clock = *self.clocks.last().unwrap();
        let mut tried_quiets = [Move::NULL; 24];
        let mut nq = 0usize;
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
            let quiet = mv.captured == 0;
            if prm.lmp > 0
                && !PV
                && quiet
                && !mv.promo
                && depth <= prm.lmp
                && searched as i32 >= prm.lmp_base + depth * depth
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
                        self.hev += tt_dep as u64;
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
                    r = r.clamp(0, (nd - 1).max(0));
                }
                score = -self.search_node::<false>(&child, -alpha - 1, -alpha, nd - r, ply + 1);
                if score > alpha && r > 0 {
                    score = -self.search_node::<false>(&child, -alpha - 1, -alpha, nd, ply + 1);
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
                        break;
                    }
                }
            }
            if quiet && nq < tried_quiets.len() {
                tried_quiets[nq] = mv;
                nq += 1;
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
        if ply > 0 || (self.root_excl.is_empty() && self.root_ban.is_empty()) {
            self.tt.store(key, score_to_tt(best, ply), depth, bound, best_mv.map(|m| (m.from, m.to)), self.hev != hev0);
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
            return (self.eval.eval_at(pos, &mut self.est, ply) / 8).clamp(-b, b);
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
        let hard = limits.time_ms;
        let mut soft = hard.map(|h| h * self.prm.soft as u64 / 100);
        let sd = limits.start_depth.min(max_depth);
        let mut depth = if sd > 1 { sd + (tid % 3) as u32 } else if tid > 0 { 1 + (tid % 3) as u32 } else { 1 };
        depth = depth.min(max_depth);
        let (mut it_last, mut it_prev, mut stable, mut last_score) = (0u64, 0u64, 0u32, None::<i32>);
        while depth <= max_depth {
            let it_start = self.start.elapsed().as_millis() as u64;
            let hev_it = self.hev;
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
            let score_delta = last_score.map_or(i32::MAX, |s| score - s);
            last_score = Some(score);
            stable = if changed { 0 } else { stable + 1 };
            res.score = score;
            res.pv = pv.clone();
            res.depth = depth;
            let forced = |s: i32| s.abs() > MATE_BOUND && (MATE - s.abs()) <= depth as i32;
            let only_move = root_len == 1 && limits.time_ms.is_some() && limits.depth.is_none();
            let decided = res.lines.iter().all(|l| forced(l.score));
            let hist_dep = self.hev != hev_it && !only_move && !decided;
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
                if let (Some(s), Some(h)) = (soft, hard) {
                    if changed && depth > 4 {
                        let ext = (s * 3 / 2).min(h * 9 / 10);
                        soft = Some(if self.prm.softfix != 0 { s.max(ext) } else { ext });
                    }
                    if ms >= soft.unwrap() {
                        break;
                    }
                }
                if let Some(h) = hard {
                    it_prev = it_last;
                    it_last = ms.saturating_sub(it_start);
                    if no_new_iteration(&self.prm, ms, h, it_last, it_prev) || easy_move_stop(&self.prm, ms, h, stable, score_delta) {
                        break;
                    }
                }
            }
            depth += 1;
        }
        res
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
    if let Some((mv, score)) = egdb.filter(|_| !want_multi && !db_root_draw).and_then(|db| db.root_pick(pos)) {
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
        partial_used: false,
        full_best: None,
        in_pc: false,
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
    fn defaults_are_off() {
        let p = Params::default();
        assert!(!no_new_iteration(&p, 9_999, 10_000, 5_000, 1_000));
        assert!(!easy_move_stop(&p, 9_999, 10_000, 100, 0));
    }

    #[test]
    fn nostart_by_fraction() {
        let p = Params::parse("nostart=60");
        assert!(!no_new_iteration(&p, 6_000, 10_000, 100, 50));
        assert!(no_new_iteration(&p, 6_001, 10_000, 100, 50));
    }

    #[test]
    fn nopred_by_growth() {
        let p = Params::parse("nopred=1");
        assert!(!no_new_iteration(&p, 7_900, 10_000, 1_000, 500));
        assert!(no_new_iteration(&p, 8_100, 10_000, 1_000, 500));
        assert!(no_new_iteration(&p, 8_600, 10_000, 1_000, 990));
        assert!(!no_new_iteration(&p, 8_400, 10_000, 1_000, 990));
        assert!(!no_new_iteration(&p, 9_999, 10_000, 0, 0));
    }

    #[test]
    fn easy_move_needs_stability_and_time() {
        let p = Params::parse("easy=3,easy_cp=20,easy_pct=40");
        assert!(!easy_move_stop(&p, 5_000, 10_000, 2, 0));
        assert!(!easy_move_stop(&p, 5_000, 10_000, 3, 25));
        assert!(!easy_move_stop(&p, 3_900, 10_000, 5, -10));
        assert!(easy_move_stop(&p, 4_000, 10_000, 3, -20));
    }

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
