
use crate::datadir;
use crate::egdb::Egdb;
use crate::eval::{Evaluator, SimpleEval};
use crate::facade::{GameConfig, VariantId};
use crate::learn::select::{load_cached, with_eval, EvalRun, EvalSpec};
use crate::position::{Position, Side as Colour};
use crate::rules::{American, Brazilian, Czech, International, Italian, Rules, Russian, Spanish, Thai, Turkish};
use crate::recall::{Recall, RecallCtx};
use crate::search::{search_tt_db, search_tt_db_recall_ctx, SearchCtx, SearchLimits};
use crate::selfplay::dispatch_cfg;
use crate::selfplay::extract::legal;
use crate::selfplay::play::{natural_end, Rng};
use crate::tt::Tt;
pub use crate::timectl::GameClock;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const SPRT_BOUND: f64 = 2.94;

pub fn elo_of(p: f64) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    -400.0 * (1.0 / p - 1.0).log10()
}

pub fn pair_moments(penta: &[u64; 5]) -> (f64, f64, f64) {
    let n: u64 = penta.iter().sum();
    if n == 0 {
        return (0.0, 0.5, 0.0);
    }
    let nf = n as f64;
    let mean: f64 = penta.iter().enumerate().map(|(i, c)| *c as f64 * i as f64 * 0.25).sum::<f64>() / nf;
    let var: f64 = penta.iter().enumerate().map(|(i, c)| *c as f64 * (i as f64 * 0.25 - mean).powi(2)).sum::<f64>() / nf;
    (nf, mean, var)
}

pub fn nelo(penta: &[u64; 5]) -> f64 {
    let (n, mean, var) = pair_moments(penta);
    let sigma = var.sqrt() * std::f64::consts::SQRT_2;
    if n <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    347.43 * (mean - 0.5) / sigma
}

pub fn elo_interval(penta: &[u64; 5]) -> (f64, f64, f64) {
    let (n, mean, var) = pair_moments(penta);
    let se = if n > 0.0 { (var / n).sqrt() } else { 0.0 };
    (elo_of(mean), elo_of(mean - 1.96 * se), elo_of(mean + 1.96 * se))
}

pub fn sprt_llr(penta: &[u64; 5], e0: f64, e1: f64) -> Option<f64> {
    let (n, mean, var) = pair_moments(penta);
    let (s0, s1) = (1.0 / (1.0 + 10f64.powf(-e0 / 400.0)), 1.0 / (1.0 + 10f64.powf(-e1 / 400.0)));
    if var > 1e-9 {
        Some(n * (s1 - s0) * (2.0 * mean - s0 - s1) / (2.0 * var))
    } else {
        None
    }
}

pub struct Throttle {
    work: Duration,
    last: Instant,
}

impl Default for Throttle {
    fn default() -> Self {
        Throttle::new()
    }
}

impl Throttle {
    pub fn new() -> Throttle {
        Throttle { work: Duration::ZERO, last: Instant::now() }
    }
    pub fn tick(&mut self, duty: u32, cancel: &AtomicBool) {
        let now = Instant::now();
        let duty = duty.clamp(1, 100);
        if duty >= 100 {
            self.work = Duration::ZERO;
            self.last = now;
            return;
        }
        self.work += now.duration_since(self.last);
        if self.work >= Duration::from_millis(20) {
            let mut left = self.work * (100 - duty) / duty;
            self.work = Duration::ZERO;
            while left > Duration::ZERO && !cancel.load(Ordering::Relaxed) {
                let step = left.min(Duration::from_millis(5));
                std::thread::sleep(step);
                left -= step;
            }
        }
        self.last = Instant::now();
    }
}

#[derive(Clone, Debug)]
pub struct Side {
    pub eval: EvalSpec,
    pub opts: String,
    pub nodes: Option<u64>,
    pub boost: Option<(u32, u32)>,
    pub egdb_root: Option<PathBuf>,
    pub egdb_max_pieces: Option<u32>,
}

impl Side {
    pub fn simple(nodes: u64) -> Side {
        Side { eval: EvalSpec::Simple, opts: String::new(), nodes: Some(nodes), boost: None, egdb_root: None, egdb_max_pieces: None }
    }
}

#[derive(Clone, Debug)]
pub struct MatchParams {
    pub a: Side,
    pub b: Side,
    pub time_ms: Option<u64>,
    pub pairs_max: u32,
    pub sprt: Option<(f64, f64)>,
    pub seed: u64,
    pub book_plies: (u32, u32),
    pub balance_cp: i32,
    pub balance_lo: i32,
    pub max_plies: u32,
    pub openings: Option<Arc<Vec<String>>>,
    pub hash_mb: usize,
    pub threads: usize,
    pub use_egdb: bool,
    pub adj_egdb_root: Option<PathBuf>,
    pub adj_decided_only: bool,
    pub state_path: Option<PathBuf>,
    pub record_moves: bool,
    pub ext: MatchExt,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatchExt {
    pub side_threads: [usize; 2],
    pub clock: Option<GameClock>,
    pub adj_eval: AdjEval,
    pub side_hash_mb: [usize; 2],
}

impl Default for MatchExt {
    fn default() -> Self {
        MatchExt { side_threads: [1, 1], clock: None, adj_eval: AdjEval::default(), side_hash_mb: [0, 0] }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdjEval {
    pub win: Option<(i32, u32)>,
    pub draw: Option<(i32, u32, u32)>,
    pub rep: Option<i32>,
}

impl AdjEval {
    fn judge<R: Rules>(&self, scores: &[i32], hist: &[Position<R>]) -> Option<i32> {
        if let Some((cp, n)) = self.win {
            let n = n.max(1) as usize;
            if scores.len() >= n {
                let tail = &scores[scores.len() - n..];
                if tail.iter().all(|&s| s >= cp) {
                    return Some(1);
                }
                if tail.iter().all(|&s| s <= -cp) {
                    return Some(-1);
                }
            }
        }
        if let Some((cp, n, after)) = self.draw {
            let n = n.max(1) as usize;
            if hist.len() > after as usize && scores.len() >= n && scores[scores.len() - n..].iter().all(|s| s.abs() <= cp) {
                return Some(0);
            }
        }
        if let (Some(cp), Some(pos), Some(&last)) = (self.rep, hist.last(), scores.last()) {
            if last.abs() <= cp && hist.iter().filter(|h| *h == pos).count() >= 2 {
                return Some(0);
            }
        }
        None
    }
}

impl Default for MatchParams {
    fn default() -> Self {
        MatchParams {
            a: Side::simple(5000),
            b: Side::simple(5000),
            time_ms: None,
            pairs_max: 100,
            sprt: None,
            seed: 12345,
            book_plies: (6, 12),
            balance_cp: 60,
            balance_lo: 0,
            max_plies: 300,
            openings: None,
            hash_mb: 8,
            threads: 0,
            use_egdb: false,
            adj_egdb_root: None,
            adj_decided_only: false,
            state_path: None,
            record_moves: false,
            ext: MatchExt::default(),
        }
    }
}

pub struct MatchCtl {
    pub cancel: AtomicBool,
    pub duty: AtomicU32,
    pub qos: AtomicU32,
    pub progress: AtomicU64,
}

impl Default for MatchCtl {
    fn default() -> Self {
        MatchCtl { cancel: AtomicBool::new(false), duty: AtomicU32::new(100), qos: AtomicU32::new(0), progress: AtomicU64::new(0) }
    }
}

impl MatchCtl {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug, Default)]
pub struct MatchReport {
    pub pairs: u32,
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
    pub penta: [u64; 5],
    pub score: f64,
    pub elo: f64,
    pub elo_lo: f64,
    pub elo_hi: f64,
    pub llr: Option<f64>,
    pub sprt: Option<bool>,
    pub reached_max: bool,
    pub cancelled: bool,
    pub resumed_pairs: u32,
    pub control: String,
    pub eval_id: [u32; 2],
    pub avg_plies: f64,
    pub secs: f64,
    pub moves: [u64; 2],
    pub depth_sum: [u64; 2],
    pub nodes_sum: [u64; 2],
    pub ms_sum: [u64; 2],
}

impl MatchReport {
    pub fn nps(&self) -> [f64; 2] {
        [0, 1].map(|i| if self.ms_sum[i] == 0 { 0.0 } else { self.nodes_sum[i] as f64 * 1000.0 / self.ms_sum[i] as f64 })
    }
}

impl MatchReport {
    pub fn decided(&self) -> bool {
        self.sprt.is_some() || self.reached_max
    }
    pub fn line(&self) -> String {
        let mut s = format!(
            "W/D/L {}/{}/{}  pairs {}  score {:.3}  Elo {:+.1} [{:+.1}, {:+.1}] (95%)  penta {:?}",
            self.wins, self.draws, self.losses, self.pairs, self.score, self.elo, self.elo_lo, self.elo_hi, self.penta
        );
        let games = 2 * self.pairs;
        let dr = if games > 0 { self.draws as f64 / games as f64 } else { 0.0 };
        s += &format!("  draws {dr:.3}  nElo {:+.1}", nelo(&self.penta));
        if let Some(l) = self.llr {
            s += &format!("  LLR {l:.2} [-2.94, 2.94]");
        }
        s
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Sess {
    moves: [u64; 2],
    depth: [u64; 2],
    nodes: [u64; 2],
    ms: [u64; 2],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PairRec {
    p: u32,
    s: [u8; 2],
    pl: [u32; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mv: Option<[String; 2]>,
    #[serde(skip)]
    sess: Sess,
}

fn fnv(h: &mut u64, b: &[u8]) {
    for &x in b {
        *h ^= x as u64;
        *h = h.wrapping_mul(0x100_0000_01B3);
    }
}

fn check_spec(cfg: &GameConfig, spec: &EvalSpec) -> Result<u32, String> {
    match spec {
        EvalSpec::Simple => Ok(0),
        EvalSpec::File(p) => {
            let l = load_cached(p)?;
            if let Some(meta) = &l.model.meta {
                if meta.variant != cfg.variant.name() || meta.forced != cfg.forced_capture {
                    return Err(format!(
                        "веса {} для {}:{}, а матч идёт в {}",
                        p.display(),
                        meta.variant,
                        if meta.forced { "forced" } else { "free" },
                        datadir::cfg_name(cfg.variant.name(), cfg.forced_capture)
                    ));
                }
            }
            Ok(l.eval_id)
        }
    }
}

pub fn params_hash(cfg: &GameConfig, p: &MatchParams, ids: [u32; 2]) -> String {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    let s = format!(
        "{}|{:08x}|{:08x}|{}|{}|{:?}|{:?}|{:?}|{}|{:?}|{}|{}|{}|{}",
        datadir::cfg_name(cfg.variant.name(), cfg.forced_capture),
        ids[0],
        ids[1],
        p.a.opts.trim(),
        p.b.opts.trim(),
        p.a.nodes,
        p.b.nodes,
        p.time_ms,
        p.seed,
        p.book_plies,
        p.balance_cp,
        p.max_plies,
        p.hash_mb,
        p.use_egdb
    );
    let mut s = s;
    if p.a.boost.is_some() || p.b.boost.is_some() {
        s += &format!("|boost {:?} {:?}", p.a.boost, p.b.boost);
    }
    if p.a.egdb_root.is_some() || p.b.egdb_root.is_some() {
        s += &format!("|egdb_side {:?} {:?}", p.a.egdb_root, p.b.egdb_root);
    }
    if p.a.egdb_max_pieces.is_some() || p.b.egdb_max_pieces.is_some() {
        s += &format!("|egdb_cap {:?} {:?}", p.a.egdb_max_pieces, p.b.egdb_max_pieces);
    }
    if p.adj_egdb_root.is_some() || p.adj_decided_only {
        s += &format!("|adj {:?} {}", p.adj_egdb_root, p.adj_decided_only);
    }
    if p.ext.side_threads != [1, 1] {
        s += &format!("|sthreads {:?}", p.ext.side_threads);
    }
    if let Some(c) = &p.ext.clock {
        s += &format!("|clock {} {} {} {} {} {}", c.base_ms, c.inc_ms, c.moves_to_go, c.front_moves, c.front_pct, c.max_pct);
    }
    if p.ext.adj_eval != AdjEval::default() {
        s += &format!("|adjeval {:?}", p.ext.adj_eval);
    }
    let sh = [side_hash_mb(p, 0), side_hash_mb(p, 1)];
    if sh != [p.hash_mb, p.hash_mb] {
        s += &format!("|shash {sh:?}");
    }
    if p.balance_lo != 0 {
        s += &format!("|balance_lo {}", p.balance_lo);
    }
    if let Some(o) = &p.openings {
        let mut oh = 0xCBF2_9CE4_8422_2325u64;
        for d in o.iter() {
            fnv(&mut oh, d.as_bytes());
            fnv(&mut oh, b"\n");
        }
        s += &format!("|openings {} {oh:016x}", o.len());
    }
    fnv(&mut h, s.as_bytes());
    format!("{h:016x}")
}

fn header_line(cfg: &GameConfig, hash: &str) -> String {
    format!("{{\"format\":1,\"hash\":\"{hash}\",\"cfg\":\"{}\"}}", datadir::cfg_name(cfg.variant.name(), cfg.forced_capture))
}

fn lines_of(recs: &BTreeMap<u32, PairRec>, header: &str) -> String {
    let mut s = String::from(header);
    s.push('\n');
    for r in recs.values() {
        s += &serde_json::to_string(r).unwrap_or_default();
        s.push('\n');
    }
    s
}

fn load_journal(path: &Path, header: &str, hash: &str) -> Result<BTreeMap<u32, PairRec>, String> {
    let mut recs = BTreeMap::new();
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => {
            datadir::write_atomic(path, format!("{header}\n").as_bytes()).map_err(|e| format!("{}: {e}", path.display()))?;
            return Ok(recs);
        }
    };
    let txt = String::from_utf8_lossy(&bytes).to_string();
    let mut lines = txt.split('\n');
    let ok_head = lines
        .next()
        .and_then(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .map(|v| v["format"] == 1 && v["hash"] == hash)
        .unwrap_or(false);
    if !ok_head {
        datadir::mark_stale(path).map_err(|e| format!("{}: {e}", path.display()))?;
        datadir::write_atomic(path, format!("{header}\n").as_bytes()).map_err(|e| format!("{}: {e}", path.display()))?;
        return Ok(recs);
    }
    let mut clean = txt.ends_with('\n');
    let rest: Vec<&str> = lines.collect();
    for (i, l) in rest.iter().enumerate() {
        if l.is_empty() && i + 1 == rest.len() {
            break;
        }
        match serde_json::from_str::<PairRec>(l) {
            Ok(r) => {
                if recs.contains_key(&r.p) {
                    clean = false;
                } else {
                    recs.insert(r.p, r);
                }
            }
            Err(_) => {
                clean = false;
                break;
            }
        }
    }
    if !clean {
        datadir::write_atomic(path, lines_of(&recs, header).as_bytes()).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(recs)
}

pub fn load_openings(cfg: &GameConfig, text: &str) -> Result<Vec<String>, String> {
    let mut out = vec![];
    for (i, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let at = |e: String| format!("строка {}: {e}", i + 1);
        crate::facade::validate_position(cfg, line).map_err(at)?;
        if crate::facade::legal_moves(cfg, line).map_err(at)?.is_empty() {
            return Err(at("нет легальных ходов".into()));
        }
        out.push(line.to_string());
    }
    if out.is_empty() {
        return Err("в файле дебютов нет позиций".into());
    }
    Ok(out)
}

fn opening<R: Rules>(seed: u64, pair: u32, book: (u32, u32), balance_cp: i32, balance_lo: i32, simple: &SimpleEval) -> Result<Vec<Position<R>>, String> {
    let mut rng = Rng::for_game(seed, u32::MAX, pair);
    let (bmin, bmax) = (book.0.min(255), book.1.max(book.0).min(255));
    let stop = AtomicBool::new(false);
    for _ in 0..500 {
        let n = bmin + (rng.next() % (bmax - bmin + 1) as u64) as u32;
        let mut hist: Vec<Position<R>> = vec![Position::<R>::start()];
        let mut ok = true;
        for _ in 0..n {
            let pos = *hist.last().unwrap();
            let list = legal::<R>(&pos);
            if natural_end(&hist, list.len != 0).is_some() {
                ok = false;
                break;
            }
            let mv = list[(rng.next() % list.len as u64) as usize];
            hist.push(pos.make(&mv));
        }
        if !ok {
            continue;
        }
        let pos = *hist.last().unwrap();
        if natural_end(&hist, legal::<R>(&pos).len != 0).is_some() {
            continue;
        }
        if balance_cp > 0 || balance_lo > 0 {
            let lim = SearchLimits { depth: Some(8), hash_mb: 4, threads: 1, ..Default::default() };
            let mut tt = Tt::new(4);
            let r = search_tt_db(&pos, &hist[..hist.len() - 1], simple, &lim, &stop, &mut |_| {}, &mut tt, None);
            if (balance_cp > 0 && r.score.abs() > balance_cp) || r.score.abs() < balance_lo {
                continue;
            }
        }
        return Ok(hist);
    }
    Err(format!("не удалось построить дебют пары {pair}"))
}

struct GameOut {
    s2: u8,
    plies: u32,
    sess: Sess,
    moves: Vec<String>,
}

fn move_time(p: &MatchParams, remaining_ms: Option<u64>, own_move: u64) -> Option<u64> {
    match (&p.ext.clock, remaining_ms) {
        (Some(c), Some(rem)) => Some(c.move_ms(rem, own_move)),
        _ => p.time_ms,
    }
}

pub fn side_hash_mb(p: &MatchParams, who: usize) -> usize {
    match p.ext.side_hash_mb[who] {
        0 => p.hash_mb,
        h => h,
    }
}

fn move_limits(p: &MatchParams, sd: &Side, who: usize, nodes: Option<u64>, time_ms: Option<u64>) -> SearchLimits {
    SearchLimits {
        nodes,
        time_ms: if sd.nodes.is_none() { time_ms } else { None },
        hash_mb: side_hash_mb(p, who),
        threads: p.ext.side_threads[who].max(1),
        opts: sd.opts.clone(),
        duty: 100,
        ..Default::default()
    }
}

#[allow(clippy::too_many_arguments)]
fn play_one<R: Rules, Ea: Evaluator<R>, Eb: Evaluator<R>>(
    ea: &Ea,
    eb: &Eb,
    side_db: [Option<&Egdb<R>>; 2],
    adj: Option<&Egdb<R>>,
    p: &MatchParams,
    ctl: &MatchCtl,
    open: &[Position<R>],
    a_white: bool,
    thr: &mut Throttle,
    (recall, ids): (&[Option<RecallSide>; 2], [u32; 2]),
) -> Option<GameOut> {
    let mut hist: Vec<Position<R>> = open.to_vec();
    let mut tt = [Tt::new(side_hash_mb(p, 0)), Tt::new(side_hash_mb(p, 1))];
    let (mut ctx_a, mut ctx_b) = (SearchCtx::<Ea::State>::default(), SearchCtx::<Eb::State>::default());
    let sides = [&p.a, &p.b];
    let mut sess = Sess::default();
    let mut moves: Vec<String> = Vec::new();
    let mut clock_rem = p.ext.clock.map(|c| [c.base_ms; 2]);
    let adj_on = p.ext.adj_eval != AdjEval::default();
    let mut scores: Vec<i32> = Vec::new();
    let res: i32 = loop {
        if ctl.cancel.load(Ordering::Relaxed) {
            return None;
        }
        thr.tick(ctl.duty.load(Ordering::Relaxed), &ctl.cancel);
        let pos = *hist.last().unwrap();
        let list = legal::<R>(&pos);
        if let Some((r, _)) = natural_end(&hist, list.len != 0) {
            break r as i32;
        }
        if let Some(db) = adj {
            if pos.occ().count_ones() <= db.max_pieces() {
                if let Some(v) = db.probe(&pos) {
                    if v != 0 || !p.adj_decided_only {
                        let stm = if v > 0 { 1 } else if v < 0 { -1 } else { 0 };
                        break if pos.side == Colour::White { stm } else { -stm };
                    }
                }
            }
        }
        if adj_on {
            if let Some(r) = p.ext.adj_eval.judge(&scores, &hist) {
                break r;
            }
        }
        if hist.len() - 1 >= p.max_plies as usize {
            break 0;
        }
        let who = if (pos.side == Colour::White) == a_white { 0 } else { 1 };
        let sd = sides[who];
        let nodes = match (sd.nodes, sd.boost) {
            (Some(n), Some((k, m))) if sess.moves[who] < m as u64 => Some(n.saturating_mul(k as u64)),
            (n, _) => n,
        };
        let lim = move_limits(p, sd, who, nodes, move_time(p, clock_rem.map(|r| r[who]), sess.moves[who]));
        let hh = &hist[..hist.len() - 1];
        let rc = recall[who].as_ref().map(|r| RecallCtx { mem: &r.mem, eval_id: ids[who], plies: r.plies, max_positions: r.max_positions, record: r.learn, mirror: r.mirror });
        let r = if who == 0 {
            search_tt_db_recall_ctx(&pos, hh, ea, &lim, &ctl.cancel, &mut |_| {}, &mut tt[0], side_db[0], rc.as_ref(), Some(&mut ctx_a))
        } else {
            search_tt_db_recall_ctx(&pos, hh, eb, &lim, &ctl.cancel, &mut |_| {}, &mut tt[1], side_db[1], rc.as_ref(), Some(&mut ctx_b))
        };
        if ctl.cancel.load(Ordering::Relaxed) {
            return None;
        }
        if let (Some(c), Some(rem)) = (&p.ext.clock, clock_rem.as_mut()) {
            rem[who] = c.after_move(rem[who], r.time_ms);
        }
        sess.moves[who] += 1;
        sess.depth[who] += r.depth as u64;
        sess.nodes[who] += r.nodes;
        sess.ms[who] += r.time_ms;
        let mv = r.best.filter(|m| list.iter().any(|x| x == m)).unwrap_or(list[0]);
        if adj_on {
            scores.push(if pos.side == Colour::White { r.score } else { -r.score });
        }
        if p.record_moves {
            moves.push(crate::facade::imp::move_notation::<R>(&pos, &mv));
        }
        hist.push(pos.make(&mv));
    };
    let s2 = if a_white { 1 + res } else { 1 - res };
    Some(GameOut { s2: s2.clamp(0, 2) as u8, plies: hist.len() as u32 - 1, sess, moves })
}

struct Shared {
    done: BTreeMap<u32, PairRec>,
    prefix: u32,
    penta: [u64; 5],
    w: u32,
    d: u32,
    l: u32,
    plies: u64,
    sess: Sess,
    stop: Option<bool>,
}

impl Shared {
    fn advance(&mut self, p: &MatchParams, session_pairs: &BTreeSet<u32>) {
        while self.stop.is_none() && self.prefix < p.pairs_max {
            let Some(rec) = self.done.get(&self.prefix) else { break };
            for s in rec.s {
                match s {
                    2 => self.w += 1,
                    0 => self.l += 1,
                    _ => self.d += 1,
                }
            }
            self.penta[(rec.s[0] + rec.s[1]) as usize] += 1;
            self.plies += (rec.pl[0] + rec.pl[1]) as u64;
            if session_pairs.contains(&rec.p) {
                for i in 0..2 {
                    self.sess.moves[i] += rec.sess.moves[i];
                    self.sess.depth[i] += rec.sess.depth[i];
                    self.sess.nodes[i] += rec.sess.nodes[i];
                    self.sess.ms[i] += rec.sess.ms[i];
                }
            }
            self.prefix += 1;
            if let Some((e0, e1)) = p.sprt {
                if let Some(l) = sprt_llr(&self.penta, e0, e1) {
                    if l >= SPRT_BOUND {
                        self.stop = Some(true);
                    } else if l <= -SPRT_BOUND {
                        self.stop = Some(false);
                    }
                }
            }
        }
    }
}

struct Core<'a> {
    cfg: &'a GameConfig,
    p: &'a MatchParams,
    ctl: &'a MatchCtl,
    recall: &'a [Option<RecallSide>; 2],
}

struct OuterRun<'a>(Core<'a>);

struct InnerRun<'a, 'e, E> {
    core: Core<'a>,
    ea: &'e E,
    id_a: u32,
}

impl<'a, R: Rules> EvalRun<R> for OuterRun<'a> {
    type Out = Result<MatchReport, String>;
    fn run<E: Evaluator<R>>(self, ea: &E, id_a: u32) -> Self::Out {
        let p = self.0.p;
        with_eval::<R, _>(&p.b.eval, InnerRun { core: self.0, ea, id_a })?
    }
}

impl<'a, 'e, R: Rules, Ea: Evaluator<R>> EvalRun<R> for InnerRun<'a, 'e, Ea> {
    type Out = Result<MatchReport, String>;
    fn run<Eb: Evaluator<R>>(self, eb: &Eb, id_b: u32) -> Self::Out {
        match_core::<R, Ea, Eb>(&self.core, self.ea, eb, [self.id_a, id_b])
    }
}

fn match_core<R: Rules, Ea: Evaluator<R>, Eb: Evaluator<R>>(core: &Core, ea: &Ea, eb: &Eb, ids: [u32; 2]) -> Result<MatchReport, String> {
    let (cfg, p, ctl) = (core.cfg, core.p, core.ctl);
    let t0 = Instant::now();
    let hash = params_hash(cfg, p, ids);
    let header = header_line(cfg, &hash);
    let loaded = match &p.state_path {
        Some(path) => load_journal(path, &header, &hash)?,
        None => BTreeMap::new(),
    };
    let resumed_pairs = loaded.keys().filter(|&&k| k < p.pairs_max).count() as u32;
    let mut sh = Shared { done: loaded, prefix: 0, penta: [0; 5], w: 0, d: 0, l: 0, plies: 0, sess: Sess::default(), stop: None };
    let session_pairs: Mutex<BTreeSet<u32>> = Mutex::new(BTreeSet::new());
    sh.advance(p, &BTreeSet::new());
    ctl.progress.store(sh.prefix as u64, Ordering::Relaxed);
    let todo: Vec<u32> = if sh.stop.is_some() { vec![] } else { (0..p.pairs_max).filter(|k| !sh.done.contains_key(k)).collect() };
    let journal = match (&p.state_path, todo.is_empty()) {
        (Some(path), false) => Some(Mutex::new(std::fs::OpenOptions::new().append(true).open(path).map_err(|e| format!("{}: {e}", path.display()))?)),
        _ => None,
    };
    let open_cap = |root: &Path, cap: Option<u32>| -> Result<Arc<Egdb<R>>, String> {
        let dir = datadir::dir_in(root, cfg.variant.name(), cfg.forced_capture, datadir::Kind::Egdb);
        let db = match cap {
            Some(n) => Egdb::<R>::open_ro_capped(&dir, n)?,
            None => Egdb::<R>::open_ro(&dir)?,
        };
        if db.max_pieces() == 0 {
            return Err(format!("в {} нет баз окончаний", dir.display()));
        }
        Ok(Arc::new(db))
    };
    let open_root = |root: &Path| open_cap(root, None);
    let global = if p.use_egdb { crate::egdb::shared::<R>() } else { None };
    let side_arc: [Option<Arc<Egdb<R>>>; 2] = [
        match &p.a.egdb_root { Some(r) => Some(open_cap(r, p.a.egdb_max_pieces)?), None => global.clone() },
        match &p.b.egdb_root { Some(r) => Some(open_cap(r, p.b.egdb_max_pieces)?), None => global.clone() },
    ];
    let adj_arc: Option<Arc<Egdb<R>>> = match &p.adj_egdb_root { Some(r) => Some(open_root(r)?), None => global.clone() };
    let side_db = [side_arc[0].as_deref(), side_arc[1].as_deref()];
    let adj_db = adj_arc.as_deref();
    let per_game = p.ext.side_threads.iter().copied().max().unwrap_or(1).max(1);
    let threads = if p.threads == 0 { (crate::search::default_threads() / per_game).max(1) } else { p.threads }.clamp(1, todo.len().max(1));
    let fixed: Option<Vec<Position<R>>> = match &p.openings {
        Some(o) if !o.is_empty() => Some(
            o.iter().enumerate().map(|(i, d)| Position::<R>::from_dfen(d).map_err(|e| format!("дебют {}: {e}", i + 1))).collect::<Result<_, _>>()?,
        ),
        Some(_) => return Err("список дебютов пуст".into()),
        None => None,
    };
    let shared = Mutex::new(sh);
    let next = AtomicUsize::new(0);
    let halt = AtomicBool::new(false);
    let err: Mutex<Option<String>> = Mutex::new(None);
    if !todo.is_empty() {
        std::thread::scope(|sc| -> Result<(), String> {
            let mut hs = Vec::new();
            for _ in 0..threads {
                let h = std::thread::Builder::new()
                    .stack_size(128 << 20)
                    .spawn_scoped(sc, || {
                        crate::qos::apply(ctl.qos.load(Ordering::Relaxed).min(255) as u8);
                        let mut thr = Throttle::new();
                        let simple = SimpleEval::new::<R>();
                        loop {
                            if halt.load(Ordering::Relaxed) || ctl.cancel.load(Ordering::Relaxed) {
                                return;
                            }
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            let Some(&pair) = todo.get(i) else { return };
                            let drawn = match &fixed {
                                Some(v) => Ok(vec![v[pair as usize % v.len()]]),
                                None => opening::<R>(p.seed, pair, p.book_plies, p.balance_cp, p.balance_lo, &simple),
                            };
                            let open = match drawn {
                                Ok(o) => o,
                                Err(e) => {
                                    *err.lock().unwrap() = Some(e);
                                    halt.store(true, Ordering::Relaxed);
                                    return;
                                }
                            };
                            let rc = (core.recall, ids);
                            let Some(g1) = play_one::<R, Ea, Eb>(ea, eb, side_db, adj_db, p, ctl, &open, true, &mut thr, rc) else { return };
                            let Some(g2) = play_one::<R, Ea, Eb>(ea, eb, side_db, adj_db, p, ctl, &open, false, &mut thr, rc) else { return };
                            let mut sess = Sess::default();
                            for k in 0..2 {
                                sess.moves[k] = g1.sess.moves[k] + g2.sess.moves[k];
                                sess.depth[k] = g1.sess.depth[k] + g2.sess.depth[k];
                                sess.nodes[k] = g1.sess.nodes[k] + g2.sess.nodes[k];
                                sess.ms[k] = g1.sess.ms[k] + g2.sess.ms[k];
                            }
                            let mv = p.record_moves.then(|| [g1.moves.join(" "), g2.moves.join(" ")]);
                            let rec = PairRec { p: pair, s: [g1.s2, g2.s2], pl: [g1.plies, g2.plies], mv, sess };
                            let mut g = shared.lock().unwrap();
                            if let Some(j) = &journal {
                                let mut line = serde_json::to_string(&rec).unwrap_or_default();
                                line.push('\n');
                                let mut f = j.lock().unwrap();
                                if let Err(e) = f.write_all(line.as_bytes()).and_then(|_| f.flush()) {
                                    *err.lock().unwrap() = Some(format!("журнал матча: {e}"));
                                    halt.store(true, Ordering::Relaxed);
                                    return;
                                }
                            }
                            let sp = {
                                let mut s = session_pairs.lock().unwrap();
                                s.insert(pair);
                                s.clone()
                            };
                            g.done.insert(pair, rec);
                            g.advance(p, &sp);
                            ctl.progress.store(g.prefix as u64, Ordering::Relaxed);
                            if g.stop.is_some() {
                                halt.store(true, Ordering::Relaxed);
                            }
                        }
                    })
                    .map_err(|e| e.to_string())?;
                hs.push(h);
            }
            for h in hs {
                h.join().map_err(|_| "поток матча завершился аварийно".to_string())?;
            }
            Ok(())
        })?;
    }
    if let Some(e) = err.lock().unwrap().take() {
        return Err(e);
    }
    let g = shared.into_inner().unwrap();
    let (elo, lo, hi) = elo_interval(&g.penta);
    let (n, mean, _) = pair_moments(&g.penta);
    let llr = p.sprt.and_then(|(e0, e1)| sprt_llr(&g.penta, e0, e1));
    let reached_max = g.stop.is_none() && g.prefix >= p.pairs_max;
    Ok(MatchReport {
        pairs: g.prefix,
        wins: g.w,
        draws: g.d,
        losses: g.l,
        penta: g.penta,
        score: mean,
        elo,
        elo_lo: lo,
        elo_hi: hi,
        llr,
        sprt: g.stop,
        reached_max,
        cancelled: ctl.cancel.load(Ordering::Relaxed) && g.stop.is_none() && !reached_max,
        resumed_pairs,
        control: control_text(p),
        eval_id: ids,
        avg_plies: if n > 0.0 { g.plies as f64 / (2.0 * n) } else { 0.0 },
        secs: t0.elapsed().as_secs_f64(),
        moves: g.sess.moves,
        depth_sum: g.sess.depth,
        nodes_sum: g.sess.nodes,
        ms_sum: g.sess.ms,
    })
}

fn control_text(p: &MatchParams) -> String {
    match (p.a.nodes, p.b.nodes) {
        (Some(a), Some(b)) => format!("nodes {a}/{b}"),
        _ => match &p.ext.clock {
            Some(c) => format!("clock {}+{}ms", c.base_ms, c.inc_ms),
            None => format!("time {}ms", p.time_ms.unwrap_or(0)),
        },
    }
}

pub fn run_match(cfg: &GameConfig, p: &MatchParams, ctl: &MatchCtl) -> Result<MatchReport, String> {
    if p.pairs_max == 0 {
        return Err("pairs_max должен быть > 0".into());
    }
    if (p.a.nodes.is_none() || p.b.nodes.is_none()) && p.time_ms.is_none() && p.ext.clock.is_none() {
        return Err("нужен бюджет: nodes у обеих сторон или time_ms".into());
    }
    check_spec(cfg, &p.a.eval)?;
    check_spec(cfg, &p.b.eval)?;
    let none = [None, None];
    let core = Core { cfg, p, ctl, recall: &none };
    run_dispatch(cfg, core)
}

#[derive(Clone)]
pub struct RecallSide {
    pub mem: Arc<Recall>,
    pub plies: u32,
    pub max_positions: usize,
    pub learn: bool,
    pub mirror: bool,
}

pub fn run_match_recall(cfg: &GameConfig, p: &MatchParams, ctl: &MatchCtl, recall: &[Option<RecallSide>; 2]) -> Result<MatchReport, String> {
    if recall.iter().all(|r| r.is_none()) {
        return run_match(cfg, p, ctl);
    }
    if p.state_path.is_some() {
        return Err("память расчётов не сочетается с журналом пар (state_path)".into());
    }
    if p.pairs_max == 0 {
        return Err("pairs_max должен быть > 0".into());
    }
    if (p.a.nodes.is_none() || p.b.nodes.is_none()) && p.time_ms.is_none() && p.ext.clock.is_none() {
        return Err("нужен бюджет: nodes у обеих сторон или time_ms".into());
    }
    check_spec(cfg, &p.a.eval)?;
    check_spec(cfg, &p.b.eval)?;
    let core = Core { cfg, p, ctl, recall };
    run_dispatch(cfg, core)
}

fn run_r<R: Rules>(core: Core) -> Result<MatchReport, String> {
    let a = core.p.a.eval.clone();
    with_eval::<R, _>(&a, OuterRun(core))?
}

fn run_dispatch(cfg: &GameConfig, core: Core) -> Result<MatchReport, String> {
    dispatch_cfg!(cfg, run_r(core))
}

fn sample_positions<R: Rules>(count: u32, plies: u32) -> Vec<Vec<Position<R>>> {
    let mut out = vec![];
    let mut pair = 0u32;
    while out.len() < count as usize && pair < count * 50 {
        let mut rng = Rng::for_game(0xBE17, u32::MAX - 1, pair);
        pair += 1;
        let mut hist: Vec<Position<R>> = vec![Position::<R>::start()];
        let mut ok = true;
        for _ in 0..plies {
            let pos = *hist.last().unwrap();
            let list = legal::<R>(&pos);
            if natural_end(&hist, list.len != 0).is_some() {
                ok = false;
                break;
            }
            let mv = list[(rng.next() % list.len as u64) as usize];
            hist.push(pos.make(&mv));
        }
        if ok && natural_end(&hist, legal::<R>(hist.last().unwrap()).len != 0).is_none() {
            out.push(hist);
        }
    }
    out
}

fn nps_of<R: Rules, E: Evaluator<R>>(ev: &E, hist: &[Position<R>], slice_ms: u64) -> (u64, f64) {
    let pos = *hist.last().unwrap();
    let stop = AtomicBool::new(false);
    let mut tt = Tt::new(8);
    let lim = SearchLimits { time_ms: Some(slice_ms), hash_mb: 8, threads: 1, ..Default::default() };
    let t = Instant::now();
    let r = search_tt_db(&pos, &hist[..hist.len() - 1], ev, &lim, &stop, &mut |_| {}, &mut tt, None);
    (r.nodes, t.elapsed().as_secs_f64())
}

struct NpsOuter {
    b: EvalSpec,
    secs: f64,
}

struct NpsInner<'e, E> {
    ea: &'e E,
    secs: f64,
}

impl<R: Rules> EvalRun<R> for NpsOuter {
    type Out = Result<(f64, f64), String>;
    fn run<E: Evaluator<R>>(self, ea: &E, _id: u32) -> Self::Out {
        with_eval::<R, _>(&self.b, NpsInner { ea, secs: self.secs })?
    }
}

impl<'e, R: Rules, Ea: Evaluator<R>> EvalRun<R> for NpsInner<'e, Ea> {
    type Out = Result<(f64, f64), String>;
    fn run<Eb: Evaluator<R>>(self, eb: &Eb, _id: u32) -> Self::Out {
        let pos = sample_positions::<R>(2, 12);
        if pos.is_empty() {
            return Err("нет позиций для замера скорости".into());
        }
        let rounds = 3u32;
        let slice = ((self.secs * 1000.0) / (rounds as f64 * pos.len() as f64)).max(1.0) as u64;
        let (mut na, mut ta, mut nb, mut tb) = (0u64, 0f64, 0u64, 0f64);
        for _ in 0..rounds {
            for h in &pos {
                let (n, t) = nps_of::<R, _>(self.ea, h, slice);
                na += n;
                ta += t;
                let (n, t) = nps_of::<R, _>(eb, h, slice);
                nb += n;
                tb += t;
            }
        }
        Ok((na as f64 / ta.max(1e-9), nb as f64 / tb.max(1e-9)))
    }
}

fn nps_pair_r<R: Rules>(a: &EvalSpec, b: &EvalSpec, secs: f64) -> Result<(f64, f64), String> {
    with_eval::<R, _>(a, NpsOuter { b: b.clone(), secs })?
}

pub fn measure_nps(cfg: &GameConfig, a: &EvalSpec, b: &EvalSpec, secs_per_side: f64) -> Result<(f64, f64), String> {
    check_spec(cfg, a)?;
    check_spec(cfg, b)?;
    dispatch_cfg!(cfg, nps_pair_r(a, b, secs_per_side))
}

pub fn time_equal_nodes(cfg: &GameConfig, a: &EvalSpec, b: &EvalSpec, ms: f64) -> Result<(u64, u64), String> {
    time_equal_nodes_secs(cfg, a, b, ms, 0.4)
}

pub fn time_equal_nodes_secs(cfg: &GameConfig, a: &EvalSpec, b: &EvalSpec, ms: f64, secs_per_side: f64) -> Result<(u64, u64), String> {
    let (na, nb) = measure_nps(cfg, a, b, secs_per_side)?;
    let f = |nps: f64| ((nps * ms / 1000.0).round() as u64).max(1);
    Ok((f(na), f(nb)))
}

struct SpeedRun {
    secs: f64,
}

impl<R: Rules> EvalRun<R> for SpeedRun {
    type Out = (f64, f64);
    fn run<E: Evaluator<R>>(self, ev: &E, _id: u32) -> Self::Out {
        let games = sample_positions::<R>(24, 40);
        let all: Vec<Position<R>> = games.iter().flat_map(|g| g.iter().copied()).collect();
        let t = Instant::now();
        let (mut n, mut chk) = (0u64, 0i64);
        while t.elapsed().as_secs_f64() < self.secs * 0.4 && !all.is_empty() {
            for p in &all {
                chk = chk.wrapping_add(ev.eval_fresh(p) as i64);
            }
            n += all.len() as u64;
        }
        std::hint::black_box(chk);
        let eps = n as f64 / t.elapsed().as_secs_f64().max(1e-9);
        let (mut nodes, mut ts) = (0u64, 0f64);
        for h in games.iter().take(2) {
            let (a, b) = nps_of::<R, _>(ev, h, ((self.secs * 600.0) / 2.0).max(1.0) as u64);
            nodes += a;
            ts += b;
        }
        (eps, nodes as f64 / ts.max(1e-9))
    }
}

fn speed_r<R: Rules>(spec: &EvalSpec, secs: f64) -> Result<(f64, f64), String> {
    with_eval::<R, _>(spec, SpeedRun { secs })
}

struct TtdRun {
    positions: Vec<String>,
    depth: u32,
    threads: Vec<usize>,
    hash_mb: usize,
    reps: u32,
}

impl<R: Rules> EvalRun<R> for TtdRun {
    type Out = Result<String, String>;
    fn run<E: Evaluator<R>>(self, ev: &E, _id: u32) -> Self::Out {
        let ps: Vec<Position<R>> =
            self.positions.iter().enumerate().map(|(i, d)| Position::<R>::from_dfen(d).map_err(|e| format!("позиция {}: {e}", i + 1))).collect::<Result<_, _>>()?;
        let mut out = String::new();
        let mut best: Vec<Vec<f64>> = vec![Vec::new(); self.threads.len()];
        for (i, pos) in ps.iter().enumerate() {
            for (k, &t) in self.threads.iter().enumerate() {
                let mut b = f64::INFINITY;
                for r in 0..self.reps.max(1) {
                    let stop = AtomicBool::new(false);
                    let mut tt = Tt::new(self.hash_mb);
                    let lim = SearchLimits { depth: Some(self.depth), hash_mb: self.hash_mb, threads: t.max(1), ..Default::default() };
                    let t0 = Instant::now();
                    let res = search_tt_db(pos, &[], ev, &lim, &stop, &mut |_| {}, &mut tt, None);
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    b = b.min(ms);
                    out += &format!(
                        "TTD pos {i} threads {t} rep {r} depth {} ms {ms:.1} nodes {} early {}\n",
                        res.depth,
                        res.nodes,
                        res.early.unwrap_or("-")
                    );
                }
                best[k].push(b);
            }
        }
        for (k, &t) in self.threads.iter().enumerate() {
            let sum: f64 = best[k].iter().sum();
            let ratios: Vec<f64> = best[k].iter().zip(&best[0]).map(|(x, y)| y / x.max(1e-9)).collect();
            out += &format!(
                "TTD-SUM threads {t} positions {} depth {} ms_sum {sum:.1} speedup_vs_{}: sum {:.3} median {:.3}\n",
                ps.len(),
                self.depth,
                self.threads[0],
                best[0].iter().sum::<f64>() / sum.max(1e-9),
                median(ratios)
            );
        }
        Ok(out)
    }
}

fn ttd_r<R: Rules>(spec: &EvalSpec, positions: &[String], depth: u32, threads: &[usize], hash_mb: usize, reps: u32) -> Result<String, String> {
    with_eval::<R, _>(spec, TtdRun { positions: positions.to_vec(), depth, threads: threads.to_vec(), hash_mb, reps })?
}

pub fn time_to_depth(cfg: &GameConfig, spec: &EvalSpec, positions: &[String], depth: u32, threads: &[usize], hash_mb: usize, reps: u32) -> Result<String, String> {
    check_spec(cfg, spec)?;
    if threads.is_empty() || positions.is_empty() {
        return Err("time_to_depth: нужны позиции и хотя бы одно число потоков".into());
    }
    dispatch_cfg!(cfg, ttd_r(spec, positions, depth, threads, hash_mb, reps))
}

pub fn measure_speed(cfg: &GameConfig, spec: &EvalSpec, secs: f64) -> Result<(f64, f64), String> {
    check_spec(cfg, spec)?;
    dispatch_cfg!(cfg, speed_r(spec, secs))
}

pub const EQ_SPEED_VERSION: u32 = 3;
pub const EQ_SPEED_POSITIONS: u32 = 24;
pub const EQ_SPEED_SEED: u64 = 0xE95E_ED23;
pub const EQ_SPEED_MT_TOL: f64 = 0.10;
pub const EQ_SPEED_SAME_TOL: f64 = 0.05;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EqSpeed {
    pub version: u32,
    pub eval_a: u32,
    pub eval_b: u32,
    pub egdb: bool,
    pub positions: u32,
    pub nps_a: f64,
    pub nps_b: f64,
    pub ratio_1t: f64,
    pub ratio_mt: f64,
    pub mt_threads: usize,
    pub ratio: f64,
    pub measured_unix: u64,
}

impl EqSpeed {
    pub fn nodes_for_ms(&self, ms: f64) -> (u64, u64) {
        let nb = ((self.nps_b * ms / 1000.0).round() as u64).max(1);
        (self.nodes_a_for(nb), nb)
    }
    pub fn nodes_a_for(&self, nb: u64) -> u64 {
        let r = if self.ratio.is_finite() && self.ratio > 0.0 { self.ratio } else { 1.0 };
        ((nb as f64 * r).round() as u64).max(1)
    }
    pub fn same_speed(&self) -> bool {
        (self.ratio - 1.0).abs() <= EQ_SPEED_SAME_TOL
    }
}

pub(crate) fn repr_positions<R: Rules>(count: u32, seed: u64) -> Vec<Vec<Position<R>>> {
    let simple = SimpleEval::new::<R>();
    let stop = AtomicBool::new(false);
    let mut out = vec![];
    let mut k = 0u32;
    while out.len() < count as usize && k < count.saturating_mul(8).max(8) {
        let i = k;
        k += 1;
        let Ok(mut hist) = opening::<R>(seed, i, (4, 8), 0, 0, &simple) else { continue };
        let mut rng = Rng::for_game(seed ^ 0x9E37, u32::MAX - 2, i);
        let target = 20 + (rng.next() % 21) as usize;
        let mut tt = Tt::new(1);
        let mut ok = true;
        while hist.len() <= target {
            let pos = *hist.last().unwrap();
            if natural_end(&hist, legal::<R>(&pos).len != 0).is_some() {
                ok = false;
                break;
            }
            let lim = SearchLimits { nodes: Some(400), hash_mb: 1, threads: 1, ..Default::default() };
            let r = search_tt_db(&pos, &hist[..hist.len() - 1], &simple, &lim, &stop, &mut |_| {}, &mut tt, None);
            let Some(mv) = r.best else {
                ok = false;
                break;
            };
            hist.push(pos.make(&mv));
        }
        if !ok {
            continue;
        }
        let last = *hist.last().unwrap();
        let n = legal::<R>(&last).len;
        if n > 1 && natural_end(&hist, n != 0).is_none() {
            out.push(hist);
        }
    }
    out
}

fn eq_slice<R: Rules, E: Evaluator<R>>(ev: &E, hist: &[Position<R>], slice_ms: u64, threads: usize, db: Option<&Egdb<R>>) -> Option<(u64, f64)> {
    let pos = *hist.last().unwrap();
    let stop = AtomicBool::new(false);
    let mut tt = Tt::new(16);
    let lim = SearchLimits { time_ms: Some(slice_ms), hash_mb: 16, threads, ..Default::default() };
    let t = Instant::now();
    let r = search_tt_db(&pos, &hist[..hist.len() - 1], ev, &lim, &stop, &mut |_| {}, &mut tt, db);
    let secs = t.elapsed().as_secs_f64();
    (r.early.is_none() && r.nodes >= 200 && secs > 0.0).then_some((r.nodes, secs))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite() && *x > 0.0);
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

struct EqOuter {
    b: EvalSpec,
    secs: f64,
    mt: usize,
    egdb: bool,
}

struct EqInner<'e, E> {
    ea: &'e E,
    ida: u32,
    secs: f64,
    mt: usize,
    egdb: bool,
}

impl<R: Rules> EvalRun<R> for EqOuter {
    type Out = Result<EqSpeed, String>;
    fn run<E: Evaluator<R>>(self, ea: &E, id: u32) -> Self::Out {
        with_eval::<R, _>(&self.b, EqInner { ea, ida: id, secs: self.secs, mt: self.mt, egdb: self.egdb })?
    }
}

impl<'e, R: Rules, Ea: Evaluator<R>> EvalRun<R> for EqInner<'e, Ea> {
    type Out = Result<EqSpeed, String>;
    fn run<Eb: Evaluator<R>>(self, eb: &Eb, idb: u32) -> Self::Out {
        let pos = repr_positions::<R>(EQ_SPEED_POSITIONS, EQ_SPEED_SEED);
        if pos.is_empty() {
            return Err("нет позиций для замера скорости".into());
        }
        let db = if self.egdb { crate::egdb::shared::<R>() } else { None };
        let db = db.as_deref();
        let slice = ((self.secs * 1000.0) / (2.0 * pos.len() as f64)).max(10.0) as u64;
        let pass = |threads: usize, slice: u64, step: usize| -> (f64, f64, f64, u32) {
            let (mut rs, mut va, mut vb) = (vec![], vec![], vec![]);
            for h in pos.iter().step_by(step) {
                let a1 = eq_slice(self.ea, h, slice, threads, db);
                let b1 = eq_slice(eb, h, slice, threads, db);
                let b2 = eq_slice(eb, h, slice, threads, db);
                let a2 = eq_slice(self.ea, h, slice, threads, db);
                if let (Some(a1), Some(a2), Some(b1), Some(b2)) = (a1, a2, b1, b2) {
                    let a = (a1.0 + a2.0) as f64 / (a1.1 + a2.1).max(1e-9);
                    let b = (b1.0 + b2.0) as f64 / (b1.1 + b2.1).max(1e-9);
                    rs.push(a / b);
                    va.push(a);
                    vb.push(b);
                }
            }
            let n = rs.len() as u32;
            (median(rs), median(va), median(vb), n)
        };
        let (r1, na, nb, n) = pass(1, slice, 1);
        if n == 0 || r1 <= 0.0 {
            return Err("замер скорости не удался: ни одной пригодной позиции".into());
        }
        let mt = self.mt;
        let rmt = if mt > 1 { pass(mt, slice * 2, 2).0 } else { 0.0 };
        let ratio = if rmt > 0.0 && rmt < r1 * (1.0 - EQ_SPEED_MT_TOL) { rmt } else { r1 };
        Ok(EqSpeed {
            version: EQ_SPEED_VERSION,
            eval_a: self.ida,
            eval_b: idb,
            egdb: db.is_some(),
            positions: n,
            nps_a: na,
            nps_b: nb,
            ratio_1t: r1,
            ratio_mt: rmt,
            mt_threads: if mt > 1 { mt } else { 0 },
            ratio,
            measured_unix: datadir::now_secs(),
        })
    }
}

fn eq_speed_r<R: Rules>(a: &EvalSpec, b: &EvalSpec, secs: f64, mt: usize, egdb: bool) -> Result<EqSpeed, String> {
    with_eval::<R, _>(a, EqOuter { b: b.clone(), secs, mt, egdb })?
}

pub fn measure_eq_speed(cfg: &GameConfig, a: &EvalSpec, b: &EvalSpec, secs_per_side: f64, mt_threads: usize, use_egdb: bool) -> Result<EqSpeed, String> {
    check_spec(cfg, a)?;
    check_spec(cfg, b)?;
    let mt = if mt_threads == 0 { crate::search::default_threads() } else { mt_threads };
    dispatch_cfg!(cfg, eq_speed_r(a, b, secs_per_side.max(0.05), mt, use_egdb))
}

pub fn spec_eval_id(cfg: &GameConfig, spec: &EvalSpec) -> Result<u32, String> {
    check_spec(cfg, spec)
}

pub fn eq_speed_key(ida: u32, idb: u32, egdb: bool, mt: usize) -> String {
    format!("{ida:08x}-{idb:08x}-egdb{}-v{EQ_SPEED_VERSION}-t{mt}", u8::from(egdb))
}

pub fn eq_speed_cached(cfg: &GameConfig, a: &EvalSpec, b: &EvalSpec, secs_per_side: f64, mt_threads: usize, use_egdb: bool, cache: &Path) -> Result<EqSpeed, String> {
    let ida = check_spec(cfg, a)?;
    let idb = check_spec(cfg, b)?;
    let mt = if mt_threads == 0 { crate::search::default_threads() } else { mt_threads };
    let key = eq_speed_key(ida, idb, use_egdb, mt);
    let mut map: BTreeMap<String, EqSpeed> = std::fs::read_to_string(cache).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
    if let Some(s) = map.get(&key).filter(|s| s.version == EQ_SPEED_VERSION && s.ratio > 0.0) {
        return Ok(s.clone());
    }
    let s = measure_eq_speed(cfg, a, b, secs_per_side, mt, use_egdb)?;
    map.retain(|_, v| v.version == EQ_SPEED_VERSION);
    map.insert(key, s.clone());
    if let Ok(js) = serde_json::to_string_pretty(&map) {
        if let Some(d) = cache.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let _ = datadir::write_atomic(cache, js.as_bytes());
    }
    Ok(s)
}

#[cfg(test)]
mod pkg2_tests {
    use super::*;

    fn cfg() -> GameConfig {
        GameConfig { variant: VariantId::Russian, forced_capture: true }
    }

    #[test]
    fn side_threads_default_one_and_hashed_only_when_set() {
        let p = MatchParams::default();
        assert_eq!(p.ext, MatchExt::default());
        for who in 0..2 {
            assert_eq!(move_limits(&p, &p.a, who, Some(100), None).threads, 1);
        }
        let h0 = params_hash(&cfg(), &p, [0, 0]);
        let mut q = p.clone();
        q.ext.side_threads = [2, 1];
        assert_eq!(move_limits(&q, &q.a, 0, Some(100), None).threads, 2);
        assert_eq!(move_limits(&q, &q.b, 1, Some(100), None).threads, 1);
        assert_ne!(params_hash(&cfg(), &q, [0, 0]), h0, "потоки сторон входят в хеш журнала");
        q.ext.side_threads = [1, 1];
        assert_eq!(params_hash(&cfg(), &q, [0, 0]), h0);
    }

    #[test]
    fn side_hash_default_shared_and_hashed_only_when_different() {
        let p = MatchParams { hash_mb: 256, ..MatchParams::default() };
        for who in 0..2 {
            assert_eq!(move_limits(&p, &p.a, who, Some(100), None).hash_mb, 256);
        }
        let h0 = params_hash(&cfg(), &p, [0, 0]);
        let mut q = p.clone();
        q.ext.side_hash_mb = [256, 0];
        assert_eq!(params_hash(&cfg(), &q, [0, 0]), h0, "таблица стороны, равная общей, хеш не меняет");
        q.ext.side_hash_mb = [2048, 0];
        assert_eq!(move_limits(&q, &q.a, 0, Some(100), None).hash_mb, 2048);
        assert_eq!(move_limits(&q, &q.b, 1, Some(100), None).hash_mb, 256);
        assert_ne!(params_hash(&cfg(), &q, [0, 0]), h0, "таблица стороны входит в хеш журнала");
        let mut r = p.clone();
        r.ext.side_hash_mb = [0, 2048];
        assert_ne!(params_hash(&cfg(), &r, [0, 0]), params_hash(&cfg(), &q, [0, 0]), "сторона различима");
    }

    #[test]
    fn clock_off_keeps_per_move_limit() {
        let mut p = MatchParams { time_ms: Some(250), ..Default::default() };
        p.a.nodes = None;
        let h0 = params_hash(&cfg(), &p, [0, 0]);
        for k in 0..200 {
            assert_eq!(move_time(&p, None, k), Some(250));
            assert_eq!(move_limits(&p, &p.a, 0, None, move_time(&p, None, k)).time_ms, Some(250));
        }
        assert_eq!(control_text(&p), "time 250ms");
        p.ext.clock = Some(GameClock { front_moves: 4, front_pct: 300, ..GameClock::new(30_000, 100) });
        assert_ne!(params_hash(&cfg(), &p, [0, 0]), h0);
        assert_eq!(move_time(&p, Some(30_000), 0), Some(3_300));
        assert_eq!(move_time(&p, Some(30_000), 4), Some(1_100));
        assert_eq!(control_text(&p), "clock 30000+100ms");
        assert_eq!(move_limits(&p, &p.b, 1, Some(500), move_time(&p, Some(30_000), 0)).time_ms, None);
    }

    #[test]
    fn time_extension_keys_keep_depth_tree() {
        type R = crate::rules::Russian<true>;
        let ev = SimpleEval::new::<R>();
        let games = sample_positions::<R>(6, 30);
        for pos in games.iter().filter_map(|g| g.last().copied()) {
            let run = |opts: &str| {
                let lim = SearchLimits { depth: Some(7), hash_mb: 1, threads: 1, opts: opts.into(), ..Default::default() };
                let r = search_tt_db(&pos, &[], &ev, &lim, &AtomicBool::new(false), &mut |_| {}, &mut Tt::new(1), None);
                (r.nodes, r.best, r.score, r.depth)
            };
            assert_eq!(run(""), run("tdrop=10,tnodes=50,text=200"));
        }
    }

    #[test]
    fn adj_eval_rules() {
        type R = crate::rules::Russian<true>;
        let h: Vec<Position<R>> = vec![Position::<R>::start(); 1];
        let off = AdjEval::default();
        assert_eq!(off.judge(&[900; 20], &h), None);
        let win = AdjEval { win: Some((700, 4)), ..off };
        assert_eq!(win.judge(&[0, 800, 750, 700, 900], &h), Some(1));
        assert_eq!(win.judge(&[0, -800, -750, -700, -900], &h), Some(-1));
        assert_eq!(win.judge(&[800, 750, 690, 900], &h), None);
        let draw = AdjEval { draw: Some((5, 3, 4)), ..off };
        let long: Vec<Position<R>> = vec![Position::<R>::start(); 5];
        assert_eq!(draw.judge(&[3, -5, 0], &h), None, "рано");
        assert_eq!(draw.judge(&[3, -5, 0], &long), Some(0));
        assert_eq!(draw.judge(&[3, -6, 0], &long), None);
        let rep = AdjEval { rep: Some(20), ..off };
        assert_eq!(rep.judge(&[10], &h), None, "позиция встретилась один раз");
        let two: Vec<Position<R>> = vec![Position::<R>::start(); 2];
        assert_eq!(rep.judge(&[10], &two), Some(0));
        assert_eq!(rep.judge(&[30], &two), None);
    }

    #[test]
    fn adj_eval_off_keeps_games_on_shortens() {
        let base = MatchParams {
            a: Side::simple(300),
            b: Side::simple(600),
            pairs_max: 6,
            seed: 7,
            max_plies: 120,
            hash_mb: 1,
            threads: 1,
            record_moves: true,
            ..Default::default()
        };
        let dir = std::env::temp_dir().join(format!("pkg2-adj-{}-{:?}", std::process::id(), std::thread::current().id()));
        std::fs::create_dir_all(&dir).unwrap();
        let run = |p: &MatchParams, tag: &str| {
            let mut p = p.clone();
            p.state_path = Some(dir.join(format!("{tag}.jsonl")));
            let _ = std::fs::remove_file(p.state_path.as_ref().unwrap());
            let r = run_match(&cfg(), &p, &MatchCtl::new()).unwrap();
            let journal = std::fs::read_to_string(p.state_path.as_ref().unwrap()).unwrap();
            (r, journal)
        };
        let (r0, j0) = run(&base, "off");
        let mut inert = base.clone();
        inert.ext.adj_eval = AdjEval { win: Some((i32::MAX, 1000)), draw: Some((-1, 1, 0)), rep: Some(-1) };
        let (r1, j1) = run(&inert, "inert");
        let moves = |j: &str| j.lines().map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["mv"].clone()).collect::<Vec<_>>();
        assert_eq!(moves(&j0), moves(&j1), "недостижимые пороги — те же ходы");
        assert_eq!((r0.wins, r0.draws, r0.losses, r0.penta, r0.avg_plies), (r1.wins, r1.draws, r1.losses, r1.penta, r1.avg_plies));
        assert_eq!(params_hash(&cfg(), &base, [0, 0]), params_hash(&cfg(), &MatchParams { ext: MatchExt::default(), ..base.clone() }, [0, 0]));
        assert_ne!(params_hash(&cfg(), &inert, [0, 0]), params_hash(&cfg(), &base, [0, 0]));
        let mut on = base.clone();
        on.ext.adj_eval = AdjEval { win: Some((150, 2)), draw: Some((30, 4, 10)), rep: Some(50) };
        let (r2, _) = run(&on, "on");
        assert!(r2.avg_plies < r0.avg_plies, "присуждение укорачивает партии: {} против {}", r2.avg_plies, r0.avg_plies);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn time_to_depth_smoke() {
        let start = crate::facade::start_position(&cfg()).unwrap();
        let out = time_to_depth(&cfg(), &EvalSpec::Simple, &[start], 3, &[1, 2], 1, 1).unwrap();
        assert_eq!(out.lines().filter(|l| l.starts_with("TTD pos")).count(), 2, "{out}");
        assert_eq!(out.lines().filter(|l| l.starts_with("TTD-SUM")).count(), 2, "{out}");
        assert!(out.contains("TTD pos 0 threads 2 rep 0 depth 3"), "{out}");
    }
}
