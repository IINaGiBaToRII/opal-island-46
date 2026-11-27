
use engine_core::{apply_move, game_status, legal_moves, start_position, GameConfig, Outcome, VariantId};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

pub struct MatchArgs {
    pub a: String,
    pub b: String,
    pub cfg: GameConfig,
    pub go: Value,
    pub games: usize,
    pub concurrency: usize,
    pub sprt: Option<(f64, f64)>,
    pub threads: usize,
    pub hash_mb: usize,
    pub seed: u64,
    pub max_plies: usize,
    pub log: String,
    pub book_plies: (u64, u64),
    pub opts: [String; 2],
    pub eval: [Option<engine_core::EvalSpec>; 2],
    pub boost: [Option<(u64, u64)>; 2],
}

struct Eng {
    eval_id: u32,
    child: Child,
    w: ChildStdin,
    r: BufReader<ChildStdout>,
}

impl Eng {
    fn spawn(bin: &str, m: &MatchArgs, who: usize) -> Result<Eng, String> {
        let mut child = Command::new(bin)
            .arg("engine")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn {bin}: {e}"))?;
        let w = child.stdin.take().unwrap();
        let r = BufReader::new(child.stdout.take().unwrap());
        let mut e = Eng { eval_id: 0, child, w, r };
        e.send(&json!({"cmd":"setoption","variant":m.cfg.variant.name(),"forced":m.cfg.forced_capture,"threads":m.threads,"hash_mb":m.hash_mb,"opts":m.opts[who]}))?;
        Ok(e)
    }
    fn send(&mut self, v: &Value) -> Result<(), String> {
        writeln!(self.w, "{v}").and_then(|_| self.w.flush()).map_err(|e| e.to_string())
    }
    fn go(&mut self, cfg: &GameConfig, fen: &str, hist: &[String], go: &Value) -> Result<(Option<String>, u64, u64), String> {
        self.send(&json!({"cmd":"position","dfen":fen,"history":hist}))?;
        let mut g = go.clone();
        g["cmd"] = json!("go");
        self.send(&g)?;
        loop {
            let mut line = String::new();
            if self.r.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
                return Err("engine closed".into());
            }
            let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
            match v["type"].as_str() {
                Some("bestmove") => {
                    self.eval_id = v["eval_id"].as_u64().unwrap_or(0) as u32;
                    let mut mv = v["move"].as_str().map(|s| s.to_string());
                    if let (None, Some(after)) = (&mv, v["dfen_after"].as_str()) {
                        let board = |f: &str| f.split_once(' ').map(|x| x.1.trim().to_string()).unwrap_or_default();
                        for li in legal_moves(cfg, fen)? {
                            if board(&apply_move(cfg, fen, &li.notation)?) == board(after) {
                                mv = Some(li.notation);
                                break;
                            }
                        }
                        if mv.is_none() {
                            return Err(format!("dfen_after {after} не достижим легальным ходом из {fen}"));
                        }
                    }
                    return Ok((mv, v["depth"].as_u64().unwrap_or(0), v["nodes"].as_u64().unwrap_or(0)));
                }
                Some("error") => return Err(format!("engine error: {}", v["msg"])),
                _ => {}
            }
        }
    }
}

impl Drop for Eng {
    fn drop(&mut self) {
        let _ = self.send(&json!({"cmd":"quit"}));
        let _ = self.child.wait();
    }
}

#[derive(Default)]
struct Stats {
    penta: [u64; 5],
    w: u64,
    d: u64,
    l: u64,
    depth_sum: [u64; 2],
    moves: [u64; 2],
    time_sum: [u64; 2],
    time_max: [u64; 2],
    over: [u64; 2],
    eval_id: [u32; 2],
    log: Vec<String>,
}

fn rng_next(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn load_book(m: &MatchArgs, pairs: usize) -> Result<Vec<(String, Vec<String>, Vec<String>)>, String> {
    let path = format!("logs/book-{}-{}-{}.txt", m.cfg.variant.name(), if m.cfg.forced_capture { "forced" } else { "free" }, m.seed);
    let mut book = vec![];
    if let Ok(txt) = std::fs::read_to_string(&path) {
        for l in txt.lines() {
            let mut it = l.split('\t');
            let (Some(_), Some(mv)) = (it.next(), it.next()) else { continue };
            let mv: Vec<String> = mv.split(' ').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();
            let mut fen = start_position(&m.cfg)?;
            let mut hist = vec![];
            for x in &mv {
                let nf = apply_move(&m.cfg, &fen, x)?;
                hist.push(std::mem::replace(&mut fen, nf));
            }
            book.push((fen, hist, mv));
        }
    }
    if book.len() >= pairs {
        return Ok(book);
    }
    for p in book.len()..pairs {
        book.push(opening(m, p)?);
    }
    let _ = std::fs::create_dir_all("logs");
    let txt: String = book.iter().map(|(f, _, mv)| format!("{f}\t{}\n", mv.join(" "))).collect();
    std::fs::write(&path, txt).map_err(|e| e.to_string())?;
    Ok(book)
}

fn balanced(m: &MatchArgs, fen: &str, hist: &[String]) -> bool {
    let lim = engine_core::SearchLimits { depth: Some(8), hash_mb: 4, threads: 1, ..Default::default() };
    let stop = AtomicBool::new(false);
    match engine_core::search(&m.cfg, fen, hist, &lim, &stop, &mut |_| {}) {
        Ok(r) => r.score.abs() <= 60,
        Err(_) => false,
    }
}

fn opening(m: &MatchArgs, pair: usize) -> Result<(String, Vec<String>, Vec<String>), String> {
    let mut s = m.seed ^ (pair as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03);
    for _ in 0..1000 {
        let n = m.book_plies.0 + rng_next(&mut s) % (m.book_plies.1 - m.book_plies.0 + 1);
        let mut fen = start_position(&m.cfg)?;
        let mut hist = vec![];
        let mut mv = vec![];
        let mut ok = true;
        for _ in 0..n {
            if game_status(&m.cfg, &fen, &hist)?.outcome != Outcome::Ongoing {
                ok = false;
                break;
            }
            let ms = legal_moves(&m.cfg, &fen)?;
            let pick = &ms[(rng_next(&mut s) % ms.len() as u64) as usize].notation;
            let nf = apply_move(&m.cfg, &fen, pick)?;
            hist.push(std::mem::replace(&mut fen, nf));
            mv.push(pick.clone());
        }
        if ok && game_status(&m.cfg, &fen, &hist)?.outcome == Outcome::Ongoing && balanced(m, &fen, &hist) {
            return Ok((fen, hist, mv));
        }
    }
    Err("cannot build opening".into())
}

fn play(
    m: &MatchArgs,
    engs: &mut [Eng; 2],
    a_white: bool,
    open: &(String, Vec<String>, Vec<String>),
    st: &Mutex<Stats>,
    tag: &str,
) -> Result<f64, String> {
    for e in engs.iter_mut() {
        e.send(&json!({"cmd":"newgame"}))?;
    }
    let (mut fen, mut hist, mut moves) = (open.0.clone(), open.1.clone(), open.2.clone());
    let (mut dsum, mut cnt) = ([0u64; 2], [0u64; 2]);
    let (mut tm, mut tmax, mut over) = ([0u64; 2], [0u64; 2], [0u64; 2]);
    let (score, reason) = loop {
        let s = game_status(&m.cfg, &fen, &hist)?;
        let white_to_move = hist.len() % 2 == 0;
        if s.outcome != Outcome::Ongoing {
            let white_won = match s.outcome {
                Outcome::WhiteWins => Some(true),
                Outcome::BlackWins => Some(false),
                _ => None,
            };
            break (
                match white_won {
                    None => 0.5,
                    Some(w) => {
                        if w == a_white {
                            1.0
                        } else {
                            0.0
                        }
                    }
                },
                s.reason,
            );
        }
        if hist.len() >= m.max_plies {
            break (0.5, "move limit".to_string());
        }
        let who = if white_to_move == a_white { 0 } else { 1 };
        let mut go = m.go.clone();
        if let (Some((k, mm)), Some(t)) = (m.boost[who], m.go["time_ms"].as_u64()) {
            if cnt[who] < mm {
                go["time_ms"] = json!(t * k);
            }
        }
        let t_mv = std::time::Instant::now();
        let res_go = engs[who].go(&m.cfg, &fen, &hist, &go);
        let el = t_mv.elapsed().as_millis() as u64;
        tm[who] += el;
        tmax[who] = tmax[who].max(el);
        if let Some(lim) = go["time_ms"].as_u64() {
            if el > lim + 40 {
                over[who] += 1;
            }
        }
        match res_go {
            Ok((Some(mv), d, _)) => {
                dsum[who] += d;
                cnt[who] += 1;
                let nf = match apply_move(&m.cfg, &fen, &mv) {
                    Ok(nf) => nf,
                    Err(e) => return Err(format!("engine {} illegal move {mv} in {fen}: {e}", if who == 0 { "A" } else { "B" })),
                };
                hist.push(std::mem::replace(&mut fen, nf));
                moves.push(mv);
            }
            Ok((None, _, _)) => return Err(format!("engine {} returned no move in {fen}", if who == 0 { "A" } else { "B" })),
            Err(e) => return Err(format!("engine {} failure in {fen}: {e}", if who == 0 { "A" } else { "B" })),
        }
    };
    let mut g = st.lock().unwrap();
    for i in 0..2 {
        g.depth_sum[i] += dsum[i];
        g.moves[i] += cnt[i];
        g.time_sum[i] += tm[i];
        g.time_max[i] = g.time_max[i].max(tmax[i]);
        g.over[i] += over[i];
        g.eval_id[i] = engs[i].eval_id;
    }
    g.log.push(format!(
        "{tag} A={} result={} reason={} plies={} moves: {}",
        if a_white { "white" } else { "black" },
        score,
        reason,
        moves.len(),
        moves.join(" ")
    ));
    Ok(score)
}

use engine_core::matchplay::{elo_interval, pair_moments, sprt_llr};

fn report(st: &Stats, sprt: Option<(f64, f64)>) -> (String, Option<f64>) {
    let (n, mean, _var) = pair_moments(&st.penta);
    let (elo, lo, hi) = elo_interval(&st.penta);
    let mut s = format!(
        "W/D/L {}/{}/{}  pairs {}  score {:.3}  Elo {:+.1} [{:+.1}, {:+.1}] (95%)  penta {:?}",
        st.w, st.d, st.l, n as u64, mean, elo, lo, hi, st.penta
    );
    let mut llr = None;
    if let Some((e0, e1)) = sprt {
        if let Some(l) = sprt_llr(&st.penta, e0, e1) {
            s += &format!("  LLR {:.2} [-2.94, 2.94] H0:{e0} H1:{e1}", l);
            llr = Some(l);
        }
    }
    s += &format!("  eval_id A/B {:#x}/{:#x}", st.eval_id[0], st.eval_id[1]);
    s += &format!(
        "  avg depth A/B {:.1}/{:.1}",
        st.depth_sum[0] as f64 / st.moves[0].max(1) as f64,
        st.depth_sum[1] as f64 / st.moves[1].max(1) as f64
    );
    s += &format!(
        "  ms/move A/B {:.0}/{:.0} max {}/{} overlimit(+40ms) {}/{}",
        st.time_sum[0] as f64 / st.moves[0].max(1) as f64,
        st.time_sum[1] as f64 / st.moves[1].max(1) as f64,
        st.time_max[0],
        st.time_max[1],
        st.over[0],
        st.over[1]
    );
    (s, llr)
}

fn describe_eval(m: &MatchArgs, spec: &Option<engine_core::EvalSpec>) -> Result<String, String> {
    match spec {
        None => Ok("simple (по умолчанию)".to_string()),
        Some(engine_core::EvalSpec::Simple) => Ok("simple (eval_id 0)".to_string()),
        Some(engine_core::EvalSpec::File(p)) => {
            let l = engine_core::learn::select::load_cached(p)?;
            if let Some(meta) = &l.model.meta {
                if meta.variant != m.cfg.variant.name() || meta.forced != m.cfg.forced_capture {
                    return Err(format!(
                        "веса {} для {}:{}, а матч идёт в {}:{}",
                        p.display(),
                        meta.variant,
                        if meta.forced { "forced" } else { "free" },
                        m.cfg.variant.name(),
                        if m.cfg.forced_capture { "forced" } else { "free" }
                    ));
                }
            }
            Ok(format!("{} (eval_id {:#010x}, архитектура {})", p.display(), l.eval_id, l.model.arch.name))
        }
    }
}

pub fn run(m: MatchArgs) -> Result<(), String> {
    let eval_desc = [describe_eval(&m, &m.eval[0])?, describe_eval(&m, &m.eval[1])?];
    println!("match evals: A = {}  B = {}", eval_desc[0], eval_desc[1]);
    let pairs = (m.games + 1) / 2;
    let st = Mutex::new(Stats::default());
    let book = load_book(&m, pairs)?;
    let next = AtomicUsize::new(0);
    let halt = AtomicBool::new(false);
    let err: Mutex<Option<String>> = Mutex::new(None);
    let t0 = std::time::Instant::now();
    println!("match: optsA=[{}] optsB=[{}] A={} B={} variant={} forced={} go={} pairs={} conc={} threads={} hash={}MB", m.opts[0], m.opts[1], m.a, m.b, m.cfg.variant.name(), m.cfg.forced_capture, m.go, pairs, m.concurrency, m.threads, m.hash_mb);
    std::thread::scope(|sc| {
        for _ in 0..m.concurrency.max(1) {
            sc.spawn(|| {
                let mut engs = match (Eng::spawn(&m.a, &m, 0), Eng::spawn(&m.b, &m, 1)) {
                    (Ok(a), Ok(b)) => [a, b],
                    (Err(e), _) | (_, Err(e)) => {
                        *err.lock().unwrap() = Some(e);
                        halt.store(true, Ordering::Relaxed);
                        return;
                    }
                };
                loop {
                    if halt.load(Ordering::Relaxed) {
                        return;
                    }
                    let p = next.fetch_add(1, Ordering::Relaxed);
                    if p >= pairs {
                        return;
                    }
                    let res = Ok::<_, String>(book[p].clone()).and_then(|op| {
                        let s1 = play(&m, &mut engs, true, &op, &st, &format!("pair {p} g1"))?;
                        let s2 = play(&m, &mut engs, false, &op, &st, &format!("pair {p} g2"))?;
                        Ok((s1, s2))
                    });
                    match res {
                        Ok((s1, s2)) => {
                            let mut g = st.lock().unwrap();
                            for s in [s1, s2] {
                                if s > 0.9 {
                                    g.w += 1
                                } else if s < 0.1 {
                                    g.l += 1
                                } else {
                                    g.d += 1
                                }
                            }
                            g.penta[((s1 + s2) * 2.0).round() as usize] += 1;
                            let done: u64 = g.penta.iter().sum();
                            let (line, llr) = report(&g, m.sprt);
                            if done % 10 == 0 {
                                println!("[{:>5.0}s] {}", t0.elapsed().as_secs_f64(), line);
                            }
                            if let Some(l) = llr {
                                if l >= 2.94 || l <= -2.94 {
                                    halt.store(true, Ordering::Relaxed);
                                }
                            }
                        }
                        Err(e) => {
                            *err.lock().unwrap() = Some(e);
                            halt.store(true, Ordering::Relaxed);
                            return;
                        }
                    }
                }
            });
        }
    });
    let g = st.lock().unwrap();
    if let Some(dir) = std::path::Path::new(&m.log).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let head = format!("# evalA={} evalB={} variant={} forced={} go={}", eval_desc[0], eval_desc[1], m.cfg.variant.name(), m.cfg.forced_capture, m.go);
    let _ = std::fs::write(&m.log, head + "\n" + &g.log.join("\n") + "\n");
    let (line, llr) = report(&g, m.sprt);
    println!("FINAL [{:.0}s] {}", t0.elapsed().as_secs_f64(), line);
    if let Some(l) = llr {
        println!("SPRT: {}", if l >= 2.94 { "H1 accepted (A stronger)" } else if l <= -2.94 { "H0 accepted (no gain)" } else { "inconclusive" });
    }
    println!("log: {}", m.log);
    if let Some(e) = err.lock().unwrap().take() {
        return Err(e);
    }
    Ok(())
}

pub struct InprocArgs {
    pub cfg: GameConfig,
    pub eval: [engine_core::EvalSpec; 2],
    pub opts: [String; 2],
    pub nodes: Option<u64>,
    pub nodes_a: Option<u64>,
    pub nodes_b: Option<u64>,
    pub time_nodes_ms: Option<f64>,
    pub time_ms: u64,
    pub pairs: u32,
    pub concurrency: usize,
    pub sprt: Option<(f64, f64)>,
    pub hash_mb: usize,
    pub seed: u64,
    pub max_plies: u32,
    pub book_plies: (u32, u32),
    pub egdb: bool,
    pub boost: [Option<(u32, u32)>; 2],
    pub egdb_side: [Option<std::path::PathBuf>; 2],
    pub adj_egdb: Option<std::path::PathBuf>,
    pub adj_decided: bool,
    pub balance: (i32, i32),
    pub openings: Option<(String, std::sync::Arc<Vec<String>>)>,
    pub state: Option<std::path::PathBuf>,
    pub recall: [bool; 2],
    pub recall_file: [Option<std::path::PathBuf>; 2],
    pub recall_plies: u32,
    pub recall_max: usize,
    pub recall_cap: usize,
    pub recall_frozen: [bool; 2],
    pub recall_mirror: bool,
    pub passes: u32,
    pub record_moves: bool,
    pub ext: engine_core::matchplay::MatchExt,
}

pub fn run_inproc(m: InprocArgs) -> Result<(), String> {
    use engine_core::matchplay::{self, MatchCtl, MatchParams, Side};
    let (mut na, mut nb) = (m.nodes_a.or(m.nodes), m.nodes_b.or(m.nodes));
    if let Some(ms) = m.time_nodes_ms {
        let (a, b) = matchplay::time_equal_nodes(&m.cfg, &m.eval[0], &m.eval[1], ms)?;
        println!("равное время {ms} мс/ход через узлы: A {a} узлов, B {b} узлов");
        na = Some(a);
        nb = Some(b);
    }
    let p = MatchParams {
        a: Side { eval: m.eval[0].clone(), opts: m.opts[0].clone(), nodes: na, boost: m.boost[0], egdb_root: m.egdb_side[0].clone(), egdb_max_pieces: None },
        b: Side { eval: m.eval[1].clone(), opts: m.opts[1].clone(), nodes: nb, boost: m.boost[1], egdb_root: m.egdb_side[1].clone(), egdb_max_pieces: None },
        time_ms: Some(m.time_ms),
        pairs_max: m.pairs,
        sprt: m.sprt,
        seed: m.seed,
        book_plies: m.book_plies,
        balance_cp: m.balance.1,
        balance_lo: m.balance.0,
        max_plies: m.max_plies,
        openings: m.openings.as_ref().map(|(_, v)| v.clone()),
        hash_mb: m.hash_mb,
        threads: m.concurrency,
        use_egdb: m.egdb,
        adj_egdb_root: m.adj_egdb.clone(),
        adj_decided_only: m.adj_decided,
        state_path: m.state.clone(),
        record_moves: m.record_moves,
        ext: m.ext.clone(),
    };
    println!("match (in-process): A={} B={} control={} pairs={} threads={} hash={}MB seed={} book={:?} egdb={} opts_a={:?} opts_b={:?} boost_a={:?} boost_b={:?} egdb_a={:?} egdb_b={:?} adj_egdb={:?} adj_decided={} balance={},{}", m.eval[0].key_value(), m.eval[1].key_value(), if na.is_some() { format!("nodes {:?}/{:?}", na, nb) } else { format!("time {}ms", m.time_ms) }, m.pairs, m.concurrency, m.hash_mb, m.seed, m.book_plies, m.egdb, m.opts[0], m.opts[1], m.boost[0], m.boost[1], m.egdb_side[0], m.egdb_side[1], m.adj_egdb, m.adj_decided, m.balance.0, m.balance.1);
    if m.ext != engine_core::matchplay::MatchExt::default() {
        println!("режимы матча (пакет 2): {:?}", m.ext);
    }
    if let Some((name, v)) = &m.openings {
        println!("дебюты из файла: {name}, позиций {} (случайные дебюты и balance не применяются)", v.len());
        if m.pairs as usize > v.len() {
            println!("предупреждение: пар {} больше, чем позиций {}: позиции пойдут по кругу", m.pairs, v.len());
        }
    }
    let ctl = MatchCtl::new();
    let recall = recall_sides(&m)?;
    let passes = m.passes.max(1);
    if recall.iter().all(|r| r.is_none()) && passes == 1 {
        let r = matchplay::run_match(&m.cfg, &p, &ctl)?;
        println!("FINAL [{:.0}s] {}  eval_id A/B {:#x}/{:#x}  resumed pairs {}", r.secs, r.line(), r.eval_id[0], r.eval_id[1], r.resumed_pairs);
        if na.is_none() && nb.is_none() {
            let s = r.nps();
            println!("speed: nps_a={:.0} nps_b={:.0} moves={}/{} nodes={}/{}", s[0], s[1], r.moves[0], r.moves[1], r.nodes_sum[0], r.nodes_sum[1]);
        }
        if m.sprt.is_some() {
            println!("SPRT: {}", match r.sprt { Some(true) => "H1 accepted (A stronger)", Some(false) => "H0 accepted (no gain)", None => "inconclusive" });
        }
        return Ok(());
    }
    if passes > 1 && m.state.is_some() {
        return Err("--passes не сочетается с --state".into());
    }
    println!("recall: A={} B={} frozen A/B={}/{} mirror={} plies={} max_positions={} cap={} passes={passes}", m.recall[0], m.recall[1], m.recall_frozen[0], m.recall_frozen[1], m.recall_mirror, m.recall_plies, m.recall_max, m.recall_cap);
    let mut tot = matchplay::MatchReport::default();
    let mut secs = 0.0;
    for k in 1..=passes {
        let before: Vec<_> = recall.iter().map(|r| r.as_ref().map(|r| r.mem.snapshot())).collect();
        let r = matchplay::run_match_recall(&m.cfg, &p, &ctl, &recall)?;
        println!("FINAL pass={k} [{:.0}s] {}  eval_id A/B {:#x}/{:#x}", r.secs, r.line(), r.eval_id[0], r.eval_id[1]);
        for (i, rs) in recall.iter().enumerate() {
            if let (Some(rs), Some(b)) = (rs, before[i]) {
                println!("  pass={k} side={} {}", ["A", "B"][i], rs.mem.snapshot().since(&b).line());
                if let (Some(f), true) = (&m.recall_file[i], rs.learn) {
                    rs.mem.save(f)?;
                }
            }
        }
        secs += r.secs;
        tot.pairs += r.pairs;
        tot.wins += r.wins;
        tot.draws += r.draws;
        tot.losses += r.losses;
        for j in 0..5 {
            tot.penta[j] += r.penta[j];
        }
        if r.cancelled {
            break;
        }
    }
    let (elo, lo, hi) = matchplay::elo_interval(&tot.penta);
    let (_, mean, _) = matchplay::pair_moments(&tot.penta);
    tot.elo = elo;
    tot.elo_lo = lo;
    tot.elo_hi = hi;
    tot.score = mean;
    println!("FINAL total [{secs:.0}s] {}", tot.line());
    Ok(())
}

fn recall_sides(m: &InprocArgs) -> Result<[Option<engine_core::matchplay::RecallSide>; 2], String> {
    use engine_core::matchplay::RecallSide;
    use engine_core::recall::Recall;
    let mk = |i: usize| -> Result<Option<RecallSide>, String> {
        if !m.recall[i] {
            return Ok(None);
        }
        let mem = Recall::new(m.recall_cap);
        if let Some(f) = &m.recall_file[i] {
            if f.exists() {
                let n = mem.load(f)?;
                println!("память {}: загружено {n} записей из {}", ["A", "B"][i], f.display());
            }
        }
        Ok(Some(RecallSide { mem: std::sync::Arc::new(mem), plies: m.recall_plies, max_positions: m.recall_max, learn: !m.recall_frozen[i], mirror: m.recall_mirror }))
    };
    Ok([mk(0)?, mk(1)?])
}

pub fn parse_variant(name: &str, free: bool) -> Result<GameConfig, String> {
    let variant = VariantId::parse(name).ok_or(format!("unknown variant '{name}'"))?;
    Ok(GameConfig { variant, forced_capture: !free })
}
