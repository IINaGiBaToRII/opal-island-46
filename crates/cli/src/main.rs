
mod engine;
mod matchrun;
mod egdb_cmd;
mod selfplay_cmd;
mod learn_relabel;
mod weights_random;
use engine_core::{
    apply_move, draw_clock, game_from_pdn, game_status, game_status_with, legal_moves, perft, perft_divide, search,
    start_position, DrawRules, GameConfig, Outcome, SearchInfoOut, SearchLimits, VariantId,
};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

struct Args {
    cmd: String,
    kv: Vec<(String, Option<String>)>,
}

impl Args {
    fn parse() -> Args {
        let mut it = std::env::args().skip(1);
        let cmd = it.next().unwrap_or_default();
        let rest: Vec<String> = it.collect();
        let mut kv = vec![];
        let mut i = 0;
        while i < rest.len() {
            let a = &rest[i];
            if let Some(name) = a.strip_prefix("--") {
                if i + 1 < rest.len() && !rest[i + 1].starts_with("--") {
                    kv.push((name.to_string(), Some(rest[i + 1].clone())));
                    i += 2;
                    continue;
                }
                kv.push((name.to_string(), None));
            }
            i += 1;
        }
        Args { cmd, kv }
    }
    fn flag(&self, k: &str) -> bool {
        self.kv.iter().any(|(n, _)| n == k)
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.kv.iter().find(|(n, _)| n == k).and_then(|(_, v)| v.as_deref())
    }
    fn num<T: std::str::FromStr>(&self, k: &str) -> Result<Option<T>, String> {
        match self.get(k) {
            None => Ok(None),
            Some(v) => v.parse().map(Some).map_err(|_| format!("bad value for --{k}: {v}")),
        }
    }
    fn config(&self) -> Result<GameConfig, String> {
        let name = self.get("variant").unwrap_or("russian");
        let variant = VariantId::parse(name).ok_or(format!("unknown variant '{name}'"))?;
        Ok(GameConfig { variant, forced_capture: !self.flag("free") })
    }
}

fn fmt_score(s: i32) -> String {
    if s.abs() > 29800 {
        let plies = 30000 - s.abs();
        format!("{}M{}", if s > 0 { "+" } else { "-" }, (plies + 1) / 2)
    } else {
        format!("{:+.2}", s as f64 / 100.0)
    }
}

fn print_info(i: &SearchInfoOut) {
    println!(
        "depth {:2}  score {:>8}  nodes {:>10}  nps {:>9}  time {:>6}ms  hashfull {:>4}  pv {}",
        i.depth,
        fmt_score(i.score),
        i.nodes,
        i.nps,
        i.time_ms,
        i.hashfull,
        i.pv.join(" ")
    );
}

fn positions_from_moves(cfg: &GameConfig, start: &str, text: &str) -> Result<Vec<String>, String> {
    let mut all = vec![start.to_string()];
    for raw in text.split_whitespace() {
        if matches!(raw, "*" | "1-0" | "0-1" | "1/2-1/2" | "2-0" | "0-2" | "1-1" | "0-0") {
            break;
        }
        let digits = raw.bytes().take_while(u8::is_ascii_digit).count();
        let tok = if digits > 0 && raw[digits..].starts_with('.') { raw[digits..].trim_start_matches('.') } else { raw };
        if tok.is_empty() {
            continue;
        }
        let cur = all.last().unwrap();
        let next = apply_move(cfg, cur, tok).map_err(|e| format!("--moves, полуход {} ('{tok}'): {e}", all.len()))?;
        all.push(next);
    }
    Ok(all)
}

fn analyse_input(a: &Args, cfg: GameConfig, fen: &str) -> Result<(GameConfig, String, Vec<String>), String> {
    let (moves, pdn) = (a.get("moves"), a.get("pdn"));
    if moves.is_none() && pdn.is_none() {
        if a.get("ply").is_some() {
            return Err("--ply только вместе с --moves или --pdn".into());
        }
        return Ok((cfg, fen.to_string(), vec![]));
    }
    if a.get("fen").is_some() {
        return Err("--fen нельзя вместе с --moves/--pdn: история идёт от начальной позиции (у PDN — от тега FEN)".into());
    }
    let (cfg, mut all) = match (moves, pdn) {
        (Some(_), Some(_)) => return Err("--moves и --pdn вместе нельзя".into()),
        (Some(m), None) => (cfg, positions_from_moves(&cfg, &start_position(&cfg)?, m)?),
        (None, Some(p)) => {
            let text = std::fs::read_to_string(p).map_err(|e| format!("--pdn {p}: {e}"))?;
            let g = game_from_pdn(&text)?;
            (g.cfg, g.positions)
        }
        (None, None) => unreachable!(),
    };
    let ply = a.num::<usize>("ply")?.unwrap_or(all.len() - 1);
    if ply >= all.len() {
        return Err(format!("--ply {ply}: в партии {} полуходов", all.len() - 1));
    }
    all.truncate(ply + 1);
    let cur = all.pop().unwrap();
    Ok((cfg, cur, all))
}

fn run() -> Result<(), String> {
    let a = Args::parse();
    if a.cmd == "engine" {
        return std::thread::Builder::new().stack_size(8 << 20).spawn(engine::run).map_err(|e| e.to_string())?.join().map_err(|_| "engine thread panicked".to_string())?;
    }
    if a.cmd == "match" {
        let go = if let Some(n) = a.num::<u64>("nodes")? {
            serde_json::json!({ "nodes": n })
        } else if let Some(d) = a.num::<u64>("depth")? {
            serde_json::json!({ "depth": d })
        } else {
            serde_json::json!({ "time_ms": a.num::<u64>("time")?.unwrap_or(100) })
        };
        let sprt = match a.get("sprt") {
            None => None,
            Some(s) => {
                let p: Vec<f64> = s.split(',').filter_map(|x| x.parse().ok()).collect();
                if p.len() != 2 {
                    return Err("--sprt elo0,elo1".into());
                }
                Some((p[0], p[1]))
            }
        };
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let eval_spec = |k: &str| -> Result<Option<engine_core::EvalSpec>, String> { a.get(k).map(engine_core::EvalSpec::parse).transpose() };
        let eval = [eval_spec("eval-a")?, eval_spec("eval-b")?];
        let mut opts = [a.get("opts-a").unwrap_or("").to_string(), a.get("opts-b").unwrap_or("").to_string()];
        for i in 0..2 {
            if let Some(s) = &eval[i] {
                opts[i] = s.add_to_opts(&opts[i]);
            }
        }
        if a.flag("inproc") || a.get("time-nodes").is_some() {
            for (o, k) in opts.iter().zip(["opts-a", "opts-b"]) {
                engine_core::search::Params::try_parse(o).map_err(|e| format!("--{k}: {e}"))?;
            }
            let book: Vec<u32> = a.get("book").unwrap_or("6,12").split(',').filter_map(|x| x.parse().ok()).collect();
            if book.len() != 2 {
                return Err("--book MIN,MAX".into());
            }
            let pair = |k: &str, def: &str| -> Result<Vec<i64>, String> {
                let v: Vec<i64> = a.get(k).unwrap_or(def).split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if v.len() != 2 || v.iter().any(|x| *x < 0) {
                    return Err(format!("--{k} A,B (два неотрицательных числа)"));
                }
                Ok(v)
            };
            let boost = |k: &str| -> Result<Option<(u32, u32)>, String> {
                if a.get(k).is_none() {
                    return Ok(None);
                }
                let v = pair(k, "")?;
                Ok(Some((v[0] as u32, v[1] as u32)))
            };
            let bal = pair("balance", "0,60")?;
            let path = |k: &str| a.get(k).map(std::path::PathBuf::from);
            let games: usize = a.num("games")?.unwrap_or(200);
            return matchrun::run_inproc(matchrun::InprocArgs {
                cfg: matchrun::parse_variant(a.get("variant").unwrap_or("russian"), a.flag("free"))?,
                eval: [eval[0].clone().unwrap_or_default(), eval[1].clone().unwrap_or_default()],
                opts: [a.get("opts-a").unwrap_or("").to_string(), a.get("opts-b").unwrap_or("").to_string()],
                nodes: a.num("nodes")?,
                nodes_a: a.num("nodes-a")?,
                nodes_b: a.num("nodes-b")?,
                time_nodes_ms: a.num("time-nodes")?,
                time_ms: a.num("time")?.unwrap_or(100),
                pairs: ((games + 1) / 2) as u32,
                concurrency: a.num("concurrency")?.unwrap_or(0),
                sprt,
                hash_mb: a.num("hash")?.unwrap_or(16),
                seed: a.num("seed")?.unwrap_or(12345),
                max_plies: a.num("max-plies")?.unwrap_or(300),
                book_plies: (book[0], book[1]),
                egdb: a.flag("egdb"),
                boost: [boost("boost-a")?, boost("boost-b")?],
                egdb_side: [path("egdb-a"), path("egdb-b")],
                adj_egdb: path("adj-egdb"),
                adj_decided: a.flag("adj-decided"),
                balance: (bal[0] as i32, bal[1] as i32),
                openings: match a.get("openings") {
                    Some(f) => {
                        let cfg = matchrun::parse_variant(a.get("variant").unwrap_or("russian"), a.flag("free"))?;
                        let text = std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?;
                        let v = engine_core::matchplay::load_openings(&cfg, &text).map_err(|e| format!("{f}: {e}"))?;
                        Some((f.to_string(), std::sync::Arc::new(v)))
                    }
                    None => None,
                },
                state: a.get("state").map(std::path::PathBuf::from),
                recall: [a.flag("recall-a") || a.flag("recall-frozen-a"), a.flag("recall-b") || a.flag("recall-frozen-b")],
                recall_frozen: [a.flag("recall-frozen-a"), a.flag("recall-frozen-b")],
                recall_mirror: a.flag("recall-mirror"),
                recall_file: [path("recall-file-a"), path("recall-file-b")],
                recall_plies: a.num("recall-plies")?.unwrap_or(2),
                recall_max: a.num("recall-max")?.unwrap_or(4000),
                recall_cap: a.num("recall-cap")?.unwrap_or(200_000),
                passes: a.num("passes")?.unwrap_or(1),
                record_moves: a.flag("moves"),
                ext: engine_core::matchplay::MatchExt {
                    side_threads: [a.num("threads-a")?.unwrap_or(1), a.num("threads-b")?.unwrap_or(1)],
                    side_hash_mb: [a.num("hash-a")?.unwrap_or(0), a.num("hash-b")?.unwrap_or(0)],
                    clock: match a.get("clock") {
                        None => None,
                        Some(_) => {
                            let bi = pair("clock", "")?;
                            let mut c = engine_core::matchplay::GameClock::new(bi[0] as u64, bi[1] as u64);
                            if a.get("clock-front").is_some() {
                                let f = pair("clock-front", "")?;
                                (c.front_moves, c.front_pct) = (f[0] as u32, f[1] as u32);
                            }
                            c.moves_to_go = a.num("clock-mtg")?.unwrap_or(c.moves_to_go).max(1);
                            c.max_pct = a.num("clock-max")?.unwrap_or(c.max_pct).clamp(1, 100);
                            Some(c)
                        }
                    },
                    adj_eval: engine_core::matchplay::AdjEval {
                        win: match a.get("adj-win") {
                            None => None,
                            Some(_) => pair("adj-win", "").map(|v| Some((v[0] as i32, v[1] as u32)))?,
                        },
                        draw: match a.get("adj-draw") {
                            None => None,
                            Some(s) => {
                                let v: Vec<i64> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                                if v.len() != 3 || v.iter().any(|x| *x < 0) {
                                    return Err("--adj-draw CP,PLIES,AFTER (три неотрицательных числа)".into());
                                }
                                Some((v[0] as i32, v[1] as u32, v[2] as u32))
                            }
                        },
                        rep: a.num("adj-rep")?,
                    },
                    minsport: match a.get("draw-rules") {
                        None => false,
                        Some(n) => DrawRules::parse(n).ok_or(format!("--draw-rules legacy|minsport, а не {n}"))? == DrawRules::Minsport,
                    },
                },
            });
        }
        if a.get("draw-rules").is_some() {
            return Err("--draw-rules действует только в матче --inproc (внешние движки судит прежний счётчик)".into());
        }
        let boost_ext = |k: &str| -> Result<Option<(u64, u64)>, String> {
            let Some(s) = a.get(k) else { return Ok(None) };
            let v: Vec<u64> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            if v.len() != 2 {
                return Err(format!("--{k} K,M"));
            }
            Ok(Some((v[0], v[1])))
        };
        let m = matchrun::MatchArgs {
            a: a.get("a").ok_or("--a <bin>")?.to_string(),
            b: a.get("b").ok_or("--b <bin>")?.to_string(),
            cfg: matchrun::parse_variant(a.get("variant").unwrap_or("russian"), a.flag("free"))?,
            go,
            games: a.num("games")?.unwrap_or(200),
            concurrency: a.num("concurrency")?.unwrap_or(4),
            sprt,
            threads: a.num("threads-per-engine")?.unwrap_or(1),
            hash_mb: a.num("hash")?.unwrap_or(32),
            seed: a.num("seed")?.unwrap_or(12345),
            max_plies: a.num("max-plies")?.unwrap_or(300),
            log: a.get("log").map(|s| s.to_string()).unwrap_or(format!("logs/match-{ts}.log")),
            book_plies: (4, 6),
            opts,
            eval,
            boost: [boost_ext("boost-a")?, boost_ext("boost-b")?],
        };
        return matchrun::run(m);
    }
    if a.cmd == "ttd" {
        let cfg = a.config()?;
        let spec = a.get("eval").map(engine_core::EvalSpec::parse).transpose()?.unwrap_or_default();
        let f = a.get("positions").ok_or("--positions F.dfen")?;
        let text = std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?;
        let pos = engine_core::matchplay::load_openings(&cfg, &text).map_err(|e| format!("{f}: {e}"))?;
        let threads: Vec<usize> = a.get("threads").unwrap_or("1,2").split(',').filter_map(|x| x.trim().parse().ok()).collect();
        let (depth, hash, reps) = (a.num("depth")?.unwrap_or(12u32), a.num("hash")?.unwrap_or(16usize), a.num("reps")?.unwrap_or(3u32));
        let h = std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(move || engine_core::matchplay::time_to_depth(&cfg, &spec, &pos, depth, &threads, hash, reps))
            .map_err(|e| e.to_string())?;
        print!("{}", h.join().map_err(|_| "ttd thread panicked".to_string())??);
        return Ok(());
    }
    if a.cmd == "weights-info" {
        let path = std::env::args().nth(2).filter(|p| !p.starts_with("--")).ok_or("usage: runner weights-info <file.dw>")?;
        print!("{}", engine_core::learn::cmd_weights_info(&path)?);
        return Ok(());
    }
    if a.cmd.starts_with("egdb-") {
        return egdb_cmd::run(&a.cmd, &|k| a.get(k).map(String::from), &|k| a.flag(k));
    }
    if a.cmd == "selfplay-data" {
        return selfplay_cmd::selfplay_data(&|k| a.get(k).map(String::from), &|k| a.flag(k));
    }
    if a.cmd == "learn" {
        let sub = std::env::args().nth(2).unwrap_or_default();
        return learn_relabel::run(&sub, &|k| a.get(k).map(String::from), &|k| a.flag(k));
    }
    if a.cmd == "weights-random" {
        return weights_random::run(&|k| a.get(k).map(String::from), &|k| a.flag(k));
    }
    if a.cmd == "games-stat" {
        return selfplay_cmd::games_stat(&|k| a.get(k).map(String::from), &|k| a.flag(k));
    }
    let cfg = a.config()?;
    let fen = match a.get("fen") {
        Some(f) => f.to_string(),
        None => start_position(&cfg)?,
    };
    match a.cmd.as_str() {
        "perft" => {
            let depth: u32 = a.num("depth")?.unwrap_or(6);
            let dedup = !a.flag("no-dedup");
            if a.flag("divide") {
                let mut total = 0;
                for (m, n) in perft_divide(&cfg, &fen, depth, dedup)? {
                    println!("{m}: {n}");
                    total += n;
                }
                println!("total: {total}");
            } else {
                let t = Instant::now();
                let n = perft(&cfg, &fen, depth, dedup)?;
                let s = t.elapsed().as_secs_f64();
                println!(
                    "perft({depth}) = {n}  dedup={dedup}  time {:.3}s  {:.1} Mnodes/s",
                    s,
                    n as f64 / s.max(1e-9) / 1e6
                );
            }
        }
        "analyse" => {
            let (cfg, fen, hist) = analyse_input(&a, cfg, &fen)?;
            if !hist.is_empty() || a.get("draw-rules").is_some() {
                let name = a.get("draw-rules").unwrap_or("legacy");
                let rules = DrawRules::parse(name).ok_or(format!("--draw-rules legacy|minsport, а не {name}"))?;
                let dc = draw_clock(&cfg, &fen, &hist)?;
                let st = game_status_with(&cfg, &fen, &hist, rules)?;
                println!(
                    "history {} plies  draw counter {}/{}  seen {}  status {:?}{}",
                    hist.len(),
                    dc.plies,
                    dc.limit,
                    dc.seen,
                    st.outcome,
                    if st.reason.is_empty() { String::new() } else { format!(" ({})", st.reason) }
                );
                println!("fen {fen}");
            }
            let limits = SearchLimits {
                depth: a.num("depth")?,
                time_ms: a.num("time")?.or(if a.get("depth").is_none() { Some(1000) } else { None }),
                nodes: a.num("nodes")?,
                hash_mb: a.num("hash")?.unwrap_or(1024),
                threads: a.num("threads")?.unwrap_or(0),
                opts: a.get("opts").unwrap_or("").to_string(),
                ..Default::default()
            };
            let stop = AtomicBool::new(false);
            let fen2 = fen.clone();
            let cfg2 = cfg;
            let h = std::thread::Builder::new()
                .stack_size(256 << 20)
                .spawn(move || search(&cfg2, &fen2, &hist, &limits, &stop, &mut |i| print_info(i)))
                .map_err(|e| e.to_string())?;
            let r = h.join().map_err(|_| "search thread panicked".to_string())??;
            println!(
                "best {}  score {}  depth {}  nodes {}  time {}ms  nps {}",
                r.best_move.unwrap_or_else(|| "(none)".into()),
                fmt_score(r.score),
                r.depth,
                r.nodes,
                r.time_ms,
                r.nodes * 1000 / r.time_ms.max(1)
            );
            println!("pv {}", r.pv.join(" "));
        }
        "moves" => {
            let ms = legal_moves(&cfg, &fen)?;
            for m in &ms {
                println!("{}{}", m.notation, if m.promotes { " (promotes)" } else { "" });
            }
            println!("{} moves", ms.len());
        }
        "selfplay" => {
            let games: u32 = a.num("games")?.unwrap_or(2);
            let time: u64 = a.num("time")?.unwrap_or(100);
            let max_plies: usize = a.num("max-plies")?.unwrap_or(300);
            let cfg2 = cfg;
            let h = std::thread::Builder::new()
                .stack_size(256 << 20)
                .spawn(move || selfplay(cfg2, games, time, max_plies))
                .map_err(|e| e.to_string())?;
            h.join().map_err(|_| "selfplay thread panicked".to_string())??;
        }
        _ => {
            println!("usage: runner <perft|analyse|selfplay|moves> --variant russian [--free] [--fen \"..\"] ...");
            println!("  perft    --depth N [--no-dedup] [--divide]");
            println!("  analyse  [--depth N | --time ms] [--nodes N] [--hash MB] [--threads N (0=auto)] [--opts k=v,..]");
            println!("           [--moves \"c3-d4 b6-a5 ..\" | --pdn f.pdn] [--ply N] [--draw-rules legacy|minsport]   (история партии: счётчик ходов и повторения; N — сыграно полуходов)");
            println!("  selfplay --games N --time ms [--max-plies N]");
            println!("  moves");
            println!("  selfplay-data --variant V [--free] --out DIR --chunks N --games-per-chunk G --nodes K --seed S --threads T [--book-min --book-max --adj-score --adj-plies --max-plies --hash --no-egdb] [--rnd-prob P --rnd-min A --rnd-max B]");
            println!("  games-stat <каталог|файл> [--variant V] [--no-egdb] [--diversity]");
            println!("  selfplay-data ... [--db-play-draws]   (ничьи по базе доигрывать)");
            println!("  learn    relabel --from DIR --out DIR --eval <f.dw|simple> --nodes N [--threads T] [--no-egdb] [--any-eval]");
            println!("  match --inproc ... [--nodes-a N] [--nodes-b N] [--boost-a K,M] [--boost-b K,M] [--egdb-a ROOT] [--egdb-b ROOT] [--adj-egdb ROOT] [--balance LO,HI] [--openings FILE.dfen] [--moves] [--passes K] [--recall-a] [--recall-b] [--recall-frozen-a] [--recall-frozen-b] [--recall-mirror] [--recall-file-a F] [--recall-file-b F] [--recall-plies R] [--recall-max N] [--recall-cap N]");
            println!("  opts поиска: dbdraw=B (ничья по базе — выбор хода поиском, B > 0)");
            println!("  ttd --positions F.dfen [--eval simple|f.dw] [--depth D] [--threads 1,2] [--hash MB] [--reps K]  (время до глубины, П1.43)");
            println!("  match ... [--eval-a simple|<f.dw>] [--eval-b simple|<f.dw>]  (ключ eval= в opts стороны)");
            if !matches!(a.cmd.as_str(), "help" | "--help" | "-h") {
                usage_exit();
            }
        }
    }
    Ok(())
}

pub(crate) fn usage_exit() -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    std::process::exit(2);
}

fn selfplay(cfg: GameConfig, games: u32, time: u64, max_plies: usize) -> Result<(), String> {
    let (mut w, mut d, mut l) = (0, 0, 0);
    let stop = AtomicBool::new(false);
    for g in 0..games {
        let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ ((g as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut fen = start_position(&cfg)?;
        let mut hist: Vec<String> = vec![];
        let mut plies = 0;
        let t = Instant::now();
        let (res, reason) = loop {
            let st = game_status(&cfg, &fen, &hist)?;
            if st.outcome != Outcome::Ongoing {
                break (st.outcome, st.reason);
            }
            if plies >= max_plies {
                break (Outcome::Draw, "move limit".to_string());
            }
            let mv = if plies < 4 {
                let ms = legal_moves(&cfg, &fen)?;
                ms[(next() % ms.len() as u64) as usize].notation.clone()
            } else {
                let lim = SearchLimits { time_ms: Some(time), hash_mb: 16, threads: 1, ..Default::default() };
                search(&cfg, &fen, &hist, &lim, &stop, &mut |_| {})?
                    .best_move
                    .ok_or("no move")?
            };
            let nf = apply_move(&cfg, &fen, &mv)?;
            hist.push(std::mem::replace(&mut fen, nf));
            plies += 1;
        };
        match res {
            Outcome::WhiteWins => w += 1,
            Outcome::BlackWins => l += 1,
            _ => d += 1,
        }
        println!("game {}: {:?} ({}) plies {} time {:.1}s", g + 1, res, reason, plies, t.elapsed().as_secs_f64());
    }
    println!("W/D/L (white/draw/black): {w}/{d}/{l}");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G1_103: &str = "1. c3-b4 f6-g5 2. b4-c5 d6:b4 3. a3:c5 b6:d4 4. e3:c5 g5-f4 5. g3:e5 c7-b6 6. c5-d6 e7:c5 7. h2-g3 c5-b4 8. g3-f4 f8-e7 9. b2-c3 b6-a5 10. f2-e3 g7-f6 11. e5:g7 h6:f8 12. a1-b2 a7-b6 13. g1-f2 b4-a3 14. c3-d4 h8-g7 15. b2-c3 g7-f6 16. f2-g3 b6-c5 17. d4:b6 a5:c7 18. g3-h4 e7-d6 19. f4-g5 d8-e7 20. g5-h6 c7-b6 21. e3-f4 b6-a5 22. d2-e3 d6-c5 23. e1-d2 b8-a7 24. f4-g5 a7-b6 25. e3-f4 c5-b4 26. c3-d4 b6-c5 27. d4:b6 a5:c7 28. f4-e5 f6:d4 29. g5-f6 e7:g5 30. h4:f6 d4-c3 31. f6-g7 c3:e1 32. g7-h8 e1-g3 33. h8-f6 g3-h4 34. f6-h8 h4-e7 35. h8-a1 c7-b6 36. a1-h8 e7-d6 37. h8-a1 d6-b8 38. a1-h8 b6-a5 39. h8-d4 b8-a7 40. d4-a1 a7-c5 41. a1-e5 c5-b6 42. e5-h8 b6-d8 43. h8-a1 d8-c7 44. a1-h8 c7-g3 45. h8-a1 g3-f4 46. a1-h8 f4-e3 47. h8-a1 e3-a7 48. a1-h8 a7-f2 49. h8-a1 f2-g3 50. a1-b2 g3-h2 51. b2-a1 h2-g1 52. a1-b2";

    #[test]
    fn moves_history_gives_counter_27_of_30() {
        let cfg = GameConfig { variant: VariantId::Russian, forced_capture: true };
        let mut all = positions_from_moves(&cfg, &start_position(&cfg).unwrap(), G1_103).unwrap();
        assert_eq!(all.len(), 104);
        let cur = all.pop().unwrap();
        let dc = draw_clock(&cfg, &cur, &all).unwrap();
        assert_eq!((dc.plies, dc.limit), (27, 30));
        assert!(positions_from_moves(&cfg, &start_position(&cfg).unwrap(), "1. c3-d4 d4-e5").is_err());
    }
}
