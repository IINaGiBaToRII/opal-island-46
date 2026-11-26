
mod engine;
mod matchrun;
mod egdb_cmd;
mod selfplay_cmd;
mod learn_relabel;
mod weights_random;
use engine_core::{
    apply_move, game_status, legal_moves, perft, perft_divide, search, start_position, GameConfig, Outcome,
    SearchInfoOut, SearchLimits, VariantId,
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
            });
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
                .spawn(move || search(&cfg2, &fen2, &[], &limits, &stop, &mut |i| print_info(i)))
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
            println!("  selfplay --games N --time ms [--max-plies N]");
            println!("  moves");
            println!("  selfplay-data --variant V [--free] --out DIR --chunks N --games-per-chunk G --nodes K --seed S --threads T [--book-min --book-max --adj-score --adj-plies --max-plies --hash --no-egdb] [--rnd-prob P --rnd-min A --rnd-max B]");
            println!("  games-stat <каталог|файл> [--variant V] [--no-egdb] [--diversity]");
            println!("  selfplay-data ... [--db-play-draws]   (ничьи по базе доигрывать)");
            println!("  learn    relabel --from DIR --out DIR --eval <f.dw|simple> --nodes N [--threads T] [--no-egdb] [--any-eval]");
            println!("  match --inproc ... [--nodes-a N] [--nodes-b N] [--boost-a K,M] [--boost-b K,M] [--egdb-a ROOT] [--egdb-b ROOT] [--adj-egdb ROOT] [--balance LO,HI] [--openings FILE.dfen] [--moves] [--passes K] [--recall-a] [--recall-b] [--recall-frozen-a] [--recall-frozen-b] [--recall-mirror] [--recall-file-a F] [--recall-file-b F] [--recall-plies R] [--recall-max N] [--recall-cap N]");
            println!("  opts поиска: dbdraw=B (ничья по базе — выбор хода поиском, B > 0)");
            println!("  match ... [--eval-a simple|<f.dw>] [--eval-b simple|<f.dw>]  (ключ eval= в opts стороны)");
        }
    }
    Ok(())
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
