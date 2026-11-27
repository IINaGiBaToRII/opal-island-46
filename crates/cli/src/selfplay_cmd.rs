
use engine_core::selfplay::{self, SelfplayCtl, SelfplayParams};
use engine_core::{GameConfig, VariantId};
use std::path::PathBuf;

type Get<'a> = &'a dyn Fn(&str) -> Option<String>;
type Flag<'a> = &'a dyn Fn(&str) -> bool;

fn num<T: std::str::FromStr>(get: Get, k: &str) -> Result<Option<T>, String> {
    match get(k) {
        None => Ok(None),
        Some(v) => v.parse().map(Some).map_err(|_| format!("bad value for --{k}: {v}")),
    }
}

fn variant_of(get: Get) -> Result<VariantId, String> {
    let name = get("variant").unwrap_or_else(|| "russian".into());
    VariantId::parse(&name).ok_or(format!("unknown variant '{name}'"))
}

fn openings_of(cfg: &GameConfig, path: Option<&str>) -> Result<Option<std::sync::Arc<Vec<String>>>, String> {
    let Some(p) = path else { return Ok(None) };
    let text = std::fs::read_to_string(p).map_err(|e| format!("дебюты {p}: {e}"))?;
    Ok(Some(std::sync::Arc::new(engine_core::matchplay::load_openings(cfg, &text).map_err(|e| format!("дебюты {p}: {e}"))?)))
}

pub fn selfplay_data(get: Get, flag: Flag) -> Result<(), String> {
    let cfg = GameConfig { variant: variant_of(get)?, forced_capture: !flag("free") };
    let out = PathBuf::from(get("out").ok_or("нужен --out <каталог>")?);
    let d = SelfplayParams::default();
    let (rnd_permille, rnd_min, rnd_max) = rnd_opts(get)?;
    let p = SelfplayParams {
        games_per_chunk: num(get, "games-per-chunk")?.unwrap_or(d.games_per_chunk),
        nodes: num(get, "nodes")?.unwrap_or(d.nodes),
        seed: num(get, "seed")?.unwrap_or(d.seed),
        book_min: num(get, "book-min")?.unwrap_or(d.book_min),
        book_max: num(get, "book-max")?.unwrap_or(d.book_max),
        adj_score: num(get, "adj-score")?.unwrap_or(d.adj_score),
        adj_plies: num(get, "adj-plies")?.unwrap_or(d.adj_plies),
        max_plies: num(get, "max-plies")?.unwrap_or(d.max_plies),
        hash_mb: num(get, "hash")?.unwrap_or(d.hash_mb),
        threads: num(get, "threads")?.unwrap_or(d.threads),
        use_egdb: !flag("no-egdb"),
        db_play_draws: flag("db-play-draws"),
        created_unix: None,
        rnd_permille,
        rnd_min,
        rnd_max,
        openings: openings_of(&cfg, get("openings").as_deref())?,
        eval: match get("eval") {
            Some(v) => engine_core::EvalSpec::parse(&v)?,
            None => engine_core::EvalSpec::Simple,
        },
    };
    let chunks: u32 = num(get, "chunks")?.unwrap_or(1);
    let first: u32 = num(get, "first-chunk")?.unwrap_or(0);
    let ctl = SelfplayCtl::new();
    let t = std::time::Instant::now();
    let (mut games, mut plies, mut bytes) = (0u64, 0u64, 0u64);
    for c in first..first + chunks {
        let r = selfplay::generate_chunk(&cfg, &out, c, &p, &ctl)?;
        if r.skipped {
            println!("кусок {c}: уже есть ({} партий, {} байт) — пропуск", r.games, r.bytes);
        } else {
            println!(
                "кусок {c}: {} партий за {:.1}с ({:.1} партий/с), полуходов {}, +{} ={} -{}, файл {} байт ({:.1} байт/партию)",
                r.games,
                r.secs,
                r.games as f64 / r.secs.max(1e-9),
                r.plies,
                r.white_wins,
                r.draws,
                r.black_wins,
                r.bytes,
                r.bytes as f64 / r.games.max(1) as f64
            );
            games += r.games as u64;
            plies += r.plies;
            bytes += r.bytes;
        }
    }
    let s = t.elapsed().as_secs_f64();
    println!("итого сыграно {games} партий, {plies} полуходов, {bytes} байт за {s:.1}с ({:.1} партий/с)", games as f64 / s.max(1e-9));
    Ok(())
}

fn rnd_opts(get: Get) -> Result<(u32, u32, u32), String> {
    let (p, a, b) = (get("rnd-prob"), num::<u32>(get, "rnd-min")?, num::<u32>(get, "rnd-max")?);
    match (p, a, b) {
        (None, None, None) => Ok((0, 0, 0)),
        (Some(p), Some(a), Some(b)) => {
            let pm = selfplay::prob_permille(&p)?;
            if a > b {
                return Err(format!("--rnd-min {a} больше --rnd-max {b}"));
            }
            Ok(if pm == 0 { (0, 0, 0) } else { (pm, a, b) })
        }
        _ => Err("случайный ход в середине партии: нужны вместе --rnd-prob P --rnd-min A --rnd-max B".into()),
    }
}

pub fn games_stat(get: Get, flag: Flag) -> Result<(), String> {
    let raw: Vec<String> = std::env::args().skip(2).collect();
    let mut path = None;
    let mut i = 0;
    while i < raw.len() {
        if raw[i] == "--variant" {
            i += 2;
            continue;
        }
        if !raw[i].starts_with("--") && path.is_none() {
            path = Some(PathBuf::from(&raw[i]));
        }
        i += 1;
    }
    let path = path.ok_or("нужен путь: games-stat <каталог|файл>")?;
    let variant = if get("variant").is_some() { Some(variant_of(get)?) } else { None };
    let mut files = vec![];
    if path.is_dir() {
        let rd = std::fs::read_dir(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "dgz") {
                files.push(p);
            }
        }
        engine_core::selfplay::sort_chunk_paths(&mut files);
    } else {
        files.push(path);
    }
    let (cfg, st) = selfplay::stats_for_paths_with(&files, variant, !flag("no-egdb"), flag("diversity"))?;
    println!("конфигурация {}:{}", cfg.variant.name(), if cfg.forced_capture { "forced" } else { "free" });
    print!("{}", st.report());
    print!("{}", st.diversity_report());
    Ok(())
}
