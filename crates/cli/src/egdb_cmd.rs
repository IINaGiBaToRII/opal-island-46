
use engine_core::egdb::{self, GenOpts};
use engine_core::{GameConfig, VariantId};
use std::time::Instant;

type Get<'a> = &'a dyn Fn(&str) -> Option<String>;
type Flag<'a> = &'a dyn Fn(&str) -> bool;

fn num<T: std::str::FromStr>(get: Get, k: &str) -> Result<Option<T>, String> {
    match get(k) {
        None => Ok(None),
        Some(v) => v.parse().map(Some).map_err(|_| format!("bad value for --{k}: {v}")),
    }
}

fn config(get: Get, flag: Flag) -> Result<GameConfig, String> {
    let name = get("variant").ok_or("нужен --variant <имя>")?;
    let variant = VariantId::parse(&name).ok_or(format!("unknown variant '{name}'"))?;
    Ok(GameConfig { variant, forced_capture: !flag("free") })
}

fn mode(cfg: &GameConfig) -> &'static str {
    if cfg.forced_capture {
        "forced"
    } else {
        "free"
    }
}

fn fmt_bytes(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.2} ГБ", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1} МБ", b as f64 / (1u64 << 20) as f64)
    } else {
        format!("{:.1} КБ", b as f64 / 1024.0)
    }
}

pub fn run(cmd: &str, get: Get, flag: Flag) -> Result<(), String> {
    match cmd {
        "egdb-gen" => {
            let cfg = config(get, flag)?;
            let pieces: u32 = num(get, "pieces")?.unwrap_or(4);
            let threads: usize = num(get, "threads")?.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
            let ram: f64 = num(get, "max-ram-gb")?.unwrap_or(8.0);
            let disk: f64 = num(get, "max-disk-gb")?.unwrap_or(2.0);
            let opts = GenOpts {
                dir: egdb::egdb_dir(&cfg),
                pieces,
                threads,
                max_ram_bytes: (ram * (1u64 << 30) as f64) as u64,
                max_disk_bytes: (disk * (1u64 << 30) as f64) as u64,
            };
            println!("egdb-gen {} {} до {} фигур, потоков {}, ОЗУ ≤ {} ГБ, диск ≤ {} ГБ, каталог {}", cfg.variant.name(), mode(&cfg), pieces, threads, ram, disk, opts.dir.display());
            let t = Instant::now();
            let rep = egdb::egdb_generate(&cfg, &opts, &mut |s| println!("[{:7.1}s] {s}", t.elapsed().as_secs_f64()))?;
            println!(
                "готово: построено {}, уже было {}, пропущено {}, остановка по диску: {}, записано {}, время {:.1} с",
                rep.built.len(),
                rep.existing,
                rep.skipped.len(),
                rep.stopped_disk,
                fmt_bytes(rep.bytes),
                rep.secs
            );
            Ok(())
        }
        "egdb-status" => {
            let cfgs: Vec<GameConfig> = match get("variant") {
                Some(n) => {
                    let variant = VariantId::parse(&n).ok_or(format!("unknown variant '{n}'"))?;
                    let modes = if flag("free") { vec![false] } else { vec![true, false] };
                    modes.into_iter().map(|f| GameConfig { variant, forced_capture: f }).collect()
                }
                None => engine_core::variants()
                    .iter()
                    .map(|v| v.id)
                    .flat_map(|variant| [true, false].into_iter().map(move |f| GameConfig { variant, forced_capture: f }))
                    .collect(),
            };
            println!("{:<14} {:<7} {:>11} {:>7} {:>11}", "вариант", "режим", "макс.фигур", "срезов", "размер");
            let mut total = 0;
            for c in &cfgs {
                let i = egdb::egdb_info(c);
                total += i.bytes;
                println!("{:<14} {:<7} {:>11} {:>7} {:>11}", c.variant.name(), mode(c), i.max_pieces, i.slices, fmt_bytes(i.bytes));
            }
            println!("итого: {}", fmt_bytes(total));
            Ok(())
        }
        "egdb-verify" => {
            let cfg = config(get, flag)?;
            let sample: usize = num(get, "sample")?.unwrap_or(2000);
            let indep: usize = num(get, "indep")?.unwrap_or(200);
            let dir = egdb::egdb_dir(&cfg);
            let t = Instant::now();
            let h = std::thread::Builder::new()
                .stack_size(128 << 20)
                .spawn(move || egdb::egdb_verify(&cfg, &dir, sample, 0x5EED, indep, &mut |s| println!("{s}")))
                .map_err(|e| e.to_string())?;
            let r = h.join().map_err(|_| "verify: паника".to_string())??;
            for m in r.mismatches.iter().chain(r.indep_contradictions.iter()) {
                println!("ОШИБКА: {m}");
            }
            println!(
                "срезов {}, проверено {}, расхождений {}, без потомков в базе {}; поиском без базы: проверено {}, подтверждено {}, не подтверждено {}, противоречий {}; {:.1} с",
                r.slices,
                r.checked,
                r.mismatches.len(),
                r.unresolved,
                r.indep_checked,
                r.indep_confirmed,
                r.indep_unconfirmed,
                r.indep_contradictions.len(),
                t.elapsed().as_secs_f64()
            );
            if r.ok() {
                println!("VERIFY OK");
                Ok(())
            } else {
                Err("VERIFY FAILED".into())
            }
        }
        "egdb-probe" => {
            let cfg = config(get, flag)?;
            let fen = get("fen").ok_or("нужен --fen \"<dfen>\"")?;
            let o = egdb::egdb_probe(&cfg, &egdb::egdb_dir(&cfg), &fen)?;
            println!("значение: {:?}  ({})", o.value, o.text);
            for (m, v) in o.moves {
                println!("  {m}: потомок {}", egdb::describe(v));
            }
            Ok(())
        }
        "egdb-repack" => {
            let src = get("src").ok_or("нужен --src <каталог срезов>")?;
            let out = get("out").ok_or("нужен --out <новый каталог>")?;
            let kib: u32 = num(get, "block-kib")?.ok_or("нужен --block-kib K (1, 2, 4, …, 64)")?;
            if !kib.is_power_of_two() {
                return Err(format!("--block-kib {kib}: нужна степень двойки"));
            }
            let t = Instant::now();
            let (n, bytes) = egdb::file::repack_dir(std::path::Path::new(&src), std::path::Path::new(&out), 10 + kib.trailing_zeros(), &mut |s| println!("[{:7.1}s] {s}", t.elapsed().as_secs_f64()))?;
            println!("готово: срезов {n}, записано {}, блок {kib} КиБ, время {:.1} с", fmt_bytes(bytes), t.elapsed().as_secs_f64());
            Ok(())
        }
        _ => Err(format!("неизвестная команда {cmd}")),
    }
}
