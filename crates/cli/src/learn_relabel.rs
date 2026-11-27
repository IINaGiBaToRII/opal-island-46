
use engine_core::GameConfig;
use std::path::PathBuf;

type Get<'a> = &'a dyn Fn(&str) -> Option<String>;
type Flag<'a> = &'a dyn Fn(&str) -> bool;

fn num<T: std::str::FromStr>(get: Get, k: &str) -> Result<Option<T>, String> {
    match get(k) {
        None => Ok(None),
        Some(v) => v.parse().map(Some).map_err(|_| format!("bad value for --{k}: {v}")),
    }
}

pub fn run(sub: &str, get: Get, flag: Flag) -> Result<(), String> {
    let name = get("variant").unwrap_or_else(|| "russian".into());
    let cfg = GameConfig { variant: engine_core::VariantId::parse(&name).ok_or(format!("unknown variant '{name}'"))?, forced_capture: !flag("free") };
    let threads: usize = num(get, "threads")?.unwrap_or(0);
    match sub {
        "relabel" => relabel(&cfg, threads, get, flag),
        _ => Err("usage: runner learn relabel --variant V [--free] --from DIR --out DIR --eval <веса|simple> --nodes N ...".into()),
    }
}

pub fn relabel(cfg: &GameConfig, threads: usize, get: Get, flag: Flag) -> Result<(), String> {
    use engine_core::learn::relabel::{self, RelabelParams};
    let p = RelabelParams {
        from: PathBuf::from(get("from").ok_or("нужен --from <каталог кусков>")?),
        out: PathBuf::from(get("out").ok_or("нужен --out <каталог>")?),
        chunks: get("chunks").map(|s| relabel::parse_range(&s)).transpose()?,
        eval: engine_core::EvalSpec::parse(&get("eval").ok_or("нужен --eval <веса|simple>")?)?,
        nodes: num(get, "nodes")?.ok_or("нужен --nodes N")?,
        threads,
        egdb: get("egdb").map(PathBuf::from),
        no_egdb: flag("no-egdb"),
        any_eval: flag("any-eval"),
        samples_only: flag("samples-only"),
        verbose: true,
    };
    let r = relabel::relabel(cfg, &p)?;
    print!("{}", r.summary());
    println!("RELABEL_JSON {}", serde_json::to_string(&r).map_err(|e| e.to_string())?);
    Ok(())
}
