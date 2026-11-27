
use engine_core::booktask::{self, Spec};
use engine_core::GameConfig;
use std::path::PathBuf;
use std::time::{Duration, Instant};

type Get<'a> = &'a dyn Fn(&str) -> Option<String>;
type Flag<'a> = &'a dyn Fn(&str) -> bool;

pub fn run(get: Get, flag: Flag) -> Result<(), String> {
    let need = |k: &str| get(k).ok_or(format!("нужен --{k}"));
    let num = |k: &str, d: u64| -> Result<u64, String> {
        match get(k) {
            None => Ok(d),
            Some(v) => v.parse().map_err(|_| format!("--{k}: не число")),
        }
    };
    let name = get("variant").unwrap_or_else(|| "russian".into());
    let cfg = GameConfig { variant: engine_core::VariantId::parse(&name).ok_or("неизвестный вариант")?, forced_capture: !flag("free") };
    let tasks_path = need("tasks")?;
    let out = PathBuf::from(need("out")?);
    let opts = get("opts").unwrap_or_default();
    engine_core::search::Params::try_parse(&opts).map_err(|_| "--opts: не разобраны".to_string())?;
    let spec = Spec { eval: need("eval")?, hash_mb: num("hash", 256)? as usize, opts };
    let tasks = booktask::parse_tasks(&std::fs::read_to_string(&tasks_path).map_err(|e| format!("файл заданий: {e}"))?)?;
    let deadline = match num("seconds", 0)? {
        0 => None,
        s => Some(Instant::now() + Duration::from_secs(s)),
    };
    let t0 = Instant::now();
    let done = booktask::run_to_file(&cfg, &tasks, &spec, num("threads", 4)? as usize, deadline, &out)?;
    println!("BOOKTASK заданий {}, посчитано {done}, {:.0} с", tasks.len(), t0.elapsed().as_secs_f64());
    if done == 0 && !tasks.is_empty() {
        return Err("не посчитано ни одного задания".into());
    }
    Ok(())
}
