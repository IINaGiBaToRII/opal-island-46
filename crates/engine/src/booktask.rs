
use crate::facade::{self, GameConfig};
use crate::search::SearchLimits;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

pub const TASKS_HEAD: &str = "# book tasks v1";
pub const MAX_MULTIPV: u32 = 10;
pub const MAX_TASK_NODES: u64 = 4_000_000_000;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Task {
    pub dfen: String,
    pub nodes: u64,
    pub multipv: u32,
}

#[derive(Clone, Debug)]
pub struct Spec {
    pub eval: String,
    pub hash_mb: usize,
    pub opts: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Head {
    pub v: u32,
    pub variant: String,
    pub forced: bool,
    pub opts: String,
    pub hash_mb: u32,
    pub egdb: u32,
    pub eval_id: u32,
    pub rules_version: u32,
    pub eval_arith: u32,
    pub build: String,
    pub draw: String,
}

pub const EVAL_ARITH: u32 = 1;

pub fn draw_rules(opts: &str) -> String {
    let p = crate::search::Params::parse(opts);
    format!("rep3={},dbdraw={},dbdraw_div={},minsport={}", p.rep3, p.dbdraw, p.dbdraw_div, p.minsport)
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MoveScore {
    pub mv: String,
    pub score: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mate: Option<i32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TaskOut {
    pub dfen: String,
    pub want: u64,
    pub multipv: u32,
    pub score: i32,
    pub depth: u32,
    pub nodes: u64,
    pub eval_id: u32,
    pub best: String,
    pub lines: Vec<MoveScore>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub early: String,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub hist_dep: bool,
}

pub fn tasks_text(tasks: &[Task]) -> String {
    let mut t = String::from(TASKS_HEAD);
    t.push('\n');
    for k in tasks {
        t.push_str(&format!("{}\t{}\t{}\n", k.dfen, k.nodes, k.multipv));
    }
    t
}

pub fn parse_tasks(text: &str) -> Result<Vec<Task>, String> {
    let mut it = text.lines();
    if it.next().map(str::trim_end) != Some(TASKS_HEAD) {
        return Err(format!("файл заданий: нет заголовка «{TASKS_HEAD}»"));
    }
    let mut out = Vec::new();
    for l in it.filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = l.split('\t').collect();
        let bad = || format!("строка заданий: {l}");
        if f.len() != 3 {
            return Err(bad());
        }
        let nodes: u64 = f[1].parse().map_err(|_| bad())?;
        let multipv: u32 = f[2].parse().map_err(|_| bad())?;
        if nodes == 0 || nodes > MAX_TASK_NODES || multipv == 0 || multipv > MAX_MULTIPV {
            return Err(bad());
        }
        out.push(Task { dfen: f[0].to_string(), nodes, multipv });
    }
    Ok(out)
}

pub fn limits(spec: &Spec, t: &Task) -> SearchLimits {
    let mut opts = format!("eval={}", spec.eval);
    if !spec.opts.trim().is_empty() {
        opts.push(',');
        opts.push_str(spec.opts.trim());
    }
    SearchLimits { nodes: Some(t.nodes), depth: None, time_ms: None, hash_mb: spec.hash_mb, threads: 1, opts, multipv: t.multipv as usize, ..Default::default() }
}

pub fn compute(cfg: &GameConfig, t: &Task, spec: &Spec, stop: &AtomicBool) -> Result<TaskOut, String> {
    let canon = facade::position_facts(cfg, &t.dfen)?.canonical;
    if canon != t.dfen {
        return Err("позиция задания не в канонической записи".into());
    }
    let r = facade::search(cfg, &t.dfen, &[], &limits(spec, t), stop, &mut |_| {})?;
    if stop.load(Ordering::Relaxed) {
        return Err("остановлено".into());
    }
    Ok(TaskOut {
        dfen: t.dfen.clone(),
        want: t.nodes,
        multipv: t.multipv,
        score: r.score,
        depth: r.depth,
        nodes: r.nodes,
        eval_id: r.eval_id,
        best: r.best_move.unwrap_or_default(),
        lines: r.lines.iter().map(|l| MoveScore { mv: l.notation.clone(), score: l.score, mate: l.mate }).collect(),
        early: r.early.unwrap_or_default(),
        hist_dep: r.hist_dep,
    })
}

pub fn head(cfg: &GameConfig, spec: &Spec) -> Head {
    let eval_id = crate::EvalSpec::parse(&spec.eval).ok().and_then(|s| crate::matchplay::spec_eval_id(cfg, &s).ok()).unwrap_or(0);
    Head {
        v: 1,
        variant: cfg.variant.name().to_string(),
        forced: cfg.forced_capture,
        opts: spec.opts.clone(),
        hash_mb: spec.hash_mb as u32,
        egdb: facade::egdb_max_pieces(cfg),
        eval_id,
        rules_version: crate::datadir::RULES_VERSION,
        eval_arith: EVAL_ARITH,
        build: option_env!("RUNNER_BUILD_HASH").unwrap_or("unknown").to_string(),
        draw: draw_rules(&spec.opts),
    }
}

pub fn run(cfg: &GameConfig, tasks: &[Task], spec: &Spec, threads: usize, deadline: Option<Instant>, sink: &(dyn Fn(&TaskOut) + Sync)) -> usize {
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let finished = AtomicBool::new(false);
    std::thread::scope(|s| {
        if let Some(d) = deadline {
            let (stop, finished) = (&stop, &finished);
            s.spawn(move || {
                while !finished.load(Ordering::Relaxed) {
                    if Instant::now() >= d {
                        stop.store(true, Ordering::Relaxed);
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
            });
        }
        let workers: Vec<_> = (0..threads.max(1).min(tasks.len().max(1)))
            .map(|_| {
                std::thread::Builder::new()
                    .stack_size(256 << 20)
                    .spawn_scoped(s, || loop {
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(t) = tasks.get(i) else { break };
                        match compute(cfg, t, spec, &stop) {
                            Ok(o) => {
                                sink(&o);
                                done.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(e) => eprintln!("задание {i}: {}", if e.contains(' ') && e.len() < 80 { e.as_str() } else { "ошибка расчёта" }),
                        }
                    })
                    .expect("spawn")
            })
            .collect();
        for w in workers {
            let _ = w.join();
        }
        finished.store(true, Ordering::Relaxed);
    });
    done.load(Ordering::Relaxed)
}

pub fn run_to_file(cfg: &GameConfig, tasks: &[Task], spec: &Spec, threads: usize, deadline: Option<Instant>, out: &Path) -> Result<usize, String> {
    let mut f = std::fs::File::create(out).map_err(|e| format!("{}: {e}", out.display()))?;
    writeln!(f, "{}", serde_json::to_string(&head(cfg, spec)).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    f.flush().map_err(|e| e.to_string())?;
    let f = Mutex::new(f);
    let n = run(cfg, tasks, spec, threads, deadline, &|o| {
        let line = serde_json::to_string(o).unwrap_or_default();
        let mut g = f.lock().unwrap();
        let _ = writeln!(g, "{line}");
        let _ = g.flush();
    });
    Ok(n)
}

pub fn parse_results(text: &str) -> Result<(Head, Vec<TaskOut>), String> {
    let mut it = text.lines().filter(|l| !l.trim().is_empty());
    let h: Head = serde_json::from_str(it.next().ok_or("файл результатов пуст")?).map_err(|e| format!("заголовок результатов: {e}"))?;
    if h.v != 1 {
        return Err(format!("версия результатов {}", h.v));
    }
    let mut out = Vec::new();
    for l in it {
        out.push(serde_json::from_str::<TaskOut>(l).map_err(|e| format!("строка результатов: {e}"))?);
    }
    Ok((h, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VariantId;

    fn cfg() -> GameConfig {
        GameConfig { variant: VariantId::parse("russian").unwrap(), forced_capture: true }
    }

    #[test]
    fn tasks_text_round_trip_and_bad_lines() {
        let start = facade::start_position(&cfg()).unwrap();
        let t = vec![Task { dfen: start.clone(), nodes: 2000, multipv: 3 }, Task { dfen: start, nodes: 500, multipv: 1 }];
        assert_eq!(parse_tasks(&tasks_text(&t)).unwrap(), t);
        assert!(parse_tasks("x\t1\t1\n").is_err(), "без заголовка");
        for bad in ["a\t0\t1", "a\t5\t0", "a\t5\t11", "a\t5", "a\tx\t1"] {
            assert!(parse_tasks(&format!("{TASKS_HEAD}\n{bad}\n")).is_err(), "{bad}");
        }
    }

    #[test]
    fn run_is_deterministic_and_file_round_trips() {
        let c = cfg();
        let start = facade::start_position(&c).unwrap();
        let kids = facade::children(&c, &start).unwrap();
        let mut tasks = vec![Task { dfen: start.clone(), nodes: 3000, multipv: 3 }];
        tasks.extend(kids.iter().take(3).map(|(_, d)| Task { dfen: d.clone(), nodes: 2000, multipv: 2 }));
        let spec = Spec { eval: "simple".into(), hash_mb: 4, opts: "ext1=1".into() };
        let dir = std::env::temp_dir().join(format!("booktask-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (a, b) = (dir.join("a.jsonl"), dir.join("b.jsonl"));
        assert_eq!(run_to_file(&c, &tasks, &spec, 3, None, &a).unwrap(), tasks.len());
        assert_eq!(run_to_file(&c, &tasks, &spec, 1, None, &b).unwrap(), tasks.len());
        let (h, mut ra) = parse_results(&std::fs::read_to_string(&a).unwrap()).unwrap();
        let (_, mut rb) = parse_results(&std::fs::read_to_string(&b).unwrap()).unwrap();
        ra.sort_by(|x, y| x.dfen.cmp(&y.dfen));
        rb.sort_by(|x, y| x.dfen.cmp(&y.dfen));
        assert_eq!(ra, rb, "результат не зависит от числа потоков и порядка");
        assert_eq!((h.v, h.opts.as_str(), h.hash_mb), (1, "ext1=1", 4));
        assert_eq!((h.eval_id, h.rules_version, h.eval_arith), (0, crate::datadir::RULES_VERSION, EVAL_ARITH));
        assert!(!h.build.is_empty() && h.draw == draw_rules("ext1=1") && h.draw != draw_rules("ext1=1,rep3=1"), "{h:?}");
        assert_eq!(draw_rules("rep3=1,dbdraw=10"), draw_rules("dbdraw=10,ext1=1,rep3=1"), "сравниваются значения, не строка opts");
        let root = ra.iter().find(|o| o.dfen == start).unwrap();
        assert_eq!((root.want, root.multipv, root.lines.len()), (3000, 3, 3));
        assert_eq!(root.best, root.lines[0].mv);
        assert!(root.lines.windows(2).all(|w| w[0].score >= w[1].score), "{:?}", root.lines);
        let _ = std::fs::remove_file(&a);
        let _ = std::fs::remove_file(&b);
    }

    #[test]
    fn non_canonical_position_is_refused_and_deadline_stops() {
        let c = cfg();
        let start = facade::start_position(&c).unwrap();
        let spec = Spec { eval: "simple".into(), hash_mb: 4, opts: String::new() };
        let stop = AtomicBool::new(false);
        assert!(compute(&c, &Task { dfen: format!("{start} "), nodes: 100, multipv: 1 }, &spec, &stop).is_err());
        let tasks = vec![Task { dfen: start, nodes: MAX_TASK_NODES, multipv: 1 }];
        let t0 = Instant::now();
        let n = run(&c, &tasks, &spec, 1, Some(Instant::now() + std::time::Duration::from_millis(300)), &|_| {});
        assert_eq!(n, 0, "задание, не уложившееся в срок, в итог не попадает");
        assert!(t0.elapsed().as_secs() < 20);
    }
}
