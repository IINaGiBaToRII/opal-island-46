
use super::select::EvalSpec;
use super::wdl::Wdl;
use super::weights::Model;
use crate::datadir::{self, dir_in, Kind};
use crate::facade::{GameConfig, VariantId};
use crate::rules::{American, Brazilian, Czech, International, Italian, Rules, Russian, Spanish, Thai, Turkish};
use crate::selfplay::dispatch_cfg;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

pub const FORMAT: u32 = 1;
pub const RECHECK: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Candidate,
    Accepted,
    Rejected,
    RolledBack,
}

impl Status {
    pub fn name(self) -> &'static str {
        match self {
            Status::Candidate => "candidate",
            Status::Accepted => "accepted",
            Status::Rejected => "rejected",
            Status::RolledBack => "rolled_back",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DataInfo {
    pub games: u64,
    pub samples: u64,
    pub unique: u64,
    pub draw_rate: f64,
    pub nodes: u64,
    pub source_eval_ids: Vec<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainInfo {
    pub k: f64,
    pub lambda: f64,
    pub epochs: u32,
    pub val_loss: f64,
    pub val_loss_simple: f64,
    pub secs: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MatchSummary {
    pub opponent: u32,
    pub control: String,
    pub games: u32,
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
    pub elo: f64,
    pub elo_lo: f64,
    pub elo_hi: f64,
    pub llr: Option<f64>,
    pub verdict: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Speed {
    pub evals_per_s: f64,
    pub nps: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Generation {
    pub id: u32,
    pub parent: u32,
    pub file: String,
    pub eval_id: u32,
    pub arch_name: String,
    pub arch_summary: String,
    pub weights: u64,
    pub bytes: u64,
    pub data: Option<DataInfo>,
    pub train: Option<TrainInfo>,
    pub match_short: Option<MatchSummary>,
    pub match_long: Option<MatchSummary>,
    pub speed: Option<Speed>,
    pub search_opts: String,
    pub wdl: Option<Wdl>,
    pub status: Status,
    pub created_unix: u64,
    pub note: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub basis: String,
}

impl Default for Generation {
    fn default() -> Self {
        Generation {
            id: 0,
            parent: 0,
            file: String::new(),
            eval_id: 0,
            arch_name: String::new(),
            arch_summary: String::new(),
            weights: 0,
            bytes: 0,
            data: None,
            train: None,
            match_short: None,
            match_long: None,
            speed: None,
            search_opts: String::new(),
            wdl: None,
            status: Status::Candidate,
            created_unix: 0,
            note: String::new(),
            basis: String::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GenInfo {
    pub data: Option<DataInfo>,
    pub train: Option<TrainInfo>,
    pub search_opts: String,
    pub note: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchKind {
    Short,
    Long,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Registry {
    pub format: u32,
    pub active: u32,
    pub generations: Vec<Generation>,
    pub history: Vec<u32>,
}

impl Default for Registry {
    fn default() -> Self {
        Registry { format: FORMAT, active: 0, generations: vec![], history: vec![] }
    }
}

pub fn weights_dir(root: &Path, cfg: &GameConfig) -> PathBuf {
    dir_in(root, cfg.variant.name(), cfg.forced_capture, Kind::Weights)
}

pub fn registry_path(root: &Path, cfg: &GameConfig) -> PathBuf {
    weights_dir(root, cfg).join("registry.json")
}

fn lock() -> std::sync::MutexGuard<'static, ()> {
    static L: Mutex<()> = Mutex::new(());
    L.lock().unwrap_or_else(|e| e.into_inner())
}

fn load_model<R: Rules>(p: &Path) -> Result<Model, String> {
    Model::load::<R>(p)
}

fn load_for(cfg: &GameConfig, p: &Path) -> Result<Model, String> {
    dispatch_cfg!(cfg, load_model(p))
}

fn weights_count(m: &Model) -> u64 {
    m.layout.blocks.iter().map(|b| b.len as u64).sum()
}

impl Registry {
    pub fn load(root: &Path, cfg: &GameConfig) -> Registry {
        let p = registry_path(root, cfg);
        if !p.exists() {
            return Registry::default();
        }
        match datadir::read_json_or_stale::<Registry>(&p) {
            Some(r) if r.format == FORMAT => r,
            Some(_) => {
                let _ = datadir::mark_stale(&p);
                Registry::default()
            }
            None => Registry::default(),
        }
    }

    fn load_for_update(root: &Path, cfg: &GameConfig) -> Result<Registry, String> {
        let p = registry_path(root, cfg);
        match datadir::read_json_strict::<Registry>(&p)? {
            Some(r) if r.format == FORMAT => Ok(r),
            Some(_) => {
                datadir::stale_or_err(&p)?;
                Ok(Registry::default())
            }
            None => Ok(Registry::default()),
        }
    }

    pub fn load_default(cfg: &GameConfig) -> Registry {
        Registry::load(&datadir::data_root(), cfg)
    }

    fn save(&self, root: &Path, cfg: &GameConfig) -> Result<(), String> {
        let p = registry_path(root, cfg);
        let js = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        datadir::write_atomic(&p, &js).map_err(|e| format!("запись {}: {e}", p.display()))?;
        invalidate_cache();
        Ok(())
    }

    fn update<T>(root: &Path, cfg: &GameConfig, f: impl FnOnce(&mut Registry) -> Result<T, String>) -> Result<T, String> {
        let _g = lock();
        let mut r = Registry::load_for_update(root, cfg)?;
        let out = f(&mut r)?;
        r.save(root, cfg)?;
        Ok(out)
    }

    pub fn get(&self, id: u32) -> Option<&Generation> {
        self.generations.iter().find(|g| g.id == id)
    }

    fn get_mut(&mut self, id: u32) -> Result<&mut Generation, String> {
        self.generations.iter_mut().find(|g| g.id == id).ok_or_else(|| format!("нет поколения {id}"))
    }

    pub fn active_generation(&self) -> Option<&Generation> {
        if self.active == 0 {
            None
        } else {
            self.get(self.active)
        }
    }

    pub fn wdl_for(&self, id: u32) -> Wdl {
        self.get(id).and_then(|g| g.wdl.clone()).unwrap_or_else(Wdl::builtin)
    }

    pub fn add_candidate(root: &Path, cfg: &GameConfig, weights: &Path, parent: u32, info: GenInfo) -> Result<u32, String> {
        let model = load_for(cfg, weights)?;
        let bytes = std::fs::read(weights).map_err(|e| format!("{}: {e}", weights.display()))?;
        Self::update(root, cfg, |r| {
            if parent != 0 && r.get(parent).is_none() {
                return Err(format!("нет родительского поколения {parent}"));
            }
            let id = r.generations.iter().map(|g| g.id).max().unwrap_or(0) + 1;
            let file = format!("gen-{id:04}.dw");
            let dst = weights_dir(root, cfg).join(&file);
            if dst.exists() {
                return Err(format!("{} уже существует (не перезаписываю)", dst.display()));
            }
            datadir::write_atomic(&dst, &bytes).map_err(|e| format!("запись {}: {e}", dst.display()))?;
            r.generations.push(Generation {
                id,
                parent,
                file,
                eval_id: model.eval_id(),
                arch_name: model.arch.name.clone(),
                arch_summary: model.arch.summary().unwrap_or_default(),
                weights: weights_count(&model),
                bytes: bytes.len() as u64,
                data: info.data,
                train: info.train,
                search_opts: info.search_opts,
                status: Status::Candidate,
                created_unix: datadir::now_secs(),
                note: info.note,
                ..Default::default()
            });
            Ok(id)
        })
    }

    pub fn set_match(root: &Path, cfg: &GameConfig, id: u32, kind: MatchKind, m: MatchSummary) -> Result<(), String> {
        Self::update(root, cfg, |r| {
            if m.opponent != 0 && r.get(m.opponent).is_none() {
                return Err(format!("нет поколения-соперника {}", m.opponent));
            }
            let g = r.get_mut(id)?;
            match kind {
                MatchKind::Short => g.match_short = Some(m),
                MatchKind::Long => g.match_long = Some(m),
            }
            Ok(())
        })
    }

    pub fn set_speed(root: &Path, cfg: &GameConfig, id: u32, s: Speed) -> Result<(), String> {
        Self::update(root, cfg, |r| {
            r.get_mut(id)?.speed = Some(s);
            Ok(())
        })
    }

    pub fn set_wdl(root: &Path, cfg: &GameConfig, id: u32, w: Wdl) -> Result<(), String> {
        Self::update(root, cfg, |r| {
            r.get_mut(id)?.wdl = Some(w);
            Ok(())
        })
    }

    pub fn set_search_opts(root: &Path, cfg: &GameConfig, id: u32, opts: &str) -> Result<(), String> {
        crate::search::Params::try_parse(opts).map_err(|e| format!("search_opts: {e}"))?;
        Self::update(root, cfg, |r| {
            r.get_mut(id)?.search_opts = opts.trim().to_string();
            Ok(())
        })
    }

    pub fn set_basis(root: &Path, cfg: &GameConfig, id: u32, basis: &str) -> Result<(), String> {
        Self::update(root, cfg, |r| {
            r.get_mut(id)?.basis = basis.trim().to_string();
            Ok(())
        })
    }

    pub fn accept(root: &Path, cfg: &GameConfig, id: u32) -> Result<(), String> {
        Self::update(root, cfg, |r| {
            let g = r.get(id).ok_or_else(|| format!("нет поколения {id}"))?;
            if g.status == Status::Rejected {
                return Err(format!("поколение {id} отклонено; принять можно только кандидата или откатённое"));
            }
            check_gen(root, cfg, g)?;
            r.get_mut(id)?.status = Status::Accepted;
            r.history.retain(|&x| x != id);
            r.history.push(id);
            r.active = id;
            Ok(())
        })
    }

    pub fn reject(root: &Path, cfg: &GameConfig, id: u32) -> Result<(), String> {
        Self::update(root, cfg, |r| {
            if r.active == id {
                return Err(format!("поколение {id} действующее: сначала откат (rollback)"));
            }
            let g = r.get_mut(id)?;
            if !matches!(g.status, Status::Candidate | Status::RolledBack) {
                return Err(format!("поколение {id} в статусе {}: отклонить можно кандидата или откатённое", g.status.name()));
            }
            g.status = Status::Rejected;
            r.history.retain(|&x| x != id);
            Ok(())
        })
    }

    pub fn rollback(root: &Path, cfg: &GameConfig, to: Option<u32>) -> Result<u32, String> {
        Self::update(root, cfg, |r| {
            let cur = r.active;
            let target = match to {
                Some(0) => 0,
                Some(n) => {
                    let g = r.get(n).ok_or_else(|| format!("нет поколения {n}"))?;
                    if !matches!(g.status, Status::Accepted | Status::RolledBack) {
                        return Err(format!("поколение {n} в статусе {}: вернуться можно к принятому или откатённому", g.status.name()));
                    }
                    check_gen(root, cfg, g)?;
                    n
                }
                None => {
                    if cur == 0 {
                        return Err("уже действует SimpleEval (поколение 0)".into());
                    }
                    r.history
                        .iter()
                        .rev()
                        .copied()
                        .find(|&x| x != cur && r.get(x).map(|g| g.status == Status::Accepted).unwrap_or(false))
                        .unwrap_or(0)
                }
            };
            if target == cur {
                return Err(format!("поколение {cur} уже действует"));
            }
            if cur != 0 {
                if let Some(g) = r.generations.iter_mut().find(|g| g.id == cur) {
                    g.status = Status::RolledBack;
                }
            }
            r.history.retain(|&x| x != cur && x != target);
            if target != 0 {
                r.get_mut(target)?.status = Status::Accepted;
                r.history.push(target);
            }
            r.active = target;
            Ok(target)
        })
    }

    pub fn archive_rejected(root: &Path, cfg: &GameConfig, id: u32) -> Result<PathBuf, String> {
        Self::update(root, cfg, |r| {
            let g = r.get_mut(id)?;
            if g.status != Status::Rejected {
                return Err(format!("поколение {id} не отклонено: в архив переносятся только отклонённые"));
            }
            if g.file.starts_with("archive/") {
                return Err(format!("поколение {id} уже в архиве"));
            }
            let wd = weights_dir(root, cfg);
            let src = wd.join(&g.file);
            let ad = wd.join("archive");
            std::fs::create_dir_all(&ad).map_err(|e| e.to_string())?;
            let mut dst = ad.join(&g.file);
            let mut i = 0;
            while dst.exists() {
                i += 1;
                dst = ad.join(format!("{}.{i}", g.file));
            }
            if src.exists() {
                std::fs::rename(&src, &dst).map_err(|e| format!("перенос {} -> {}: {e}", src.display(), dst.display()))?;
            }
            g.file = format!("archive/{}", dst.file_name().and_then(|n| n.to_str()).unwrap_or(""));
            Ok(dst)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn import_accepted(
        root: &Path,
        cfg: &GameConfig,
        imp: &ImportFile,
        note: &str,
        ext: &ExternalMatch,
        min_id: u32,
        reject: Option<u32>,
        speed: Option<Speed>,
        after: impl FnOnce(&Registry, u32) -> Result<(), String>,
    ) -> Result<u32, String> {
        let _g = lock();
        let mut r = Registry::load_for_update(root, cfg)?;
        if let Some(g) = r.generations.iter().find(|g| g.eval_id == imp.eval_id) {
            return Err(format!("оценка с eval_id {:#010x} уже в реестре: поколение {} ({}); повторный импорт не нужен", imp.eval_id, g.id, g.status.name()));
        }
        let wd = weights_dir(root, cfg);
        let id = (r.generations.iter().map(|g| g.id).max().unwrap_or(0).max(max_gen_file(&wd)).max(max_gen_file(&wd.join("archive"))) + 1).max(min_id);
        let file = format!("gen-{id:04}.dw");
        let dst = wd.join(&file);
        if dst.exists() {
            return Err(format!("{} уже существует (не перезаписываю)", dst.display()));
        }
        let parent = r.active;
        if let Some(c) = reject {
            if let Some(g) = r.generations.iter_mut().find(|g| g.id == c && g.status == Status::Candidate) {
                g.status = Status::Rejected;
                r.history.retain(|&x| x != c);
            }
        }
        r.generations.push(Generation {
            id,
            parent,
            file: file.clone(),
            eval_id: imp.eval_id,
            arch_name: imp.arch_name.clone(),
            arch_summary: imp.arch_summary.clone(),
            weights: imp.weights,
            bytes: imp.bytes.len() as u64,
            match_long: Some(MatchSummary {
                opponent: parent,
                control: format!("nodes {} (внешний матч)", ext.nodes),
                games: ext.games,
                elo: ext.elo,
                elo_lo: ext.elo_lo,
                elo_hi: ext.elo_hi,
                verdict: EXTERNAL_VERDICT.into(),
                ..Default::default()
            }),
            speed,
            status: Status::Accepted,
            created_unix: datadir::now_secs(),
            note: note.to_string(),
            ..Default::default()
        });
        r.history.retain(|&x| x != id);
        r.history.push(id);
        r.active = id;
        datadir::write_atomic(&dst, &imp.bytes).map_err(|e| format!("запись {}: {e}", dst.display()))?;
        let res = check_gen(root, cfg, r.get(id).expect("только что добавлено")).map(|_| ()).and_then(|_| after(&r, id)).and_then(|_| r.save(root, cfg));
        if let Err(e) = res {
            let _ = datadir::mark_stale(&dst);
            invalidate_cache();
            return Err(e);
        }
        Ok(id)
    }
}

pub const EXTERNAL_VERDICT: &str = "external";

#[derive(Clone, Debug, Default)]
pub struct ExternalMatch {
    pub elo: f64,
    pub elo_lo: f64,
    pub elo_hi: f64,
    pub games: u32,
    pub nodes: u64,
}

#[derive(Clone, Debug)]
pub struct ImportFile {
    pub bytes: Vec<u8>,
    pub eval_id: u32,
    pub arch_name: String,
    pub arch_summary: String,
    pub weights: u64,
}

pub fn check_import_file(cfg: &GameConfig, p: &Path) -> Result<ImportFile, String> {
    let m = load_for(cfg, p)?;
    if m.meta.is_none() {
        return Err(format!("{}: в файле нет метаданных (вариант, режим взятия, версия правил) — импорт запрещён", p.display()));
    }
    if m.arch.name.is_empty() {
        return Err(format!("{}: в файле нет имени архитектуры", p.display()));
    }
    let bytes = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(ImportFile { eval_id: m.eval_id(), arch_name: m.arch.name.clone(), arch_summary: m.arch.summary().unwrap_or_default(), weights: weights_count(&m), bytes })
}

fn max_gen_file(dir: &Path) -> u32 {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| {
                    let n = e.file_name().to_string_lossy().to_string();
                    n.strip_prefix("gen-")?.split('.').next()?.parse::<u32>().ok()
                })
                .max()
                .unwrap_or(0)
        })
        .unwrap_or(0)
}

fn check_gen(root: &Path, cfg: &GameConfig, g: &Generation) -> Result<Model, String> {
    let p = weights_dir(root, cfg).join(&g.file);
    let m = load_for(cfg, &p)?;
    if m.eval_id() != g.eval_id {
        return Err(format!("файл {} не совпадает с реестром (eval_id {:#010x}, записано {:#010x})", p.display(), m.eval_id(), g.eval_id));
    }
    Ok(m)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActiveInfo {
    pub generation: u32,
    pub eval_id: u32,
    pub fallback_reason: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Active {
    pub spec: EvalSpec,
    pub opts: String,
    pub info: ActiveInfo,
    pub wdl: Option<Wdl>,
}

type Stamp = Option<(SystemTime, u64)>;

fn stamp(p: &Path) -> Stamp {
    std::fs::metadata(p).ok().map(|m| (m.modified().unwrap_or(SystemTime::UNIX_EPOCH), m.len()))
}

struct Entry {
    checked: Instant,
    reg: Stamp,
    wfile: Option<PathBuf>,
    wst: Stamp,
    active: Arc<Active>,
}

type Cache = HashMap<(PathBuf, String), Entry>;

fn cache() -> &'static Mutex<Cache> {
    static C: OnceLock<Mutex<Cache>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn invalidate_cache() {
    cache().lock().unwrap_or_else(|e| e.into_inner()).clear();
}

fn simple_active(reason: Option<String>) -> Active {
    Active { spec: EvalSpec::Simple, opts: String::new(), info: ActiveInfo { generation: 0, eval_id: 0, fallback_reason: reason }, wdl: Some(Wdl::builtin()) }
}

fn disabled_by_env() -> bool {
    std::env::var("RUNNER_EVAL").map(|v| v.trim().eq_ignore_ascii_case("simple")).unwrap_or(false)
}

fn compute(root: &Path, cfg: &GameConfig) -> (Active, Option<PathBuf>) {
    let reg = Registry::load(root, cfg);
    let Some(g) = reg.active_generation() else {
        if reg.active != 0 {
            return (simple_active(Some(format!("в реестре нет действующего поколения {}", reg.active))), None);
        }
        return (simple_active(None), None);
    };
    let path = weights_dir(root, cfg).join(&g.file);
    match check_gen(root, cfg, g) {
        Ok(_) => (
            Active {
                spec: EvalSpec::File(path.clone()),
                opts: g.search_opts.clone(),
                info: ActiveInfo { generation: g.id, eval_id: g.eval_id, fallback_reason: None },
                wdl: g.wdl.clone().filter(|w| w.own),
            },
            Some(path),
        ),
        Err(e) => (simple_active(Some(format!("поколение {} не загрузилось, играет SimpleEval: {e}", g.id))), Some(path)),
    }
}

pub fn active_in(root: &Path, cfg: &GameConfig) -> Arc<Active> {
    if disabled_by_env() {
        return Arc::new(simple_active(Some("RUNNER_EVAL=simple: реестр отключён".into())));
    }
    let key = (root.to_path_buf(), datadir::cfg_name(cfg.variant.name(), cfg.forced_capture));
    let mut c = cache().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(e) = c.get_mut(&key) {
        if e.checked.elapsed() < RECHECK {
            return e.active.clone();
        }
        let rs = stamp(&registry_path(root, cfg));
        let ws = e.wfile.as_ref().map(|p| stamp(p)).unwrap_or(None);
        if rs == e.reg && ws == e.wst {
            e.checked = Instant::now();
            return e.active.clone();
        }
    }
    let rs = stamp(&registry_path(root, cfg));
    let (a, wfile) = compute(root, cfg);
    let wst = wfile.as_ref().map(|p| stamp(p)).unwrap_or(None);
    let a = Arc::new(a);
    c.insert(key, Entry { checked: Instant::now(), reg: rs, wfile, wst, active: a.clone() });
    a
}

pub fn active(cfg: &GameConfig) -> Arc<Active> {
    active_in(&datadir::data_root(), cfg)
}

pub fn active_info(cfg: &GameConfig) -> ActiveInfo {
    active(cfg).info.clone()
}

pub fn resolve_search(cfg: &GameConfig, opts: &str) -> Result<(EvalSpec, String), String> {
    let (spec, rest) = super::select::split_eval_key(opts)?;
    if let Some(s) = spec {
        return Ok((s, rest));
    }
    let a = active(cfg);
    let merged = match (a.opts.trim().is_empty(), rest.is_empty()) {
        (true, _) => rest,
        (false, true) => a.opts.clone(),
        (false, false) => format!("{},{}", a.opts, rest),
    };
    Ok((a.spec.clone(), merged))
}

pub fn headers_info(dir: &Path) -> (u64, Vec<u32>) {
    let mut ids = std::collections::BTreeSet::new();
    let mut nodes = 0;
    if let Ok(v) = super::dataset::list_chunks(dir) {
        for p in v {
            if let Ok(h) = crate::selfplay::format::read_header(&p) {
                ids.insert(h.eval_id);
                nodes = nodes.max(h.nodes);
            }
        }
    }
    (nodes, ids.into_iter().collect())
}

