
use super::hybrid::{HybridEval, HybridEvalInc};
use super::nnfast::NnFast;
use super::linear::{LinearEval, LinearEvalInc};
use super::refeval::RefEval;
use super::weights::Model;
use crate::eval::{Evaluator, SimpleEval};
use crate::facade::GameConfig;
use crate::rules::Rules;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum EvalSpec {
    #[default]
    Simple,
    File(PathBuf),
}

impl EvalSpec {
    pub fn parse(s: &str) -> Result<EvalSpec, String> {
        let s = s.trim();
        if s.is_empty() {
            return Err("пустое значение eval (нужно simple или путь к файлу весов)".into());
        }
        if s.eq_ignore_ascii_case("simple") {
            return Ok(EvalSpec::Simple);
        }
        if s.contains(',') {
            return Err(format!("путь к весам не должен содержать запятых: '{s}'"));
        }
        Ok(EvalSpec::File(PathBuf::from(s)))
    }

    pub fn key_value(&self) -> String {
        match self {
            EvalSpec::Simple => "simple".into(),
            EvalSpec::File(p) => p.display().to_string(),
        }
    }

    pub fn add_to_opts(&self, opts: &str) -> String {
        let eval = format!("eval={}", self.key_value());
        if opts.trim().is_empty() {
            eval
        } else {
            format!("{opts},{eval}")
        }
    }
}

pub fn split_eval_key(opts: &str) -> Result<(Option<EvalSpec>, String), String> {
    let mut spec = None;
    let mut rest: Vec<&str> = vec![];
    for kv in opts.split(',') {
        match kv.split_once('=') {
            Some((k, v)) if k.trim() == "eval" => spec = Some(EvalSpec::parse(v)?),
            _ => {
                if !kv.is_empty() {
                    rest.push(kv);
                }
            }
        }
    }
    Ok((spec, rest.join(",")))
}

pub fn default_eval_spec(cfg: &GameConfig) -> EvalSpec {
    super::registry::active(cfg).spec.clone()
}

#[derive(Clone)]
pub struct Loaded {
    pub model: Arc<Model>,
    pub eval_id: u32,
}

type CacheMap = HashMap<PathBuf, (Option<SystemTime>, u64, Loaded)>;

fn cache() -> &'static Mutex<CacheMap> {
    static C: OnceLock<Mutex<CacheMap>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn load_cached(path: &Path) -> Result<Loaded, String> {
    let md = std::fs::metadata(path).map_err(|e| format!("файл весов {}: {e}", path.display()))?;
    let (mt, len) = (md.modified().ok(), md.len());
    let mut c = cache().lock().unwrap_or_else(|e| e.into_inner());
    if let Some((m, l, loaded)) = c.get(path) {
        if *m == mt && *l == len {
            return Ok(loaded.clone());
        }
    }
    let model = Model::load_any(path).map_err(|e| format!("файл весов {}: {e}", path.display()))?;
    let loaded = Loaded { eval_id: model.eval_id(), model: Arc::new(model) };
    c.insert(path.to_path_buf(), (mt, len, loaded.clone()));
    Ok(loaded)
}

type PrepMap = HashMap<(u32, std::any::TypeId, u8), Arc<dyn std::any::Any + Send + Sync>>;

fn prep_cache() -> &'static Mutex<PrepMap> {
    static C: OnceLock<Mutex<PrepMap>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn hybrid_cached<R: Rules>(l: &Loaded) -> Result<Arc<HybridEval<R>>, String> {
    let key = (l.eval_id, std::any::TypeId::of::<R>(), 0u8);
    if let Some(a) = prep_cache().lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        if let Ok(h) = a.clone().downcast::<HybridEval<R>>() {
            return Ok(h);
        }
    }
    let h = Arc::new(HybridEval::<R>::from_arc(l.model.clone())?);
    let mut c = prep_cache().lock().unwrap_or_else(|e| e.into_inner());
    if c.len() >= 16 {
        c.clear();
    }
    c.insert(key, h.clone());
    Ok(h)
}

pub fn hybrid_inc_cached<R: Rules>(l: &Loaded) -> Result<Arc<HybridEvalInc<R>>, String> {
    let key = (l.eval_id, std::any::TypeId::of::<R>(), 1u8);
    if let Some(a) = prep_cache().lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        if let Ok(h) = a.clone().downcast::<HybridEvalInc<R>>() {
            return Ok(h);
        }
    }
    let h = Arc::new(HybridEvalInc::<R>::from_arc(l.model.clone())?);
    let mut c = prep_cache().lock().unwrap_or_else(|e| e.into_inner());
    if c.len() >= 16 {
        c.clear();
    }
    c.insert(key, h.clone());
    Ok(h)
}

pub fn nnfast_cached<R: Rules>(l: &Loaded) -> Result<Arc<NnFast<R>>, String> {
    let key = (l.eval_id, std::any::TypeId::of::<R>(), 3u8);
    if let Some(a) = prep_cache().lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        if let Ok(h) = a.clone().downcast::<NnFast<R>>() {
            return Ok(h);
        }
    }
    let h = Arc::new(NnFast::<R>::from_arc(l.model.clone())?);
    let mut c = prep_cache().lock().unwrap_or_else(|e| e.into_inner());
    if c.len() >= 16 {
        c.clear();
    }
    c.insert(key, h.clone());
    Ok(h)
}

pub fn old_path_requested(p: &Path) -> bool {
    match std::env::var_os("RUNNER_EVAL_OLD_PATHS") {
        Some(v) => std::env::split_paths(&v).any(|q| !q.as_os_str().is_empty() && q == p),
        None => false,
    }
}

pub fn linear_cached<R: Rules>(l: &Loaded) -> Result<Arc<LinearEval<R>>, String> {
    let key = (l.eval_id, std::any::TypeId::of::<R>(), 2u8);
    if let Some(a) = prep_cache().lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        if let Ok(h) = a.clone().downcast::<LinearEval<R>>() {
            return Ok(h);
        }
    }
    let h = Arc::new(LinearEval::<R>::from_arc(l.model.clone())?);
    let mut c = prep_cache().lock().unwrap_or_else(|e| e.into_inner());
    if c.len() >= 16 {
        c.clear();
    }
    c.insert(key, h.clone());
    Ok(h)
}

pub const LINEAR_INCREMENTAL: bool = false;

pub const HYBRID_INC_MIN_WIDTH: usize = 64;

pub trait EvalRun<R: Rules> {
    type Out;
    fn run<E: Evaluator<R>>(self, eval: &E, eval_id: u32) -> Self::Out;
}

pub fn with_eval<R: Rules, V: EvalRun<R>>(spec: &EvalSpec, v: V) -> Result<V::Out, String> {
    match spec {
        EvalSpec::Simple => Ok(v.run(&SimpleEval::new::<R>(), 0)),
        EvalSpec::File(p) => {
            let l = load_cached(p)?;
            let ctx = |e: String| format!("оценка eval={}: {e}", p.display());
            if l.model.arch.deep.is_none() {
                if LINEAR_INCREMENTAL {
                    let ev = LinearEvalInc::<R>::from_arc(l.model).map_err(ctx)?;
                    Ok(v.run(&ev, l.eval_id))
                } else {
                    let ev = linear_cached::<R>(&l).map_err(ctx)?;
                    Ok(v.run(&*ev, l.eval_id))
                }
            } else {
                if !old_path_requested(p) {
                    if let Ok(ev) = nnfast_cached::<R>(&l) {
                        return Ok(v.run(&*ev, l.eval_id));
                    }
                }
                let wide_acc = l.model.arch.deep.as_ref().map_or(false, |d| d.width >= HYBRID_INC_MIN_WIDTH);
                if wide_acc {
                    if let Ok(ev) = hybrid_inc_cached::<R>(&l) {
                        if ev.is_incremental() {
                            return Ok(v.run(&*ev, l.eval_id));
                        }
                    }
                }
                match hybrid_cached::<R>(&l) {
                    Ok(ev) => Ok(v.run(&*ev, l.eval_id)),
                    Err(_) => {
                        let ev = RefEval::<R>::from_arc(l.model).map_err(ctx)?;
                        Ok(v.run(&ev, l.eval_id))
                    }
                }
            }
        }
    }
}
