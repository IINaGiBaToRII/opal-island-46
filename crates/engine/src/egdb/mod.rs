
pub mod api;
pub mod cache;
pub mod file;
pub mod gen;
pub mod index;
pub mod probe;
pub mod verify;

pub use api::*;
pub use verify::VerifyReport;
pub use gen::{GenCtl, GenOpts, GenReport, LayerPlan, SlicePlan};
pub use probe::Egdb;

use crate::datadir::{self, HeaderCheck};
use crate::position::{Position, Side};
use crate::rules::Rules;
use crate::search::{MATE, MATE_BOUND};
use index::{sig_name, Sig};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

pub fn slice_path(dir: &Path, sig: Sig) -> PathBuf {
    dir.join(format!("{}.egdb", sig_name(sig)))
}

pub fn fixed_value<R: Rules>(p: &Position<R>) -> Option<i8> {
    let wc = p.white.count_ones();
    let bc = p.black.count_ones();
    let own = if p.side == Side::White { wc } else { bc };
    if own == 0 {
        return Some(-1);
    }
    if R::ONE_V_ONE_DRAW && wc == 1 && bc == 1 {
        return Some(0);
    }
    None
}

pub const SAT: i8 = 127;
pub const SAT_SPREAD: i32 = 200;
pub const SAT_SCORE: i32 = MATE_BOUND - 1 - 300;
const _: () = assert!(SAT_SCORE + SAT_SPREAD < MATE_BOUND && SAT_SCORE - SAT_SPREAD >= 29000);

pub fn to_score(v: i8, ply: usize) -> i32 {
    let m = (v as i32).abs();
    if v == SAT || v == -SAT {
        SAT_SCORE * (v as i32).signum()
    } else if v > 0 {
        (MATE - ply as i32 - m).max(MATE_BOUND + 1)
    } else if v < 0 {
        (-MATE + ply as i32 + m - 1).min(-MATE_BOUND - 1)
    } else {
        0
    }
}

pub fn is_sat_score(s: i32) -> bool {
    (SAT_SCORE - SAT_SPREAD..=SAT_SCORE + SAT_SPREAD).contains(&s.abs())
}

pub fn sat_bonus<R: Rules>(p: &Position<R>) -> i32 {
    use crate::geometry::Geometry;
    use crate::movegen::{generate, ListSink, MoveList};
    let n = <R::Geo as Geometry>::N as i32;
    let side_terms = |own: u64, white: bool| -> i32 {
        let (men, kings) = (own & !p.kings, own & p.kings);
        let mut adv = 0;
        let mut x = men;
        while x != 0 {
            let (r, _) = <R::Geo as Geometry>::coords(x.trailing_zeros());
            adv += if white { r as i32 } else { n - 1 - r as i32 };
            x &= x - 1;
        }
        men.count_ones() as i32 * 30 + kings.count_ones() as i32 * 40 + adv * 4
    };
    let (own, opp, white) = if p.side == Side::White { (p.white, p.black, true) } else { (p.black, p.white, false) };
    let mobility = |q: &Position<R>| {
        let mut l = MoveList::new();
        generate(q, &mut ListSink::<true>(&mut l));
        l.len as i32
    };
    let other = Position::<R>::new(p.white, p.black, p.kings, if white { Side::Black } else { Side::White });
    let s = side_terms(own, white) - side_terms(opp, !white) + 4 * (mobility(p) - mobility(&other));
    s.clamp(-SAT_SPREAD, SAT_SPREAD)
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SliceStat {
    pub name: String,
    pub positions: u64,
    pub bytes: u64,
    pub secs: f64,
    pub max_dist: u32,
    pub wins: u64,
    pub losses: u64,
    pub draws: u64,
}

#[derive(Serialize, Deserialize, Debug)]
struct Manifest {
    format: u16,
    rules_version: u32,
    variant: String,
    forced: bool,
    total_bytes: u64,
    slices: Vec<SliceStat>,
}

pub fn quarantine_stale(dir: &Path) -> usize {
    let mut n = 0;
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().map(|x| x != "egdb").unwrap_or(true) {
            continue;
        }
        let bad = match datadir::read_prefix(&p, file::HEADER_LEN) {
            Err(_) => true,
            Ok(b) => match file::decode_header(&b) {
                Err(HeaderCheck::Ok) => false,
                Err(_) => true,
                Ok(h) => !file::validate_len(&h, e.metadata().map(|m| m.len()).unwrap_or(0)),
            },
        };
        if bad && datadir::mark_stale(&p).is_ok() {
            n += 1;
        }
    }
    n
}

pub fn update_manifest(dir: &Path, variant: &str, forced: bool, new: Option<SliceStat>) {
    let mpath = dir.join("manifest.json");
    let mut m: Manifest = std::fs::read_to_string(&mpath)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .filter(|m: &Manifest| m.rules_version == datadir::RULES_VERSION)
        .unwrap_or(Manifest { format: file::FORMAT as u16, rules_version: datadir::RULES_VERSION, variant: variant.into(), forced, total_bytes: 0, slices: vec![] });
    if let Some(s) = new {
        m.slices.retain(|x| x.name != s.name);
        m.slices.push(s);
    }
    let mut present: HashMap<String, u64> = HashMap::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().map(|x| x == "egdb").unwrap_or(false) {
                if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                    present.insert(stem.to_string(), e.metadata().map(|m| m.len()).unwrap_or(0));
                }
            }
        }
    }
    m.slices.retain(|x| present.contains_key(&x.name));
    for (name, len) in &present {
        if !m.slices.iter().any(|x| &x.name == name) {
            m.slices.push(SliceStat { name: name.clone(), positions: 0, bytes: *len, secs: 0.0, max_dist: 0, wins: 0, losses: 0, draws: 0 });
        }
    }
    m.slices.sort_by(|a, b| a.name.cmp(&b.name));
    m.total_bytes = present.values().sum();
    if let Ok(s) = serde_json::to_string_pretty(&m) {
        let tmp = dir.join("manifest.json.tmp");
        if std::fs::write(&tmp, s).is_ok() {
            let _ = std::fs::rename(&tmp, &mpath);
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct EgdbInfo {
    pub max_pieces: u32,
    pub slices: u32,
    pub bytes: u64,
}

type SharedMap = Mutex<HashMap<(&'static str, bool), (Instant, Option<Arc<dyn Any + Send + Sync>>)>>;

pub fn shared<R: Rules>() -> Option<Arc<Egdb<R>>> {
    if std::env::var("RUNNER_EGDB").map(|v| v == "off" || v == "0").unwrap_or(false) {
        return None;
    }
    static CACHE: OnceLock<SharedMap> = OnceLock::new();
    let map = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = map.lock().ok()?;
    let key = (R::NAME, R::FORCED);
    if let Some((t, v)) = g.get(&key) {
        if t.elapsed() < Duration::from_secs(30) {
            return v.clone().and_then(|a| a.downcast::<Egdb<R>>().ok());
        }
    }
    let db = Egdb::<R>::open().ok().filter(|d| d.max_pieces() > 0).map(Arc::new);
    g.insert(key, (Instant::now(), db.clone().map(|a| a as Arc<dyn Any + Send + Sync>)));
    db
}
