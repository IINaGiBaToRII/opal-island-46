
use crate::learn::ViewTables;
use crate::position::Position;
use crate::rules::Rules;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecallEntry {
    pub score: i32,
    pub depth: u8,
    pub bound: u8,
    pub mv: Option<(u8, u8)>,
    pub nodes: u64,
    pub eval_id: u32,
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    white: u64,
    black: u64,
    kings: u64,
    side: u8,
    e: RecallEntry,
}

#[derive(Default, Debug)]
pub struct RecallStats {
    pub searches: AtomicU64,
    pub searches_hit: AtomicU64,
    pub root_hits: AtomicU64,
    pub placed: AtomicU64,
    pub probed: AtomicU64,
    pub used_pv: AtomicU64,
    pub used_cut: AtomicU64,
    pub searches_used: AtomicU64,
    pub recorded: AtomicU64,
    pub rejected: AtomicU64,
    pub rej_hist: AtomicU64,
    pub rej_incomplete: AtomicU64,
    pub rej_early: AtomicU64,
    pub rej_partial: AtomicU64,
    pub kept_deeper: AtomicU64,
    pub mirror_placed: AtomicU64,
    pub rej_clock: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RecallSnapshot {
    pub searches: u64,
    pub searches_hit: u64,
    pub root_hits: u64,
    pub placed: u64,
    pub probed: u64,
    pub used_pv: u64,
    pub used_cut: u64,
    pub searches_used: u64,
    pub recorded: u64,
    pub rejected: u64,
    pub rej_hist: u64,
    pub rej_incomplete: u64,
    pub rej_early: u64,
    pub rej_partial: u64,
    pub kept_deeper: u64,
    pub mirror_placed: u64,
    pub rej_clock: u64,
    pub entries: u64,
}

impl RecallSnapshot {
    pub fn since(&self, b: &RecallSnapshot) -> RecallSnapshot {
        RecallSnapshot {
            searches: self.searches - b.searches,
            searches_hit: self.searches_hit - b.searches_hit,
            root_hits: self.root_hits - b.root_hits,
            placed: self.placed - b.placed,
            probed: self.probed - b.probed,
            used_pv: self.used_pv - b.used_pv,
            used_cut: self.used_cut - b.used_cut,
            searches_used: self.searches_used - b.searches_used,
            recorded: self.recorded - b.recorded,
            rejected: self.rejected - b.rejected,
            rej_hist: self.rej_hist - b.rej_hist,
            rej_incomplete: self.rej_incomplete - b.rej_incomplete,
            rej_early: self.rej_early - b.rej_early,
            rej_partial: self.rej_partial - b.rej_partial,
            kept_deeper: self.kept_deeper - b.kept_deeper,
            mirror_placed: self.mirror_placed - b.mirror_placed,
            rej_clock: self.rej_clock - b.rej_clock,
            entries: self.entries,
        }
    }
    pub fn line(&self) -> String {
        let pct = |a: u64| if self.searches > 0 { 100.0 * a as f64 / self.searches as f64 } else { 0.0 };
        format!(
            "recall: entries {}  searches {}  with_hit {} ({:.1}%)  root_hit {} ({:.1}%)  used {} ({:.1}%)  placed {}  probed {}  used_pv {}  used_cut {}  recorded {}  rejected {} (hist {}  incomplete {}  early {}  partial {})  kept_deeper {}  mirror_placed {}  rej_clock {}",
            self.entries,
            self.searches,
            self.searches_hit,
            pct(self.searches_hit),
            self.root_hits,
            pct(self.root_hits),
            self.searches_used,
            pct(self.searches_used),
            self.placed,
            self.probed,
            self.used_pv,
            self.used_cut,
            self.recorded,
            self.rejected,
            self.rej_hist,
            self.rej_incomplete,
            self.rej_early,
            self.rej_partial,
            self.kept_deeper,
            self.mirror_placed,
            self.rej_clock
        )
    }
}

const SHARDS: usize = 16;
const MAGIC: &[u8; 4] = b"DRCL";
const VERSION: u32 = 1;
const REC_BYTES: usize = 8 * 5 + 1 + 4 + 1 + 1 + 1 + 1 + 1 + 4;

pub struct Recall {
    shards: Vec<Mutex<HashMap<u64, Slot>>>,
    cap: usize,
    len: AtomicUsize,
    evict: Mutex<()>,
    pub stats: RecallStats,
}

impl Recall {
    pub fn new(cap: usize) -> Recall {
        Recall {
            shards: (0..SHARDS).map(|_| Mutex::new(HashMap::new())).collect(),
            cap: cap.max(16),
            len: AtomicUsize::new(0),
            evict: Mutex::new(()),
            stats: RecallStats::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.len.load(Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    #[inline]
    fn shard(&self, key: u64) -> &Mutex<HashMap<u64, Slot>> {
        &self.shards[(key >> 59) as usize % SHARDS]
    }

    pub fn get<R: Rules>(&self, pos: &Position<R>, eval_id: u32) -> Option<RecallEntry> {
        let g = self.shard(pos.hash).lock().unwrap();
        let s = g.get(&pos.hash)?;
        let same = s.white == pos.white && s.black == pos.black && s.kings == pos.kings && s.side == pos.side as u8;
        (same && s.e.eval_id == eval_id).then_some(s.e)
    }

    pub fn put<R: Rules>(&self, pos: &Position<R>, e: RecallEntry) -> bool {
        self.put_raw(pos.hash, pos.white, pos.black, pos.kings, pos.side as u8, e)
    }

    fn put_raw(&self, key: u64, white: u64, black: u64, kings: u64, side: u8, e: RecallEntry) -> bool {
        let slot = Slot { white, black, kings, side, e };
        let grown = {
            let mut g = self.shard(key).lock().unwrap();
            match g.get_mut(&key) {
                Some(old) => {
                    let same = old.white == white && old.black == black && old.kings == kings && old.side == side && old.e.eval_id == e.eval_id;
                    if same && old.e.depth > e.depth {
                        return false;
                    }
                    *old = slot;
                    false
                }
                None => {
                    g.insert(key, slot);
                    true
                }
            }
        };
        if grown && self.len.fetch_add(1, Relaxed) + 1 > self.cap {
            self.evict_shallow();
        }
        true
    }

    fn evict_shallow(&self) {
        let Ok(_g) = self.evict.try_lock() else { return };
        let len = self.len();
        if len <= self.cap {
            return;
        }
        let target = self.cap * 9 / 10;
        let mut need = len.saturating_sub(target);
        let mut hist = [0usize; 256];
        for s in &self.shards {
            for v in s.lock().unwrap().values() {
                hist[v.e.depth as usize] += 1;
            }
        }
        let mut t = 0;
        let mut below = 0;
        while t < 255 && below + hist[t] < need {
            below += hist[t];
            t += 1;
        }
        let mut at_t = need - below;
        for s in &self.shards {
            let mut g = s.lock().unwrap();
            let before = g.len();
            g.retain(|_, v| {
                let d = v.e.depth as usize;
                if d < t {
                    false
                } else if d == t && at_t > 0 {
                    at_t -= 1;
                    false
                } else {
                    true
                }
            });
            let gone = before - g.len();
            need = need.saturating_sub(gone);
            self.len.fetch_sub(gone, Relaxed);
        }
    }

    pub fn snapshot(&self) -> RecallSnapshot {
        let s = &self.stats;
        RecallSnapshot {
            searches: s.searches.load(Relaxed),
            searches_hit: s.searches_hit.load(Relaxed),
            root_hits: s.root_hits.load(Relaxed),
            placed: s.placed.load(Relaxed),
            probed: s.probed.load(Relaxed),
            used_pv: s.used_pv.load(Relaxed),
            used_cut: s.used_cut.load(Relaxed),
            searches_used: s.searches_used.load(Relaxed),
            recorded: s.recorded.load(Relaxed),
            rejected: s.rejected.load(Relaxed),
            rej_hist: s.rej_hist.load(Relaxed),
            rej_incomplete: s.rej_incomplete.load(Relaxed),
            rej_early: s.rej_early.load(Relaxed),
            rej_partial: s.rej_partial.load(Relaxed),
            kept_deeper: s.kept_deeper.load(Relaxed),
            mirror_placed: s.mirror_placed.load(Relaxed),
            rej_clock: s.rej_clock.load(Relaxed),
            entries: self.len() as u64,
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let mut recs: Vec<(u64, Slot)> = Vec::with_capacity(self.len());
        for s in &self.shards {
            recs.extend(s.lock().unwrap().iter().map(|(k, v)| (*k, *v)));
        }
        recs.sort_by_key(|r| r.0);
        let mut b = Vec::with_capacity(16 + recs.len() * REC_BYTES);
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&VERSION.to_le_bytes());
        b.extend_from_slice(&(recs.len() as u64).to_le_bytes());
        for (k, s) in &recs {
            for x in [*k, s.white, s.black, s.kings, s.e.nodes] {
                b.extend_from_slice(&x.to_le_bytes());
            }
            b.push(s.side);
            b.extend_from_slice(&s.e.score.to_le_bytes());
            b.push(s.e.depth);
            b.push(s.e.bound);
            let (has, f, t) = match s.e.mv {
                Some((f, t)) => (1u8, f, t),
                None => (0, 0, 0),
            };
            b.extend_from_slice(&[has, f, t]);
            b.extend_from_slice(&s.e.eval_id.to_le_bytes());
        }
        crate::datadir::write_atomic(path, &b).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn load(&self, path: &Path) -> Result<usize, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let bad = |w: &str| format!("{}: {w}", path.display());
        if b.len() < 16 || &b[..4] != MAGIC {
            return Err(bad("не файл памяти расчётов"));
        }
        let ver = u32::from_le_bytes(b[4..8].try_into().unwrap());
        if ver != VERSION {
            return Err(bad(&format!("версия {ver}, ожидалась {VERSION}")));
        }
        let n = u64::from_le_bytes(b[8..16].try_into().unwrap()) as usize;
        if b.len() != 16 + n * REC_BYTES {
            return Err(bad("размер не сходится"));
        }
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        for i in 0..n {
            let o = 16 + i * REC_BYTES;
            let (k, white, black, kings, nodes) = (u64_at(o), u64_at(o + 8), u64_at(o + 16), u64_at(o + 24), u64_at(o + 32));
            let side = b[o + 40];
            let score = i32::from_le_bytes(b[o + 41..o + 45].try_into().unwrap());
            let (depth, bound, has, f, t) = (b[o + 45], b[o + 46], b[o + 47], b[o + 48], b[o + 49]);
            let eval_id = u32::from_le_bytes(b[o + 50..o + 54].try_into().unwrap());
            if side > 1 || bound > 2 {
                return Err(bad(&format!("запись {i} повреждена")));
            }
            let e = RecallEntry { score, depth, bound, mv: (has != 0).then_some((f, t)), nodes, eval_id };
            self.put_raw(k, white, black, kings, side, e);
        }
        Ok(n)
    }
}

#[derive(Clone, Copy)]
pub struct RecallCtx<'a> {
    pub mem: &'a Recall,
    pub eval_id: u32,
    pub plies: u32,
    pub max_positions: usize,
    pub record: bool,
    pub mirror: bool,
}

pub fn color_tables<R: Rules>() -> Option<&'static ViewTables> {
    static M: std::sync::OnceLock<Mutex<HashMap<&'static str, Option<&'static ViewTables>>>> = std::sync::OnceLock::new();
    let m = M.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = m.lock().unwrap();
    *g.entry(std::any::type_name::<R>()).or_insert_with(|| {
        let t = ViewTables::new::<R>();
        t.check_rotation().is_ok().then(|| &*Box::leak(Box::new(t)))
    })
}

pub fn color_mirror<R: Rules>(pos: &Position<R>) -> Option<Position<R>> {
    color_tables::<R>().map(|t| t.flip(pos))
}

pub fn mirror_move(t: &ViewTables, mv: (u8, u8)) -> Option<(u8, u8)> {
    let (f, to) = (t.rot[mv.0 as usize & 63], t.rot[mv.1 as usize & 63]);
    (f != 255 && to != 255).then_some((f, to))
}

impl Recall {
    pub fn get_or_mirror<R: Rules>(&self, pos: &Position<R>, eval_id: u32, mirror: Option<&ViewTables>) -> Option<(RecallEntry, bool)> {
        if let Some(e) = self.get(pos, eval_id) {
            return Some((e, false));
        }
        let t = mirror?;
        let e = self.get(&t.flip(pos), eval_id)?;
        Some((RecallEntry { mv: e.mv.and_then(|m| mirror_move(t, m)), ..e }, true))
    }
}
