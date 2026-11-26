
use super::file::BLOCK;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

pub type Block = Arc<Vec<u8>>;

const SHARDS: usize = 64;
const TL_SLOTS: usize = 64;
const BLOCK_BITS: u32 = 40;
pub const DEFAULT_CACHE_MB: usize = 512;
pub const MAX_DEFAULT_CACHE_MB: usize = 2048;

pub fn default_mb_for(phys: Option<u64>) -> usize {
    match phys {
        Some(b) => ((b >> 20) / 8).clamp(DEFAULT_CACHE_MB as u64, MAX_DEFAULT_CACHE_MB as u64) as usize,
        None => DEFAULT_CACHE_MB,
    }
}

static CAP_BLOCKS: AtomicUsize = AtomicUsize::new(0);

fn blocks_for(mb: usize) -> usize {
    (mb.saturating_mul(1 << 20) / BLOCK).max(SHARDS)
}

pub fn set_cache_mb(mb: usize) {
    CAP_BLOCKS.store(blocks_for(mb), Relaxed);
}

pub fn cache_mb() -> usize {
    cap_blocks() * BLOCK >> 20
}

fn cap_blocks() -> usize {
    let c = CAP_BLOCKS.load(Relaxed);
    if c != 0 {
        return c;
    }
    let mb = std::env::var("RUNNER_EGDB_CACHE_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or_else(|| default_mb_for(crate::learn::memplan::physical_memory()));
    let c = blocks_for(mb);
    CAP_BLOCKS.store(c, Relaxed);
    c
}

#[cfg(unix)]
type FileId = (u64, u64, u64, i64, u32);
#[cfg(not(unix))]
type FileId = (std::path::PathBuf, u64, i64, u32);

fn file_id(f: &File, _path: &Path) -> Option<FileId> {
    let m = f.metadata().ok()?;
    let t = m.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Some((m.dev(), m.ino(), m.len(), t.as_secs() as i64, t.subsec_nanos()))
    }
    #[cfg(not(unix))]
    {
        Some((_path.to_path_buf(), m.len(), t.as_secs() as i64, t.subsec_nanos()))
    }
}

pub fn file_uid(f: &File, path: &Path) -> u32 {
    static NEXT: AtomicU32 = AtomicU32::new(1);
    static REG: OnceLock<Mutex<HashMap<FileId, u32>>> = OnceLock::new();
    let Some(id) = file_id(f, path) else {
        return NEXT.fetch_add(1, Relaxed);
    };
    let mut g = lock(REG.get_or_init(|| Mutex::new(HashMap::new())));
    *g.entry(id).or_insert_with(|| NEXT.fetch_add(1, Relaxed))
}

pub fn key(uid: u32, b: u64) -> u64 {
    debug_assert!(b < 1 << BLOCK_BITS);
    ((uid as u64) << BLOCK_BITS) | b
}

struct Entry {
    key: u64,
    blk: Block,
    used: bool,
}

#[derive(Default)]
struct Shard {
    map: HashMap<u64, usize>,
    ring: Vec<Entry>,
    hand: usize,
}

impl Shard {
    fn get(&mut self, key: u64) -> Option<Block> {
        let i = *self.map.get(&key)?;
        let e = &mut self.ring[i];
        e.used = true;
        Some(e.blk.clone())
    }

    fn insert(&mut self, key: u64, blk: Block, cap: usize) -> Block {
        if let Some(b) = self.get(key) {
            return b;
        }
        if self.ring.len() < cap {
            self.map.insert(key, self.ring.len());
            self.ring.push(Entry { key, blk: blk.clone(), used: true });
            return blk;
        }
        loop {
            let i = self.hand;
            self.hand = (self.hand + 1) % self.ring.len();
            let e = &mut self.ring[i];
            if e.used {
                e.used = false;
                continue;
            }
            self.map.remove(&e.key);
            *e = Entry { key, blk: blk.clone(), used: true };
            self.map.insert(key, i);
            return blk;
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn shards() -> &'static [Mutex<Shard>] {
    static S: OnceLock<Vec<Mutex<Shard>>> = OnceLock::new();
    S.get_or_init(|| (0..SHARDS).map(|_| Mutex::new(Shard::default())).collect())
}

fn mix(k: u64) -> u64 {
    (k ^ (k >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

thread_local! {
    static TL: RefCell<Vec<(u64, Option<Block>)>> = const { RefCell::new(Vec::new()) };
}

pub mod stats {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
    pub static ON: AtomicBool = AtomicBool::new(false);
    pub static C: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
    #[inline(always)]
    pub fn on() -> bool {
        ON.load(Relaxed)
    }
    #[inline(always)]
    pub fn bump(i: usize) {
        if on() {
            C[i].fetch_add(1, Relaxed);
        }
    }
    pub fn enable(v: bool) {
        ON.store(v, Relaxed);
    }
    pub fn take() -> [u64; 7] {
        std::array::from_fn(|i| C[i].swap(0, Relaxed))
    }
}

pub fn get_or_load(key: u64, load: impl FnOnce() -> Option<Vec<u8>>) -> Option<Block> {
    let h = mix(key);
    let ti = (h >> 32) as usize & (TL_SLOTS - 1);
    let hit = TL.try_with(|t| match t.borrow().get(ti) {
        Some((k, Some(b))) if *k == key => Some(b.clone()),
        _ => None,
    });
    if let Ok(Some(b)) = hit {
        stats::bump(4);
        return Some(b);
    }
    let sh = &shards()[(h >> 58) as usize % SHARDS];
    let found = lock(sh).get(key);
    let b = match found {
        Some(b) => {
            stats::bump(5);
            b
        }
        None => {
            stats::bump(6);
            let v = Arc::new(load()?);
            let cap = (cap_blocks() / SHARDS).max(1);
            lock(sh).insert(key, v, cap)
        }
    };
    let _ = TL.try_with(|t| {
        let mut t = t.borrow_mut();
        if t.is_empty() {
            t.resize(TL_SLOTS, (0, None));
        }
        t[ti] = (key, Some(b.clone()));
    });
    Some(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_mb_by_memory() {
        const G: u64 = 1 << 30;
        assert_eq!(default_mb_for(None), 512);
        for (gb, mb) in [(2, 512), (4, 512), (8, 1024), (16, 2048), (24, 2048), (64, 2048)] {
            assert_eq!(default_mb_for(Some(gb * G)), mb, "{gb} ГБ");
        }
        assert_eq!(default_mb_for(Some(0)), 512);
        assert_eq!(default_mb_for(Some(12 * G)), 1536);
    }

    #[test]
    fn clock_evicts_and_keeps_used() {
        let mut s = Shard::default();
        for k in 0..4u64 {
            s.insert(k, Arc::new(vec![k as u8]), 4);
        }
        s.insert(4, Arc::new(vec![4]), 4);
        assert_eq!(s.ring.len(), 4);
        assert!(s.get(0).is_none());
        for k in 1..5u64 {
            assert_eq!(s.get(k).unwrap()[0], k as u8);
        }
        let b = s.insert(3, Arc::new(vec![99]), 4);
        assert_eq!(b[0], 3);
    }
}
