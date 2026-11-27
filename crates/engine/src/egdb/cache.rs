
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
pub const MAX_TL_SLOTS: usize = 1 << 16;
const BLOCK_BITS: u32 = 40;
pub const DEFAULT_CACHE_MB: usize = 512;
pub const MAX_DEFAULT_CACHE_MB: usize = 2048;

pub fn default_mb_for(phys: Option<u64>) -> usize {
    match phys {
        Some(b) => ((b >> 20) / 8).clamp(DEFAULT_CACHE_MB as u64, MAX_DEFAULT_CACHE_MB as u64) as usize,
        None => DEFAULT_CACHE_MB,
    }
}

static TL_N: AtomicUsize = AtomicUsize::new(0);

pub fn set_tl_slots(n: usize) {
    TL_N.store(n.clamp(1, MAX_TL_SLOTS).next_power_of_two(), Relaxed);
}

#[inline(always)]
pub fn tl_slots() -> usize {
    let n = TL_N.load(Relaxed);
    if n != 0 {
        return n;
    }
    let n = std::env::var("RUNNER_EGDB_TL_SLOTS").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(TL_SLOTS).clamp(1, MAX_TL_SLOTS).next_power_of_two();
    TL_N.store(n, Relaxed);
    n
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
    charge: usize,
}

#[derive(Default)]
struct Shard {
    map: HashMap<u64, usize>,
    ring: Vec<Entry>,
    hand: usize,
    bytes: usize,
}

impl Shard {
    fn get(&mut self, key: u64) -> Option<Block> {
        let i = *self.map.get(&key)?;
        let e = &mut self.ring[i];
        e.used = true;
        Some(e.blk.clone())
    }

    fn insert(&mut self, key: u64, blk: Block, cap: usize, charge: usize) -> Block {
        if let Some(b) = self.get(key) {
            return b;
        }
        if self.ring.is_empty() || self.bytes + charge <= cap {
            self.map.insert(key, self.ring.len());
            self.ring.push(Entry { key, blk: blk.clone(), used: true, charge });
            self.bytes += charge;
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
            self.bytes = self.bytes - e.charge + charge;
            *e = Entry { key, blk: blk.clone(), used: true, charge };
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
    pub const LAYERS: usize = 17;
    pub const KINDS: usize = 9;
    pub const K_CALL: usize = 0;
    pub const K_PROBE: usize = 1;
    pub const K_FIXED: usize = 2;
    pub const K_NONE: usize = 3;
    pub const K_TL: usize = 4;
    pub const K_SHARED: usize = 5;
    pub const K_ZSTD: usize = 6;
    pub const K_RAM: usize = 7;
    pub const K_ZSTD_NS: usize = 8;
    pub static L: [[AtomicU64; KINDS]; LAYERS] = [const { [const { AtomicU64::new(0) }; KINDS] }; LAYERS];
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
    #[inline(always)]
    pub fn add_layer(layer: u32, k: usize, v: u64) {
        L[(layer as usize).min(LAYERS - 1)][k].fetch_add(v, Relaxed);
    }
    pub fn enable(v: bool) {
        ON.store(v, Relaxed);
    }
    pub fn take() -> [u64; 7] {
        std::array::from_fn(|i| C[i].swap(0, Relaxed))
    }
    pub fn take_layers() -> [[u64; KINDS]; LAYERS] {
        std::array::from_fn(|l| std::array::from_fn(|k| L[l][k].swap(0, Relaxed)))
    }
    pub fn format_layers(l: &[[u64; KINDS]; LAYERS], nodes: u64, tag: &str) -> String {
        let nn = nodes.max(1) as f64;
        let mut s = String::new();
        for (t, c) in l.iter().enumerate() {
            if c.iter().all(|&x| x == 0) {
                continue;
            }
            let blk = (c[K_TL] + c[K_SHARED] + c[K_ZSTD] + c[K_RAM]).max(1) as f64;
            s += &format!(
                "STATS-L {tag}{t} вызовов {} проб {} ({:.4} на узел) fixed {} нет {} поточный {} ({:.2}%) общий {} ({:.2}%) zstd {} ({:.3}%; {:.1} мс, {:.1} мкс/распаковку) ОЗУ {} ({:.2}%)\n",
                c[K_CALL],
                c[K_PROBE],
                c[K_PROBE] as f64 / nn,
                c[K_FIXED],
                c[K_NONE],
                c[K_TL],
                100.0 * c[K_TL] as f64 / blk,
                c[K_SHARED],
                100.0 * c[K_SHARED] as f64 / blk,
                c[K_ZSTD],
                100.0 * c[K_ZSTD] as f64 / blk,
                c[K_ZSTD_NS] as f64 / 1e6,
                c[K_ZSTD_NS] as f64 / 1e3 / c[K_ZSTD].max(1) as f64,
                c[K_RAM],
                100.0 * c[K_RAM] as f64 / blk,
            );
        }
        s
    }
}

pub mod ram {
    use super::{lock, Arc, HashMap, Mutex, OnceLock};
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering::Relaxed};

    pub const RAM_DEFAULT_MB: usize = 1024;
    static SET: AtomicBool = AtomicBool::new(false);
    static PIECES: AtomicU32 = AtomicU32::new(0);
    static MB: AtomicUsize = AtomicUsize::new(RAM_DEFAULT_MB);

    pub fn set_ram(pieces: u32, mb: usize) {
        PIECES.store(pieces, Relaxed);
        MB.store(mb, Relaxed);
        SET.store(true, Relaxed);
    }

    pub fn opts() -> (u32, usize) {
        if SET.load(Relaxed) {
            return (PIECES.load(Relaxed), MB.load(Relaxed));
        }
        let num = |k: &str| std::env::var(k).ok().and_then(|v| v.trim().parse::<usize>().ok());
        (num("RUNNER_EGDB_RAM").unwrap_or(0) as u32, num("RUNNER_EGDB_RAM_MB").unwrap_or(RAM_DEFAULT_MB))
    }

    pub fn get_or_build(uid: u32, build: impl FnOnce() -> Option<Vec<u8>>) -> Option<Arc<Vec<u8>>> {
        static REG: OnceLock<Mutex<HashMap<u32, Arc<Vec<u8>>>>> = OnceLock::new();
        let mut g = lock(REG.get_or_init(|| Mutex::new(HashMap::new())));
        if let Some(v) = g.get(&uid) {
            return Some(v.clone());
        }
        let v = Arc::new(build()?);
        g.insert(uid, v.clone());
        Some(v)
    }
}

pub fn get_or_load(key: u64, layer: u32, charge: usize, load: impl FnOnce() -> Option<Vec<u8>>) -> Option<Block> {
    let h = mix(key);
    let slots = tl_slots();
    let ti = (h >> 32) as usize & (slots - 1);
    let hit = TL.try_with(|t| match t.borrow().get(ti) {
        Some((k, Some(b))) if *k == key => Some(b.clone()),
        _ => None,
    });
    if let Ok(Some(b)) = hit {
        if stats::on() {
            stats::C[4].fetch_add(1, Relaxed);
            stats::add_layer(layer, stats::K_TL, 1);
        }
        return Some(b);
    }
    let sh = &shards()[(h >> 58) as usize % SHARDS];
    let found = lock(sh).get(key);
    let b = match found {
        Some(b) => {
            if stats::on() {
                stats::C[5].fetch_add(1, Relaxed);
                stats::add_layer(layer, stats::K_SHARED, 1);
            }
            b
        }
        None => {
            let v = if stats::on() {
                stats::C[6].fetch_add(1, Relaxed);
                stats::add_layer(layer, stats::K_ZSTD, 1);
                let t0 = std::time::Instant::now();
                let v = load();
                stats::add_layer(layer, stats::K_ZSTD_NS, t0.elapsed().as_nanos() as u64);
                v
            } else {
                load()
            };
            let v = Arc::new(v?);
            let cap = (cap_blocks() / SHARDS).max(1) * BLOCK;
            lock(sh).insert(key, v, cap, charge)
        }
    };
    let _ = TL.try_with(|t| {
        let mut t = t.borrow_mut();
        if t.len() != slots {
            t.clear();
            t.resize(slots, (0, None));
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
            s.insert(k, Arc::new(vec![k as u8]), 4, 1);
        }
        s.insert(4, Arc::new(vec![4]), 4, 1);
        assert_eq!(s.ring.len(), 4);
        assert!(s.get(0).is_none());
        for k in 1..5u64 {
            assert_eq!(s.get(k).unwrap()[0], k as u8);
        }
        let b = s.insert(3, Arc::new(vec![99]), 4, 1);
        assert_eq!(b[0], 3);
    }

    #[test]
    fn byte_capacity_small_blocks() {
        let mut s = Shard::default();
        for k in 0..100u64 {
            s.insert(k, Arc::new(vec![0]), 4 * BLOCK, BLOCK / 16);
        }
        assert_eq!(s.ring.len(), 64);
        assert_eq!(s.bytes, 4 * BLOCK);
        let mut s = Shard::default();
        for k in 0..10u64 {
            s.insert(k, Arc::new(vec![0]), 4 * BLOCK, BLOCK);
        }
        assert_eq!(s.ring.len(), 4);
    }}
