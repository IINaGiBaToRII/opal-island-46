
use std::alloc::{alloc_zeroed, handle_alloc_error, Layout};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};

pub const BOUND_EXACT: u8 = 0;
pub const BOUND_LOWER: u8 = 1;
pub const BOUND_UPPER: u8 = 2;

#[derive(Clone, Copy, Debug)]
pub struct TtEntry {
    pub score: i32,
    pub depth: i32,
    pub bound: u8,
    pub mv: Option<(u8, u8)>,
    pub hist: bool,
    pub recall: bool,
    pub eval: Option<i32>,
}

#[repr(align(64))]
struct Bucket([AtomicU64; 8]);

pub struct Tt {
    buckets: Box<[Bucket]>,
    age: AtomicU8,
}

const VALID_BIT: u64 = 1 << 63;
const HIST_BIT: u64 = 1 << 49;
const RECALL_BIT: u64 = 1 << 50;
const EVAL_SHIFT: u32 = 51;
const EVAL_MASK: u64 = 0xFFF << EVAL_SHIFT;
const EVAL_BIAS: i32 = 2048;
pub const EVAL_MAX: i32 = 2047;

#[inline(always)]
fn eval_bits(eval: Option<i32>) -> u64 {
    match eval {
        Some(v) if (-EVAL_MAX..=EVAL_MAX).contains(&v) => ((v + EVAL_BIAS) as u64) << EVAL_SHIFT,
        _ => 0,
    }
}

#[inline(always)]
fn unpack(w: u64) -> TtEntry {
    let has_mv = (w >> 48) & 1 != 0;
    TtEntry {
        score: (w & 0xFFFF) as u16 as i16 as i32,
        depth: ((w >> 16) & 0xFF) as i32,
        bound: ((w >> 24) & 3) as u8,
        mv: if has_mv { Some((((w >> 32) & 0xFF) as u8, ((w >> 40) & 0xFF) as u8)) } else { None },
        hist: w & HIST_BIT != 0,
        recall: w & RECALL_BIT != 0,
        eval: match (w & EVAL_MASK) >> EVAL_SHIFT {
            0 => None,
            f => Some(f as i32 - EVAL_BIAS),
        },
    }
}

impl Tt {
    pub fn new(mb: usize) -> Tt {
        let n = (mb.max(1) * 1024 * 1024 / 64).max(16);
        let layout = Layout::array::<Bucket>(n).expect("tt layout");
        let buckets = unsafe {
            let p = alloc_zeroed(layout) as *mut Bucket;
            if p.is_null() {
                handle_alloc_error(layout);
            }
            Box::from_raw(std::slice::from_raw_parts_mut(p, n))
        };
        Tt { buckets, age: AtomicU8::new(1) }
    }

    pub fn new_search(&self) {
        let mut a = (self.age.load(Relaxed) + 1) & 0x3F;
        if a == 0 {
            a = 1;
        }
        self.age.store(a, Relaxed);
    }

    #[inline(always)]
    fn idx(&self, key: u64) -> usize {
        ((key as u128 * self.buckets.len() as u128) >> 64) as usize
    }

    #[inline(always)]
    pub fn prefetch(&self, key: u64) {
        let p = &self.buckets[self.idx(key)] as *const Bucket;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, preserves_flags, readonly));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = p;
    }

    #[inline(always)]
    pub fn probe(&self, key: u64) -> Option<TtEntry> {
        let b = &self.buckets[self.idx(key)].0;
        for j in 0..4 {
            let d = b[2 * j + 1].load(Relaxed);
            if d & VALID_BIT != 0 && b[2 * j].load(Relaxed) ^ d == key {
                return Some(unpack(d));
            }
        }
        None
    }

    #[inline]
    pub fn store(&self, key: u64, score: i32, depth: i32, bound: u8, mv: Option<(u8, u8)>, hist: bool) {
        self.store_flags(key, score, depth, bound, mv, if hist { HIST_BIT } else { 0 });
    }

    #[inline]
    pub fn store_eval(&self, key: u64, score: i32, depth: i32, bound: u8, mv: Option<(u8, u8)>, hist: bool, eval: Option<i32>) {
        self.store_flags(key, score, depth, bound, mv, if hist { HIST_BIT } else { 0 } | eval_bits(eval));
    }

    pub fn store_recall(&self, key: u64, score: i32, depth: i32, bound: u8, mv: Option<(u8, u8)>) -> bool {
        if let Some(e) = self.probe(key) {
            if e.bound == BOUND_EXACT && e.depth >= depth {
                return false;
            }
        }
        self.store_flags(key, score, depth, bound, mv, RECALL_BIT);
        true
    }

    #[inline(always)]
    fn store_flags(&self, key: u64, score: i32, depth: i32, bound: u8, mv: Option<(u8, u8)>, flags: u64) {
        let b = &self.buckets[self.idx(key)].0;
        let age = self.age.load(Relaxed) as u64;
        let mut victim = 0;
        let mut worst = i32::MAX;
        let mut old_mv = None;
        let mut old_eval = 0u64;
        let mut same = false;
        for j in 0..4 {
            let d = b[2 * j + 1].load(Relaxed);
            if d & VALID_BIT == 0 {
                if worst > i32::MIN {
                    worst = i32::MIN;
                    victim = j;
                }
                continue;
            }
            let k = b[2 * j].load(Relaxed) ^ d;
            if k == key {
                let e = unpack(d);
                let e_age = (d >> 26) & 0x3F;
                if e_age == age && bound != BOUND_EXACT && e.depth > depth + 3 {
                    return;
                }
                old_mv = e.mv;
                old_eval = d & EVAL_MASK;
                victim = j;
                same = true;
                break;
            }
            let e_age = (d >> 26) & 0x3F;
            let dist = ((age + 64 - e_age) & 0x3F) as i32;
            let val = ((d >> 16) & 0xFF) as i32 - 8 * dist;
            if val < worst {
                worst = val;
                victim = j;
            }
        }
        let _ = same;
        let mv = mv.or(old_mv);
        let mut w = (score.clamp(-32000, 32000) as i16 as u16 as u64)
            | ((depth.clamp(0, 255) as u64) << 16)
            | ((bound as u64) << 24)
            | (age << 26)
            | VALID_BIT;
        if let Some((f, t)) = mv {
            w |= ((f as u64) << 32) | ((t as u64) << 40) | (1 << 48);
        }
        if flags & EVAL_MASK == 0 {
            w |= old_eval;
        }
        w |= flags;
        b[2 * victim].store(key ^ w, Relaxed);
        b[2 * victim + 1].store(w, Relaxed);
    }

    pub fn hashfull(&self) -> u32 {
        let age = self.age.load(Relaxed) as u64;
        let n = self.buckets.len().min(250);
        let mut c = 0;
        for bk in &self.buckets[..n] {
            for j in 0..4 {
                let d = bk.0[2 * j + 1].load(Relaxed);
                if d & VALID_BIT != 0 && (d >> 26) & 0x3F == age {
                    c += 1;
                }
            }
        }
        (c * 1000 / (n * 4)) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hist_bit_roundtrip() {
        let tt = Tt::new(1);
        for (i, &(score, depth, bound, mv)) in
            [(-31999, 0, BOUND_EXACT, None), (123, 17, BOUND_LOWER, Some((5u8, 63u8))), (32000, 255, BOUND_UPPER, Some((0, 1))), (-1, 1, BOUND_EXACT, Some((63, 0)))].iter().enumerate()
        {
            for hist in [false, true] {
                let key = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(2 * i as u64 + hist as u64 + 1);
                tt.store(key, score, depth, bound, mv, hist);
                let e = tt.probe(key).expect("запись есть");
                assert_eq!((e.score, e.depth, e.bound, e.mv, e.hist), (score, depth, bound, mv, hist));
            }
        }
        let key = 0xABCD_EF01_2345_6789u64;
        tt.store(key, 50, 10, BOUND_EXACT, Some((3, 7)), true);
        tt.store(key, 60, 12, BOUND_EXACT, None, false);
        let e = tt.probe(key).unwrap();
        assert_eq!((e.score, e.depth, e.bound, e.mv, e.hist), (60, 12, BOUND_EXACT, Some((3, 7)), false));
    }

    #[test]
    fn eval_field_roundtrip() {
        let tt = Tt::new(1);
        for (i, ev) in [Some(0), Some(-1), Some(1), Some(EVAL_MAX), Some(-EVAL_MAX), Some(EVAL_MAX + 1), Some(-30000), None].into_iter().enumerate() {
            let key = 0x1234_5678_9ABC_DEF1u64.wrapping_mul(2 * i as u64 + 3);
            tt.store_eval(key, -31999, 255, BOUND_UPPER, Some((63, 62)), true, ev);
            let e = tt.probe(key).unwrap();
            let want = ev.filter(|v| v.abs() <= EVAL_MAX);
            assert_eq!((e.score, e.depth, e.bound, e.mv, e.hist, e.recall, e.eval), (-31999, 255, BOUND_UPPER, Some((63, 62)), true, false, want));
        }
        let key = 0xFEDC_BA98_7654_3210u64;
        tt.store(key, 5, 3, BOUND_EXACT, None, false);
        assert_eq!(tt.probe(key).unwrap().eval, None);
        tt.store_eval(key, 6, 4, BOUND_EXACT, None, false, Some(-77));
        tt.store(key, 7, 5, BOUND_LOWER, None, false);
        let e = tt.probe(key).unwrap();
        assert_eq!((e.score, e.depth, e.eval), (7, 5, Some(-77)));
        assert_eq!(std::mem::size_of::<Bucket>(), 64, "корзина — строка кеша, 4 записи по 16 байт");
    }
}
