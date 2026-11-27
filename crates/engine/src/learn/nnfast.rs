
use super::arch::EVAL_LIMIT;
use super::arch::ViewKind;
use super::linear::LinearEval;
use super::refeval::RefEval;
use super::view::ViewTables;
use super::weights::Model;
use crate::eval::Evaluator;
use crate::movegen::Move;
use crate::position::{Position, Side};
use crate::rules::Rules;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Scalar,
    Neon,
    Avx2,
}

impl Kernel {
    pub fn name(self) -> &'static str {
        match self {
            Kernel::Scalar => "scalar",
            Kernel::Neon => "neon+dotprod",
            Kernel::Avx2 => "avx2",
        }
    }

    pub fn parse(s: &str) -> Option<Kernel> {
        match s.trim().to_ascii_lowercase().as_str() {
            "scalar" => Some(Kernel::Scalar),
            "neon" | "neon+dotprod" | "dotprod" => Some(Kernel::Neon),
            "avx2" => Some(Kernel::Avx2),
            _ => None,
        }
    }

    pub fn is_available(self) -> bool {
        match self {
            Kernel::Scalar => true,
            Kernel::Neon => {
                #[cfg(target_arch = "aarch64")]
                {
                    std::arch::is_aarch64_feature_detected!("neon") && std::arch::is_aarch64_feature_detected!("dotprod")
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    false
                }
            }
            Kernel::Avx2 => {
                #[cfg(target_arch = "x86_64")]
                {
                    std::arch::is_x86_feature_detected!("avx2") || std::env::var("RUNNER_NN_ASSUME_AVX2").map_or(false, |v| v == "1")
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    false
                }
            }
        }
    }

    pub fn available() -> Vec<Kernel> {
        [Kernel::Scalar, Kernel::Neon, Kernel::Avx2].into_iter().filter(|k| k.is_available()).collect()
    }

    pub fn best() -> Kernel {
        if let Ok(v) = std::env::var("RUNNER_NN_KERNEL") {
            if let Some(k) = Kernel::parse(&v) {
                if k.is_available() {
                    return k;
                }
            }
        }
        if Kernel::Neon.is_available() {
            Kernel::Neon
        } else if Kernel::Avx2.is_available() {
            Kernel::Avx2
        } else {
            Kernel::Scalar
        }
    }
}

const LANES: usize = 32;
pub const NN_MAX_IN: usize = 2048;
pub const NN_MAX_D: usize = 1024;
const LOOKBACK: usize = 8;
const NONE: u32 = u32::MAX;

#[repr(C, align(64))]
struct BufU8([u8; NN_MAX_IN]);
#[repr(C, align(64))]
struct BufI32([i32; NN_MAX_IN]);
#[repr(C, align(64))]
struct BufI16([i16; 2 * NN_MAX_D]);

type Un<T> = std::mem::MaybeUninit<T>;

fn round_up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

struct Layer {
    inp: usize,
    outp: usize,
    wplain: Vec<i8>,
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    wdot: Vec<i8>,
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    w16: Vec<i16>,
    bias: Vec<i32>,
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    bias_s: Vec<i32>,
}

impl Layer {
    fn new(w: &[i8], bias: &[i32], inn: usize, out: usize, inp: usize, remap: &dyn Fn(usize) -> usize) -> Layer {
        let outp = round_up(out.max(1), 8);
        let mut wplain = vec![0i8; outp * inp];
        for o in 0..out {
            for i in 0..inn {
                wplain[o * inp + remap(i)] = w[o * inn + i];
            }
        }
        let mut b = vec![0i32; outp];
        b[..out].copy_from_slice(&bias[..out]);
        let mut bias_s = b.clone();
        for o in 0..outp {
            let sw: i32 = wplain[o * inp..(o + 1) * inp].iter().map(|&x| x as i32).sum();
            bias_s[o] = b[o].wrapping_add(sw.wrapping_mul(128));
        }
        let mut wdot = Vec::with_capacity(outp * inp);
        let nob = outp / 4;
        let mut ob0 = 0;
        while ob0 < nob {
            let cnt = (nob - ob0).min(8);
            for g in 0..inp / 16 {
                for l in 0..4 {
                    for k in 0..cnt {
                        for r in 0..4 {
                            let o = (ob0 + k) * 4 + r;
                            for t in 0..4 {
                                wdot.push(wplain[o * inp + g * 16 + l * 4 + t]);
                            }
                        }
                    }
                }
            }
            ob0 += cnt;
        }
        let mut w16 = Vec::with_capacity(outp * inp);
        let nreg = outp / 8;
        let mut r0 = 0;
        while r0 < nreg {
            let cnt = (nreg - r0).min(8);
            for ip in 0..inp / 2 {
                for k in 0..cnt {
                    for q in 0..8 {
                        let o = (r0 + k) * 8 + q;
                        w16.push(wplain[o * inp + 2 * ip] as i16);
                        w16.push(wplain[o * inp + 2 * ip + 1] as i16);
                    }
                }
            }
            r0 += cnt;
        }
        Layer { inp, outp, wplain, wdot, w16, bias: b, bias_s }
    }

    #[inline(always)]
    unsafe fn run(&self, k: Kernel, x: &[u8], s: *mut i32) {
        match k {
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => neon::layer(self, x, s),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => avx2::layer(self, x, s),
            _ => self.run_scalar(x, s),
        }
    }

    unsafe fn run_scalar(&self, x: &[u8], s: *mut i32) {
        let inp = self.inp;
        for o in 0..self.outp {
            let w = &self.wplain[o * inp..(o + 1) * inp];
            let mut a = 0i32;
            for (wi, &xi) in w.iter().zip(&x[..inp]) {
                a = a.wrapping_add(*wi as i32 * xi as i32);
            }
            *s.add(o) = self.bias[o].wrapping_add(a);
        }
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C, align(32))]
struct CacheEnt {
    w: u64,
    b: u64,
    k: u64,
    tag: u32,
    v: i32,
}

#[derive(Clone, Copy, Default)]
#[repr(C, align(16))]
struct CacheEnt16 {
    key: u64,
    chk: u32,
    v: i32,
}

#[inline(always)]
fn board_chk(w: u64, b: u64, k: u64) -> u32 {
    let x = w.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ k.wrapping_mul(0x1656_67B1_9E37_79F9);
    ((x ^ (x >> 29)) >> 32) as u32 | 1
}

pub struct EvalCache {
    owner: u64,
    ents: Box<[CacheEnt]>,
    ents16: Box<[CacheEnt16]>,
    pub hits: u64,
    pub probes: u64,
}

thread_local! {
    static CACHE_POOL: std::cell::RefCell<Vec<Box<EvalCache>>> = const { std::cell::RefCell::new(Vec::new()) };
}

static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl Drop for NnState {
    fn drop(&mut self) {
        if let Some(c) = self.cache.take() {
            let _ = CACHE_POOL.try_with(|p| {
                if let Ok(mut p) = p.try_borrow_mut() {
                    if p.len() < 4 {
                        p.push(c);
                    }
                }
            });
        }
    }
}

fn take_cache(owner: u64, n: usize, small: bool) -> Box<EvalCache> {
    let (n32, n16) = if small { (0, n) } else { (n, 0) };
    let fits = |c: &EvalCache| c.ents.len() == n32 && c.ents16.len() == n16;
    let got = CACHE_POOL
        .try_with(|p| {
            let mut p = p.try_borrow_mut().ok()?;
            let i = p.iter().position(|c| c.owner == owner && fits(c)).or_else(|| p.iter().position(|c| fits(c)))?;
            Some(p.swap_remove(i))
        })
        .ok()
        .flatten();
    match got {
        Some(mut c) => {
            if c.owner != owner {
                c.ents.fill(CacheEnt::default());
                c.ents16.fill(CacheEnt16::default());
                c.owner = owner;
            }
            c
        }
        None => Box::new(EvalCache {
            owner,
            ents: vec![CacheEnt::default(); n32].into_boxed_slice(),
            ents16: vec![CacheEnt16::default(); n16].into_boxed_slice(),
            hits: 0,
            probes: 0,
        }),
    }
}

pub struct NnState {
    valid: bool,
    w: u64,
    b: u64,
    k: u64,
    acc: Box<[i16]>,
    pix: Box<[u16]>,
    cache: Option<Box<EvalCache>>,
}

impl Clone for NnState {
    fn clone(&self) -> Self {
        NnState { valid: self.valid, w: self.w, b: self.b, k: self.k, acc: self.acc.clone(), pix: self.pix.clone(), cache: None }
    }
    fn clone_from(&mut self, src: &Self) {
        self.valid = src.valid;
        self.w = src.w;
        self.b = src.b;
        self.k = src.k;
        if self.acc.len() != src.acc.len() {
            self.acc = src.acc.clone();
        } else if src.valid {
            self.acc.copy_from_slice(&src.acc);
        }
        self.cache = None;
    }
}

impl NnState {
    pub fn is_valid(&self) -> bool {
        self.valid
    }
    pub fn acc_slice(&self) -> &[i16] {
        &self.acc
    }
    pub fn cache(&self) -> Option<&EvalCache> {
        self.cache.as_deref()
    }
    pub fn diff_bits<R: Rules>(&self, pos: &Position<R>) -> u32 {
        if !self.valid {
            return u32::MAX;
        }
        ((self.w ^ pos.white) | (self.b ^ pos.black) | (self.k ^ pos.kings)).count_ones()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Ready,
    Delta { j: usize, bits: u32 },
    Full,
}

pub struct NnFast<R: Rules> {
    model: Arc<Model>,
    refe: Option<RefEval<R>>,
    lin: Option<LinearEval<R>>,
    kernel: Kernel,
    has_wide: bool,
    nviews: usize,
    d: usize,
    dp: usize,
    roff: Box<[[[u32; 5]; 64]; 2]>,
    dtab: Vec<i16>,
    base0: Vec<i16>,
    heads: Vec<Vec<Layer>>,
    head_tab: [u8; 65],
    parity: Option<[u64; 2]>,
    cache_bits: u32,
    cache16: bool,
    id: u64,
    acc32: Option<Acc32>,
    pat: Option<PatPart>,
    _r: PhantomData<fn() -> R>,
}

const PAT_MAX: usize = 64;
const ADDS_MAX: usize = 64 + PAT_MAX;

struct Rows {
    p: [std::mem::MaybeUninit<*const i16>; ADDS_MAX],
    n: usize,
}

impl Rows {
    #[inline(always)]
    fn new() -> Rows {
        Rows { p: [std::mem::MaybeUninit::uninit(); ADDS_MAX], n: 0 }
    }
    #[inline(always)]
    fn push(&mut self, r: *const i16) {
        debug_assert!(self.n < ADDS_MAX);
        unsafe { self.p.get_unchecked_mut(self.n).write(r) };
        self.n += 1;
    }
    #[inline(always)]
    fn clear(&mut self) {
        self.n = 0;
    }
    #[inline(always)]
    fn as_slice(&self) -> &[*const i16] {
        unsafe { std::slice::from_raw_parts(self.p.as_ptr() as *const *const i16, self.n) }
    }
}
const PG: usize = 16;

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([i16; LANES]);

struct PatPart {
    n: usize,
    ngroups: usize,
    bytes: [Vec<(u32, u32)>; 2],
    itab: Vec<[u16; PG]>,
    rbase: Vec<u32>,
    vbase: Vec<u32>,
    ptab: Vec<Line>,
    rmap: Vec<u32>,
    sqo: [Vec<u32>; 2],
    sqe: [Vec<(u8, u16)>; 2],
}

impl PatPart {
    fn new(m: &Model, multi: &[usize], t: &ViewTables, dp: usize, pack: bool) -> PatPart {
        let (l, d) = (&m.layout, m.arch.deep.as_ref().map_or(0, |x| x.width));
        let n = multi.len();
        let ngroups = n.div_ceil(PG);
        let mut first_row: std::collections::BTreeMap<usize, u32> = Default::default();
        let mut ptab: Vec<Line> = vec![];
        let lines = dp / LANES;
        let (mut rbase, mut vbase) = (vec![], vec![]);
        for &i in multi {
            let p = &m.arch.patterns[i];
            let vb = l.v_off[p.table];
            let r0 = *first_row.entry(p.table).or_insert_with(|| {
                let r0 = (ptab.len() / lines) as u32;
                for ix in 0..m.arch.tables[p.table].size {
                    let mut row = vec![0i16; dp];
                    row[..d].copy_from_slice(&m.deep_v[vb + ix * d..vb + (ix + 1) * d]);
                    for c in row.chunks(LANES) {
                        let mut ln = Line([0; LANES]);
                        ln.0.copy_from_slice(c);
                        ptab.push(ln);
                    }
                }
                r0
            });
            rbase.push(r0);
            vbase.push(vb as u32);
        }
        let mut itab: Vec<[u16; PG]> = vec![];
        let mut bytes = [vec![], vec![]];
        let (mut sqo, mut sqe): ([Vec<u32>; 2], [Vec<(u8, u16)>; 2]) = Default::default();
        for o in 0..2 {
            for bit in 0..64usize {
                sqo[o].push(sqe[o].len() as u32);
                for (j, &i) in multi.iter().enumerate() {
                    let mut mul = 1u32;
                    for &s in &m.arch.patterns[i].squares {
                        if t.from_dense[o][s as usize] as usize == bit {
                            sqe[o].push((j as u8, mul as u16));
                        }
                        mul *= 3;
                    }
                }
            }
            sqo[o].push(sqe[o].len() as u32);
        }
        for o in 0..2 {
            let mut ent: Vec<(usize, usize, usize, u16)> = vec![];
            for (j, &i) in multi.iter().enumerate() {
                let mut mul = 1u32;
                for &s in &m.arch.patterns[i].squares {
                    let bit = t.from_dense[o][s as usize] as usize;
                    ent.push((bit / 8, bit % 8, j, mul as u16));
                    mul *= 3;
                }
            }
            let mut cs: Vec<usize> = ent.iter().map(|e| e.0).collect();
            cs.sort_unstable();
            cs.dedup();
            for c in cs {
                let toff = itab.len() as u32;
                for g in 0..ngroups {
                    for x in 0..256usize {
                        let mut e = [0u16; PG];
                        for &(_, b, j, mul) in ent.iter().filter(|e| e.0 == c && e.2 / PG == g) {
                            if x >> b & 1 != 0 {
                                e[j % PG] += mul;
                            }
                        }
                        itab.push(e);
                    }
                }
                bytes[o].push(((c * 8) as u32, toff));
            }
        }
        let mut rmap = vec![];
        if pack {
            let mut packed: Vec<Line> = vec![];
            for row in ptab.chunks(lines) {
                if row.iter().all(|ln| ln.0.iter().all(|&x| x == 0)) {
                    rmap.push(NONE);
                } else {
                    rmap.push((packed.len() / lines) as u32);
                    packed.extend_from_slice(row);
                }
            }
            ptab = packed;
            ptab.shrink_to_fit();
        }
        PatPart { n, ngroups, bytes, itab, rbase, vbase, ptab, rmap, sqo, sqe }
    }

    fn rows_stat(&self, dp: usize) -> (usize, usize) {
        let lines = dp / LANES;
        let full = if self.rmap.is_empty() { self.ptab.len() / lines } else { self.rmap.len() };
        (self.ptab.len() / lines, full)
    }

    #[inline(always)]
    fn push_row(&self, i: usize, ix: u16, dp: usize, rows: &mut Rows) {
        if self.rmap.is_empty() {
            rows.push(self.row(i, ix, dp));
        } else {
            let r = unsafe { *self.rmap.get_unchecked(self.rbase[i] as usize + ix as usize) };
            if r != NONE {
                rows.push(unsafe { (self.ptab.as_ptr() as *const i16).add(r as usize * dp) });
            }
        }
    }

    #[inline(always)]
    fn indices(&self, o: usize, w: u64, b: u64, k: u64, ix: &mut [u16]) {
        let (own, opp) = if o == 0 { (w & !k, b & !k) } else { (b & !k, w & !k) };
        for g in 0..self.ngroups {
            let mut a = [0u16; PG];
            for &(sh, toff) in &self.bytes[o] {
                let tb = toff as usize + g * 256;
                let x = &self.itab[tb + (own >> sh & 255) as usize];
                let y = &self.itab[tb + (opp >> sh & 255) as usize];
                for l in 0..PG {
                    a[l] = a[l].wrapping_add(x[l]).wrapping_add(y[l].wrapping_add(y[l]));
                }
            }
            let hi = (self.n - g * PG).min(PG);
            ix[g * PG..g * PG + hi].copy_from_slice(&a[..hi]);
        }
    }

    #[inline(always)]
    fn row(&self, i: usize, ix: u16, dp: usize) -> *const i16 {
        unsafe { (self.ptab.as_ptr() as *const i16).add((self.rbase[i] as usize + ix as usize) * dp) }
    }

    #[inline(always)]
    fn full_rows(&self, o: usize, w: u64, b: u64, k: u64, dp: usize, ix: &mut [u16], adds: &mut Rows) {
        self.indices(o, w, b, k, ix);
        for (i, &x) in ix[..self.n].iter().enumerate() {
            self.push_row(i, x, dp, adds);
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn delta_rows(&self, o: usize, (sw, sb, sk): (u64, u64, u64), (w, b, k): (u64, u64, u64), dp: usize, ix: &mut [u16], adds: &mut Rows, subs: &mut Rows) {
        let men = |w: u64, b: u64, k: u64| if o == 0 { (w & !k, b & !k) } else { (b & !k, w & !k) };
        let ((own0, opp0), (own1, opp1)) = (men(sw, sb, sk), men(w, b, k));
        let mut x = (own0 ^ own1) | (opp0 ^ opp1);
        if x == 0 {
            return;
        }
        let mut old = [std::mem::MaybeUninit::<u16>::uninit(); PAT_MAX];
        let mut touched = 0u64;
        let (sqo, sqe) = (&self.sqo[o], &self.sqe[o]);
        while x != 0 {
            let bit = x.trailing_zeros() as usize;
            x &= x - 1;
            let dg = |own: u64, opp: u64| (own >> bit & 1) as u16 + 2 * (opp >> bit & 1) as u16;
            let (d0, d1) = (dg(own0, opp0), dg(own1, opp1));
            for &(j, mul) in &sqe[sqo[bit] as usize..sqo[bit + 1] as usize] {
                let j = j as usize;
                if touched >> j & 1 == 0 {
                    touched |= 1u64 << j;
                    old[j].write(ix[j]);
                }
                ix[j] = ix[j].wrapping_add(mul.wrapping_mul(d1)).wrapping_sub(mul.wrapping_mul(d0));
            }
        }
        while touched != 0 {
            let j = touched.trailing_zeros() as usize;
            touched &= touched - 1;
            let o = unsafe { old[j].assume_init() };
            if ix[j] != o {
                self.push_row(j, o, dp, subs);
                self.push_row(j, ix[j], dp, adds);
            }
        }
    }
}

struct Acc32 {
    base: Vec<i32>,
    dtab: Vec<i32>,
}

pub const CACHE_BITS: u32 = 13;

impl<R: Rules> NnFast<R> {
    pub fn new(model: Model) -> Result<Self, String> {
        Self::from_arc(Arc::new(model))
    }

    pub fn from_arc(model: Arc<Model>) -> Result<Self, String> {
        Self::with_kernel(model, Kernel::best())
    }

    pub fn with_kernel(model: Arc<Model>, kernel: Kernel) -> Result<Self, String> {
        if !kernel.is_available() {
            return Err(format!("NnFast: ядро {} недоступно на этой машине", kernel.name()));
        }
        let dpp = model.arch.deep.as_ref().ok_or("NnFast: у архитектуры нет глубокой части")?;
        let multi: Vec<usize> = (0..model.arch.patterns.len()).filter(|&i| model.arch.patterns[i].squares.len() != 1).collect();
        if !multi.is_empty() {
            let a = &model.arch;
            let hand = dpp.head_flags & super::arch::FLAG_HAND != 0;
            if a.wide || (!a.globals.is_empty() && !hand) || a.phase.kind != super::arch::PhaseKind::None {
                return Err("NnFast: многопольные шаблоны только без широкой части, глобальных признаков (без флага hand_features) и стадий".into());
            }
            if multi.len() > PAT_MAX {
                return Err(format!("NnFast: многопольных шаблонов больше {PAT_MAX}"));
            }
            for &i in &multi {
                let p = &a.patterns[i];
                if p.alphabet != 3 || a.tables[p.table].size > 1 << 16 || p.squares.is_empty() {
                    return Err("NnFast: многопольный шаблон не с алфавитом 3 или индекс не меньше 2^16".into());
                }
            }
        }
        let l = &model.layout;
        let d = dpp.width;
        if d == 0 || d > NN_MAX_D || l.nviews == 0 || l.nviews > 2 {
            return Err(format!("NnFast: ширина накопителя {d} или число видов {} вне пределов", l.nviews));
        }
        if l.dims.is_empty() || l.dims.iter().any(|&(i, o)| i > NN_MAX_IN || o > NN_MAX_IN) || l.dims.last().map(|x| x.1) != Some(1) {
            return Err("NnFast: размеры слоёв вне пределов".into());
        }
        if model.arch.quant.qa != 255 || model.arch.quant.qb != 64 {
            return Err("NnFast: квантование не DWT1 (QA 255, QB 64)".into());
        }
        let m = &*model;
        let mut bound: Vec<i64> = m.deep_b.iter().map(|&b| (b as i64).abs()).collect();
        for p in &m.arch.patterns {
            let size = m.arch.tables[p.table].size;
            let base = l.v_off[p.table];
            for (j, bj) in bound.iter_mut().enumerate() {
                *bj += (0..size).map(|i| (m.deep_v[base + i * d + j] as i64).abs()).max().unwrap_or(0);
            }
        }
        let acc32 = bound.iter().any(|&b| b > 32767);
        let dp = round_up(d, LANES);
        let t = ViewTables::new::<R>();
        let nsq = l.nsq;
        let mut base0 = vec![0i16; dp];
        base0[..d].copy_from_slice(&m.deep_b);
        let mut base32: Vec<i32> = (0..dp).map(|j| if j < d { m.deep_b[j] as i32 } else { 0 }).collect();
        let mut rows = vec![vec![vec![0i16; dp]; 5]; nsq];
        let mut rows32 = vec![vec![vec![0i32; dp]; 5]; if acc32 { nsq } else { 0 }];
        let mut used = vec![false; nsq];
        for p in m.arch.patterns.iter().filter(|p| p.squares.len() == 1) {
            let q = p.squares[0] as usize;
            used[q] = true;
            let vb = l.v_off[p.table];
            let zero = &m.deep_v[vb..vb + d];
            for (a, &z) in base0.iter_mut().zip(zero) {
                *a = a.wrapping_add(z);
            }
            for (a, &z) in base32.iter_mut().zip(zero) {
                *a += z as i32;
            }
            for s in 1..5usize {
                let dg = if p.alphabet == 5 || s <= 2 { s } else { 0 };
                if dg >= p.alphabet as usize {
                    continue;
                }
                let v = &m.deep_v[vb + dg * d..vb + (dg + 1) * d];
                for j in 0..d {
                    rows[q][s][j] = rows[q][s][j].wrapping_add(v[j].wrapping_sub(zero[j]));
                }
                if acc32 {
                    for j in 0..d {
                        rows32[q][s][j] += v[j] as i32 - zero[j] as i32;
                    }
                }
            }
        }
        let mut dtab: Vec<i16> = vec![];
        let mut dtab32: Vec<i32> = vec![];
        let mut qoff = vec![[NONE; 5]; nsq];
        for q in 0..nsq {
            if !used[q] {
                continue;
            }
            for s in 1..5 {
                if acc32 {
                    if rows32[q][s].iter().all(|&x| x == 0) {
                        continue;
                    }
                    qoff[q][s] = dtab32.len() as u32;
                    dtab32.extend_from_slice(&rows32[q][s]);
                } else {
                    if rows[q][s].iter().all(|&x| x == 0) {
                        continue;
                    }
                    qoff[q][s] = dtab.len() as u32;
                    dtab.extend_from_slice(&rows[q][s]);
                }
            }
        }
        let acc32 = if acc32 { Some(Acc32 { base: base32, dtab: dtab32 }) } else { None };
        let mut roff = Box::new([[[NONE; 5]; 64]; 2]);
        for o in 0..2 {
            for bit in 0..64 {
                let q = t.to_dense[o][bit];
                if q != 255 && (q as usize) < nsq {
                    roff[o][bit] = qoff[q as usize];
                }
            }
        }
        let nv = l.nviews;
        let extra = dpp.head_extra();
        let mut heads = vec![];
        for h in 0..l.heads {
            let mut layers = vec![];
            for (li, &(inn, out)) in l.dims.iter().enumerate() {
                let (w, b) = (&m.deep_w[li][h * inn * out..(h + 1) * inn * out], &m.deep_bias[li][h * out..(h + 1) * out]);
                let layer = if li == 0 {
                    if inn != nv * d + extra {
                        return Err("NnFast: вход слоя 0 не равен 2D".into());
                    }
                    Layer::new(w, b, inn, out, nv * dp + 16 * extra, &|i| if i < nv * d { (i / d) * dp + i % d } else { nv * dp })
                } else {
                    Layer::new(w, b, inn, out, round_up(inn, 16), &|i| i)
                };
                if layer.inp > NN_MAX_IN || layer.outp > NN_MAX_IN {
                    return Err("NnFast: слой шире буферов".into());
                }
                layers.push(layer);
            }
            for w in layers.windows(2) {
                if w[1].inp < w[0].outp.min(w[1].inp) {
                    return Err("NnFast: несогласованные слои".into());
                }
            }
            heads.push(layers);
        }
        let mut head_tab = [0u8; 65];
        for (n, x) in head_tab.iter_mut().enumerate() {
            *x = dpp.head_of(n as u32).min(l.heads - 1) as u8;
        }
        let parity = (extra > 0).then(|| {
            let mut sys = [0u64; 2];
            for (o, s) in sys.iter_mut().enumerate() {
                for bit in 0..64 {
                    let q = t.to_dense[o][bit];
                    if q != 255 && t.row[q as usize] & 1 == 0 {
                        *s |= 1u64 << bit;
                    }
                }
            }
            sys
        });
        let has_wide = m.arch.wide || !m.arch.globals.is_empty();
        let (refe, lin) = if has_wide {
            let mut a = m.arch.clone();
            a.deep = None;
            a.views = vec![ViewKind::Stm];
            let mut m2 = Model::zeros(a)?;
            m2.wide = m.wide.clone();
            m2.globals = m.globals.clone();
            m2.meta = m.meta.clone();
            (Some(RefEval::<R>::from_arc(model.clone())?), LinearEval::<R>::new(m2).ok())
        } else {
            (None, None)
        };
        let cache_bits = match std::env::var("RUNNER_NN_CACHE") {
            Ok(v) => v.trim().parse::<u32>().unwrap_or(CACHE_BITS).min(24),
            Err(_) => CACHE_BITS,
        };
        let cache16 = std::env::var("RUNNER_NN_CACHE16").map_or(false, |v| v.trim() == "1");
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let pack = std::env::var("RUNNER_NN_PACK").map_or(false, |v| v.trim() == "1");
        let pat = if multi.is_empty() { None } else { Some(PatPart::new(m, &multi, &t, dp, pack)) };
        Ok(NnFast { model, refe, lin, kernel, cache_bits, cache16, acc32, id, has_wide, nviews: nv, d, dp, roff, dtab, base0, heads, head_tab, parity, pat, _r: PhantomData })
    }

    pub fn kernel(&self) -> Kernel {
        self.kernel
    }

    pub fn pattern_count(&self) -> usize {
        self.pat.as_ref().map_or(0, |p| p.n)
    }

    pub fn pattern_indices(&self, pos: &Position<R>, o: usize) -> Vec<usize> {
        match &self.pat {
            None => vec![],
            Some(pp) => {
                let mut ix = [0u16; PAT_MAX];
                pp.indices(o, pos.white, pos.black, pos.kings, &mut ix);
                ix[..pp.n].iter().map(|&x| x as usize).collect()
            }
        }
    }

    pub fn is_acc32(&self) -> bool {
        self.acc32.is_some()
    }

    fn eval_full32(&self, pos: &Position<R>, a: &Acc32) -> i32 {
        let dp = self.dp;
        let o_stm = if pos.side == Side::White { 0 } else { 1 };
        let occ = pos.white | pos.black;
        let mut xa = Un::<BufU8>::uninit();
        let xp = xa.as_mut_ptr() as *mut u8;
        let mut acc = vec![0i32; dp];
        for v in 0..self.nviews {
            let o = o_stm ^ v;
            acc.copy_from_slice(&a.base);
            let mut x = occ;
            while x != 0 {
                let bit = x.trailing_zeros() as usize;
                x &= x - 1;
                let off = self.roff[o][bit][Self::state_of(o, pos.white, pos.black, pos.kings, bit)];
                if off != NONE {
                    for (s, &r) in acc.iter_mut().zip(&a.dtab[off as usize..off as usize + dp]) {
                        *s += r;
                    }
                }
            }
            if let Some(pp) = &self.pat {
                let (m, d) = (&*self.model, self.d);
                let mut ix = [0u16; PAT_MAX];
                pp.indices(o, pos.white, pos.black, pos.kings, &mut ix);
                for i in 0..pp.n {
                    let vb = pp.vbase[i] as usize + ix[i] as usize * d;
                    for (s, &r) in acc[..d].iter_mut().zip(&m.deep_v[vb..vb + d]) {
                        *s += r as i32;
                    }
                }
            }
            for (j, &s) in acc.iter().enumerate() {
                unsafe { *xp.add(v * dp + j) = s.clamp(0, 255) as u8 };
            }
        }
        unsafe { self.put_parity(pos, xp) };
        let deep = unsafe { self.layers_out(self.head(pos), xp) };
        let wide = if self.has_wide { self.wide_cp(pos) } else { 0 };
        wide.wrapping_add(deep).clamp(-EVAL_LIMIT, EVAL_LIMIT)
    }

    #[inline(always)]
    unsafe fn put_parity(&self, pos: &Position<R>, x: *mut u8) {
        if let Some(sys) = &self.parity {
            let occ = pos.white | pos.black;
            let n = occ.count_ones();
            let o = if pos.side == Side::White { 0 } else { 1 };
            let bit = (super::arch::PARITY_MIN..=super::arch::PARITY_MAX).contains(&n) && (occ & sys[o]).count_ones() & 1 != 0;
            let at = x.add(self.nviews * self.dp);
            std::ptr::write_bytes(at, 0, 16);
            *at = if bit { 255 } else { 0 };
        }
    }

    #[inline(always)]
    fn head(&self, pos: &Position<R>) -> usize {
        self.head_tab[(pos.white | pos.black).count_ones() as usize] as usize
    }

    pub fn set_cache_bits(&mut self, bits: u32) {
        self.cache_bits = bits.min(24);
    }

    pub fn set_pack(&mut self, on: bool) {
        if self.pat.as_ref().map_or(true, |p| !p.rmap.is_empty() == on) {
            return;
        }
        let multi: Vec<usize> = (0..self.model.arch.patterns.len()).filter(|&i| self.model.arch.patterns[i].squares.len() != 1).collect();
        let t = ViewTables::new::<R>();
        self.pat = Some(PatPart::new(&self.model, &multi, &t, self.dp, on));
    }

    pub fn pattern_rows(&self) -> (usize, usize) {
        self.pat.as_ref().map_or((0, 0), |p| p.rows_stat(self.dp))
    }

    pub fn set_cache16(&mut self, on: bool) {
        self.cache16 = on;
    }

    pub fn cache16(&self) -> bool {
        self.cache16
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    #[inline(always)]
    fn wide_cp(&self, pos: &Position<R>) -> i32 {
        if let Some(lin) = &self.lin {
            let w = lin.eval_full(pos);
            if w.abs() < EVAL_LIMIT {
                return w;
            }
        }
        match &self.refe {
            Some(r) => r.wide_cp(&r.features(pos)),
            None => 0,
        }
    }

    pub fn wide_path(&self) -> &'static str {
        match (self.has_wide, self.lin.is_some()) {
            (false, _) => "none",
            (true, true) => "linear",
            (true, false) => "ref",
        }
    }

    pub fn widths(&self) -> (usize, usize) {
        (self.d, self.dp)
    }

    #[inline(always)]
    fn state_of(o: usize, w: u64, b: u64, k: u64, bit: usize) -> usize {
        let (own, opp) = if o == 0 { (w, b) } else { (b, w) };
        let king = (k >> bit & 1) as usize * 2;
        if own >> bit & 1 != 0 {
            1 + king
        } else if opp >> bit & 1 != 0 {
            2 + king
        } else {
            0
        }
    }

    #[inline(always)]
    fn full_into(&self, pos: &Position<R>, dst: *mut i16, pix: &mut [u16]) {
        let dp = self.dp;
        let occ = pos.white | pos.black;
        let mut adds = Rows::new();
        for o in 0..2usize {
            adds.clear();
            let mut x = occ;
            while x != 0 {
                let bit = x.trailing_zeros() as usize;
                x &= x - 1;
                let off = self.roff[o][bit][Self::state_of(o, pos.white, pos.black, pos.kings, bit)];
                if off != NONE {
                    adds.push(unsafe { self.dtab.as_ptr().add(off as usize) });
                }
            }
            if let Some(pp) = &self.pat {
                pp.full_rows(o, pos.white, pos.black, pos.kings, dp, &mut pix[o * pp.n..(o + 1) * pp.n], &mut adds);
            }
            unsafe { acc_apply(self.kernel, dst.add(o * dp), self.base0.as_ptr(), dp, adds.as_slice(), &[]) };
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn delta_into(&self, pos: &Position<R>, sw: u64, sb: u64, sk: u64, src: *const i16, dst: *mut i16, spix: *const u16, dpix: *mut u16) {
        let dp = self.dp;
        let diff = (sw ^ pos.white) | (sb ^ pos.black) | (sk ^ pos.kings);
        let (mut adds, mut subs) = (Rows::new(), Rows::new());
        if let Some(pp) = &self.pat {
            if spix != dpix as *const u16 {
                unsafe { std::ptr::copy_nonoverlapping(spix, dpix, 2 * pp.n) };
            }
        }
        for o in 0..2usize {
            adds.clear();
            subs.clear();
            let mut x = diff;
            while x != 0 {
                let bit = x.trailing_zeros() as usize;
                x &= x - 1;
                let r = &self.roff[o][bit];
                let (s0, s1) = (Self::state_of(o, sw, sb, sk, bit), Self::state_of(o, pos.white, pos.black, pos.kings, bit));
                let (a, b) = (r[s0], r[s1]);
                if a == b {
                    continue;
                }
                if a != NONE {
                    subs.push(unsafe { self.dtab.as_ptr().add(a as usize) });
                }
                if b != NONE {
                    adds.push(unsafe { self.dtab.as_ptr().add(b as usize) });
                }
            }
            if let Some(pp) = &self.pat {
                let ix = unsafe { std::slice::from_raw_parts_mut(dpix.add(o * pp.n), pp.n) };
                pp.delta_rows(o, (sw, sb, sk), (pos.white, pos.black, pos.kings), dp, ix, &mut adds, &mut subs);
            }
            unsafe { acc_apply(self.kernel, dst.add(o * dp), src.add(o * dp), dp, adds.as_slice(), subs.as_slice()) };
        }
    }

    #[inline(always)]
    fn finish(&self, pos: &Position<R>, acc: &[i16]) -> i32 {
        let dp = self.dp;
        let o_stm = if pos.side == Side::White { 0 } else { 1 };
        let mut xa = Un::<BufU8>::uninit();
        let xp = xa.as_mut_ptr() as *mut u8;
        for v in 0..self.nviews {
            let o = o_stm ^ v;
            unsafe { clamp_u8(self.kernel, &acc[o * dp..(o + 1) * dp], xp.add(v * dp)) };
        }
        unsafe { self.put_parity(pos, xp) };
        let deep = unsafe { self.layers_out(self.head(pos), xp) };
        let wide = if self.has_wide { self.wide_cp(pos) } else { 0 };
        wide.wrapping_add(deep).clamp(-EVAL_LIMIT, EVAL_LIMIT)
    }

    #[inline(always)]
    unsafe fn layers_out(&self, h: usize, x0: *mut u8) -> i32 {
        let mut s = Un::<BufI32>::uninit();
        let sp = s.as_mut_ptr() as *mut i32;
        let mut xb = Un::<BufU8>::uninit();
        let layers = self.heads.get_unchecked(h);
        let nl = layers.len();
        let (mut cur, mut nxt) = (x0, xb.as_mut_ptr() as *mut u8);
        for (li, l) in layers.iter().enumerate() {
            l.run(self.kernel, std::slice::from_raw_parts(cur, l.inp), sp);
            if li + 1 == nl {
                return *sp >> 6;
            }
            let ninp = layers[li + 1].inp;
            for o in 0..l.outp.min(ninp) {
                *nxt.add(o) = (*sp.add(o) >> 6).clamp(0, 255) as u8;
            }
            for o in l.outp.min(ninp)..ninp {
                *nxt.add(o) = 0;
            }
            std::mem::swap(&mut cur, &mut nxt);
        }
        0
    }

    #[inline(always)]
    pub fn plan(&self, pos: &Position<R>, stack: &[NnState], ply: usize) -> Source {
        let (mut best_j, mut best_bits) = (usize::MAX, u32::MAX);
        let lo = ply.saturating_sub(LOOKBACK);
        let mut j = ply + 1;
        while j > lo {
            j -= 1;
            let st = &stack[j];
            if !st.valid {
                continue;
            }
            let bits = ((st.w ^ pos.white) | (st.b ^ pos.black) | (st.k ^ pos.kings)).count_ones();
            if bits == 0 && j == ply {
                return Source::Ready;
            }
            if bits < best_bits {
                best_bits = bits;
                best_j = j;
            }
        }
        let occ = (pos.white | pos.black).count_ones();
        let npat = self.pat.as_ref().map_or(0, |p| p.n as u32);
        if best_j != usize::MAX && 2 * best_bits + 2 * npat.min(2 * best_bits) < occ + npat {
            Source::Delta { j: best_j, bits: best_bits }
        } else {
            Source::Full
        }
    }

    #[inline(always)]
    fn materialize(&self, pos: &Position<R>, stack: &mut [NnState], ply: usize) {
        match self.plan(pos, stack, ply) {
            Source::Ready => return,
            Source::Full => {
                let me = &mut stack[ply];
                self.full_into(pos, me.acc.as_mut_ptr(), &mut me.pix);
            }
            Source::Delta { j, .. } => {
                let (lo, hi) = stack.split_at_mut(ply);
                let me = &mut hi[0];
                let dst = me.acc.as_mut_ptr();
                let dpix = me.pix.as_mut_ptr();
                if j == ply {
                    let (sw, sb, sk) = (me.w, me.b, me.k);
                    self.delta_into(pos, sw, sb, sk, dst as *const i16, dst, dpix as *const u16, dpix);
                } else {
                    let s = &lo[j];
                    self.delta_into(pos, s.w, s.b, s.k, s.acc.as_ptr(), dst, s.pix.as_ptr(), dpix);
                }
            }
        }
        let me = &mut stack[ply];
        me.valid = true;
        me.w = pos.white;
        me.b = pos.black;
        me.k = pos.kings;
    }

    pub fn eval_full(&self, pos: &Position<R>) -> i32 {
        if let Some(a32) = &self.acc32 {
            return self.eval_full32(pos, a32);
        }
        let mut a = Un::<BufI16>::uninit();
        let ap = a.as_mut_ptr() as *mut i16;
        let mut pix = [0u16; 2 * PAT_MAX];
        self.full_into(pos, ap, &mut pix);
        self.finish(pos, unsafe { std::slice::from_raw_parts(ap, 2 * self.dp) })
    }

    pub fn state_consistent(&self, st: &NnState) -> bool {
        if !st.valid {
            return true;
        }
        let mut p = Position::<R>::start();
        p.white = st.w;
        p.black = st.b;
        p.kings = st.k;
        let mut a = Un::<BufI16>::uninit();
        let ap = a.as_mut_ptr() as *mut i16;
        let mut pix = vec![0u16; 2 * self.pattern_count()];
        self.full_into(&p, ap, &mut pix);
        let got = unsafe { std::slice::from_raw_parts(ap, 2 * self.dp) };
        got == &st.acc[..] && pix[..] == st.pix[..]
    }

    pub fn layers_only(&self, x: &[u8]) -> i32 {
        self.layers_only_head(0, x)
    }

    pub fn layers_only_head(&self, h: usize, x: &[u8]) -> i32 {
        let h = h.min(self.heads.len() - 1);
        let mut xa = BufU8([0; NN_MAX_IN]);
        xa.0[..x.len()].copy_from_slice(x);
        unsafe { self.layers_out(h, xa.0.as_mut_ptr()) }
    }

    #[inline(always)]
    fn eval_at16(&self, pos: &Position<R>, stack: &mut [NnState], ply: usize, idx: usize) -> i32 {
        let chk = board_chk(pos.white, pos.black, pos.kings);
        {
            let c = stack[0].cache.get_or_insert_with(|| take_cache(self.id, 1 << self.cache_bits, true));
            c.probes += 1;
            let e = &c.ents16[idx];
            if e.key == pos.hash && e.chk == chk {
                c.hits += 1;
                return e.v;
            }
        }
        let v = match &self.acc32 {
            Some(a32) => self.eval_full32(pos, a32),
            None => {
                self.materialize(pos, stack, ply);
                self.finish(pos, &stack[ply].acc)
            }
        };
        if let Some(c) = stack[0].cache.as_mut() {
            c.ents16[idx] = CacheEnt16 { key: pos.hash, chk, v };
        }
        v
    }

    pub fn head_count(&self) -> usize {
        self.heads.len()
    }
}

impl<R: Rules> Evaluator<R> for NnFast<R> {
    type State = NnState;
    fn root(&self, _pos: &Position<R>) -> NnState {
        NnState { valid: false, w: 0, b: 0, k: 0, acc: vec![0i16; 2 * self.dp].into_boxed_slice(), pix: vec![0u16; 2 * self.pattern_count()].into_boxed_slice(), cache: None }
    }
    fn child(&self, parent: &NnState, _before: &Position<R>, _mv: &Move, _after: &Position<R>) -> NnState {
        parent.clone()
    }
    #[inline]
    fn eval(&self, pos: &Position<R>, st: &mut NnState) -> i32 {
        if let Some(a32) = &self.acc32 {
            return self.eval_full32(pos, a32);
        }
        self.materialize(pos, std::slice::from_mut(st), 0);
        self.finish(pos, &st.acc)
    }
    #[inline(always)]
    fn child_at(&self, _stack: &mut [NnState], _ply: usize, _before: &Position<R>, _mv: &Move, _after: &Position<R>) {}
    #[inline]
    fn eval_at(&self, pos: &Position<R>, stack: &mut [NnState], ply: usize) -> i32 {
        if self.cache_bits == 0 {
            if let Some(a32) = &self.acc32 {
                return self.eval_full32(pos, a32);
            }
            self.materialize(pos, stack, ply);
            return self.finish(pos, &stack[ply].acc);
        }
        let idx = (pos.hash as usize) & ((1usize << self.cache_bits) - 1);
        if self.cache16 {
            return self.eval_at16(pos, stack, ply, idx);
        }
        let tag = 1 | ((pos.side == Side::White) as u32) << 1;
        {
            let c = stack[0].cache.get_or_insert_with(|| take_cache(self.id, 1 << self.cache_bits, false));
            c.probes += 1;
            let e = &c.ents[idx];
            if e.tag == tag && e.w == pos.white && e.b == pos.black && e.k == pos.kings {
                c.hits += 1;
                return e.v;
            }
        }
        let v = match &self.acc32 {
            Some(a32) => self.eval_full32(pos, a32),
            None => {
                self.materialize(pos, stack, ply);
                self.finish(pos, &stack[ply].acc)
            }
        };
        if let Some(c) = stack[0].cache.as_mut() {
            c.ents[idx] = CacheEnt { w: pos.white, b: pos.black, k: pos.kings, tag, v };
        }
        v
    }
    fn eval_fresh(&self, pos: &Position<R>) -> i32 {
        self.eval_full(pos)
    }
}

#[inline(always)]
unsafe fn acc_apply(k: Kernel, dst: *mut i16, src: *const i16, dp: usize, adds: &[*const i16], subs: &[*const i16]) {
    match k {
        #[cfg(target_arch = "aarch64")]
        Kernel::Neon => neon::acc_apply(dst, src, dp, adds, subs),
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => avx2::acc_apply(dst, src, dp, adds, subs),
        _ => acc_apply_scalar(dst, src, dp, adds, subs),
    }
}

unsafe fn acc_apply_scalar(dst: *mut i16, src: *const i16, dp: usize, adds: &[*const i16], subs: &[*const i16]) {
    if dst as *const i16 != src {
        std::ptr::copy_nonoverlapping(src, dst, dp);
    }
    let d = std::slice::from_raw_parts_mut(dst, dp);
    for &p in adds {
        for (a, &r) in d.iter_mut().zip(std::slice::from_raw_parts(p, dp)) {
            *a = a.wrapping_add(r);
        }
    }
    for &p in subs {
        for (a, &r) in d.iter_mut().zip(std::slice::from_raw_parts(p, dp)) {
            *a = a.wrapping_sub(r);
        }
    }
}

#[inline(always)]
unsafe fn clamp_u8(k: Kernel, a: &[i16], x: *mut u8) {
    match k {
        #[cfg(target_arch = "aarch64")]
        Kernel::Neon => neon::clamp_u8(a, x),
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => avx2::clamp_u8(a, x),
        _ => {
            for (i, &v) in a.iter().enumerate() {
                *x.add(i) = v.clamp(0, 255) as u8;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::Layer;
    use std::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub unsafe fn acc_apply(dst: *mut i16, src: *const i16, dp: usize, adds: &[*const i16], subs: &[*const i16]) {
        let mut c = 0;
        while c < dp {
            let mut a0 = vld1q_s16(src.add(c));
            let mut a1 = vld1q_s16(src.add(c + 8));
            let mut a2 = vld1q_s16(src.add(c + 16));
            let mut a3 = vld1q_s16(src.add(c + 24));
            for &p in adds {
                let p = p.add(c);
                a0 = vaddq_s16(a0, vld1q_s16(p));
                a1 = vaddq_s16(a1, vld1q_s16(p.add(8)));
                a2 = vaddq_s16(a2, vld1q_s16(p.add(16)));
                a3 = vaddq_s16(a3, vld1q_s16(p.add(24)));
            }
            for &p in subs {
                let p = p.add(c);
                a0 = vsubq_s16(a0, vld1q_s16(p));
                a1 = vsubq_s16(a1, vld1q_s16(p.add(8)));
                a2 = vsubq_s16(a2, vld1q_s16(p.add(16)));
                a3 = vsubq_s16(a3, vld1q_s16(p.add(24)));
            }
            vst1q_s16(dst.add(c), a0);
            vst1q_s16(dst.add(c + 8), a1);
            vst1q_s16(dst.add(c + 16), a2);
            vst1q_s16(dst.add(c + 24), a3);
            c += 32;
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn clamp_u8(a: &[i16], x: *mut u8) {
        let n = a.len();
        let mut i = 0;
        while i + 16 <= n {
            let lo = vqmovun_s16(vld1q_s16(a.as_ptr().add(i)));
            let hi = vqmovun_s16(vld1q_s16(a.as_ptr().add(i + 8)));
            vst1q_u8(x.add(i), vcombine_u8(lo, hi));
            i += 16;
        }
        while i < n {
            *x.add(i) = a[i].clamp(0, 255) as u8;
            i += 1;
        }
    }

    #[inline(always)]
    unsafe fn group<const C: usize>(acc: &mut [int32x4_t; C], xv: int8x16_t, w: *const i8) {
        for k in 0..C {
            acc[k] = vdotq_laneq_s32::<0>(acc[k], vld1q_s8(w.add(16 * k)), xv);
        }
        for k in 0..C {
            acc[k] = vdotq_laneq_s32::<1>(acc[k], vld1q_s8(w.add(16 * (C + k))), xv);
        }
        for k in 0..C {
            acc[k] = vdotq_laneq_s32::<2>(acc[k], vld1q_s8(w.add(16 * (2 * C + k))), xv);
        }
        for k in 0..C {
            acc[k] = vdotq_laneq_s32::<3>(acc[k], vld1q_s8(w.add(16 * (3 * C + k))), xv);
        }
    }

    #[inline(always)]
    unsafe fn block<const C: usize, const S: usize>(xs: *const i8, ngroups: usize, wp: &mut *const i8, out: *mut i32) {
        let mut acc = [[vdupq_n_s32(0); C]; S];
        let step = 64 * C;
        let mut g = 0;
        while g + S <= ngroups {
            for s in 0..S {
                group::<C>(&mut acc[s], vld1q_s8(xs.add((g + s) * 16)), wp.add(s * step));
            }
            *wp = wp.add(S * step);
            g += S;
        }
        while g < ngroups {
            group::<C>(&mut acc[0], vld1q_s8(xs.add(g * 16)), *wp);
            *wp = wp.add(step);
            g += 1;
        }
        for k in 0..C {
            let mut a = acc[0][k];
            for s in 1..S {
                a = vaddq_s32(a, acc[s][k]);
            }
            vst1q_s32(out.add(k * 4), a);
        }
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn layer(l: &Layer, x: &[u8], s: *mut i32) {
        let inp = l.inp;
        let mut xsb = super::Un::<super::BufU8>::uninit();
        let xw = xsb.as_mut_ptr() as *mut u8;
        let m80 = vdupq_n_u8(0x80);
        let mut i = 0;
        while i < inp {
            vst1q_u8(xw.add(i), veorq_u8(vld1q_u8(x.as_ptr().add(i)), m80));
            i += 16;
        }
        let xs = xw as *const i8;
        let ng = inp / 16;
        let nob = l.outp / 4;
        let mut wp = l.wdot.as_ptr();
        let mut ob0 = 0;
        while ob0 < nob {
            let cnt = (nob - ob0).min(8);
            let out = s.add(ob0 * 4);
            match cnt {
                8 => block::<8, 2>(xs, ng, &mut wp, out),
                7 => block::<7, 2>(xs, ng, &mut wp, out),
                6 => block::<6, 2>(xs, ng, &mut wp, out),
                5 => block::<5, 2>(xs, ng, &mut wp, out),
                4 => block::<4, 3>(xs, ng, &mut wp, out),
                3 => block::<3, 4>(xs, ng, &mut wp, out),
                2 => block::<2, 4>(xs, ng, &mut wp, out),
                _ => block::<1, 4>(xs, ng, &mut wp, out),
            }
            ob0 += cnt;
        }
        for o in 0..l.outp {
            *s.add(o) = l.bias_s[o].wrapping_add(*s.add(o));
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::Layer;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn acc_apply(dst: *mut i16, src: *const i16, dp: usize, adds: &[*const i16], subs: &[*const i16]) {
        let mut c = 0;
        while c < dp {
            let mut a0 = _mm256_loadu_si256(src.add(c) as *const __m256i);
            let mut a1 = _mm256_loadu_si256(src.add(c + 16) as *const __m256i);
            for &p in adds {
                let p = p.add(c);
                a0 = _mm256_add_epi16(a0, _mm256_loadu_si256(p as *const __m256i));
                a1 = _mm256_add_epi16(a1, _mm256_loadu_si256(p.add(16) as *const __m256i));
            }
            for &p in subs {
                let p = p.add(c);
                a0 = _mm256_sub_epi16(a0, _mm256_loadu_si256(p as *const __m256i));
                a1 = _mm256_sub_epi16(a1, _mm256_loadu_si256(p.add(16) as *const __m256i));
            }
            _mm256_storeu_si256(dst.add(c) as *mut __m256i, a0);
            _mm256_storeu_si256(dst.add(c + 16) as *mut __m256i, a1);
            c += 32;
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn clamp_u8(a: &[i16], x: *mut u8) {
        let n = a.len();
        let mut i = 0;
        while i + 32 <= n {
            let lo = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let hi = _mm256_loadu_si256(a.as_ptr().add(i + 16) as *const __m256i);
            let p = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi16(lo, hi));
            _mm256_storeu_si256(x.add(i) as *mut __m256i, p);
            i += 32;
        }
        while i < n {
            *x.add(i) = a[i].clamp(0, 255) as u8;
            i += 1;
        }
    }

    #[inline(always)]
    unsafe fn block<const C: usize>(xp: *const u32, npairs: usize, wp: &mut *const i16, out: *mut i32) {
        let mut acc = [_mm256_setzero_si256(); C];
        for ip in 0..npairs {
            let xv = _mm256_set1_epi32(*xp.add(ip) as i32);
            for k in 0..C {
                acc[k] = _mm256_add_epi32(acc[k], _mm256_madd_epi16(xv, _mm256_loadu_si256(*wp as *const __m256i)));
                *wp = wp.add(16);
            }
        }
        for k in 0..C {
            _mm256_storeu_si256(out.add(k * 8) as *mut __m256i, acc[k]);
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn layer(l: &Layer, x: &[u8], s: *mut i32) {
        let inp = l.inp;
        #[repr(C, align(64))]
        struct B16([i16; super::NN_MAX_IN]);
        let mut x16 = super::Un::<B16>::uninit();
        let xw = x16.as_mut_ptr() as *mut i16;
        let mut i = 0;
        while i < inp {
            let v = _mm_loadu_si128(x.as_ptr().add(i) as *const __m128i);
            _mm256_storeu_si256(xw.add(i) as *mut __m256i, _mm256_cvtepu8_epi16(v));
            i += 16;
        }
        let xp = xw as *const u32;
        let np = inp / 2;
        let nreg = l.outp / 8;
        let mut wp = l.w16.as_ptr();
        let mut r0 = 0;
        while r0 < nreg {
            let cnt = (nreg - r0).min(8);
            let out = s.add(r0 * 8);
            match cnt {
                8 => block::<8>(xp, np, &mut wp, out),
                7 => block::<7>(xp, np, &mut wp, out),
                6 => block::<6>(xp, np, &mut wp, out),
                5 => block::<5>(xp, np, &mut wp, out),
                4 => block::<4>(xp, np, &mut wp, out),
                3 => block::<3>(xp, np, &mut wp, out),
                2 => block::<2>(xp, np, &mut wp, out),
                _ => block::<1>(xp, np, &mut wp, out),
            }
            r0 += cnt;
        }
        for o in 0..l.outp {
            *s.add(o) = l.bias[o].wrapping_add(*s.add(o));
        }
    }
}
