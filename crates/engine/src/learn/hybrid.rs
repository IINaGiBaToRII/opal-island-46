
use super::arch::*;
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

pub const MAX_FAST_WIDTH: usize = 512;

#[derive(Clone, Copy)]
struct Ent {
    sh: u8,
    toff: u32,
}

#[derive(Clone, Copy)]
struct PatTab {
    e0: u32,
    e1: u32,
    alph5: bool,
    vbase: u32,
}

pub(crate) struct IdxTabs {
    pats: [Vec<PatTab>; 2],
    ents: [Vec<Ent>; 2],
    tab: Vec<u32>,
}

impl IdxTabs {
    fn new(arch: &Arch, l: &Layout, t: &ViewTables) -> IdxTabs {
        let mut tab: Vec<u32> = vec![];
        let (mut pats, mut ents) = ([vec![], vec![]], [vec![], vec![]]);
        for o in 0..2usize {
            for p in &arch.patterns {
                let e0 = ents[o].len() as u32;
                let mut by_chunk: std::collections::BTreeMap<usize, Vec<(usize, u32)>> = Default::default();
                let mut mul = 1u32;
                for &s in &p.squares {
                    let bit = t.from_dense[o][s as usize] as usize;
                    by_chunk.entry(bit / 8).or_default().push((bit % 8, mul));
                    mul *= p.alphabet as u32;
                }
                for (c, list) in by_chunk {
                    let toff = tab.len() as u32;
                    for board in 0..4u32 {
                        for x in 0..256usize {
                            let mut v = 0u32;
                            for &(b, m) in &list {
                                if x >> b & 1 != 0 {
                                    v += (board + 1) * m;
                                }
                            }
                            tab.push(v);
                        }
                    }
                    ents[o].push(Ent { sh: (c * 8) as u8, toff });
                }
                pats[o].push(PatTab { e0, e1: ents[o].len() as u32, alph5: p.alphabet == 5, vbase: l.v_off[p.table] as u32 });
            }
        }
        IdxTabs { pats, ents, tab }
    }

    #[inline(always)]
    fn index(&self, o: usize, p: &PatTab, b: &[u64; 4]) -> usize {
        let mut ix = 0u32;
        for e in &self.ents[o][p.e0 as usize..p.e1 as usize] {
            let (sh, t) = (e.sh as u32, e.toff as usize);
            ix += self.tab[t + (b[0] >> sh & 255) as usize] + self.tab[t + 256 + (b[1] >> sh & 255) as usize];
            if p.alph5 {
                ix += self.tab[t + 512 + (b[2] >> sh & 255) as usize] + self.tab[t + 768 + (b[3] >> sh & 255) as usize];
            }
        }
        ix as usize
    }
}

#[inline(always)]
fn boards<R: Rules>(pos: &Position<R>, o: usize) -> [u64; 4] {
    let (own, opp) = if o == 0 { (pos.white, pos.black) } else { (pos.black, pos.white) };
    let k = pos.kings;
    [own & !k, opp & !k, own & k, opp & k]
}

pub struct HybridEval<R: Rules> {
    model: Arc<Model>,
    refe: RefEval<R>,
    lin: Option<LinearEval<R>>,
    tabs: IdxTabs,
    d: usize,
    nviews: usize,
    safe16: bool,
    wmax: usize,
    _r: PhantomData<fn() -> R>,
}

impl<R: Rules> HybridEval<R> {
    pub fn new(model: Model) -> Result<Self, String> {
        Self::from_arc(Arc::new(model))
    }

    pub fn from_arc(model: Arc<Model>) -> Result<Self, String> {
        let refe = RefEval::<R>::from_arc(model.clone())?;
        let dp = model.arch.deep.as_ref().ok_or("HybridEval: у архитектуры нет глубокой части (используйте LinearEval)")?;
        let l = &model.layout;
        if dp.width > MAX_FAST_WIDTH || dp.width * l.nviews > MAX_FAST_WIDTH || l.dims.iter().any(|&(i, o)| i > MAX_FAST_WIDTH || o > MAX_FAST_WIDTH) {
            return Err(format!("HybridEval: ширина слоёв больше {MAX_FAST_WIDTH}"));
        }
        let t = ViewTables::new::<R>();
        let tabs = IdxTabs::new(&model.arch, l, &t);
        let lin = if model.arch.wide || !model.arch.globals.is_empty() {
            let mut a = model.arch.clone();
            a.deep = None;
            a.views = vec![ViewKind::Stm];
            let mut m2 = Model::zeros(a)?;
            m2.wide = model.wide.clone();
            m2.globals = model.globals.clone();
            m2.meta = model.meta.clone();
            LinearEval::<R>::new(m2).ok()
        } else {
            None
        };
        let d = dp.width;
        let mut bound = vec![0i64; d];
        for (j, b) in bound.iter_mut().enumerate() {
            *b = (model.deep_b[j] as i64).abs();
        }
        for p in &model.arch.patterns {
            let size = model.arch.tables[p.table].size;
            let base = l.v_off[p.table];
            for j in 0..d {
                let mx = (0..size).map(|i| (model.deep_v[base + i * d + j] as i64).abs()).max().unwrap_or(0);
                bound[j] += mx;
            }
        }
        let safe16 = bound.iter().all(|&b| b <= 32767);
        let need = l.dims.iter().map(|&(i, o)| i.max(o)).max().unwrap_or(0).max(d * l.nviews);
        let wmax = [32usize, 64, 128, 256, 512].into_iter().find(|&w| w >= need).unwrap_or(512);
        Ok(HybridEval { nviews: l.nviews, d, model, refe, lin, tabs, safe16, wmax, _r: PhantomData })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn acc_is_i16(&self) -> bool {
        self.safe16
    }

    #[inline(always)]
    fn wide_cp(&self, pos: &Position<R>) -> i32 {
        match &self.lin {
            Some(lin) => {
                let w = lin.eval_full(pos);
                if w.abs() < EVAL_LIMIT {
                    w
                } else {
                    self.refe.wide_cp(&self.refe.features(pos))
                }
            }
            None if self.model.arch.wide || !self.model.arch.globals.is_empty() => self.refe.wide_cp(&self.refe.features(pos)),
            None => 0,
        }
    }

    #[inline(always)]
    fn view_x<const W: usize>(&self, pos: &Position<R>, o: usize, xs: &mut [i16]) {
        let m = &*self.model;
        let d = self.d;
        let b = boards(pos, o);
        if self.safe16 {
            let mut acc = [0i16; W];
            acc[..d].copy_from_slice(&m.deep_b);
            for p in &self.tabs.pats[o] {
                let ix = self.tabs.index(o, p, &b);
                let row = &m.deep_v[p.vbase as usize + ix * d..][..d];
                for (a, &r) in acc[..d].iter_mut().zip(row) {
                    *a = a.wrapping_add(r);
                }
            }
            for (xo, &a) in xs.iter_mut().zip(&acc[..d]) {
                *xo = a.clamp(0, 255);
            }
        } else {
            let mut acc = [0i32; W];
            for (a, &bb) in acc[..d].iter_mut().zip(&m.deep_b) {
                *a = bb as i32;
            }
            for p in &self.tabs.pats[o] {
                let ix = self.tabs.index(o, p, &b);
                let row = &m.deep_v[p.vbase as usize + ix * d..][..d];
                for (a, &r) in acc[..d].iter_mut().zip(row) {
                    *a += r as i32;
                }
            }
            for (xo, &a) in xs.iter_mut().zip(&acc[..d]) {
                *xo = a.clamp(0, 255) as i16;
            }
        }
    }

    #[inline(always)]
    fn tail<const W: usize>(&self, mut x: [i16; W]) -> i32 {
        let m = &*self.model;
        let dims = &m.layout.dims;
        let nl = dims.len();
        for li in 0..nl {
            let (inn, out) = dims[li];
            let w = &m.deep_w[li];
            let bias = &m.deep_bias[li];
            let mut y = [0i16; W];
            for o in 0..out {
                let wr = &w[o * inn..(o + 1) * inn];
                let mut s = 0i32;
                for (a, b) in wr.iter().zip(&x[..inn]) {
                    s += *a as i32 * *b as i32;
                }
                let s = bias[o].wrapping_add(s);
                if li + 1 < nl {
                    y[o] = (s >> 6).clamp(0, 255) as i16;
                } else {
                    return s >> 6;
                }
            }
            x = y;
        }
        0
    }

    #[inline(always)]
    fn deep_w<const W: usize>(&self, pos: &Position<R>) -> i32 {
        let d = self.d;
        let o_stm = if pos.side == Side::White { 0 } else { 1 };
        let mut x = [0i16; W];
        for v in 0..self.nviews {
            self.view_x::<W>(pos, o_stm ^ v, &mut x[v * d..(v + 1) * d]);
        }
        self.tail::<W>(x)
    }

    #[inline(always)]
    fn deep_cp(&self, pos: &Position<R>) -> i32 {
        match self.wmax {
            32 => self.deep_w::<32>(pos),
            64 => self.deep_w::<64>(pos),
            128 => self.deep_w::<128>(pos),
            256 => self.deep_w::<256>(pos),
            _ => self.deep_w::<512>(pos),
        }
    }

    #[inline]
    pub fn eval_full(&self, pos: &Position<R>) -> i32 {
        self.wide_cp(pos).wrapping_add(self.deep_cp(pos)).clamp(-EVAL_LIMIT, EVAL_LIMIT)
    }
}

impl<R: Rules> Evaluator<R> for HybridEval<R> {
    type State = ();
    #[inline(always)]
    fn root(&self, _pos: &Position<R>) {}
    #[inline(always)]
    fn child(&self, _p: &(), _b: &Position<R>, _mv: &Move, _a: &Position<R>) {}
    #[inline]
    fn eval(&self, pos: &Position<R>, _st: &mut ()) -> i32 {
        self.eval_full(pos)
    }
}

const APPLY_MAX_BITS: u32 = 10;

pub const INC_MAX_D: usize = 256;

#[derive(Clone)]
pub struct HybState {
    kind: u8,
    bw: u64,
    bb: u64,
    bk: u64,
    acc: [[i16; INC_MAX_D]; 2],
}

impl HybState {
    pub fn is_full(&self) -> bool {
        self.kind == 2
    }
    pub fn is_pending(&self) -> bool {
        self.kind == 1
    }
    pub fn diff_bits<R: Rules>(&self, pos: &Position<R>) -> u32 {
        ((self.bw ^ pos.white) | (self.bb ^ pos.black) | (self.bk ^ pos.kings)).count_ones()
    }
}

pub struct HybridEvalInc<R: Rules> {
    inner: HybridEval<R>,
    ok: bool,
    pat_of_bit: [[u8; 64]; 2],
    bit_off: [[(u32, bool); 64]; 2],
    dtab: Vec<i16>,
    base0: Vec<i16>,
}

impl<R: Rules> HybridEvalInc<R> {
    pub fn new(model: Model) -> Result<Self, String> {
        Self::from_arc(Arc::new(model))
    }

    pub fn from_arc(model: Arc<Model>) -> Result<Self, String> {
        let inner = HybridEval::<R>::from_arc(model)?;
        Ok(Self::wrap(inner))
    }

    pub fn wrap(inner: HybridEval<R>) -> Self {
        let m = &inner.model;
        let singles = m.arch.patterns.iter().all(|p| p.squares.len() == 1) && m.arch.patterns.len() <= 254;
        let ok = singles && inner.safe16 && inner.d <= INC_MAX_D;
        let mut pat_of_bit = [[255u8; 64]; 2];
        if ok {
            let t = ViewTables::new::<R>();
            for o in 0..2 {
                for (i, p) in m.arch.patterns.iter().enumerate() {
                    pat_of_bit[o][t.from_dense[o][p.squares[0] as usize] as usize] = i as u8;
                }
            }
        }
        let d = inner.d;
        let mut bit_off = [[(u32::MAX, false); 64]; 2];
        let mut dtab: Vec<i16> = vec![];
        let mut base0: Vec<i16> = vec![];
        if ok {
            let t = ViewTables::new::<R>();
            base0 = m.deep_b.clone();
            for (i, p) in m.arch.patterns.iter().enumerate() {
                let vb = m.layout.v_off[p.table];
                let zero = &m.deep_v[vb..vb + d];
                for (a, &z) in base0.iter_mut().zip(zero) {
                    *a = a.wrapping_add(z);
                }
                for dg in 0..5usize {
                    let row: Vec<i16> = if dg < p.alphabet as usize {
                        m.deep_v[vb + dg * d..vb + (dg + 1) * d].iter().zip(zero).map(|(&v, &z)| v.wrapping_sub(z)).collect()
                    } else {
                        vec![0; d]
                    };
                    dtab.extend_from_slice(&row);
                }
                for o in 0..2 {
                    bit_off[o][t.from_dense[o][p.squares[0] as usize] as usize] = ((i * 5 * d) as u32, p.alphabet == 5);
                }
            }
        }
        HybridEvalInc { inner, ok, pat_of_bit, bit_off, dtab, base0 }
    }

    pub fn inner(&self) -> &HybridEval<R> {
        &self.inner
    }

    pub fn is_incremental(&self) -> bool {
        self.ok
    }

    #[inline(always)]
    fn digit(alph5: bool, w: u64, b: u64, kings: u64, o: usize, bit: usize) -> usize {
        let (own, opp) = if o == 0 { (w, b) } else { (b, w) };
        let k = kings >> bit & 1 != 0;
        if own >> bit & 1 != 0 {
            if !k {
                1
            } else if alph5 {
                3
            } else {
                0
            }
        } else if opp >> bit & 1 != 0 {
            if !k {
                2
            } else if alph5 {
                4
            } else {
                0
            }
        } else {
            0
        }
    }

    fn empty() -> HybState {
        HybState { kind: 0, bw: 0, bb: 0, bk: 0, acc: [[0; INC_MAX_D]; 2] }
    }

    fn compute_both(&self, pos: &Position<R>, st: &mut HybState) {
        let d = self.inner.d;
        let occ = pos.white | pos.black;
        for o in 0..2usize {
            let acc = &mut st.acc[o][..d];
            acc.copy_from_slice(&self.base0);
            let mut x = occ;
            while x != 0 {
                let bit = x.trailing_zeros() as usize;
                x &= x - 1;
                let (off, alph5) = self.bit_off[o][bit];
                if off == u32::MAX {
                    continue;
                }
                let dg = Self::digit(alph5, pos.white, pos.black, pos.kings, o, bit);
                if dg == 0 {
                    continue;
                }
                let row = &self.dtab[off as usize + dg * d..][..d];
                for (a, &r) in acc.iter_mut().zip(row) {
                    *a = a.wrapping_add(r);
                }
            }
        }
        st.kind = 2;
    }

    fn apply(&self, pos: &Position<R>, st: &mut HybState) {
        let d = self.inner.d;
        let diff = (st.bw ^ pos.white) | (st.bb ^ pos.black) | (st.bk ^ pos.kings);
        for o in 0..2usize {
            let mut x = diff;
            while x != 0 {
                let bit = x.trailing_zeros() as usize;
                x &= x - 1;
                let (off, alph5) = self.bit_off[o][bit];
                if off == u32::MAX {
                    continue;
                }
                let (d0, d1) = (Self::digit(alph5, st.bw, st.bb, st.bk, o, bit), Self::digit(alph5, pos.white, pos.black, pos.kings, o, bit));
                if d0 == d1 {
                    continue;
                }
                let (r0, r1) = (&self.dtab[off as usize + d0 * d..][..d], &self.dtab[off as usize + d1 * d..][..d]);
                for ((a, &x0), &x1) in st.acc[o][..d].iter_mut().zip(r0).zip(r1) {
                    *a = a.wrapping_sub(x0).wrapping_add(x1);
                }
            }
        }
        st.kind = 2;
    }

    #[inline(always)]
    fn finish_w<const W: usize>(&self, pos: &Position<R>, st: &HybState) -> i32 {
        let d = self.inner.d;
        let o_stm = if pos.side == Side::White { 0 } else { 1 };
        let mut x = [0i16; W];
        for v in 0..self.inner.nviews {
            let a = &st.acc[o_stm ^ v][..d];
            for (xo, &s) in x[v * d..(v + 1) * d].iter_mut().zip(a) {
                *xo = s.clamp(0, 255);
            }
        }
        self.inner.wide_cp(pos).wrapping_add(self.inner.tail::<W>(x)).clamp(-EVAL_LIMIT, EVAL_LIMIT)
    }
}

impl<R: Rules> Evaluator<R> for HybridEvalInc<R> {
    type State = HybState;
    #[inline]
    fn root(&self, _pos: &Position<R>) -> HybState {
        Self::empty()
    }
    #[inline]
    fn child(&self, parent: &HybState, before: &Position<R>, _mv: &Move, _after: &Position<R>) -> HybState {
        if self.ok && parent.kind == 2 {
            HybState { kind: 1, bw: before.white, bb: before.black, bk: before.kings, acc: parent.acc }
        } else if self.ok && parent.kind == 1 {
            parent.clone()
        } else {
            Self::empty()
        }
    }
    #[inline]
    fn eval(&self, pos: &Position<R>, st: &mut HybState) -> i32 {
        if !self.ok {
            return self.inner.eval_full(pos);
        }
        match st.kind {
            2 => {}
            1 if ((st.bw ^ pos.white) | (st.bb ^ pos.black) | (st.bk ^ pos.kings)).count_ones() <= APPLY_MAX_BITS => self.apply(pos, st),
            _ => self.compute_both(pos, st),
        }
        match self.inner.wmax {
            32 => self.finish_w::<32>(pos, st),
            64 => self.finish_w::<64>(pos, st),
            128 => self.finish_w::<128>(pos, st),
            256 => self.finish_w::<256>(pos, st),
            _ => self.finish_w::<512>(pos, st),
        }
    }
    fn eval_fresh(&self, pos: &Position<R>) -> i32 {
        self.inner.eval_full(pos)
    }
}
