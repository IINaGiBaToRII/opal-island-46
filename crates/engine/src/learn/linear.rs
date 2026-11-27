
use super::arch::*;
use super::globals::quiet_moves;
use super::view::ViewTables;
use super::weights::Model;
use crate::eval::Evaluator;
use crate::movegen::Move;
use crate::position::{Position, Side};
use crate::rules::Rules;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

pub const INC_MAX_PATTERNS: usize = 24;

#[derive(Clone, Copy)]
struct Grp {
    mask: u64,
    shift: u32,
}

struct PatPlan {
    base: usize,
    stride: usize,
    alph: u8,
    tab: [u32; 2],
    gs: [(u32, u32); 2],
}

#[derive(Clone, Copy)]
struct Ent {
    pat: u16,
    alph: u8,
    mul: u32,
}

struct Plan {
    wide: bool,
    shift: u32,
    phase: Phase,
    n: i32,
    pats: Vec<PatPlan>,
    tabs: Vec<Vec<u32>>,
    groups: Vec<Grp>,
    g_men: Option<usize>,
    g_kown: Option<usize>,
    g_kopp: Option<usize>,
    g_ksq: Option<usize>,
    g_kmob: Option<usize>,
    g_mob: Option<usize>,
    g_tempo: Option<usize>,
    g_bal: Option<usize>,
    g_row: usize,
    has_globals: bool,
    need_adv: bool,
    need_mob: bool,
    dense: [[u8; 64]; 2],
    row: [[u8; 64]; 2],
    left: [u64; 2],
    ent_range: [[(u32, u32); 64]; 2],
    ents: [Vec<Ent>; 2],
    inc_ok: bool,
    avg_groups: f64,
    ntabs: usize,
}

struct Boards {
    om: u64,
    pm: u64,
    ok: u64,
    pk: u64,
    own: u64,
    opp: u64,
    kings: bool,
}

#[inline(always)]
fn gather(gs: &[Grp], x: u64) -> usize {
    let mut r = 0u64;
    for g in gs {
        r |= (x & g.mask) >> g.shift;
    }
    r as usize
}

impl Plan {
    fn new<R: Rules>(model: &Model) -> Result<Plan, String> {
        let a = &model.arch;
        if a.deep.is_some() {
            return Err("LinearEval не поддерживает глубокую часть (deep); нужен RefEval".into());
        }
        let t = ViewTables::new::<R>();
        let l = &model.layout;
        let mut groups: Vec<Grp> = vec![];
        let mut tabs: Vec<Vec<u32>> = vec![];
        let mut tab_ids: HashMap<(u8, Vec<u8>), u32> = HashMap::new();
        let mut pats = vec![];
        let mut total_groups = 0usize;
        if a.wide {
            for p in &a.patterns {
                let len = p.squares.len();
                let mut tab = [0u32; 2];
                let mut gs = [(0u32, 0u32); 2];
                for o in 0..2 {
                    let mut f: Vec<(u8, u8)> = p.squares.iter().enumerate().map(|(j, &s)| (t.from_dense[o][s as usize], j as u8)).collect();
                    f.sort();
                    let perm: Vec<u8> = f.iter().map(|x| x.1).collect();
                    let start = groups.len() as u32;
                    let mut k = 0usize;
                    while k < len {
                        let sh = f[k].0 as u32 - k as u32;
                        let mut mask = 0u64;
                        while k < len && f[k].0 as u32 - k as u32 == sh {
                            mask |= 1u64 << f[k].0;
                            k += 1;
                        }
                        groups.push(Grp { mask, shift: sh });
                    }
                    gs[o] = (start, groups.len() as u32);
                    total_groups += groups.len() - start as usize;
                    let key = (p.alphabet, perm.clone());
                    let id = *tab_ids.entry(key).or_insert_with(|| {
                        let mut v = vec![0u32; 1usize << len];
                        for (x, e) in v.iter_mut().enumerate() {
                            let mut s = 0u32;
                            for (b, &j) in perm.iter().enumerate() {
                                if (x >> b) & 1 != 0 {
                                    s += (p.alphabet as u32).pow(j as u32);
                                }
                            }
                            *e = s;
                        }
                        tabs.push(v);
                        (tabs.len() - 1) as u32
                    });
                    tab[o] = id;
                }
                pats.push(PatPlan { base: l.wide_off[p.table], stride: a.tables[p.table].size, alph: p.alphabet, tab, gs });
            }
        }
        let off = |g: super::globals::Global| l.globals.iter().position(|&x| x == g).map(|i| l.g_off[i]);
        use super::globals::Global::*;
        let (g_tempo, g_kmob, g_mob) = (off(Tempo), off(KingMobility), off(Mobility));
        let need_adv = g_tempo.is_some() || (a.phase.kind != PhaseKind::None && a.phase.by == PhaseBy::Tempo);
        let n = t.n as i32;
        let mut dense = [[255u8; 64]; 2];
        let mut row = [[0u8; 64]; 2];
        let mut left = [0u64; 2];
        for o in 0..2 {
            for b in 0..64 {
                let d = t.to_dense[o][b];
                dense[o][b] = d;
                if d != 255 {
                    row[o][b] = t.row[d as usize];
                    if (t.file[d as usize] as usize) < t.n / 2 {
                        left[o] |= 1u64 << b;
                    }
                }
            }
        }
        let mut ent_range = [[(0u32, 0u32); 64]; 2];
        let mut ents: [Vec<Ent>; 2] = [vec![], vec![]];
        let inc_ok = a.wide && pats.len() <= INC_MAX_PATTERNS;
        if inc_ok {
            for o in 0..2 {
                let mut per: Vec<Vec<Ent>> = vec![vec![]; 64];
                for (pi, p) in a.patterns.iter().enumerate() {
                    for (j, &s) in p.squares.iter().enumerate() {
                        let r = t.from_dense[o][s as usize] as usize;
                        per[r].push(Ent { pat: pi as u16, alph: p.alphabet, mul: (p.alphabet as u32).pow(j as u32) });
                    }
                }
                for (r, v) in per.into_iter().enumerate() {
                    let s = ents[o].len() as u32;
                    ents[o].extend(v);
                    ent_range[o][r] = (s, ents[o].len() as u32);
                }
            }
        }
        Ok(Plan {
            wide: a.wide,
            shift: PHASE_SHIFT + a.quant.wide.trailing_zeros(),
            phase: a.phase.clone(),
            n,
            avg_groups: if pats.is_empty() { 0.0 } else { total_groups as f64 / (2 * pats.len()) as f64 },
            ntabs: tabs.len(),
            pats,
            tabs,
            groups,
            g_men: off(MenDiff),
            g_kown: off(KingsOwn),
            g_kopp: off(KingsOpp),
            g_ksq: off(KingSq),
            g_kmob,
            g_mob,
            g_tempo,
            g_bal: off(Balance),
            g_row: l.g_row,
            has_globals: !l.globals.is_empty(),
            need_adv,
            need_mob: g_kmob.is_some() || g_mob.is_some(),
            dense,
            row,
            left,
            ent_range,
            ents,
            inc_ok,
        })
    }
}

const MAGIC_MAXLEN: usize = 16;

#[derive(Clone, Copy)]
struct FPat {
    mask: u64,
    magic: u64,
    sh: u32,
    alph5: bool,
    toff: usize,
    poff: usize,
}

struct Fast {
    pats: [Vec<FPat>; 2],
    tabs: Vec<u32>,
    packed: Vec<i64>,
    p: usize,
    gpk: Vec<i64>,
    gr: usize,
    ptab: Vec<(usize, i32, i32)>,
    mode: u8,
    need_adv: bool,
    n1: i32,
    rm: [[u64; 4]; 2],
    rbits: usize,
    ksq: [[u32; 64]; 2],
}

fn find_magic(mask: u64, subs: &[(u64, u32)], k: u32, tries: u32, rng: &mut u64) -> Option<(u64, Vec<u32>)> {
    let size = 1usize << k;
    let mut stamp = vec![0u32; size];
    for att in 1..=tries {
        let mut r = || {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            *rng
        };
        let magic = r() & r() & r();
        if (mask.wrapping_mul(magic) >> 56).count_ones() < 2 && mask.count_ones() > 3 {
            continue;
        }
        let mut ok = true;
        for &(x, _) in subs {
            let h = (x.wrapping_mul(magic) >> (64 - k)) as usize;
            if stamp[h] == att {
                ok = false;
                break;
            }
            stamp[h] = att;
        }
        if ok {
            let mut t = vec![0u32; size];
            for &(x, c) in subs {
                t[(x.wrapping_mul(magic) >> (64 - k)) as usize] = c;
            }
            return Some((magic, t));
        }
    }
    None
}

impl Fast {
    fn new(pl: &Plan, m: &Model, t: &ViewTables) -> Option<Fast> {
        let a = &m.arch;
        let l = &m.layout;
        let ph = &pl.phase;
        let kst = if ph.kind == PhaseKind::None { 1usize } else { ph.k as usize };
        let interp = ph.kind != PhaseKind::None && ph.kind != PhaseKind::Buckets;
        if interp && kst < 2 || ph.kind != PhaseKind::None && ph.norm <= 0 {
            return None;
        }
        let p = if ph.kind == PhaseKind::None { 1 } else if interp { kst - 1 } else { kst };
        if (pl.pats.len() + 1200) as u64 * 32768 >= (1u64 << 31) {
            return None;
        }
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut pats: [Vec<FPat>; 2] = [vec![], vec![]];
        let mut tabs: Vec<u32> = vec![];
        let mut packed: Vec<i64> = vec![];
        let mut poffs: HashMap<usize, usize> = HashMap::new();
        if a.wide {
            for pt in &a.patterns {
                let len = pt.squares.len();
                if len > MAGIC_MAXLEN || len == 0 {
                    return None;
                }
                let size = a.tables[pt.table].size;
                if size != (pt.alphabet as usize).pow(len as u32) {
                    return None;
                }
                let poff = match poffs.get(&pt.table) {
                    Some(&o) => o,
                    None => {
                        let o = packed.len();
                        let base = l.wide_off[pt.table];
                        for ix in 0..size {
                            for s in 0..p {
                                let w0 = *m.wide.get(base + s * size + ix)? as i64;
                                let w1 = if interp { *m.wide.get(base + (s + 1) * size + ix)? as i64 } else { 0 };
                                packed.push(w0.wrapping_add(w1 << 32));
                            }
                        }
                        poffs.insert(pt.table, o);
                        o
                    }
                };
                for o in 0..2 {
                    let mut mask = 0u64;
                    let mut pw = [0u32; 64];
                    for (j, &s) in pt.squares.iter().enumerate() {
                        let raw = t.from_dense[o][s as usize] as usize;
                        mask |= 1u64 << raw;
                        pw[raw] = (pt.alphabet as u32).pow(j as u32);
                    }
                    if mask.count_ones() as usize != len {
                        return None;
                    }
                    let mut subs: Vec<(u64, u32)> = Vec::with_capacity(1 << len);
                    let mut sub = 0u64;
                    loop {
                        let mut c = 0u32;
                        let mut y = sub;
                        while y != 0 {
                            c += pw[y.trailing_zeros() as usize];
                            y &= y - 1;
                        }
                        subs.push((sub, c));
                        sub = sub.wrapping_sub(mask) & mask;
                        if sub == 0 {
                            break;
                        }
                    }
                    let mut found = None;
                    for (extra, tries) in [(0u32, 3000u32), (1, 30000), (2, 100000), (3, 300000)] {
                        let k = len as u32 + extra;
                        if let Some((mg, tb)) = find_magic(mask, &subs, k, tries, &mut rng) {
                            found = Some((mg, k, tb));
                            break;
                        }
                    }
                    let (magic, k, tb) = found?;
                    let toff = tabs.len();
                    tabs.extend_from_slice(&tb);
                    pats[o].push(FPat { mask, magic, sh: 64 - k, alph5: pt.alphabet == 5, toff, poff });
                }
            }
        }
        let gr = l.g_row;
        let mut gpk: Vec<i64> = vec![];
        if gr > 0 {
            for s in 0..p {
                for off in 0..gr {
                    let w0 = *m.globals.get(s * gr + off)? as i64;
                    let w1 = if interp { *m.globals.get((s + 1) * gr + off)? as i64 } else { 0 };
                    gpk.push(w0.wrapping_add(w1 << 32));
                }
            }
        }
        let one = PHASE_ONE as i64;
        let from_s = |s: i32| -> (usize, i32, i32) {
            let k = ph.k as i32;
            if ph.kind == PhaseKind::Buckets {
                ((((s * k) / (PHASE_ONE + 1)).min(k - 1)) as usize, PHASE_ONE, 0)
            } else {
                let x = s * (k - 1);
                let i = (x / PHASE_ONE).min(k - 2);
                let frac = x - i * PHASE_ONE;
                (i as usize, PHASE_ONE - frac, frac)
            }
        };
        let n = t.n as i32;
        let (mode, ptab) = if ph.kind == PhaseKind::None {
            (0u8, vec![(0usize, PHASE_ONE, 0i32)])
        } else if ph.by == PhaseBy::Pieces {
            let norm = ph.norm as i64;
            (1u8, (0..=64i64).map(|c| from_s((one - (one * c / norm).min(one)) as i32)).collect())
        } else {
            let norm = ph.norm as i64;
            (2u8, (0..=(64 * (n as i64 - 1).max(1))).map(|adv| from_s((one * adv / norm).min(one) as i32)).collect())
        };
        let mut rm = [[0u64; 4]; 2];
        let mut ksq = [[0u32; 64]; 2];
        for o in 0..2 {
            for b in 0..64 {
                let d = pl.dense[o][b];
                if d == 255 {
                    continue;
                }
                let r = pl.row[o][b] as usize;
                if r >= 16 {
                    return None;
                }
                for kb in 0..4 {
                    if (r >> kb) & 1 != 0 {
                        rm[o][kb] |= 1u64 << b;
                    }
                }
                if let Some(off) = pl.g_ksq {
                    ksq[o][b] = (off + d as usize) as u32;
                }
            }
        }
        let rbits = (32 - ((n - 1).max(1) as u32).leading_zeros()) as usize;
        Some(Fast { pats, tabs, packed, p, gpk, gr, ptab, mode, need_adv: pl.need_adv, n1: n - 1, rm, rbits, ksq })
    }

    #[inline(always)]
    fn row_sum(&self, o: usize, x: u64) -> i32 {
        let mut s = 0i32;
        for kb in 0..self.rbits {
            s += (((x & self.rm[o][kb]).count_ones()) as i32) << kb;
        }
        s
    }
}

pub struct LinearEval<R: Rules> {
    model: Arc<Model>,
    plan: Plan,
    fast: Option<Fast>,
    _r: PhantomData<fn() -> R>,
}

impl<R: Rules> LinearEval<R> {
    pub fn new(model: Model) -> Result<Self, String> {
        Self::from_arc(Arc::new(model))
    }

    pub fn from_arc(model: Arc<Model>) -> Result<Self, String> {
        model.arch.validate()?;
        model.check_shapes()?;
        model.check_rules::<R>()?;
        let t = ViewTables::new::<R>();
        if model.layout.nsq != t.nsq {
            return Err("число полей геометрии не совпадает с описанием".into());
        }
        let plan = Plan::new::<R>(&model)?;
        let fast = Fast::new(&plan, &model, &t);
        Ok(LinearEval { model, plan, fast, _r: PhantomData })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn stats(&self) -> (f64, usize) {
        (self.plan.avg_groups, self.plan.ntabs)
    }

    #[inline(always)]
    fn boards(pos: &Position<R>) -> (usize, Boards) {
        let (own, opp, o) = if pos.side == Side::White { (pos.white, pos.black, 0) } else { (pos.black, pos.white, 1) };
        let k = pos.kings;
        (o, Boards { om: own & !k, pm: opp & !k, ok: own & k, pk: opp & k, own, opp, kings: k != 0 })
    }

    #[inline(always)]
    fn pat_index(&self, p: &PatPlan, o: usize, b: &Boards) -> u32 {
        let (s, e) = p.gs[o];
        let gs = &self.plan.groups[s as usize..e as usize];
        let t = &self.plan.tabs[p.tab[o] as usize];
        let mut idx = t[gather(gs, b.om)] + 2 * t[gather(gs, b.pm)];
        if p.alph == 5 && b.kings {
            idx += 3 * t[gather(gs, b.ok)] + 4 * t[gather(gs, b.pk)];
        }
        idx
    }

    #[inline(always)]
    fn row_sum(&self, o: usize, mut x: u64) -> i32 {
        let mut s = 0i32;
        while x != 0 {
            s += self.plan.row[o][x.trailing_zeros() as usize] as i32;
            x &= x - 1;
        }
        s
    }

    #[inline(always)]
    fn finish<F: Fn(usize) -> u32>(&self, pos: &Position<R>, o: usize, b: &Boards, f: F) -> i32 {
        let pl = &self.plan;
        let m = &*self.model;
        let (mut a_own, mut a_opp) = (0i32, 0i32);
        if pl.need_adv {
            a_own = self.row_sum(o, b.om);
            a_opp = b.pm.count_ones() as i32 * (pl.n - 1) - self.row_sum(o, b.pm);
        }
        let (st0, w0, st1, w1);
        if pl.phase.kind == PhaseKind::None {
            (st0, w0, st1, w1) = (0usize, PHASE_ONE, 0usize, 0i32);
        } else {
            let one = PHASE_ONE as i64;
            let norm = pl.phase.norm as i64;
            let s = match pl.phase.by {
                PhaseBy::Pieces => one - (one * ((b.own | b.opp).count_ones() as i64) / norm).min(one),
                PhaseBy::Tempo => {
                    let adv = a_own as i64 + a_opp as i64 + pos.kings.count_ones() as i64 * (pl.n as i64 - 1);
                    (one * adv / norm).min(one)
                }
            } as i32;
            let k = pl.phase.k as i32;
            if pl.phase.kind == PhaseKind::Buckets {
                (st0, w0, st1, w1) = ((((s * k) / (PHASE_ONE + 1)).min(k - 1)) as usize, PHASE_ONE, 0usize, 0i32);
            } else {
                let x = s * (k - 1);
                let i = (x / PHASE_ONE).min(k - 2);
                let frac = x - i * PHASE_ONE;
                (st0, w0, st1, w1) = (i as usize, PHASE_ONE - frac, i as usize + 1, frac);
            }
        }
        let (mut s0, mut s1) = (0i32, 0i32);
        if pl.wide {
            for (i, p) in pl.pats.iter().enumerate() {
                let ix = f(i) as usize;
                s0 = s0.wrapping_add(m.wide[p.base + st0 * p.stride + ix] as i32);
                if w1 != 0 {
                    s1 = s1.wrapping_add(m.wide[p.base + st1 * p.stride + ix] as i32);
                }
            }
        }
        if pl.has_globals {
            let mut gv = [(0usize, 0i32); 7];
            let mut ng = 0usize;
            macro_rules! push {
                ($off:expr, $v:expr) => {
                    if let Some(off) = $off {
                        gv[ng] = (off, $v);
                        ng += 1;
                    }
                };
            }
            push!(pl.g_men, b.om.count_ones() as i32 - b.pm.count_ones() as i32);
            push!(pl.g_kown, b.ok.count_ones() as i32);
            push!(pl.g_kopp, b.pk.count_ones() as i32);
            if pl.need_mob {
                let w = pos.side == Side::White;
                let (m1, k1) = quiet_moves::<R>(b.own, b.opp, pos.kings, w);
                let (m2, k2) = quiet_moves::<R>(b.opp, b.own, pos.kings, !w);
                push!(pl.g_kmob, k1 - k2);
                push!(pl.g_mob, m1 + k1 - m2 - k2);
            }
            push!(pl.g_tempo, a_own - a_opp);
            if pl.g_bal.is_some() {
                let l = pl.left[o];
                let imb = |x: u64| (2 * (x & l).count_ones() as i32 - x.count_ones() as i32).abs();
                push!(pl.g_bal, imb(b.own) - imb(b.opp));
            }
            let gw = &m.globals;
            let one_stage = |st: usize| -> i32 {
                let row = st * pl.g_row;
                let mut s = 0i32;
                for &(off, v) in &gv[..ng] {
                    s = s.wrapping_add(v.wrapping_mul(gw[row + off] as i32));
                }
                if let Some(off) = pl.g_ksq {
                    let mut x = b.ok;
                    while x != 0 {
                        let d = pl.dense[o][x.trailing_zeros() as usize] as usize;
                        s = s.wrapping_add(gw[row + off + d] as i32);
                        x &= x - 1;
                    }
                    let mut x = b.pk;
                    while x != 0 {
                        let d = pl.dense[o][x.trailing_zeros() as usize] as usize;
                        s = s.wrapping_sub(gw[row + off + d] as i32);
                        x &= x - 1;
                    }
                }
                s
            };
            s0 = s0.wrapping_add(one_stage(st0));
            if w1 != 0 {
                s1 = s1.wrapping_add(one_stage(st1));
            }
        }
        let raw = w0.wrapping_mul(s0).wrapping_add(w1.wrapping_mul(s1));
        (raw >> pl.shift).clamp(-EVAL_LIMIT, EVAL_LIMIT)
    }

    #[inline]
    pub fn eval_full(&self, pos: &Position<R>) -> i32 {
        if let Some(f) = &self.fast {
            return self.eval_fast(f, pos);
        }
        self.eval_slow(pos)
    }

    #[inline]
    pub fn eval_slow(&self, pos: &Position<R>) -> i32 {
        let (o, b) = Self::boards(pos);
        self.finish(pos, o, &b, |i| self.pat_index(&self.plan.pats[i], o, &b))
    }
}

impl<R: Rules> LinearEval<R> {
    #[inline(always)]
    fn eval_fast(&self, f: &Fast, pos: &Position<R>) -> i32 {
        let (o, b) = Self::boards(pos);
        let pl = &self.plan;
        let (nom, npm, nok, npk) = (b.om.count_ones() as i32, b.pm.count_ones() as i32, b.ok.count_ones() as i32, b.pk.count_ones() as i32);
        let mut tempo = 0i32;
        let (slot, w0, w1);
        if f.mode == 2 {
            let a_own = f.row_sum(o, b.om);
            let a_opp = npm * f.n1 - f.row_sum(o, b.pm);
            tempo = a_own - a_opp;
            (slot, w0, w1) = f.ptab[(a_own + a_opp + (nok + npk) * f.n1) as usize];
        } else {
            if f.need_adv {
                tempo = f.row_sum(o, b.om | b.pm) - npm * f.n1;
            }
            (slot, w0, w1) = if f.mode == 0 { f.ptab[0] } else { f.ptab[(nom + npm + nok + npk) as usize] };
        }
        let mut acc = 0i64;
        let kings = b.kings;
        for p in f.pats[o].iter() {
            unsafe {
                let hm = (((b.om & p.mask).wrapping_mul(p.magic)) >> p.sh) as usize;
                let hp = (((b.pm & p.mask).wrapping_mul(p.magic)) >> p.sh) as usize;
                let mut ix = *f.tabs.get_unchecked(p.toff + hm) + 2 * *f.tabs.get_unchecked(p.toff + hp);
                if p.alph5 && kings {
                    let hk = (((b.ok & p.mask).wrapping_mul(p.magic)) >> p.sh) as usize;
                    let hq = (((b.pk & p.mask).wrapping_mul(p.magic)) >> p.sh) as usize;
                    ix += 3 * *f.tabs.get_unchecked(p.toff + hk) + 4 * *f.tabs.get_unchecked(p.toff + hq);
                }
                acc = acc.wrapping_add(*f.packed.get_unchecked(p.poff + ix as usize * f.p + slot));
            }
        }
        if pl.has_globals {
            let g = &f.gpk[slot * f.gr..(slot + 1) * f.gr];
            if let Some(off) = pl.g_men {
                acc = acc.wrapping_add(((nom - npm) as i64).wrapping_mul(g[off]));
            }
            if let Some(off) = pl.g_kown {
                acc = acc.wrapping_add((nok as i64).wrapping_mul(g[off]));
            }
            if let Some(off) = pl.g_kopp {
                acc = acc.wrapping_add((npk as i64).wrapping_mul(g[off]));
            }
            if pl.need_mob {
                let w = pos.side == Side::White;
                let (m1, k1) = quiet_moves::<R>(b.own, b.opp, pos.kings, w);
                let (m2, k2) = quiet_moves::<R>(b.opp, b.own, pos.kings, !w);
                if let Some(off) = pl.g_kmob {
                    acc = acc.wrapping_add(((k1 - k2) as i64).wrapping_mul(g[off]));
                }
                if let Some(off) = pl.g_mob {
                    acc = acc.wrapping_add(((m1 + k1 - m2 - k2) as i64).wrapping_mul(g[off]));
                }
            }
            if let Some(off) = pl.g_tempo {
                acc = acc.wrapping_add((tempo as i64).wrapping_mul(g[off]));
            }
            if pl.g_bal.is_some() {
                let l = pl.left[o];
                let imb = (2 * (b.own & l).count_ones() as i32 - nom - nok).abs() - (2 * (b.opp & l).count_ones() as i32 - npm - npk).abs();
                if let Some(off) = pl.g_bal {
                    acc = acc.wrapping_add((imb as i64).wrapping_mul(g[off]));
                }
            }
            if pl.g_ksq.is_some() {
                let mut x = b.ok;
                while x != 0 {
                    acc = acc.wrapping_add(g[f.ksq[o][x.trailing_zeros() as usize] as usize]);
                    x &= x - 1;
                }
                let mut x = b.pk;
                while x != 0 {
                    acc = acc.wrapping_sub(g[f.ksq[o][x.trailing_zeros() as usize] as usize]);
                    x &= x - 1;
                }
            }
        }
        let s0 = acc as i32;
        let s1 = ((acc - s0 as i64) >> 32) as i32;
        let raw = w0.wrapping_mul(s0).wrapping_add(w1.wrapping_mul(s1));
        (raw >> pl.shift).clamp(-EVAL_LIMIT, EVAL_LIMIT)
    }
}

impl<R: Rules> Evaluator<R> for LinearEval<R> {
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

#[derive(Clone, Copy)]
struct Pend {
    bw: u64,
    bb: u64,
    bk: u64,
    aw: u64,
    ab: u64,
    ak: u64,
}

#[derive(Clone)]
pub struct IncState {
    full: bool,
    pend: bool,
    mv: Pend,
    idx: [[u32; INC_MAX_PATTERNS]; 2],
}

impl IncState {
    pub fn is_full(&self) -> bool {
        self.full
    }
    pub fn is_pending(&self) -> bool {
        self.pend && !self.full
    }
}

pub struct LinearEvalInc<R: Rules> {
    inner: LinearEval<R>,
}

impl<R: Rules> LinearEvalInc<R> {
    pub fn new(model: Model) -> Result<Self, String> {
        Ok(LinearEvalInc { inner: LinearEval::new(model)? })
    }
    pub fn from_arc(model: Arc<Model>) -> Result<Self, String> {
        Ok(LinearEvalInc { inner: LinearEval::from_arc(model)? })
    }
    pub fn inner(&self) -> &LinearEval<R> {
        &self.inner
    }

    fn empty() -> IncState {
        IncState { full: false, pend: false, mv: Pend { bw: 0, bb: 0, bk: 0, aw: 0, ab: 0, ak: 0 }, idx: [[0; INC_MAX_PATTERNS]; 2] }
    }

    fn compute_both(&self, pos: &Position<R>, st: &mut IncState) {
        let pl = &self.inner.plan;
        for p in 0..2usize {
            let (own, opp) = if p == 0 { (pos.white, pos.black) } else { (pos.black, pos.white) };
            let k = pos.kings;
            let b = Boards { om: own & !k, pm: opp & !k, ok: own & k, pk: opp & k, own, opp, kings: k != 0 };
            for (i, pp) in pl.pats.iter().enumerate() {
                st.idx[p][i] = self.inner.pat_index(pp, p, &b);
            }
        }
        st.full = true;
    }

    fn apply(&self, st: &mut IncState) {
        let pl = &self.inner.plan;
        let m = st.mv;
        let before = [m.bw & !m.bk, m.bw & m.bk, m.bb & !m.bk, m.bb & m.bk];
        let after = [m.aw & !m.ak, m.aw & m.ak, m.ab & !m.ak, m.ab & m.ak];
        for c in 0..4usize {
            let white = c < 2;
            let king = c & 1 == 1;
            let (rem, add) = (before[c] & !after[c], after[c] & !before[c]);
            for p in 0..2usize {
                let own = white == (p == 0);
                let (mut r, mut a) = (rem, add);
                let idx = &mut st.idx[p];
                while r != 0 {
                    let sq = r.trailing_zeros() as usize;
                    r &= r - 1;
                    let (s, e) = pl.ent_range[p][sq];
                    for en in &pl.ents[p][s as usize..e as usize] {
                        idx[en.pat as usize] = idx[en.pat as usize].wrapping_sub(digit(en.alph, own, king) * en.mul);
                    }
                }
                while a != 0 {
                    let sq = a.trailing_zeros() as usize;
                    a &= a - 1;
                    let (s, e) = pl.ent_range[p][sq];
                    for en in &pl.ents[p][s as usize..e as usize] {
                        idx[en.pat as usize] = idx[en.pat as usize].wrapping_add(digit(en.alph, own, king) * en.mul);
                    }
                }
            }
        }
        st.full = true;
    }
}

#[inline(always)]
fn digit(alph: u8, own: bool, king: bool) -> u32 {
    if king {
        if alph == 5 {
            3 + (!own) as u32
        } else {
            0
        }
    } else {
        1 + (!own) as u32
    }
}

impl<R: Rules> Evaluator<R> for LinearEvalInc<R> {
    type State = IncState;
    #[inline]
    fn root(&self, _pos: &Position<R>) -> IncState {
        Self::empty()
    }
    #[inline]
    fn child(&self, parent: &IncState, before: &Position<R>, _mv: &Move, after: &Position<R>) -> IncState {
        if self.inner.plan.inc_ok && parent.full {
            IncState {
                full: false,
                pend: true,
                mv: Pend { bw: before.white, bb: before.black, bk: before.kings, aw: after.white, ab: after.black, ak: after.kings },
                idx: parent.idx,
            }
        } else {
            Self::empty()
        }
    }
    #[inline]
    fn eval(&self, pos: &Position<R>, st: &mut IncState) -> i32 {
        if !self.inner.plan.inc_ok {
            return self.inner.eval_full(pos);
        }
        if !st.full {
            if st.pend {
                self.apply(st);
            } else {
                self.compute_both(pos, st);
            }
        }
        let (o, b) = LinearEval::<R>::boards(pos);
        self.inner.finish(pos, o, &b, |i| st.idx[o][i])
    }
    fn eval_fresh(&self, pos: &Position<R>) -> i32 {
        self.inner.eval_full(pos)
    }
}
