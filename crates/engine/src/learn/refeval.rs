
use super::arch::*;
use super::view::{View, ViewTables};
use super::weights::Model;
use crate::eval::Evaluator;
use crate::movegen::Move;
use crate::position::Position;
use crate::rules::Rules;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Features {
    pub phase_value: i32,
    pub stages: Vec<(usize, i32)>,
    pub table_indices: Vec<Vec<(usize, usize)>>,
    pub globals: Vec<i32>,
    pub head: usize,
}

pub const HEAD_MASK: usize = 63;
pub const HEAD_PARITY: usize = 64;

pub struct RefEval<R: Rules> {
    model: Arc<Model>,
    t: ViewTables,
    _r: PhantomData<fn() -> R>,
}

pub fn pattern_index(p: &Pattern, v: &View) -> usize {
    let mut idx = 0usize;
    let mut mul = 1usize;
    for &s in &p.squares {
        let digit = if (v.own_men >> s) & 1 != 0 {
            1
        } else if (v.opp_men >> s) & 1 != 0 {
            2
        } else if p.alphabet == 5 && (v.own_kings >> s) & 1 != 0 {
            3
        } else if p.alphabet == 5 && (v.opp_kings >> s) & 1 != 0 {
            4
        } else {
            0
        };
        idx += digit * mul;
        mul *= p.alphabet as usize;
    }
    idx
}

pub fn phase_of(ph: &Phase, v: &View, t: &ViewTables) -> (i32, Vec<(usize, i32)>) {
    if ph.kind == PhaseKind::None {
        return (0, vec![(0, PHASE_ONE)]);
    }
    let norm = ph.norm as i64;
    let one = PHASE_ONE as i64;
    let s = match ph.by {
        PhaseBy::Pieces => {
            let total = v.count() as i64;
            one - (one * total / norm).min(one)
        }
        PhaseBy::Tempo => {
            let adv = super::globals::adv_sum(v.own_men, t, false) as i64
                + super::globals::adv_sum(v.opp_men, t, true) as i64
                + (v.own_kings | v.opp_kings).count_ones() as i64 * (t.n as i64 - 1);
            (one * adv / norm).min(one)
        }
    } as i32;
    let k = ph.k as i32;
    match ph.kind {
        PhaseKind::None => unreachable!(),
        PhaseKind::Buckets => (s, vec![(((s * k) / (PHASE_ONE + 1)).min(k - 1) as usize, PHASE_ONE)]),
        PhaseKind::Interp => {
            let x = s * (k - 1);
            let i = (x / PHASE_ONE).min(k - 2);
            let frac = x - i * PHASE_ONE;
            (s, vec![(i as usize, PHASE_ONE - frac), (i as usize + 1, frac)])
        }
    }
}

impl<R: Rules> RefEval<R> {
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
        Ok(RefEval { model, t, _r: PhantomData })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn features(&self, pos: &Position<R>) -> Features {
        let a = &self.model.arch;
        let v0 = self.t.stm(pos);
        let mut views = vec![v0];
        if a.views.len() == 2 {
            views.push(self.t.opp(pos));
        }
        let table_indices = views.iter().map(|v| a.patterns.iter().map(|p| (p.table, pattern_index(p, v))).collect()).collect();
        let (phase_value, stages) = phase_of(&a.phase, &v0, &self.t);
        let l = &self.model.layout;
        let mut globals = vec![0i32; l.g_row];
        for (g, &off) in l.globals.iter().zip(&l.g_off) {
            g.compute(pos, &v0, &self.t, &mut globals[off..off + g.nweights(l.nsq)]);
        }
        let head = a.deep.as_ref().map_or(0, |d| {
            let par = d.head_flags & FLAG_PARITY != 0 && parity_bit(&v0, &self.t);
            d.head_of(v0.count()) | if par { HEAD_PARITY } else { 0 }
        });
        Features { phase_value, stages, table_indices, globals, head }
    }

    pub fn wide_cp(&self, f: &Features) -> i32 {
        let m = &self.model;
        let l = &m.layout;
        let mut raw = 0i32;
        for &(st, w) in &f.stages {
            let mut s = 0i32;
            if m.arch.wide {
                for &(t, i) in &f.table_indices[0] {
                    s = s.wrapping_add(m.wide[l.wide_off[t] + st * m.arch.tables[t].size + i] as i32);
                }
            }
            for (j, &val) in f.globals.iter().enumerate() {
                s = s.wrapping_add(val.wrapping_mul(m.globals[st * l.g_row + j] as i32));
            }
            raw = raw.wrapping_add(w.wrapping_mul(s));
        }
        raw >> (PHASE_SHIFT + m.arch.quant.wide.trailing_zeros())
    }

    pub fn deep_cp(&self, f: &Features) -> i32 {
        let m = &self.model;
        let d = match &m.arch.deep {
            Some(d) => d,
            None => return 0,
        };
        let l = &m.layout;
        let qa = m.arch.quant.qa;
        let mut x: Vec<i32> = vec![];
        for idxs in &f.table_indices {
            let mut acc: Vec<i32> = m.deep_b.iter().map(|&b| b as i32).collect();
            for &(t, i) in idxs {
                let row = &m.deep_v[l.v_off[t] + i * d.width..][..d.width];
                for (a, &r) in acc.iter_mut().zip(row) {
                    *a += r as i32;
                }
            }
            x.extend(acc.iter().map(|&a| (a.clamp(-32768, 32767)).clamp(0, qa)));
        }
        if d.head_extra() > 0 {
            x.push(if f.head & HEAD_PARITY != 0 { qa } else { 0 });
        }
        let nl = l.dims.len();
        let mut out_raw = 0i32;
        let h = (f.head & HEAD_MASK).min(l.heads - 1);
        for (li, &(inn, out)) in l.dims.iter().enumerate() {
            let w = &m.deep_w[li][h * inn * out..(h + 1) * inn * out];
            let bias = &m.deep_bias[li][h * out..(h + 1) * out];
            let mut y = Vec::with_capacity(out);
            for o in 0..out {
                let mut s = bias[o];
                for i in 0..inn {
                    s = s.wrapping_add((w[o * inn + i] as i32).wrapping_mul(x[i]));
                }
                if li + 1 < nl {
                    y.push((s >> 6).clamp(0, qa));
                } else {
                    out_raw = s;
                }
            }
            x = y;
        }
        out_raw >> 6
    }

    pub fn score(&self, f: &Features) -> i32 {
        self.wide_cp(f).wrapping_add(self.deep_cp(f)).clamp(-EVAL_LIMIT, EVAL_LIMIT)
    }
}

impl<R: Rules> Evaluator<R> for RefEval<R> {
    type State = ();
    fn root(&self, _pos: &Position<R>) {}
    fn child(&self, _p: &(), _b: &Position<R>, _mv: &Move, _a: &Position<R>) {}
    fn eval(&self, pos: &Position<R>, _st: &mut ()) -> i32 {
        self.score(&self.features(pos))
    }
}
