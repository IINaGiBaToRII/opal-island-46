
use crate::facade::{GameConfig, VariantId};
use crate::rules::{American, Brazilian, Czech, International, Italian, Rules, Russian, Spanish, Thai, Turkish};
use crate::selfplay::format::read_header;
use crate::selfplay::{dispatch_cfg, for_each_sample, Chunk};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

pub const MATE_SCORE: i32 = 29000;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Wdl {
    pub d: f64,
    pub c: f64,
    pub samples: u64,
    #[serde(default)]
    pub own: bool,
}

impl Default for Wdl {
    fn default() -> Self {
        Wdl::builtin()
    }
}

impl Wdl {
    pub fn builtin() -> Wdl {
        Wdl { d: 352.3, c: 206.7, samples: 0, own: true }
    }
}

fn sigmoid(x: f64) -> f64 {
    super::detmath::sigmoid(x)
}

pub fn probs(w: &Wdl, score: i32) -> (f64, f64, f64) {
    if score >= MATE_SCORE {
        return (1.0, 0.0, 0.0);
    }
    if score <= -MATE_SCORE {
        return (0.0, 0.0, 1.0);
    }
    let s = score as f64;
    let pw = sigmoid((s - w.d) / w.c);
    let pl = sigmoid((-s - w.d) / w.c);
    let pd = (1.0 - pw - pl).max(0.0);
    (pw, pd, pl)
}

fn nll(counts: &[(i32, [u64; 3])], d: f64, c: f64) -> f64 {
    let w = Wdl { d, c, samples: 0, own: false };
    let mut s = 0.0;
    for &(sc, n) in counts {
        let (a, b, l) = probs(&w, sc);
        let f = |p: f64, n: u64| if n == 0 { 0.0 } else { n as f64 * super::detmath::ln_f64(p.max(1e-12)) };
        s -= f(a, n[0]) + f(b, n[1]) + f(l, n[2]);
    }
    s
}

pub fn fit_counts(counts: &[(i32, [u64; 3])]) -> Result<Wdl, String> {
    let counts: Vec<(i32, [u64; 3])> = counts.iter().copied().filter(|(s, n)| s.abs() < MATE_SCORE && n.iter().sum::<u64>() > 0).collect();
    let samples: u64 = counts.iter().map(|(_, n)| n.iter().sum::<u64>()).sum();
    if samples < 20 {
        return Err(format!("слишком мало выборок для калибровки: {samples}"));
    }
    let f = |x: [f64; 2]| nll(&counts, x[0] * x[0], super::detmath::exp_f64(x[1]));
    let mut best = [5.0, super::detmath::ln_f64(100.0)];
    let mut step = 1.0;
    for _ in 0..3 {
        best = nelder_mead(&f, best, step);
        step = 0.3;
    }
    Ok(Wdl { d: best[0] * best[0], c: super::detmath::exp_f64(best[1]), samples, own: false })
}

fn nelder_mead(f: &dyn Fn([f64; 2]) -> f64, start: [f64; 2], step: f64) -> [f64; 2] {
    let mut p = [start, [start[0] + step * 3.0, start[1]], [start[0], start[1] + step * 0.5]];
    let mut v = [f(p[0]), f(p[1]), f(p[2])];
    for _ in 0..500 {
        let mut idx = [0usize, 1, 2];
        idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
        let (b, m, w) = (idx[0], idx[1], idx[2]);
        if (v[w] - v[b]).abs() < 1e-12 * (1.0 + v[b].abs()) && (p[w][0] - p[b][0]).abs() + (p[w][1] - p[b][1]).abs() < 1e-9 {
            break;
        }
        let c = [(p[b][0] + p[m][0]) / 2.0, (p[b][1] + p[m][1]) / 2.0];
        let at = |t: f64| [c[0] + t * (p[w][0] - c[0]), c[1] + t * (p[w][1] - c[1])];
        let r = at(-1.0);
        let fr = f(r);
        if fr < v[b] {
            let e = at(-2.0);
            let fe = f(e);
            if fe < fr {
                p[w] = e;
                v[w] = fe;
            } else {
                p[w] = r;
                v[w] = fr;
            }
        } else if fr < v[m] {
            p[w] = r;
            v[w] = fr;
        } else {
            let k = if fr < v[w] { at(-0.5) } else { at(0.5) };
            let fk = f(k);
            if fk < v[w].min(fr) {
                p[w] = k;
                v[w] = fk;
            } else {
                for i in [m, w] {
                    p[i] = [(p[i][0] + p[b][0]) / 2.0, (p[i][1] + p[b][1]) / 2.0];
                    v[i] = f(p[i]);
                }
            }
        }
    }
    let mut bi = 0;
    for i in 1..3 {
        if v[i] < v[bi] {
            bi = i;
        }
    }
    p[bi]
}

#[derive(Clone, Debug)]
pub struct BinRow {
    pub label: String,
    pub n: u64,
    pub emp: (f64, f64, f64),
    pub model: (f64, f64, f64),
}

#[derive(Clone, Debug)]
pub struct FitReport {
    pub wdl: Wdl,
    pub bins: Vec<BinRow>,
    pub nll_per_sample: f64,
    pub nll_const_per_sample: f64,
}

const EDGES: [i32; 8] = [-300, -150, -75, -25, 25, 75, 150, 300];

fn bin_of(s: i32) -> usize {
    EDGES.iter().take_while(|&&e| s >= e).count()
}

fn bin_label(i: usize) -> String {
    match i {
        0 => format!("< {}", EDGES[0]),
        8 => format!(">= {}", EDGES[7]),
        _ => format!("[{}, {})", EDGES[i - 1], EDGES[i]),
    }
}

pub fn fit_report(counts: &[(i32, [u64; 3])]) -> Result<FitReport, String> {
    let wdl = fit_counts(counts)?;
    let used: Vec<(i32, [u64; 3])> = counts.iter().copied().filter(|(s, n)| s.abs() < MATE_SCORE && n.iter().sum::<u64>() > 0).collect();
    let mut n = [[0u64; 3]; 9];
    let mut m = [[0f64; 3]; 9];
    for &(s, c) in &used {
        let b = bin_of(s);
        let (pw, pd, pl) = probs(&wdl, s);
        let t: u64 = c.iter().sum();
        for k in 0..3 {
            n[b][k] += c[k];
        }
        m[b][0] += pw * t as f64;
        m[b][1] += pd * t as f64;
        m[b][2] += pl * t as f64;
    }
    let bins = (0..9)
        .filter_map(|i| {
            let t: u64 = n[i].iter().sum();
            if t == 0 {
                return None;
            }
            let tf = t as f64;
            Some(BinRow {
                label: bin_label(i),
                n: t,
                emp: (n[i][0] as f64 / tf, n[i][1] as f64 / tf, n[i][2] as f64 / tf),
                model: (m[i][0] / tf, m[i][1] / tf, m[i][2] / tf),
            })
        })
        .collect();
    let total: f64 = wdl.samples as f64;
    let mut tot = [0u64; 3];
    for (_, c) in &used {
        for k in 0..3 {
            tot[k] += c[k];
        }
    }
    let nll_const: f64 = -(0..3).map(|k| if tot[k] == 0 { 0.0 } else { tot[k] as f64 * super::detmath::ln_f64(tot[k] as f64 / total) }).sum::<f64>();
    Ok(FitReport { nll_per_sample: nll(&used, wdl.d, wdl.c) / total, nll_const_per_sample: nll_const / total, wdl, bins })
}

pub fn format_report(r: &FitReport) -> String {
    let pc = |x: f64| format!("{:5.1}", x * 100.0);
    let w = &r.wdl;
    let mut s = format!(
        "калибровка: d = {:.1}, c = {:.1} (выборок {}); -logL на выборку {:.4} (без оценки {:.4})\n",
        w.d, w.c, w.samples, r.nll_per_sample, r.nll_const_per_sample
    );
    s += "  оценка -> % выигрыш / ничья / проигрыш\n";
    for sc in [0, 50, -50, 100, -100, 200, -200, 400, -400] {
        let (a, b, c) = probs(w, sc);
        s += &format!("    {:>5}: {} / {} / {}\n", sc, pc(a), pc(b), pc(c));
    }
    s += "  корзины: выборок | эмпирика В/Н/П % | модель В/Н/П %\n";
    for b in &r.bins {
        s += &format!(
            "    {:<12} {:>9} | {} {} {} | {} {} {}\n",
            b.label, b.n, pc(b.emp.0), pc(b.emp.1), pc(b.emp.2), pc(b.model.0), pc(b.model.1), pc(b.model.2)
        );
    }
    s
}

fn collect<R: Rules>(dir: &Path, eval_id: Option<u32>) -> Result<Vec<(i32, [u64; 3])>, String> {
    let mut map: HashMap<i32, [u64; 3]> = HashMap::new();
    for p in super::dataset::list_chunks(dir)? {
        if let Some(id) = eval_id {
            if read_header(&p)?.eval_id != id {
                continue;
            }
        }
        let ch = Chunk::read(&p)?;
        for g in ch.games() {
            let g = g?;
            for_each_sample::<R>(&g, None, |_, s| {
                let k = match s.result_stm {
                    1 => 0,
                    0 => 1,
                    _ => 2,
                };
                map.entry(s.score as i32).or_insert([0; 3])[k] += 1;
            })?;
        }
    }
    let mut v: Vec<_> = map.into_iter().collect();
    v.sort_by_key(|x| x.0);
    Ok(v)
}

pub fn counts_from_dir(cfg: &GameConfig, dir: &Path, eval_id: Option<u32>) -> Result<Vec<(i32, [u64; 3])>, String> {
    dispatch_cfg!(cfg, collect(dir, eval_id))
}

