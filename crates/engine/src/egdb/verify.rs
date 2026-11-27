
use super::index::*;
use super::probe::Egdb;
use crate::eval::SimpleEval;
use crate::movegen::{generate, ListSink, MoveList};
use crate::position::Side;
use crate::rules::Rules;
use crate::search::{search, SearchLimits, MATE, MATE_BOUND};
use std::path::Path;
use std::sync::atomic::AtomicBool;

#[derive(Default, Debug)]
pub struct VerifyReport {
    pub slices: usize,
    pub checked: u64,
    pub mismatches: Vec<String>,
    pub unresolved: u64,
    pub indep_checked: u64,
    pub indep_confirmed: u64,
    pub indep_unconfirmed: u64,
    pub indep_contradictions: Vec<String>,
}

impl VerifyReport {
    pub fn ok(&self) -> bool {
        self.mismatches.is_empty() && self.indep_contradictions.is_empty()
    }
}

pub fn expected<R: Rules>(db: &Egdb<R>, p: &crate::position::Position<R>) -> Option<i8> {
    let mut list = MoveList::new();
    generate(p, &mut ListSink::<true>(&mut list));
    if list.len == 0 {
        return Some(-1);
    }
    let mut loss_min: Option<i32> = None;
    let mut all_win = true;
    let mut maxw = 0i32;
    for mv in list.as_slice() {
        let cv = db.probe(&p.make(mv))? as i32;
        if cv < 0 {
            let a = -cv;
            loss_min = Some(loss_min.map_or(a, |x| x.min(a)));
        }
        if cv <= 0 {
            all_win = false;
        } else {
            maxw = maxw.max(cv);
        }
    }
    Some(if let Some(m) = loss_min {
        m.min(127) as i8
    } else if all_win {
        -((maxw + 2).min(127)) as i8
    } else {
        0
    })
}

#[derive(Default, Debug)]
pub struct IdentityReport {
    pub slices: usize,
    pub values: u64,
    pub probes: u64,
    pub mismatches: Vec<String>,
}

impl IdentityReport {
    pub fn ok(&self) -> bool {
        self.mismatches.is_empty() && self.slices > 0
    }
}

pub fn identity<R: Rules>(a: &Egdb<R>, b: &Egdb<R>, max_pieces: u32) -> Result<IdentityReport, String> {
    use rayon::prelude::*;
    let sigs: Vec<Sig> = a.sigs().into_iter().filter(|s| s.iter().map(|&x| x as u32).sum::<u32>() <= max_pieces).collect();
    let parts: Result<Vec<(u64, u64, Vec<String>)>, String> = sigs
        .par_iter()
        .map(|&sig| {
            if !b.has_slice(sig) {
                return Ok((0, 0, vec![format!("{}: нет среза во второй базе", sig_name(sig))]));
            }
            let ix = Indexer::new::<R::Geo>(sig)?;
            let (mut vals, mut probes, mut bad) = (0u64, 0u64, vec![]);
            for idx in 0..ix.n {
                let m = ix.unrank(idx);
                let valid = ix.valid(&m);
                for side in [Side::White, Side::Black] {
                    let (x, y) = (a.value_at(sig, side, idx), b.value_at(sig, side, idx));
                    vals += 1;
                    if x != y && bad.len() < 5 {
                        bad.push(format!("{} {:?} индекс {}: {:?} != {:?}", sig_name(sig), side, idx, x, y));
                    }
                    if valid {
                        let p = ix.to_position::<R>(&m, side);
                        let (x, y) = (a.probe(&p), b.probe(&p));
                        probes += 1;
                        if x != y && bad.len() < 5 {
                            bad.push(format!("{} {}: probe {:?} != {:?}", sig_name(sig), p.to_dfen(), x, y));
                        }
                    }
                }
            }
            Ok((vals, probes, bad))
        })
        .collect();
    let mut rep = IdentityReport { slices: sigs.len(), ..Default::default() };
    for (v, p, bad) in parts? {
        rep.values += v;
        rep.probes += p;
        for s in bad {
            if rep.mismatches.len() < 20 {
                rep.mismatches.push(s);
            }
        }
    }
    Ok(rep)
}

fn rng(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

pub fn verify<R: Rules>(dir: &Path, sample: usize, seed: u64, indep_max: usize, log: &mut dyn FnMut(&str)) -> Result<VerifyReport, String> {
    let db = Egdb::<R>::open_at(dir)?;
    let mut rep = VerifyReport::default();
    let eval = SimpleEval::new::<R>();
    let stop = AtomicBool::new(false);
    let mut s = seed | 1;
    for sig in db.sigs() {
        rep.slices += 1;
        let ix = Indexer::new::<R::Geo>(sig)?;
        let total = 2 * ix.n;
        let all = sample == 0 || (sample as u64) >= total;
        let k = if all { total } else { sample as u64 };
        let (c0, m0, i0) = (rep.checked, rep.mismatches.len(), rep.indep_checked);
        for j in 0..k {
            let g = if all { j } else { rng(&mut s) % total };
            let (side, idx) = if g >= ix.n { (Side::Black, g - ix.n) } else { (Side::White, g) };
            let m = ix.unrank(idx);
            if !ix.valid(&m) {
                continue;
            }
            let pos = ix.to_position::<R>(&m, side);
            let v = db.probe(&pos).ok_or("probe: нет значения в существующем срезе")?;
            let Some(e) = expected(&db, &pos) else {
                rep.unresolved += 1;
                continue;
            };
            rep.checked += 1;
            let sat = v.unsigned_abs() >= 127 || e.unsigned_abs() >= 127;
            if v != e && !(sat && v.signum() == e.signum()) {
                if rep.mismatches.len() < 20 {
                    rep.mismatches.push(format!("{} {}: база {} != минимакс {}", sig_name(sig), pos.to_dfen(), v, e));
                }
                continue;
            }
            if !R::ONE_V_ONE_DRAW && rep.indep_checked < indep_max as u64 && (v as i32).abs() <= 9 && (j % 7 == 0 || v != 0) {
                let d = if v == 0 { 6 } else { (v as i32).unsigned_abs() + 2 };
                let lim = SearchLimits { depth: Some(d), hash_mb: 4, ..Default::default() };
                let r = search::<R, _>(&pos, &[], &eval, &lim, &stop, &mut |_| {});
                rep.indep_checked += 1;
                let sc = r.score;
                if v > 0 {
                    if sc > MATE_BOUND {
                        if MATE - sc >= v as i32 {
                            rep.indep_confirmed += 1;
                        } else {
                            rep.indep_contradictions.push(format!("{} {}: поиск нашёл мат быстрее базы ({} < {})", sig_name(sig), pos.to_dfen(), MATE - sc, v));
                        }
                    } else if sc < -MATE_BOUND {
                        rep.indep_contradictions.push(format!("{} {}: база выигрыш {}, поиск проигрыш", sig_name(sig), pos.to_dfen(), v));
                    } else {
                        rep.indep_unconfirmed += 1;
                    }
                } else if v < 0 {
                    if sc < -MATE_BOUND {
                        if MATE + sc >= -(v as i32) - 1 {
                            rep.indep_confirmed += 1;
                        } else {
                            rep.indep_contradictions.push(format!("{} {}: поиск видит поражение быстрее базы", sig_name(sig), pos.to_dfen()));
                        }
                    } else if sc > MATE_BOUND {
                        rep.indep_contradictions.push(format!("{} {}: база проигрыш {}, поиск выигрыш", sig_name(sig), pos.to_dfen(), v));
                    } else {
                        rep.indep_unconfirmed += 1;
                    }
                } else if sc.abs() > MATE_BOUND {
                    rep.indep_contradictions.push(format!("{} {}: база ничья, поиск видит мат ({})", sig_name(sig), pos.to_dfen(), sc));
                } else {
                    rep.indep_confirmed += 1;
                }
            }
        }
        log(&format!(
            "{}: проверено {}, расхождений {}, поиском {}",
            sig_name(sig),
            rep.checked - c0,
            rep.mismatches.len() - m0,
            rep.indep_checked - i0
        ));
    }
    Ok(rep)
}
