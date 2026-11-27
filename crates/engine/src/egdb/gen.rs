
use super::file;
use super::index::*;
use super::{fixed_value, quarantine_stale, slice_path, update_manifest, SliceStat};
use crate::movegen::{generate, ListSink, MoveList};
use crate::position::{Position, Side};
use crate::rules::Rules;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const CH: usize = 4096;
pub const VERIFY_SAMPLES: usize = 2000;
const DRAW: u8 = 0x80;
const TERM: u8 = 0xFF;

pub struct SliceTab {
    pub ix: Indexer,
    pub vals: Vec<u8>,
}

type DepMap = HashMap<Sig, Arc<SliceTab>>;

pub struct GenOpts {
    pub dir: PathBuf,
    pub pieces: u32,
    pub threads: usize,
    pub max_ram_bytes: u64,
    pub max_disk_bytes: u64,
}

pub struct GenCtl {
    pub cancel: AtomicBool,
    pub duty: AtomicU32,
    pub qos: u8,
    pub one_slice: bool,
    pub strict: bool,
    pub verify_samples: usize,
    pub debug_corrupt: bool,
    cache: Mutex<DepMap>,
    current: Mutex<Option<(String, u64, Instant)>>,
}

impl Default for GenCtl {
    fn default() -> Self {
        GenCtl { cancel: AtomicBool::new(false), duty: AtomicU32::new(100), qos: 0, one_slice: false, strict: false, verify_samples: VERIFY_SAMPLES, debug_corrupt: false, cache: Mutex::new(HashMap::new()), current: Mutex::new(None) }
    }
}

impl GenCtl {
    pub fn new() -> Arc<GenCtl> {
        Arc::new(GenCtl::default())
    }
    pub fn with(one_slice: bool, strict: bool, qos: u8) -> GenCtl {
        GenCtl { one_slice, strict, qos, ..Default::default() }
    }
    pub fn current(&self) -> Option<(String, u64, Instant)> {
        self.current.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

struct Th {
    n: u32,
    dead: bool,
}

thread_local! {
    static WORK: std::cell::Cell<Option<(Instant, Duration)>> = const { std::cell::Cell::new(None) };
}

impl Th {
    fn new() -> Th {
        Th { n: 0, dead: false }
    }
    #[inline]
    fn tick(&mut self, ctl: &GenCtl) -> bool {
        if self.dead {
            return true;
        }
        self.n = self.n.wrapping_add(1);
        if self.n & 255 != 0 {
            return false;
        }
        if ctl.cancel.load(Relaxed) {
            self.dead = true;
            return true;
        }
        let duty = ctl.duty.load(Relaxed).clamp(1, 100) as u64;
        if duty < 100 {
            let now = Instant::now();
            let (last, mut work) = WORK.with(|w| w.get()).unwrap_or((now, Duration::ZERO));
            work += now.duration_since(last).min(Duration::from_millis(10));
            if work >= Duration::from_millis(20) {
                std::thread::sleep(work * (100 - duty as u32) / duty as u32);
                work = Duration::ZERO;
                WORK.with(|w| w.set(Some((Instant::now(), work))));
            } else {
                WORK.with(|w| w.set(Some((now, work))));
            }
        }
        false
    }
}

#[derive(Clone, Debug)]
pub struct SlicePlan {
    pub name: String,
    pub pieces: u32,
    pub positions: u64,
    pub ram_bytes: u64,
}

#[derive(Default, Debug)]
pub struct GenReport {
    pub cancelled: bool,
    pub peak_cache_bytes: u64,
    pub built: Vec<SliceStat>,
    pub existing: usize,
    pub skipped: Vec<String>,
    pub stopped_disk: bool,
    pub bytes: u64,
    pub secs: f64,
}

fn is_one_v_one<R: Rules>(s: Sig) -> bool {
    R::ONE_V_ONE_DRAW && s[0] + s[1] == 1 && s[2] + s[3] == 1
}

fn child_value<R: Rules>(deps: &DepMap, c: &Position<R>) -> Option<i8> {
    if let Some(v) = fixed_value(c) {
        return Some(v);
    }
    let tab = deps.get(&sig_of(c))?;
    let m = tab.ix.masks(c);
    if !tab.ix.valid(&m) {
        return None;
    }
    let idx = tab.ix.rank(&m)?;
    let g = idx + if c.side == Side::Black { tab.ix.n } else { 0 };
    Some(tab.vals[g as usize] as i8)
}

#[inline]
fn split(g: usize, n: usize) -> (Side, u64) {
    if g >= n {
        (Side::Black, (g - n) as u64)
    } else {
        (Side::White, g as u64)
    }
}

pub struct Computed {
    pub vals: Vec<u8>,
    pub wins: u64,
    pub losses: u64,
    pub draws: u64,
    pub max_dist: u32,
    pub levels: u32,
    pub anomalies: u64,
    pub saturated: bool,
}

pub fn compute<R: Rules>(pool: &rayon::ThreadPool, ix: &Indexer, deps: &DepMap) -> Computed {
    compute_ctl::<R>(pool, ix, deps, &GenCtl::default()).expect("без отмены расчёт всегда завершается")
}

pub fn compute_ctl<R: Rules>(pool: &rayon::ThreadPool, ix: &Indexer, deps: &DepMap, ctl: &GenCtl) -> Option<Computed> {
    let n = ix.n as usize;
    let total = 2 * n;
    let mut work = vec![0u8; total];
    let mut ew = vec![0u8; total];
    let mut ml = vec![0u8; total];
    let maxext = AtomicU32::new(0);
    let anomalies = AtomicU64::new(0);
    pool.install(|| {
        work.par_chunks_mut(CH)
            .zip(ew.par_chunks_mut(CH))
            .zip(ml.par_chunks_mut(CH))
            .enumerate()
            .for_each_init(|| (MoveList::new(), Th::new()), |st, (ci, ((w, e), l))| {
                let (list, th) = (&mut st.0, &mut st.1);
                let base = ci * CH;
                let mut mx_local = 0u32;
                let mut anom = 0u64;
                for j in 0..w.len() {
                    if th.tick(ctl) {
                        break;
                    }
                    let (side, idx) = split(base + j, n);
                    let m = ix.unrank(idx);
                    if !ix.valid(&m) {
                        w[j] = DRAW;
                        continue;
                    }
                    let pos = ix.to_position::<R>(&m, side);
                    list.clear();
                    generate(&pos, &mut ListSink::<true>(&mut *list));
                    if list.len == 0 {
                        w[j] = TERM;
                        continue;
                    }
                    let (mut has_ins, mut ew_, mut ml_, mut cant) = (false, 0u32, 0u32, false);
                    for mv in list.as_slice() {
                        if mv.captured == 0 && !mv.promo {
                            has_ins = true;
                            continue;
                        }
                        let ch = pos.make(mv);
                        let v = match child_value(deps, &ch) {
                            Some(v) => v,
                            None => {
                                anom += 1;
                                0
                            }
                        };
                        let a = (v as i32).unsigned_abs();
                        mx_local = mx_local.max(a);
                        if v < 0 {
                            if ew_ == 0 || a < ew_ {
                                ew_ = a;
                            }
                        } else if v == 0 {
                            cant = true;
                        } else {
                            ml_ = ml_.max(a);
                        }
                    }
                    if !has_ins {
                        w[j] = if ew_ != 0 {
                            ew_ as u8
                        } else if cant {
                            DRAW
                        } else {
                            (-((ml_ + 2).min(127) as i32)) as i8 as u8
                        };
                    } else {
                        e[j] = ew_.min(127) as u8;
                        l[j] = if cant || ew_ != 0 { 255 } else { ml_ as u8 };
                    }
                }
                maxext.fetch_max(mx_local, Relaxed);
                anomalies.fetch_add(anom, Relaxed);
            });
    });
    if ctl.cancel.load(Relaxed) {
        return None;
    }
    let maxext = maxext.load(Relaxed);
    let wa: &[AtomicU8] = unsafe { std::slice::from_raw_parts(work.as_ptr() as *const AtomicU8, total) };

    let child_idx = |ch: &Position<R>| -> Option<usize> {
        let r = ix.rank_pos(ch)?;
        Some((r + if ch.side == Side::Black { ix.n } else { 0 }) as usize)
    };

    let pass_a = |m: u32| -> u64 {
        let want = (-(m as i32)) as i8 as u8;
        pool.install(|| {
            (0..total)
                .into_par_iter()
                .with_min_len(2048)
                .map_init(|| (MoveList::new(), Th::new()), |st, g| -> u64 {
                    let (list, th) = (&mut st.0, &mut st.1);
                    if th.tick(ctl) {
                        return 0;
                    }
                    if wa[g].load(Relaxed) != 0 {
                        return 0;
                    }
                    if ew[g] as u32 == m {
                        wa[g].store(m as u8, Relaxed);
                        return 1;
                    }
                    let (side, idx) = split(g, n);
                    let pos = ix.to_position::<R>(&ix.unrank(idx), side);
                    list.clear();
                    generate(&pos, &mut ListSink::<true>(&mut *list));
                    for mv in list.as_slice() {
                        if mv.captured != 0 || mv.promo {
                            continue;
                        }
                        if let Some(ci) = child_idx(&pos.make(mv)) {
                            if wa[ci].load(Relaxed) == want {
                                wa[g].store(m as u8, Relaxed);
                                return 1;
                            }
                        }
                    }
                    0
                })
                .sum()
        })
    };
    let pass_b = |m: u32| -> u64 {
        let set = (-((m + 2) as i32)) as i8 as u8;
        pool.install(|| {
            (0..total)
                .into_par_iter()
                .with_min_len(2048)
                .map_init(|| (MoveList::new(), Th::new()), |st, g| -> u64 {
                    let (list, th) = (&mut st.0, &mut st.1);
                    if th.tick(ctl) {
                        return 0;
                    }
                    if wa[g].load(Relaxed) != 0 || ml[g] == 255 {
                        return 0;
                    }
                    let (side, idx) = split(g, n);
                    let pos = ix.to_position::<R>(&ix.unrank(idx), side);
                    list.clear();
                    generate(&pos, &mut ListSink::<true>(&mut *list));
                    let mut mx = ml[g] as i32;
                    for mv in list.as_slice() {
                        if mv.captured != 0 || mv.promo {
                            continue;
                        }
                        match child_idx(&pos.make(mv)) {
                            Some(ci) => {
                                let v = wa[ci].load(Relaxed) as i8 as i32;
                                if v <= 0 {
                                    return 0;
                                }
                                mx = mx.max(v);
                            }
                            None => return 0,
                        }
                    }
                    if mx == m as i32 {
                        wa[g].store(set, Relaxed);
                        1
                    } else {
                        0
                    }
                })
                .sum()
        })
    };

    let mut m = 1u32;
    let mut levels = 0u32;
    let mut saturated = false;
    loop {
        if m > 125 {
            saturated = true;
            break;
        }
        let ca = pass_a(m);
        let cb = pass_b(m);
        if ctl.cancel.load(Relaxed) {
            return None;
        }
        levels += 1;
        if std::env::var_os("RUNNER_EGDB_DEBUG").is_some() {
            eprintln!("  m={m} A={ca} B={cb} maxext={maxext}");
        }
        if ca == 0 && cb == 0 && m >= maxext + 2 {
            break;
        }
        m += 2;
    }
    if saturated {
        loop {
            let a: u64 = pool.install(|| {
                (0..total)
                    .into_par_iter()
                    .map_init(|| (MoveList::new(), Th::new()), |st, g| -> u64 {
                    let (list, th) = (&mut st.0, &mut st.1);
                    if th.tick(ctl) {
                        return 0;
                    }
                        if wa[g].load(Relaxed) != 0 {
                            return 0;
                        }
                        if ew[g] != 0 {
                            wa[g].store(127, Relaxed);
                            return 1;
                        }
                        let (side, idx) = split(g, n);
                        let pos = ix.to_position::<R>(&ix.unrank(idx), side);
                        list.clear();
                        generate(&pos, &mut ListSink::<true>(&mut *list));
                        for mv in list.as_slice() {
                            if mv.captured != 0 || mv.promo {
                                continue;
                            }
                            if let Some(ci) = child_idx(&pos.make(mv)) {
                                let v = wa[ci].load(Relaxed);
                                if v != DRAW && (v as i8) < 0 {
                                    wa[g].store(127, Relaxed);
                                    return 1;
                                }
                            }
                        }
                        0
                    })
                    .sum()
            });
            let b: u64 = pool.install(|| {
                (0..total)
                    .into_par_iter()
                    .map_init(|| (MoveList::new(), Th::new()), |st, g| -> u64 {
                    let (list, th) = (&mut st.0, &mut st.1);
                    if th.tick(ctl) {
                        return 0;
                    }
                        if wa[g].load(Relaxed) != 0 || ml[g] == 255 {
                            return 0;
                        }
                        let (side, idx) = split(g, n);
                        let pos = ix.to_position::<R>(&ix.unrank(idx), side);
                        list.clear();
                        generate(&pos, &mut ListSink::<true>(&mut *list));
                        for mv in list.as_slice() {
                            if mv.captured != 0 || mv.promo {
                                continue;
                            }
                            match child_idx(&pos.make(mv)) {
                                Some(ci) if (wa[ci].load(Relaxed) as i8) > 0 => {}
                                _ => return 0,
                            }
                        }
                        wa[g].store((-127i8) as u8, Relaxed);
                        1
                    })
                    .sum()
            });
            if ctl.cancel.load(Relaxed) {
                return None;
            }
            if a == 0 && b == 0 {
                break;
            }
        }
    }
    let (mut wins, mut losses, mut draws, mut maxd) = (0u64, 0u64, 0u64, 0u32);
    for v in work.iter_mut() {
        if *v == DRAW {
            *v = 0;
        }
        let s = *v as i8;
        if s > 0 {
            wins += 1;
        } else if s < 0 {
            losses += 1;
        } else {
            draws += 1;
        }
        maxd = maxd.max((s as i32).unsigned_abs());
    }
    Some(Computed { vals: work, wins, losses, draws, max_dist: maxd, levels, anomalies: anomalies.load(Relaxed), saturated })
}

pub fn verify_sample<R: Rules>(ix: &Indexer, vals: &[u8], deps: &DepMap, samples: usize) -> Result<usize, String> {
    let n = ix.n as usize;
    let total = 2 * n;
    if total == 0 || vals.len() != total {
        return Err(format!("размер среза {} вместо {}", vals.len(), total));
    }
    let mut list = MoveList::new();
    let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ total as u64;
    let mut next = || {
        rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let all = total <= samples;
    let count = if all { total } else { samples };
    let mut checked = 0usize;
    for k in 0..count {
        let g = if all { k } else { (next() % total as u64) as usize };
        let (side, idx) = split(g, n);
        let m = ix.unrank(idx);
        if !ix.valid(&m) {
            continue;
        }
        let pos = ix.to_position::<R>(&m, side);
        let my_sig = sig_of(&pos);
        let v = vals[g] as i8;
        if v.unsigned_abs() >= 125 {
            continue;
        }
        list.clear();
        generate(&pos, &mut ListSink::<true>(&mut list));
        if list.len == 0 {
            if v != -1 {
                return Err(format!("позиция {g}: нет ходов, значение {v} вместо -1"));
            }
            checked += 1;
            continue;
        }
        let (mut min_loss, mut has_draw, mut max_win, mut skip) = (0u32, false, 0u32, false);
        for mv in list.as_slice() {
            let ch = pos.make(mv);
            let cv = if let Some(f) = fixed_value(&ch) {
                Some(f)
            } else if sig_of(&ch) == my_sig {
                ix.rank_pos(&ch).map(|r| vals[(r + if ch.side == Side::Black { ix.n } else { 0 }) as usize] as i8)
            } else {
                child_value(deps, &ch)
            };
            let Some(cv) = cv else {
                skip = true;
                break;
            };
            let a = cv.unsigned_abs() as u32;
            if a >= 125 {
                skip = true;
                break;
            }
            if cv < 0 {
                if min_loss == 0 || a < min_loss {
                    min_loss = a;
                }
            } else if cv == 0 {
                has_draw = true;
            } else {
                max_win = max_win.max(a);
            }
        }
        if skip {
            continue;
        }
        let expect: i32 = if min_loss != 0 {
            min_loss as i32
        } else if has_draw {
            0
        } else {
            -((max_win + 2) as i32)
        };
        if v as i32 != expect {
            return Err(format!("позиция {g}: значение {v}, по потомкам должно быть {expect}"));
        }
        checked += 1;
    }
    Ok(checked)
}

pub fn next_slice<R: Rules>(dir: &std::path::Path, pieces: u32, max_ram_bytes: u64) -> (Option<SlicePlan>, Option<SlicePlan>) {
    let mut too_big = None;
    for t in 2..=pieces {
        for sig in sigs_of_total(t) {
            if is_one_v_one::<R>(sig) || slice_path(dir, sig).exists() {
                continue;
            }
            let Ok(ix) = Indexer::new::<R::Geo>(sig) else { continue };
            let deps: Vec<Sig> = dep_sigs(sig).into_iter().filter(|s| !is_one_v_one::<R>(*s)).collect();
            if deps.iter().any(|d| !slice_path(dir, *d).exists()) {
                continue;
            }
            let dep_bytes: u64 = deps.iter().filter_map(|d| Indexer::new::<R::Geo>(*d).ok()).map(|i| 2 * i.n).sum();
            let plan = SlicePlan { name: sig_name(sig), pieces: t, positions: 2 * ix.n, ram_bytes: 6 * ix.n + dep_bytes };
            if plan.ram_bytes > max_ram_bytes {
                too_big.get_or_insert(plan);
                continue;
            }
            return (Some(plan), too_big);
        }
    }
    (None, too_big)
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayerPlan {
    pub pieces: u32,
    pub slices_total: usize,
    pub slices_left: usize,
    pub positions_total: u64,
    pub positions_left: u64,
    pub max_ram_bytes: u64,
    pub max_ram_slice: String,
}

pub fn layer_plan<R: Rules>(dir: &std::path::Path, pieces: u32) -> Result<LayerPlan, String> {
    let mut p = LayerPlan { pieces, ..Default::default() };
    for sig in sigs_of_total(pieces) {
        if is_one_v_one::<R>(sig) {
            continue;
        }
        let ix = Indexer::new::<R::Geo>(sig).map_err(|e| format!("срез {}: {e}", sig_name(sig)))?;
        p.slices_total += 1;
        p.positions_total += 2 * ix.n;
        if slice_path(dir, sig).exists() {
            continue;
        }
        let mut dep_bytes = 0u64;
        for d in dep_sigs(sig).into_iter().filter(|s| !is_one_v_one::<R>(*s)) {
            dep_bytes += 2 * Indexer::new::<R::Geo>(d).map_err(|e| format!("срез {}: {e}", sig_name(d)))?.n;
        }
        let ram = 6 * ix.n + dep_bytes;
        p.slices_left += 1;
        p.positions_left += 2 * ix.n;
        if ram > p.max_ram_bytes {
            p.max_ram_bytes = ram;
            p.max_ram_slice = sig_name(sig);
        }
    }
    Ok(p)
}

fn dir_bytes(dir: &std::path::Path) -> u64 {
    let mut t = 0;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if e.path().extension().map(|x| x == "egdb").unwrap_or(false) {
                t += e.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    t
}

pub fn generate_all<R: Rules>(o: &GenOpts, log: &mut dyn FnMut(&str)) -> Result<GenReport, String> {
    generate_all_ctl::<R>(o, &GenCtl::default(), log)
}

pub fn generate_all_ctl<R: Rules>(o: &GenOpts, ctl: &GenCtl, log: &mut dyn FnMut(&str)) -> Result<GenReport, String> {
    let t0 = Instant::now();
    std::fs::create_dir_all(&o.dir).map_err(|e| format!("{}: {e}", o.dir.display()))?;
    let q = quarantine_stale(&o.dir);
    if q > 0 {
        log(&format!("устаревших/повреждённых файлов отложено (*.stale): {q}"));
    }
    let qos = ctl.qos;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(o.threads.max(1))
        .start_handler(move |_| {
            if qos != 0 {
                crate::qos::apply(qos);
            }
        })
        .build()
        .map_err(|e| e.to_string())?;
    let mut rep = GenReport::default();
    let mut disk = dir_bytes(&o.dir);
    let mut cache_guard = ctl.cache.lock().unwrap_or_else(|e| e.into_inner());
    let cache: &mut DepMap = &mut cache_guard;
    'outer: for t in 2..=o.pieces {
        for sig in sigs_of_total(t) {
            if is_one_v_one::<R>(sig) {
                continue;
            }
            let path = slice_path(&o.dir, sig);
            if path.exists() {
                rep.existing += 1;
                continue;
            }
            if disk >= o.max_disk_bytes {
                log(&format!("лимит диска ({} МБ) достигнут: остановка перед {}", o.max_disk_bytes >> 20, sig_name(sig)));
                rep.stopped_disk = true;
                break 'outer;
            }
            let ix = Indexer::new::<R::Geo>(sig)?;
            let own = 3 * 2 * ix.n;
            let deps: Vec<Sig> = dep_sigs(sig).into_iter().filter(|s| !is_one_v_one::<R>(*s)).collect();
            let mut missing = None;
            let mut dep_bytes = 0u64;
            for d in &deps {
                if !slice_path(&o.dir, *d).exists() {
                    missing = Some(*d);
                    break;
                }
                dep_bytes += 2 * Indexer::new::<R::Geo>(*d)?.n;
            }
            if let Some(d) = missing {
                let msg = format!("{}: пропуск, нет подсреза {}", sig_name(sig), sig_name(d));
                log(&msg);
                rep.skipped.push(msg);
                continue;
            }
            if own + dep_bytes > o.max_ram_bytes {
                let msg = format!(
                    "{}: пропуск, нужно ~{} МБ ОЗУ (> лимита {} МБ)",
                    sig_name(sig),
                    (own + dep_bytes) >> 20,
                    o.max_ram_bytes >> 20
                );
                log(&msg);
                rep.skipped.push(msg);
                continue;
            }
            let missing_bytes: u64 = deps.iter().filter(|d| !cache.contains_key(*d)).filter_map(|d| Indexer::new::<R::Geo>(*d).ok()).map(|i| 2 * i.n).sum();
            loop {
                let cached: u64 = cache.values().map(|t| t.vals.len() as u64).sum();
                if cached + missing_bytes + own <= o.max_ram_bytes {
                    break;
                }
                let victim = cache.iter().filter(|(k, _)| !deps.contains(k)).max_by_key(|(_, t)| t.vals.len()).map(|(k, _)| *k);
                match victim {
                    Some(k) => {
                        cache.remove(&k);
                    }
                    None => break,
                }
            }
            for d in &deps {
                if !cache.contains_key(d) {
                    let (_, vals) = file::read_all(&slice_path(&o.dir, *d), R::FORCED, *d)?;
                    cache.insert(*d, Arc::new(SliceTab { ix: Indexer::new::<R::Geo>(*d)?, vals }));
                }
            }
            rep.peak_cache_bytes = rep.peak_cache_bytes.max(cache.values().map(|t| t.vals.len() as u64).sum::<u64>() + own);
            let ts = Instant::now();
            *ctl.current.lock().unwrap_or_else(|e| e.into_inner()) = Some((sig_name(sig), 2 * ix.n, ts));
            let computed = compute_ctl::<R>(&pool, &ix, cache, ctl);
            *ctl.current.lock().unwrap_or_else(|e| e.into_inner()) = None;
            let Some(c) = computed else {
                log(&format!("{}: расчёт прерван, срез не записан", sig_name(sig)));
                rep.cancelled = true;
                break 'outer;
            };
            if ctl.strict && c.anomalies > 0 {
                return Err(format!("{}: аномалий {} (потомок вне загруженных подсрезов) — срез не записан", sig_name(sig), c.anomalies));
            }
            let mut c = c;
            if ctl.debug_corrupt {
                for (i, v) in c.vals.iter_mut().enumerate() {
                    if i % 3 == 0 && *v != 0 && *v != TERM {
                        *v = if (*v as i8) > 0 { v.wrapping_add(2) } else { v.wrapping_sub(2) };
                    }
                }
            }
            if ctl.verify_samples > 0 {
                if let Err(e) = verify_sample::<R>(&ix, &c.vals, cache, ctl.verify_samples) {
                    return Err(format!("{}: самопроверка среза не пройдена ({e}) — файл не записан", sig_name(sig)));
                }
            }
            let packed = pool.install(|| file::pack(sig, R::FORCED, ix.nsq, ix.n, &c.vals))?;
            let bytes = packed.total_bytes();
            if disk + bytes > o.max_disk_bytes {
                log(&format!(
                    "{}: размер {} МБ превысил бы лимит диска ({} МБ): остановка",
                    sig_name(sig),
                    bytes >> 20,
                    o.max_disk_bytes >> 20
                ));
                rep.stopped_disk = true;
                break 'outer;
            }
            file::write_atomic(&path, &packed).map_err(|e| format!("{}: {e}", path.display()))?;
            disk += bytes;
            let secs = ts.elapsed().as_secs_f64();
            let st = SliceStat {
                name: sig_name(sig),
                positions: 2 * ix.n,
                bytes,
                secs,
                max_dist: c.max_dist,
                wins: c.wins,
                losses: c.losses,
                draws: c.draws,
            };
            log(&format!(
                "{}: {} поз., {:.1} КБ, {:.2} с, уровней {}, макс. {} , В/П/Н {}/{}/{}{}{}",
                st.name,
                st.positions,
                bytes as f64 / 1024.0,
                secs,
                c.levels,
                c.max_dist,
                c.wins,
                c.losses,
                c.draws,
                if c.anomalies > 0 { format!(" АНОМАЛИЙ {}", c.anomalies) } else { String::new() },
                if c.saturated { " [насыщение]" } else { "" }
            ));
            cache.insert(sig, Arc::new(SliceTab { ix, vals: c.vals }));
            update_manifest(&o.dir, R::NAME, R::FORCED, Some(st.clone()));
            rep.bytes += bytes;
            rep.built.push(st);
            if ctl.one_slice {
                break 'outer;
            }
        }
    }
    update_manifest(&o.dir, R::NAME, R::FORCED, None);
    rep.secs = t0.elapsed().as_secs_f64();
    Ok(rep)
}
