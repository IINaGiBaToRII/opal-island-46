
use super::cache::stats;
use super::index::*;
use super::probe::Egdb;
use super::{egdb_generate, GenOpts};
use crate::position::{Position, Side};
use crate::rules::{Russian, Rules};
use crate::{GameConfig, VariantId};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

type R = Russian<true>;

pub(super) static STATS_LOCK: Mutex<()> = Mutex::new(());

fn tmp_dir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("runner-egdb-speed-{tag}-{}-{nanos}", std::process::id()))
}

pub(super) fn db_dir(pieces: u32) -> &'static Path {
    static D3: OnceLock<PathBuf> = OnceLock::new();
    static D4: OnceLock<PathBuf> = OnceLock::new();
    let cell = if pieces == 3 { &D3 } else { &D4 };
    cell.get_or_init(|| {
        let dir = tmp_dir(&format!("db{pieces}"));
        let cfg = GameConfig { variant: VariantId::Russian, forced_capture: true };
        let o = GenOpts { dir: dir.clone(), pieces, threads: 4, max_ram_bytes: 4 << 30, max_disk_bytes: 1 << 30 };
        egdb_generate(&cfg, &o, &mut |_| {}).unwrap();
        dir
    })
}

pub(super) fn positions(sig: Sig, step: usize) -> Vec<Position<R>> {
    let ix = Indexer::new::<<R as Rules>::Geo>(sig).unwrap();
    let mut v = vec![];
    for i in (0..ix.n).step_by(step) {
        let m = ix.unrank(i);
        if ix.valid(&m) {
            v.push(ix.to_position::<R>(&m, Side::White));
            v.push(ix.to_position::<R>(&m, Side::Black));
        }
    }
    v
}

#[test]
fn stats_by_layer_on_changes_off_silent() {
    let _g = STATS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let db = Egdb::<R>::open_ro(db_dir(3)).unwrap();
    assert_eq!(db.max_pieces(), 3);
    let p3 = positions([1, 0, 1, 1], 5);
    let p4 = positions([1, 1, 1, 1], 4001);
    assert!(!p3.is_empty() && !p4.is_empty());
    let run = || -> Vec<Option<i32>> { p3.iter().chain(p4.iter()).map(|p| db.probe_score(p, 1)).collect() };
    stats::enable(false);
    stats::take();
    stats::take_layers();
    let off = run();
    let l = stats::take_layers();
    assert!(l.iter().flatten().all(|&x| x == 0), "выключенные счётчики считают");
    stats::enable(true);
    let on = run();
    stats::enable(false);
    let l = stats::take_layers();
    stats::take();
    assert_eq!(off, on, "счётчики изменили ответы базы");
    let n3 = p3.len() as u64;
    assert!(l[3][stats::K_CALL] >= n3 && l[3][stats::K_PROBE] >= n3, "слой 3: {:?}", l[3]);
    let blk3 = l[3][stats::K_TL] + l[3][stats::K_SHARED] + l[3][stats::K_ZSTD] + l[3][stats::K_RAM];
    assert!(blk3 >= n3, "слой 3, блоки: {:?}", l[3]);
    assert!(l[4][stats::K_CALL] >= p4.len() as u64, "слой 4 (выше базы): {:?}", l[4]);
    let s = stats::format_layers(&l, 1000, "");
    assert!(s.contains("STATS-L 3 ") && s.contains("STATS-L 4 "), "{s}");
    assert!(stats::format_layers(&l, 1000, "REP 1 ").contains("STATS-L REP 1 3 "));
}

pub(super) fn search_lines(db: &Egdb<R>, ps: &[Position<R>], depth: u32) -> Vec<String> {
    use crate::eval::SimpleEval;
    use crate::search::{search_tt_db, SearchLimits};
    use crate::tt::Tt;
    use std::sync::atomic::AtomicBool;
    let eval = SimpleEval::new::<R>();
    let stop = AtomicBool::new(false);
    let lim = SearchLimits { depth: Some(depth), hash_mb: 1, threads: 1, ..Default::default() };
    ps.iter()
        .map(|p| {
            let mut tt = Tt::new(1);
            let r = search_tt_db(p, &[], &eval, &lim, &stop, &mut |_| {}, &mut tt, Some(db));
            format!("{} {} {:?} {} {:?}", r.nodes, r.depth, r.best, r.score, r.pv)
        })
        .collect()
}

pub(super) fn search_positions() -> Vec<Position<R>> {
    let mut v = vec![];
    for sig in [[2u8, 0, 2, 1], [1, 1, 2, 1], [2, 1, 2, 1], [1, 1, 1, 2]] {
        let ix = Indexer::new::<<R as Rules>::Geo>(sig).unwrap();
        let step = (ix.n / 5).max(1) as usize;
        let mut k = 0;
        for i in (7..ix.n).step_by(step) {
            let m = ix.unrank(i);
            if ix.valid(&m) && k < 4 {
                v.push(ix.to_position::<R>(&m, if k % 2 == 0 { Side::White } else { Side::Black }));
                k += 1;
            }
        }
    }
    v
}

#[test]
fn ram_layers_identity_and_path() {
    let _g = STATS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = db_dir(4);
    let plain = Egdb::<R>::open_ro_ram(dir, 0, 0).unwrap();
    let ram = Egdb::<R>::open_ro_ram(dir, 4, 64).unwrap();
    assert_eq!(plain.max_pieces(), 4);
    assert_eq!(plain.ram_info().0, 0);
    let (n, bytes) = ram.ram_info();
    assert_eq!(n as usize, plain.sigs().len());
    assert_eq!(bytes, plain.sigs().iter().map(|s| 2 * plain.slice_n(*s).unwrap()).sum::<u64>());
    let small = Egdb::<R>::open_ro_ram(dir, 4, 1).unwrap();
    assert!(small.ram_info().0 > 0 && small.ram_info().0 < n && small.ram_info().1 <= 1 << 20, "{:?}", small.ram_info());
    let r3 = Egdb::<R>::open_ro_ram(dir, 3, 64).unwrap();
    assert_eq!(r3.ram_info().0 as usize, plain.sigs().iter().filter(|s| s.iter().sum::<u8>() <= 3).count());
    for other in [&ram, &small, &r3] {
        let rep = super::verify::identity(&plain, other, 4).unwrap();
        assert!(rep.ok(), "{:?}", rep.mismatches);
        assert_eq!(rep.values, plain.sigs().iter().map(|s| 2 * plain.slice_n(*s).unwrap()).sum::<u64>());
        assert!(rep.probes > 0);
    }
    let ps = positions([1, 1, 1, 1], 997);
    stats::take_layers();
    stats::enable(true);
    let a: Vec<_> = ps.iter().map(|p| plain.probe(p)).collect();
    let la = stats::take_layers();
    let b: Vec<_> = ps.iter().map(|p| ram.probe(p)).collect();
    stats::enable(false);
    let lb = stats::take_layers();
    assert_eq!(a, b);
    assert_eq!(la[4][stats::K_RAM], 0);
    assert!(lb[4][stats::K_RAM] >= ps.len() as u64, "{:?}", lb[4]);
    let sp = search_positions();
    assert!(sp.len() >= 8);
    let base = search_lines(&plain, &sp, 7);
    assert_eq!(base, search_lines(&plain, &sp, 7), "поиск сам недетерминирован");
    assert_eq!(base, search_lines(&ram, &sp, 7));
}

fn repacked(shift: u32) -> &'static Path {
    static D: Mutex<Vec<(u32, &'static Path)>> = Mutex::new(Vec::new());
    let mut g = D.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, p)) = g.iter().find(|x| x.0 == shift) {
        return p;
    }
    let dst: &'static Path = Box::leak(tmp_dir(&format!("rp{shift}")).into_boxed_path());
    super::file::repack_dir(db_dir(4), dst, shift, &mut |_| {}).unwrap();
    g.push((shift, dst));
    dst
}

#[test]
fn small_blocks_identity_and_format() {
    let _g = STATS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let plain = Egdb::<R>::open_ro_ram(db_dir(4), 0, 0).unwrap();
    for s in plain.sigs() {
        assert_eq!(plain.slice_shift(s), Some(super::file::BLOCK_SHIFT));
    }
    for shift in [10u32, 12] {
        let small = Egdb::<R>::open_ro_ram(repacked(shift), 0, 0).unwrap();
        assert_eq!(small.max_pieces(), 4);
        assert_eq!(small.sigs(), plain.sigs());
        let mut more = 0;
        for s in plain.sigs() {
            assert_eq!(small.slice_shift(s), Some(shift));
            more += (small.slice_nblocks(s).unwrap() > plain.slice_nblocks(s).unwrap()) as u32;
        }
        assert!(more > 0);
        let rep = super::verify::identity(&plain, &small, 4).unwrap();
        assert!(rep.ok(), "{:?}", rep.mismatches);
        assert_eq!(rep.values, plain.sigs().iter().map(|s| 2 * plain.slice_n(*s).unwrap()).sum::<u64>());
    }
    assert!(super::file::repack_dir(db_dir(4), db_dir(4), 12, &mut |_| {}).is_err());
    assert!(super::file::repack_dir(db_dir(4), &tmp_dir("bad"), 9, &mut |_| {}).is_err());
    let small = Egdb::<R>::open_ro_ram(repacked(12), 0, 0).unwrap();
    let sp = search_positions();
    assert_eq!(search_lines(&plain, &sp, 7), search_lines(&small, &sp, 7));
}

#[test]
fn tl_slots_change_path_not_answers() {
    use super::cache::{set_tl_slots, tl_slots};
    let _g = STATS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let db = Egdb::<R>::open_ro_ram(db_dir(4), 0, 0).unwrap();
    let ps: Vec<Position<R>> = [[1u8, 1, 1, 1], [2, 0, 1, 1], [0, 2, 0, 2]].iter().flat_map(|&s| positions(s, 509)).collect();
    let run = |slots: usize| -> (Vec<Option<i8>>, u64) {
        set_tl_slots(slots);
        stats::take_layers();
        stats::enable(true);
        let a: Vec<_> = ps.iter().map(|p| db.probe(p)).collect();
        let b: Vec<_> = ps.iter().map(|p| db.probe(p)).collect();
        stats::enable(false);
        let l = stats::take_layers();
        assert_eq!(a, b);
        (a, l[4][stats::K_TL])
    };
    let def = tl_slots();
    assert_eq!(def, 64, "умолчание поточного кэша изменилось");
    let (v1, h1) = run(1);
    let (v4k, h4k) = run(4096);
    set_tl_slots(def);
    let (v64, _) = run(64);
    assert_eq!(v1, v4k);
    assert_eq!(v1, v64);
    assert!(h4k > h1, "больший поточный кэш не дал больше попаданий: {h4k} <= {h1}");
}
