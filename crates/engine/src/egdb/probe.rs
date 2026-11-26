
use super::cache::{self, Block};
use super::file::{self, Header, BLOCK};
use super::index::*;
use super::{fixed_value, EgdbInfo};
use crate::datadir::{self, Kind};
use crate::movegen::{generate, ListSink, Move, MoveList};
use crate::position::{Position, Side};
use crate::rules::Rules;
use memmap2::Mmap;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

struct Slice {
    mmap: Mmap,
    h: Header,
    ix: Indexer,
    uid: u32,
}

pub struct Egdb<R: Rules> {
    dir: PathBuf,
    slices: HashMap<Sig, Slice>,
    max_pieces: u32,
    bytes: u64,
    _r: PhantomData<R>,
}

impl<R: Rules> Egdb<R> {
    pub fn open() -> Result<Self, String> {
        Self::open_at(&datadir::knowledge_dir(R::NAME, R::FORCED, Kind::Egdb))
    }

    pub fn open_at(dir: &Path) -> Result<Self, String> {
        super::quarantine_stale(dir);
        Self::open_ro(dir)
    }

    pub fn open_ro(dir: &Path) -> Result<Self, String> {
        Self::open_ro_inner(dir, None)
    }

    pub fn open_ro_capped(dir: &Path, max_pieces: u32) -> Result<Self, String> {
        Self::open_ro_inner(dir, Some(max_pieces))
    }

    fn open_ro_inner(dir: &Path, cap: Option<u32>) -> Result<Self, String> {
        let mut slices = HashMap::new();
        let mut bytes = 0u64;
        let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().map(|x| x != "egdb").unwrap_or(true) {
                continue;
            }
            let Some(sig) = p.file_stem().and_then(|s| s.to_str()).and_then(parse_sig_name) else { continue };
            if cap.is_some_and(|c| sig.iter().map(|&x| x as u32).sum::<u32>() > c) {
                continue;
            }
            let f = std::fs::File::open(&p).map_err(|e| e.to_string())?;
            let mmap = unsafe { Mmap::map(&f) }.map_err(|e| e.to_string())?;
            let Ok(h) = file::decode_header(&mmap) else { continue };
            let ix = Indexer::new::<R::Geo>(sig)?;
            if h.sig != sig || h.forced != R::FORCED || h.n != ix.n || h.nsq as usize != ix.nsq || !file::validate_len(&h, mmap.len() as u64) {
                continue;
            }
            bytes += mmap.len() as u64;
            let uid = cache::file_uid(&f, &p);
            slices.insert(sig, Slice { mmap, h, ix, uid });
        }
        let mut max_pieces = 0;
        'o: for t in 2..=16u32 {
            for s in sigs_of_total(t) {
                if !(R::ONE_V_ONE_DRAW && s[0] + s[1] == 1 && s[2] + s[3] == 1) && !slices.contains_key(&s) {
                    break 'o;
                }
            }
            max_pieces = t;
        }
        Ok(Egdb {
            dir: dir.to_path_buf(),
            slices,
            max_pieces,
            bytes,
            _r: PhantomData,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn max_pieces(&self) -> u32 {
        self.max_pieces
    }

    pub fn info(&self) -> EgdbInfo {
        EgdbInfo { max_pieces: self.max_pieces, slices: self.slices.len() as u32, bytes: self.bytes }
    }

    pub fn has_slice(&self, s: Sig) -> bool {
        self.slices.contains_key(&s)
    }

    pub fn sigs(&self) -> Vec<Sig> {
        let mut v: Vec<Sig> = self.slices.keys().copied().collect();
        v.sort_by_key(|s| (s[0] + s[1] + s[2] + s[3], s[0] + s[2], *s));
        v
    }

    pub fn slice_n(&self, s: Sig) -> Option<u64> {
        self.slices.get(&s).map(|x| x.h.n)
    }

    fn block(&self, sl: &Slice, b: u64) -> Option<Block> {
        cache::get_or_load(cache::key(sl.uid, b), || file::unpack_block(&sl.mmap, &sl.h, b).ok())
    }

    pub fn value_at(&self, s: Sig, side: Side, idx: u64) -> Option<i8> {
        let sl = self.slices.get(&s)?;
        let g = idx + if side == Side::Black { sl.h.n } else { 0 };
        let blk = self.block(sl, g / BLOCK as u64)?;
        blk.get((g % BLOCK as u64) as usize).map(|v| *v as i8)
    }

    pub fn probe(&self, p: &Position<R>) -> Option<i8> {
        if let Some(v) = fixed_value(p) {
            cache::stats::bump(2);
            return Some(v);
        }
        let sig = sig_of(p);
        let Some(sl) = self.slices.get(&sig) else {
            cache::stats::bump(3);
            return None;
        };
        let m = sl.ix.masks(p);
        if !sl.ix.valid(&m) {
            cache::stats::bump(3);
            return None;
        }
        let Some(idx) = sl.ix.rank(&m) else {
            cache::stats::bump(3);
            return None;
        };
        self.value_at(sig, p.side, idx)
    }

    pub fn probe_score(&self, p: &Position<R>, ply: usize) -> Option<i32> {
        cache::stats::bump(0);
        if p.occ().count_ones() > self.max_pieces {
            return None;
        }
        cache::stats::bump(1);
        self.probe(p).map(|v| {
            let s = super::to_score(v, ply);
            if v == super::SAT || v == -super::SAT { s + super::sat_bonus(p) } else { s }
        })
    }

    pub fn root_pick(&self, p: &Position<R>) -> Option<(Move, i32)> {
        if p.occ().count_ones() > self.max_pieces {
            return None;
        }
        self.best_move(p).map(|(m, v)| (m, super::to_score(v, 0)))
    }

    pub fn best_move(&self, p: &Position<R>) -> Option<(Move, i8)> {
        let v = self.probe(p)?;
        if v == super::SAT || v == -super::SAT {
            return None;
        }
        let mut list = MoveList::new();
        generate(p, &mut ListSink::<true>(&mut list));
        let mut best: Option<(i32, Move)> = None;
        for mv in list.as_slice() {
            let cv = self.probe(&p.make(mv))? as i32;
            let f = if cv < 0 { 1000 - (-cv) } else if cv == 0 { 0 } else { -1000 + cv };
            if best.map(|b| f > b.0).unwrap_or(true) {
                best = Some((f, *mv));
            }
        }
        best.map(|b| (b.1, v))
    }
}
