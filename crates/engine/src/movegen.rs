
use crate::geometry::Geometry;
use crate::position::{Position, Side};
use crate::rules::{CapturePriority, KingLanding, PromotionInCapture, Removal, Rules};
use std::mem::MaybeUninit;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Move {
    pub captured: u64,
    pub from: u8,
    pub to: u8,
    pub promo: bool,
}

impl Move {
    pub const NULL: Move = Move { captured: 0, from: 0, to: 0, promo: false };
    #[inline(always)]
    pub fn is_capture(&self) -> bool {
        self.captured != 0
    }
}

pub const MAX_MOVES: usize = 512;

pub struct MoveList {
    moves: [MaybeUninit<Move>; MAX_MOVES],
    pub len: usize,
    pub overflowed: bool,
}

impl MoveList {
    #[inline(always)]
    pub fn new() -> Self {
        MoveList {
            moves: unsafe { MaybeUninit::<[MaybeUninit<Move>; MAX_MOVES]>::uninit().assume_init() },
            len: 0,
            overflowed: false,
        }
    }
    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        unsafe { std::slice::from_raw_parts(self.moves.as_ptr() as *const Move, self.len) }
    }
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [Move] {
        unsafe { std::slice::from_raw_parts_mut(self.moves.as_mut_ptr() as *mut Move, self.len) }
    }
    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        if self.len < MAX_MOVES {
            self.moves[self.len] = MaybeUninit::new(m);
            self.len += 1;
        } else {
            self.overflowed = true;
        }
    }
    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

impl std::ops::Deref for MoveList {
    type Target = [Move];
    #[inline(always)]
    fn deref(&self) -> &[Move] {
        self.as_slice()
    }
}

impl std::ops::DerefMut for MoveList {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [Move] {
        self.as_mut_slice()
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

pub trait Sink {
    const WANT_PATH: bool;
    const ALL_LANDINGS: bool = false;
    fn push(&mut self, from: u8, to: u8, captured: u64, promo: bool, path: &[u8]);
    fn clear(&mut self);
}

pub struct ListSink<'a, const DEDUP: bool>(pub &'a mut MoveList);

impl<'a, const DEDUP: bool> Sink for ListSink<'a, DEDUP> {
    const WANT_PATH: bool = false;
    const ALL_LANDINGS: bool = !DEDUP;
    #[inline(always)]
    fn push(&mut self, from: u8, to: u8, captured: u64, promo: bool, _path: &[u8]) {
        let m = Move { captured, from, to, promo };
        if DEDUP && captured != 0 && self.0.as_slice().contains(&m) {
            return;
        }
        self.0.push(m);
    }
    #[inline(always)]
    fn clear(&mut self) {
        self.0.clear();
    }
}

#[derive(Default)]
pub struct CountSink {
    pub count: u64,
}

impl Sink for CountSink {
    const WANT_PATH: bool = false;
    const ALL_LANDINGS: bool = true;
    #[inline(always)]
    fn push(&mut self, _f: u8, _t: u8, _c: u64, _p: bool, _path: &[u8]) {
        self.count += 1;
    }
    #[inline(always)]
    fn clear(&mut self) {
        self.count = 0;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathMove {
    pub mv: Move,
    pub path: Vec<u8>,
}

#[derive(Default)]
pub struct PathSink {
    pub out: Vec<PathMove>,
}

impl Sink for PathSink {
    const WANT_PATH: bool = true;
    fn push(&mut self, from: u8, to: u8, captured: u64, promo: bool, path: &[u8]) {
        self.out.push(PathMove { mv: Move { captured, from, to, promo }, path: path.to_vec() });
    }
    fn clear(&mut self) {
        self.out.clear();
    }
}

const MAX_PATH: usize = 40;

struct Gen<'a, R: Rules, S: Sink, const W: bool> {
    opp: u64,
    opp_kings: u64,
    empty: u64,
    from: u8,
    mover_king: bool,
    kord: u32,
    best: u64,
    path: [u8; MAX_PATH],
    sink: &'a mut S,
    _r: std::marker::PhantomData<R>,
}

type Gm<R> = <R as Rules>::Geo;

impl<'a, R: Rules, S: Sink, const W: bool> Gen<'a, R, S, W> {
    const IMM: bool = matches!(R::REMOVAL, Removal::Immediate);
    const PRIO: CapturePriority = if R::FORCED { R::PRIORITY } else { CapturePriority::None };
    const HAS_PRIO: bool = !matches!(Self::PRIO, CapturePriority::None);
    const ITALIAN: bool = matches!(Self::PRIO, CapturePriority::Italian);
    const REVERSE_OK: bool = Self::IMM && !R::NO_U_TURN;
    const LAND_ANY: bool = R::KING_FLYING && matches!(R::KING_LANDING, KingLanding::Any);

    #[inline(always)]
    fn promo_row() -> u64 {
        if W {
            Gm::<R>::WHITE_PROMO
        } else {
            Gm::<R>::BLACK_PROMO
        }
    }

    #[inline(always)]
    fn emp(&self, cap: u64) -> u64 {
        if Self::IMM {
            self.empty | cap
        } else {
            self.empty
        }
    }

    #[inline(always)]
    fn key(&self, cap: u64) -> u64 {
        match Self::PRIO {
            CapturePriority::None => 0,
            CapturePriority::MaxCount => cap.count_ones() as u64,
            CapturePriority::Spanish => ((cap.count_ones() as u64) << 8) | (cap & self.opp_kings).count_ones() as u64,
            CapturePriority::Czech => self.mover_king as u64,
            CapturePriority::Italian => {
                ((cap.count_ones() as u64) << 40)
                    | ((self.mover_king as u64) << 39)
                    | (((cap & self.opp_kings).count_ones() as u64) << 32)
                    | self.kord as u64
            }
        }
    }

    #[inline(always)]
    fn emit(&mut self, to_bit: u64, cap: u64, promo: bool, plen: usize) {
        if Self::HAS_PRIO {
            let k = self.key(cap);
            if k < self.best {
                return;
            }
            if k > self.best {
                self.best = k;
                self.sink.clear();
            }
        }
        let to = to_bit.trailing_zeros() as u8;
        if S::WANT_PATH {
            self.path[plen - 1] = to;
        }
        let from = self.from;
        let path: &[u8] = if S::WANT_PATH { &self.path[..plen] } else { &[] };
        self.sink.push(from, to, cap, promo, path);
    }

    #[inline(always)]
    fn set_path(&mut self, bit: u64, idx: usize) {
        if S::WANT_PATH {
            self.path[idx] = bit.trailing_zeros() as u8;
        }
    }

    #[inline(always)]
    fn slide(p: u64, d: usize, em: u64) -> u64 {
        let mut x = Gm::<R>::shift(p, d);
        while x & em != 0 {
            x = Gm::<R>::shift(x, d);
        }
        x
    }

    #[inline(always)]
    fn hit(p: u64, d: usize, em: u64) -> u64 {
        if R::KING_FLYING {
            Self::slide(p, d, em)
        } else {
            Gm::<R>::shift(p, d)
        }
    }

    #[inline(always)]
    fn king_can_step(&self, p: u64, d: usize, cap: u64) -> bool {
        let em = self.emp(cap);
        let v = Self::hit(p, d, em) & self.opp & !cap;
        v != 0 && Gm::<R>::shift(v, d) & em != 0
    }

    fn man_cap(&mut self, p: u64, cap: u64, plen: usize) -> bool {
        let mut any = false;
        let em = self.emp(cap);
        for d in 0..Gm::<R>::NDIRS {
            if !R::MEN_CAPTURE_BACKWARD {
                let fwd = if W { Gm::<R>::FWD_WHITE } else { Gm::<R>::FWD_BLACK };
                if !fwd.contains(&d) {
                    continue;
                }
            }
            let mut v = Gm::<R>::shift(p, d) & self.opp & !cap;
            if !R::MAN_CAPTURES_KING {
                v &= !self.opp_kings;
            }
            if v == 0 {
                continue;
            }
            let l = Gm::<R>::shift(v, d) & em;
            if l == 0 {
                continue;
            }
            any = true;
            let ncap = cap | v;
            let sk = self.kord;
            if Self::ITALIAN {
                self.kord = (sk << 1) | (v & self.opp_kings != 0) as u32;
            }
            self.set_path(l, plen);
            if l & Self::promo_row() != 0 {
                match R::PROMOTION {
                    PromotionInCapture::ContinueAsKing => {
                        let mut cont = false;
                        for e in 0..Gm::<R>::NDIRS {
                            if R::NO_U_TURN && e == (d + Gm::<R>::NDIRS / 2) % Gm::<R>::NDIRS {
                                continue;
                            }
                            cont |= self.king_step(l, e, ncap, true, plen + 1);
                        }
                        if !cont {
                            self.emit(l, ncap, true, plen + 1);
                        }
                    }
                    PromotionInCapture::Stop => {
                        self.emit(l, ncap, true, plen + 1);
                    }
                    PromotionInCapture::Transit => {
                        let cont = self.man_cap(l, ncap, plen + 1);
                        if !cont {
                            self.emit(l, ncap, true, plen + 1);
                        }
                    }
                }
            } else {
                let cont = self.man_cap(l, ncap, plen + 1);
                if !cont {
                    self.emit(l, ncap, false, plen + 1);
                }
            }
            self.kord = sk;
        }
        any
    }

    fn king_step(&mut self, p: u64, d: usize, cap: u64, promo: bool, plen: usize) -> bool {
        let em = self.emp(cap);
        let v = Self::hit(p, d, em) & self.opp & !cap;
        if v == 0 {
            return false;
        }
        let first = Gm::<R>::shift(v, d) & em;
        if first == 0 {
            return false;
        }
        let ncap = cap | v;
        let sk = self.kord;
        if Self::ITALIAN {
            self.kord = (sk << 1) | (v & self.opp_kings != 0) as u32;
        }
        let (lands, last) = if Self::LAND_ANY {
            let mut lands = 0u64;
            let mut last = 0u64;
            let mut y = first;
            while y != 0 {
                lands |= y;
                last = y;
                y = Gm::<R>::shift(y, d) & em;
            }
            (lands, last)
        } else {
            (first, first)
        };
        let [e1, e2] = Gm::<R>::perp(d);
        let rev = (d + Gm::<R>::NDIRS / 2) % Gm::<R>::NDIRS;
        let straight = self.king_can_step(last, d, ncap);
        let mut turns = 0u64;
        let mut x = lands;
        while x != 0 {
            let b = x & x.wrapping_neg();
            x &= x - 1;
            if self.king_can_step(b, e1, ncap)
                || self.king_can_step(b, e2, ncap)
                || (Self::REVERSE_OK && self.king_can_step(b, rev, ncap))
            {
                turns |= b;
            }
        }
        if straight {
            if S::ALL_LANDINGS {
                let mut x = lands;
                while x != 0 {
                    let b = x & x.wrapping_neg();
                    x &= x - 1;
                    self.set_path(b, plen);
                    self.king_step(b, d, ncap, promo, plen + 1);
                }
            } else {
                self.king_step(last, d, ncap, promo, plen);
            }
        }
        let mut x = turns;
        while x != 0 {
            let b = x & x.wrapping_neg();
            x &= x - 1;
            self.set_path(b, plen);
            self.king_step(b, e1, ncap, promo, plen + 1);
            self.king_step(b, e2, ncap, promo, plen + 1);
            if Self::REVERSE_OK {
                self.king_step(b, rev, ncap, promo, plen + 1);
            }
        }
        if !straight {
            let stop = if R::KING_MUST_CONTINUE {
                if turns == 0 {
                    lands
                } else {
                    0
                }
            } else {
                lands & !turns
            };
            let mut x = stop;
            while x != 0 {
                let b = x & x.wrapping_neg();
                x &= x - 1;
                self.emit(b, ncap, promo, plen + 1);
            }
        }
        self.kord = sk;
        true
    }
}

#[inline]
pub fn generate<R: Rules, S: Sink>(pos: &Position<R>, sink: &mut S) {
    match pos.side {
        Side::White => gen_side::<R, S, true, false>(pos, sink),
        Side::Black => gen_side::<R, S, false, false>(pos, sink),
    }
}

#[inline]
pub fn generate_captures<R: Rules, S: Sink>(pos: &Position<R>, sink: &mut S) {
    match pos.side {
        Side::White => gen_side::<R, S, true, true>(pos, sink),
        Side::Black => gen_side::<R, S, false, true>(pos, sink),
    }
}

fn gen_side<R: Rules, S: Sink, const W: bool, const CAPS_ONLY: bool>(pos: &Position<R>, sink: &mut S) {
    let (own, opp) = if W { (pos.white, pos.black) } else { (pos.black, pos.white) };
    let occ = pos.white | pos.black;
    let free = Gm::<R>::VALID & !occ;
    let kings = pos.kings & own;
    let men = own & !kings;
    let mut found = false;
    {
        let mut g = Gen::<R, S, W> {
            opp,
            opp_kings: opp & pos.kings,
            empty: 0,
            from: 0,
            mover_king: false,
            kord: 0,
            best: 0,
            path: [0; MAX_PATH],
            sink: &mut *sink,
            _r: std::marker::PhantomData,
        };
        let targets = if R::MAN_CAPTURES_KING { opp } else { opp & !pos.kings };
        let mut mm = 0u64;
        for d in 0..Gm::<R>::NDIRS {
            if !R::MEN_CAPTURE_BACKWARD {
                let fwd = if W { Gm::<R>::FWD_WHITE } else { Gm::<R>::FWD_BLACK };
                if !fwd.contains(&d) {
                    continue;
                }
            }
            let rev = (d + Gm::<R>::NDIRS / 2) % Gm::<R>::NDIRS;
            mm |= Gm::<R>::shift(Gm::<R>::shift(free, rev) & targets, rev);
        }
        let mut x = kings | (mm & men);
        while x != 0 {
            let p = x & x.wrapping_neg();
            x &= x - 1;
            g.empty = free | p;
            g.from = p.trailing_zeros() as u8;
            if p & kings != 0 {
                g.mover_king = true;
                for d in 0..Gm::<R>::NDIRS {
                    found |= g.king_step(p, d, 0, false, 0);
                }
            } else {
                g.mover_king = false;
                found |= g.man_cap(p, 0, 0);
            }
        }
    }
    if CAPS_ONLY || (R::FORCED && found) {
        return;
    }
    let promo_row = if W { Gm::<R>::WHITE_PROMO } else { Gm::<R>::BLACK_PROMO };
    let fwd = if W { Gm::<R>::FWD_WHITE } else { Gm::<R>::FWD_BLACK };
    for &d in fwd {
        let back = (d + Gm::<R>::NDIRS / 2) % Gm::<R>::NDIRS;
        let mut t = Gm::<R>::shift(men, d) & free;
        while t != 0 {
            let b = t & t.wrapping_neg();
            t &= t - 1;
            let from = Gm::<R>::shift(b, back).trailing_zeros() as u8;
            let to = b.trailing_zeros() as u8;
            sink.push(from, to, 0, b & promo_row != 0, &[to]);
        }
    }
    let mut k = kings;
    while k != 0 {
        let p = k & k.wrapping_neg();
        k &= k - 1;
        let from = p.trailing_zeros() as u8;
        for d in 0..Gm::<R>::NDIRS {
            let mut x = Gm::<R>::shift(p, d) & free;
            while x != 0 {
                let to = x.trailing_zeros() as u8;
                sink.push(from, to, 0, false, &[to]);
                if !R::KING_FLYING {
                    break;
                }
                x = Gm::<R>::shift(x, d) & free;
            }
        }
    }
}

#[inline]
pub fn has_quiet<R: Rules>(pos: &Position<R>) -> bool {
    let (own, w) = match pos.side {
        Side::White => (pos.white, true),
        Side::Black => (pos.black, false),
    };
    let free = Gm::<R>::VALID & !(pos.white | pos.black);
    let kings = pos.kings & own;
    let men = own & !kings;
    let fwd = if w { Gm::<R>::FWD_WHITE } else { Gm::<R>::FWD_BLACK };
    for &d in fwd {
        if Gm::<R>::shift(men, d) & free != 0 {
            return true;
        }
    }
    for d in 0..Gm::<R>::NDIRS {
        if Gm::<R>::shift(kings, d) & free != 0 {
            return true;
        }
    }
    false
}

pub fn perft<R: Rules, const DEDUP: bool>(pos: &Position<R>, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    if depth == 1 && !DEDUP {
        let mut s = CountSink::default();
        generate(pos, &mut s);
        return s.count;
    }
    let mut list = MoveList::new();
    generate(pos, &mut ListSink::<DEDUP>(&mut list));
    if depth == 1 {
        return list.len as u64;
    }
    let mut n = 0;
    for m in list.as_slice() {
        n += perft::<R, DEDUP>(&pos.make(m), depth - 1);
    }
    n
}
