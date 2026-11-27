
use super::view::{View, ViewTables};
use crate::geometry::Geometry;
use crate::position::{Position, Side};
use crate::rules::Rules;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Global {
    MenDiff,
    KingsOwn,
    KingsOpp,
    KingSq,
    KingMobility,
    Mobility,
    Tempo,
    Balance,
}

pub const ALL_GLOBALS: [Global; 8] = [
    Global::MenDiff,
    Global::KingsOwn,
    Global::KingsOpp,
    Global::KingSq,
    Global::KingMobility,
    Global::Mobility,
    Global::Tempo,
    Global::Balance,
];

impl Global {
    pub fn name(self) -> &'static str {
        match self {
            Global::MenDiff => "men_diff",
            Global::KingsOwn => "kings_own",
            Global::KingsOpp => "kings_opp",
            Global::KingSq => "king_sq",
            Global::KingMobility => "king_mobility",
            Global::Mobility => "mobility",
            Global::Tempo => "tempo",
            Global::Balance => "balance",
        }
    }

    pub fn parse(s: &str) -> Option<Global> {
        ALL_GLOBALS.iter().copied().find(|g| g.name() == s)
    }

    pub fn nweights(self, nsq: usize) -> usize {
        match self {
            Global::KingSq => nsq,
            _ => 1,
        }
    }

    pub fn compute<R: Rules>(self, pos: &Position<R>, v: &View, t: &ViewTables, out: &mut [i32]) {
        match self {
            Global::MenDiff => out[0] = v.own_men.count_ones() as i32 - v.opp_men.count_ones() as i32,
            Global::KingsOwn => out[0] = v.own_kings.count_ones() as i32,
            Global::KingsOpp => out[0] = v.opp_kings.count_ones() as i32,
            Global::KingSq => {
                for (k, o) in out.iter_mut().enumerate() {
                    *o = ((v.own_kings >> k) & 1) as i32 - ((v.opp_kings >> k) & 1) as i32;
                }
            }
            Global::KingMobility | Global::Mobility => {
                let (own, opp) = match pos.side {
                    Side::White => (pos.white, pos.black),
                    Side::Black => (pos.black, pos.white),
                };
                let w = pos.side == Side::White;
                let (m1, k1) = quiet_moves::<R>(own, opp, pos.kings, w);
                let (m2, k2) = quiet_moves::<R>(opp, own, pos.kings, !w);
                out[0] = if self == Global::KingMobility { k1 - k2 } else { m1 + k1 - m2 - k2 };
            }
            Global::Tempo => out[0] = adv_sum(v.own_men, t, false) - adv_sum(v.opp_men, t, true),
            Global::Balance => {
                out[0] = imbalance(v.own_men | v.own_kings, t) - imbalance(v.opp_men | v.opp_kings, t);
            }
        }
    }
}

pub fn adv_sum(mut x: u64, t: &ViewTables, flip: bool) -> i32 {
    let mut s = 0i32;
    while x != 0 {
        let k = x.trailing_zeros() as usize;
        x &= x - 1;
        let r = t.row[k] as i32;
        s += if flip { t.n as i32 - 1 - r } else { r };
    }
    s
}

fn imbalance(mut x: u64, t: &ViewTables) -> i32 {
    let (mut l, mut r) = (0i32, 0i32);
    while x != 0 {
        let k = x.trailing_zeros() as usize;
        x &= x - 1;
        if (t.file[k] as usize) < t.n / 2 {
            l += 1;
        } else {
            r += 1;
        }
    }
    (l - r).abs()
}

pub(crate) fn quiet_moves<R: Rules>(own: u64, opp: u64, kings: u64, white: bool) -> (i32, i32) {
    type G<R> = <R as Rules>::Geo;
    let empty = G::<R>::VALID & !(own | opp);
    let men = own & !kings;
    let ks = own & kings;
    let dirs = if white { G::<R>::FWD_WHITE } else { G::<R>::FWD_BLACK };
    let mut m = 0i32;
    for &d in dirs {
        m += (G::<R>::shift(men, d) & empty).count_ones() as i32;
    }
    let mut k = 0i32;
    for d in 0..G::<R>::NDIRS {
        let mut front = G::<R>::shift(ks, d) & empty;
        if R::KING_FLYING {
            while front != 0 {
                k += front.count_ones() as i32;
                front = G::<R>::shift(front, d) & empty;
            }
        } else {
            k += front.count_ones() as i32;
        }
    }
    (m, k)
}

pub fn names() -> Vec<&'static str> {
    ALL_GLOBALS.iter().map(|g| g.name()).collect()
}
