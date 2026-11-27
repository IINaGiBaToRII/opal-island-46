
use crate::geometry::Geometry;
use crate::position::{Position, Side};
use crate::rules::Rules;

pub type Sig = [u8; 4];
pub const MAX_GROUP: usize = 8;

const fn binom_table() -> [[u64; MAX_GROUP + 1]; 65] {
    let mut t = [[0u64; MAX_GROUP + 1]; 65];
    let mut n = 0;
    while n < 65 {
        t[n][0] = 1;
        let mut k = 1;
        while k <= MAX_GROUP {
            if n > 0 {
                t[n][k] = t[n - 1][k - 1] + t[n - 1][k];
            }
            k += 1;
        }
        n += 1;
    }
    t
}
static BINOM: [[u64; MAX_GROUP + 1]; 65] = binom_table();

#[inline(always)]
fn c(n: usize, k: usize) -> u64 {
    if k > n || k > MAX_GROUP {
        0
    } else {
        BINOM[n][k]
    }
}

pub type Masks = [u64; 4];

#[derive(Clone)]
pub struct Indexer {
    pub sig: Sig,
    pub nsq: usize,
    sq_of_bit: [u8; 64],
    bit_of_sq: [u8; 64],
    wm_list: Vec<u8>,
    wm_idx: [u8; 64],
    bad_black: u64,
    sizes: [u64; 4],
    ks: [usize; 4],
    pub n: u64,
}

impl Indexer {
    pub fn new<G: Geometry>(sig: Sig) -> Result<Indexer, String> {
        let mut sq_of_bit = [0xFFu8; 64];
        let mut bit_of_sq = [0xFFu8; 64];
        let mut id = 0usize;
        for b in 0..64 {
            if (G::VALID >> b) & 1 == 1 {
                sq_of_bit[b] = id as u8;
                bit_of_sq[id] = b as u8;
                id += 1;
            }
        }
        let nsq = id;
        let ortho = nsq == 64;
        let mut wm_list = vec![];
        let mut wm_idx = [0xFFu8; 64];
        let mut bad_black = 0u64;
        for sq in 0..nsq {
            let b = bit_of_sq[sq] as u32;
            let (row, _) = G::coords(b);
            let wpromo = (G::WHITE_PROMO >> b) & 1 == 1;
            let bpromo = (G::BLACK_PROMO >> b) & 1 == 1;
            let w_ok = !wpromo && !(ortho && row == 0);
            if w_ok {
                wm_idx[sq] = wm_list.len() as u8;
                wm_list.push(sq as u8);
            }
            if bpromo || (ortho && row == 7) {
                bad_black |= 1u64 << sq;
            }
        }
        let ks = [sig[0] as usize, sig[2] as usize, sig[1] as usize, sig[3] as usize];
        if ks.iter().any(|&k| k > MAX_GROUP) || ks.iter().sum::<usize>() > nsq {
            return Err("слишком много фигур".into());
        }
        let sizes = [
            c(wm_list.len(), ks[0]),
            c(nsq - ks[0], ks[1]),
            c(nsq - ks[0] - ks[1], ks[2]),
            c(nsq - ks[0] - ks[1] - ks[2], ks[3]),
        ];
        let mut n = 1u64;
        for s in sizes {
            n = n.checked_mul(s).ok_or("переполнение размера среза")?;
        }
        Ok(Indexer { sig, nsq, sq_of_bit, bit_of_sq, wm_list, wm_idx, bad_black, sizes, ks, n })
    }

    #[inline]
    fn to_sq(&self, mut x: u64) -> u64 {
        let mut r = 0u64;
        while x != 0 {
            let b = x.trailing_zeros() as usize;
            x &= x - 1;
            r |= 1u64 << self.sq_of_bit[b];
        }
        r
    }

    #[inline]
    fn to_bits(&self, mut x: u64) -> u64 {
        let mut r = 0u64;
        while x != 0 {
            let s = x.trailing_zeros() as usize;
            x &= x - 1;
            r |= 1u64 << self.bit_of_sq[s];
        }
        r
    }

    #[inline]
    pub fn masks<R: Rules>(&self, p: &Position<R>) -> Masks {
        let wk = p.white & p.kings;
        let bk = p.black & p.kings;
        [self.to_sq(p.white & !wk), self.to_sq(p.black & !bk), self.to_sq(wk), self.to_sq(bk)]
    }

    #[inline]
    pub fn valid(&self, m: &Masks) -> bool {
        m[1] & self.bad_black == 0
    }

    pub fn rank(&self, m: &Masks) -> Option<u64> {
        for g in 0..4 {
            if m[g].count_ones() as usize != self.ks[g] {
                return None;
            }
        }
        let mut r = [0u64; 4];
        let mut x = m[0];
        let mut i = 0;
        while x != 0 {
            let sq = x.trailing_zeros() as usize;
            x &= x - 1;
            let ci = self.wm_idx[sq];
            if ci == 0xFF {
                return None;
            }
            r[0] += c(ci as usize, i + 1);
            i += 1;
        }
        let mut occ = m[0];
        for g in 1..4 {
            if m[g] & occ != 0 {
                return None;
            }
            let mut x = m[g];
            let mut i = 0;
            while x != 0 {
                let sq = x.trailing_zeros() as usize;
                x &= x - 1;
                let ci = sq - (occ & ((1u64 << sq) - 1)).count_ones() as usize;
                r[g] += c(ci, i + 1);
                i += 1;
            }
            occ |= m[g];
        }
        Some(r[0] + self.sizes[0] * (r[1] + self.sizes[1] * (r[2] + self.sizes[2] * r[3])))
    }

    pub fn rank_pos<R: Rules>(&self, p: &Position<R>) -> Option<u64> {
        self.rank(&self.masks(p))
    }

    fn comb_decode(mut r: u64, k: usize, cs: &mut [usize; MAX_GROUP]) {
        let mut hi = 64usize;
        for i in (1..=k).rev() {
            let mut ch = hi - 1;
            while c(ch, i) > r {
                ch -= 1;
            }
            r -= c(ch, i);
            cs[i - 1] = ch;
            hi = ch;
        }
    }

    pub fn unrank(&self, idx: u64) -> Masks {
        let mut rem = idx;
        let mut rs = [0u64; 4];
        for g in 0..4 {
            rs[g] = rem % self.sizes[g];
            rem /= self.sizes[g];
        }
        let mut out = [0u64; 4];
        let mut cs = [0usize; MAX_GROUP];
        Self::comb_decode(rs[0], self.ks[0], &mut cs);
        for j in 0..self.ks[0] {
            out[0] |= 1u64 << self.wm_list[cs[j]];
        }
        let mut occ = out[0];
        for g in 1..4 {
            let k = self.ks[g];
            Self::comb_decode(rs[g], k, &mut cs);
            let mut mask = 0u64;
            let (mut j, mut free) = (0usize, 0usize);
            if k > 0 {
                for sq in 0..self.nsq {
                    if (occ >> sq) & 1 == 1 {
                        continue;
                    }
                    if cs[j] == free {
                        mask |= 1u64 << sq;
                        j += 1;
                        if j == k {
                            break;
                        }
                    }
                    free += 1;
                }
            }
            out[g] = mask;
            occ |= mask;
        }
        out
    }

    pub fn to_position<R: Rules>(&self, m: &Masks, side: Side) -> Position<R> {
        let white = self.to_bits(m[0] | m[2]);
        let black = self.to_bits(m[1] | m[3]);
        let kings = self.to_bits(m[2] | m[3]);
        Position::new(white, black, kings, side)
    }
}

pub fn sig_of<R: Rules>(p: &Position<R>) -> Sig {
    let wk = (p.white & p.kings).count_ones() as u8;
    let bk = (p.black & p.kings).count_ones() as u8;
    [p.white.count_ones() as u8 - wk, wk, p.black.count_ones() as u8 - bk, bk]
}

pub fn sig_name(s: Sig) -> String {
    format!("{}m{}k-{}m{}k", s[0], s[1], s[2], s[3])
}

pub fn parse_sig_name(name: &str) -> Option<Sig> {
    let b = name.as_bytes();
    if b.len() != 9 || b[1] != b'm' || b[3] != b'k' || b[4] != b'-' || b[6] != b'm' || b[8] != b'k' {
        return None;
    }
    let d = |i: usize| if b[i].is_ascii_digit() { Some(b[i] - b'0') } else { None };
    Some([d(0)?, d(2)?, d(5)?, d(7)?])
}

pub fn sigs_of_total(t: u32) -> Vec<Sig> {
    let mut v = vec![];
    for wm in 0..=t {
        for wk in 0..=(t - wm) {
            for bm in 0..=(t - wm - wk) {
                let bk = t - wm - wk - bm;
                if wm + wk >= 1 && bm + bk >= 1 && [wm, wk, bm, bk].iter().all(|&x| x <= 9) {
                    v.push([wm as u8, wk as u8, bm as u8, bk as u8]);
                }
            }
        }
    }
    v.sort_by_key(|s| (s[0] + s[2], *s));
    v
}

pub fn dep_sigs(sig: Sig) -> Vec<Sig> {
    let mut set = std::collections::BTreeSet::new();
    let [wm, wk, bm, bk] = sig;
    for w2 in 0..=wm {
        let wk2 = wk + (wm - w2);
        for bm2 in 0..=bm {
            for bk2 in 0..=bk {
                if w2 == wm && bm2 == bm && bk2 == bk {
                    continue;
                }
                if bm2 + bk2 == 0 {
                    continue;
                }
                set.insert([w2, wk2, bm2, bk2]);
            }
        }
    }
    for b2 in 0..=bm {
        let bk2 = bk + (bm - b2);
        for wm2 in 0..=wm {
            for wk2 in 0..=wk {
                if b2 == bm && wm2 == wm && wk2 == wk {
                    continue;
                }
                if wm2 + wk2 == 0 {
                    continue;
                }
                set.insert([wm2, wk2, b2, bk2]);
            }
        }
    }
    set.into_iter().collect()
}
