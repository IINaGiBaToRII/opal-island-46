
use crate::geometry::Geometry;
use crate::movegen::Move;
use crate::rules::Rules;
use std::marker::PhantomData;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Side {
    White = 0,
    Black = 1,
}

impl Side {
    #[inline(always)]
    pub fn other(self) -> Side {
        match self {
            Side::White => Side::Black,
            Side::Black => Side::White,
        }
    }
}

const fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const fn gen_zobrist() -> [[u64; 64]; 4] {
    let mut t = [[0u64; 64]; 4];
    let mut s = 0x1234_5678_9ABC_DEF0u64;
    let mut p = 0;
    while p < 4 {
        let mut i = 0;
        while i < 64 {
            s = splitmix(s);
            t[p][i] = s;
            i += 1;
        }
        p += 1;
    }
    t
}

pub static ZOBRIST: [[u64; 64]; 4] = gen_zobrist();
pub const SIDE_KEY: u64 = splitmix(0xDEAD_BEEF_CAFE_F00D);

pub struct Position<R: Rules> {
    pub white: u64,
    pub black: u64,
    pub kings: u64,
    pub side: Side,
    pub hash: u64,
    _r: PhantomData<R>,
}

impl<R: Rules> Clone for Position<R> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<R: Rules> Copy for Position<R> {}
impl<R: Rules> PartialEq for Position<R> {
    fn eq(&self, o: &Self) -> bool {
        self.white == o.white && self.black == o.black && self.kings == o.kings && self.side == o.side
    }
}
impl<R: Rules> Eq for Position<R> {}
impl<R: Rules> std::fmt::Debug for Position<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_dfen())
    }
}

type G<R> = <R as Rules>::Geo;

impl<R: Rules> Position<R> {
    pub fn new(white: u64, black: u64, kings: u64, side: Side) -> Self {
        let mut p = Position { white, black, kings, side, hash: 0, _r: PhantomData };
        p.hash = p.compute_hash();
        p
    }

    pub fn start() -> Self {
        Self::from_dfen(R::START_DFEN).expect("START_DFEN valid")
    }

    #[inline(always)]
    pub fn occ(&self) -> u64 {
        self.white | self.black
    }

    #[inline(always)]
    pub fn own(&self) -> u64 {
        match self.side {
            Side::White => self.white,
            Side::Black => self.black,
        }
    }

    #[inline(always)]
    pub fn opp(&self) -> u64 {
        match self.side {
            Side::White => self.black,
            Side::Black => self.white,
        }
    }

    pub fn compute_hash(&self) -> u64 {
        let mut h = if self.side == Side::Black { SIDE_KEY } else { 0 };
        for (bb, base) in [(self.white, 0usize), (self.black, 2usize)] {
            let mut x = bb;
            while x != 0 {
                let b = x.trailing_zeros() as usize;
                x &= x - 1;
                let k = ((self.kings >> b) & 1) as usize;
                h ^= ZOBRIST[base + k][b];
            }
        }
        h
    }

    #[inline(always)]
    pub fn hash_after(&self, mv: &Move) -> u64 {
        let was_king = self.kings & (1u64 << mv.from) != 0;
        let now_king = was_king || mv.promo;
        let (cbase, obase) = match self.side {
            Side::White => (0usize, 2usize),
            Side::Black => (2usize, 0usize),
        };
        let mut h = self.hash ^ SIDE_KEY;
        h ^= ZOBRIST[cbase + was_king as usize][mv.from as usize];
        h ^= ZOBRIST[cbase + now_king as usize][mv.to as usize];
        let mut x = mv.captured;
        while x != 0 {
            let b = x.trailing_zeros() as usize;
            x &= x - 1;
            h ^= ZOBRIST[obase + ((self.kings >> b) & 1) as usize][b];
        }
        h
    }

    #[inline]
    pub fn make(&self, mv: &Move) -> Self {
        let from = 1u64 << mv.from;
        let to = 1u64 << mv.to;
        let was_king = self.kings & from != 0;
        let now_king = was_king || mv.promo;
        let (own, opp, cbase, obase) = match self.side {
            Side::White => (self.white, self.black, 0usize, 2usize),
            Side::Black => (self.black, self.white, 2usize, 0usize),
        };
        let mut h = self.hash ^ SIDE_KEY;
        h ^= ZOBRIST[cbase + was_king as usize][mv.from as usize];
        h ^= ZOBRIST[cbase + now_king as usize][mv.to as usize];
        let mut x = mv.captured;
        while x != 0 {
            let b = x.trailing_zeros() as usize;
            x &= x - 1;
            h ^= ZOBRIST[obase + ((self.kings >> b) & 1) as usize][b];
        }
        let new_own = (own & !from) | to;
        let new_opp = opp & !mv.captured;
        let mut kings = self.kings & !(from | mv.captured);
        if now_king {
            kings |= to;
        }
        let (white, black) = match self.side {
            Side::White => (new_own, new_opp),
            Side::Black => (new_opp, new_own),
        };
        Position { white, black, kings, side: self.side.other(), hash: h, _r: PhantomData }
    }

    pub fn check_invariants(&self) -> Result<(), String> {
        if self.white & self.black != 0 {
            return Err("white & black overlap".into());
        }
        if (self.white | self.black) & !G::<R>::VALID != 0 {
            return Err("piece outside VALID".into());
        }
        if self.kings & !(self.white | self.black) != 0 {
            return Err("king without piece".into());
        }
        if self.hash != self.compute_hash() {
            return Err("hash mismatch".into());
        }
        Ok(())
    }

    pub fn parse_square(s: &str) -> Result<u32, String> {
        let b = s.as_bytes();
        if b.len() < 2 || !b[0].is_ascii_lowercase() {
            return Err(format!("bad square '{s}'"));
        }
        let file = (b[0] - b'a') as usize;
        let rank: usize = s[1..].parse().map_err(|_| format!("bad square '{s}'"))?;
        if rank == 0 {
            return Err(format!("bad square '{s}'"));
        }
        G::<R>::bit(rank - 1, file).ok_or_else(|| format!("not a playable square '{s}'"))
    }

    pub fn square_name(bit: u32) -> String {
        let (r, f) = G::<R>::coords(bit);
        format!("{}{}", (b'a' + f as u8) as char, r + 1)
    }

    pub fn from_dfen(s: &str) -> Result<Self, String> {
        let n = G::<R>::N;
        let mut it = s.split_whitespace();
        let side = match it.next() {
            Some("w") => Side::White,
            Some("b") => Side::Black,
            _ => return Err("DFEN: first token must be 'w' or 'b'".into()),
        };
        let rows_s = it.next().ok_or("DFEN: missing rows")?;
        if it.next().is_some() {
            return Err("DFEN: trailing tokens".into());
        }
        let rows: Vec<&str> = rows_s.split('/').collect();
        if rows.len() != n {
            return Err(format!("DFEN: expected {n} rows, got {}", rows.len()));
        }
        let (mut white, mut black, mut kings) = (0u64, 0u64, 0u64);
        for (i, row) in rows.iter().enumerate() {
            let r = n - 1 - i;
            let chars: Vec<char> = row.chars().collect();
            if chars.len() != n {
                return Err(format!("DFEN: row {} has {} cells, expected {n}", r + 1, chars.len()));
            }
            for (f, c) in chars.iter().enumerate() {
                if *c == '.' {
                    continue;
                }
                if !matches!(c, 'w' | 'b' | 'W' | 'B') {
                    return Err(format!("DFEN: bad char '{c}'"));
                }
                let bit = G::<R>::bit(r, f)
                    .ok_or_else(|| format!("DFEN: piece on non-playable square {}{}", (b'a' + f as u8) as char, r + 1))?;
                let m = 1u64 << bit;
                if matches!(c, 'w' | 'W') {
                    white |= m;
                } else {
                    black |= m;
                }
                if c.is_ascii_uppercase() {
                    kings |= m;
                }
            }
        }
        Ok(Self::new(white, black, kings, side))
    }

    pub fn to_dfen(&self) -> String {
        let n = G::<R>::N;
        let mut s = String::with_capacity(n * n + 4);
        s.push_str(if self.side == Side::White { "w " } else { "b " });
        for r in (0..n).rev() {
            for f in 0..n {
                let c = match G::<R>::bit(r, f) {
                    None => '.',
                    Some(b) => {
                        let m = 1u64 << b;
                        let k = self.kings & m != 0;
                        if self.white & m != 0 {
                            if k { 'W' } else { 'w' }
                        } else if self.black & m != 0 {
                            if k { 'B' } else { 'b' }
                        } else {
                            '.'
                        }
                    }
                };
                s.push(c);
            }
            if r > 0 {
                s.push('/');
            }
        }
        s
    }
}
