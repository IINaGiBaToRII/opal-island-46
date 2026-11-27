
use crate::geometry::Geometry;
use crate::position::{Position, Side};
use crate::rules::Rules;

pub fn geometry_name<G: Geometry>() -> &'static str {
    if G::N == 10 {
        "diag10"
    } else if G::NSQ == 64 {
        "ortho8"
    } else if G::bit(0, 0).is_some() {
        "diag8"
    } else {
        "diag8m"
    }
}

pub fn geometry_nsq(name: &str) -> Option<usize> {
    match name {
        "diag8" | "diag8m" => Some(32),
        "diag10" => Some(50),
        "ortho8" => Some(64),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct View {
    pub own_men: u64,
    pub opp_men: u64,
    pub own_kings: u64,
    pub opp_kings: u64,
}

impl View {
    pub fn count(&self) -> u32 {
        (self.own_men | self.opp_men | self.own_kings | self.opp_kings).count_ones()
    }
}

#[derive(Clone, Debug)]
pub struct ViewTables {
    pub n: usize,
    pub nsq: usize,
    pub to_dense: [[u8; 64]; 2],
    pub from_dense: [[u8; 64]; 2],
    pub rot: [u8; 64],
    pub row: [u8; 64],
    pub file: [u8; 64],
    pub valid: u64,
    pub white_promo: u64,
    pub black_promo: u64,
}

impl ViewTables {
    pub fn new<R: Rules>() -> Self {
        Self::for_geo::<R::Geo>()
    }

    pub fn for_geo<G: Geometry>() -> Self {
        let n = G::N;
        let mut t = ViewTables {
            n,
            nsq: G::NSQ,
            to_dense: [[255; 64]; 2],
            from_dense: [[255; 64]; 2],
            rot: [255; 64],
            row: [0; 64],
            file: [0; 64],
            valid: G::VALID,
            white_promo: G::WHITE_PROMO,
            black_promo: G::BLACK_PROMO,
        };
        let mut k = 0usize;
        for r in 0..n {
            for f in 0..n {
                if let Some(b) = G::bit(r, f) {
                    t.to_dense[0][b as usize] = k as u8;
                    t.from_dense[0][k] = b as u8;
                    t.row[k] = r as u8;
                    t.file[k] = f as u8;
                    if let Some(rb) = G::bit(n - 1 - r, n - 1 - f) {
                        t.rot[b as usize] = rb as u8;
                    }
                    k += 1;
                }
            }
        }
        debug_assert_eq!(k, G::NSQ);
        for b in 0..64usize {
            if t.to_dense[0][b] != 255 && t.rot[b] != 255 {
                t.to_dense[1][b] = t.to_dense[0][t.rot[b] as usize];
            }
        }
        for k in 0..G::NSQ {
            let b = t.from_dense[0][k] as usize;
            t.from_dense[1][k] = t.rot[b];
        }
        t
    }

    pub fn check_rotation(&self) -> Result<(), String> {
        let mut x = self.valid;
        while x != 0 {
            let b = x.trailing_zeros() as usize;
            x &= x - 1;
            let r = self.rot[b];
            if r == 255 {
                return Err(format!("поле {b} не переходит в игровое"));
            }
            if (self.valid >> r) & 1 == 0 {
                return Err(format!("образ поля {b} вне VALID"));
            }
            if self.rot[r as usize] as usize != b {
                return Err(format!("поворот не инволюция на {b}"));
            }
        }
        if self.rotate_bits(self.white_promo) != self.black_promo || self.rotate_bits(self.black_promo) != self.white_promo {
            return Err("последние ряды не меняются местами".into());
        }
        Ok(())
    }

    pub fn rotate_bits(&self, mut x: u64) -> u64 {
        let mut out = 0u64;
        while x != 0 {
            let b = x.trailing_zeros() as usize;
            x &= x - 1;
            let r = self.rot[b];
            if r != 255 {
                out |= 1u64 << r;
            }
        }
        out
    }

    pub fn to_dense_bits(&self, mut x: u64, o: usize) -> u64 {
        let mut out = 0u64;
        while x != 0 {
            let b = x.trailing_zeros() as usize;
            x &= x - 1;
            let d = self.to_dense[o][b];
            if d != 255 {
                out |= 1u64 << d;
            }
        }
        out
    }

    pub fn view_of<R: Rules>(&self, pos: &Position<R>, persp: Side) -> View {
        let (own, opp, o) = match persp {
            Side::White => (pos.white, pos.black, 0),
            Side::Black => (pos.black, pos.white, 1),
        };
        View {
            own_men: self.to_dense_bits(own & !pos.kings, o),
            opp_men: self.to_dense_bits(opp & !pos.kings, o),
            own_kings: self.to_dense_bits(own & pos.kings, o),
            opp_kings: self.to_dense_bits(opp & pos.kings, o),
        }
    }

    pub fn stm<R: Rules>(&self, pos: &Position<R>) -> View {
        self.view_of(pos, pos.side)
    }

    pub fn opp<R: Rules>(&self, pos: &Position<R>) -> View {
        self.view_of(pos, pos.side.other())
    }

    pub fn flip<R: Rules>(&self, pos: &Position<R>) -> Position<R> {
        let white = self.rotate_bits(pos.black);
        let black = self.rotate_bits(pos.white);
        let kings = self.rotate_bits(pos.kings);
        Position::new(white, black, kings, pos.side.other())
    }
}

pub fn stm_view<R: Rules>(pos: &Position<R>) -> View {
    ViewTables::new::<R>().stm(pos)
}

pub fn opp_view<R: Rules>(pos: &Position<R>) -> View {
    ViewTables::new::<R>().opp(pos)
}

pub fn color_flip<R: Rules>(pos: &Position<R>) -> Position<R> {
    ViewTables::new::<R>().flip(pos)
}
