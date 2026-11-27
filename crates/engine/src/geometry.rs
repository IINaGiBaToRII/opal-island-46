
pub trait Geometry: 'static + Copy + Default + Send + Sync + std::fmt::Debug {
    const N: usize;
    const NSQ: usize;
    const VALID: u64;
    const WHITE_PROMO: u64;
    const BLACK_PROMO: u64;
    const FWD_WHITE: &'static [usize];
    const FWD_BLACK: &'static [usize];
    const NDIRS: usize;
    fn shift(x: u64, d: usize) -> u64;
    fn perp(d: usize) -> [usize; 2];
    fn bit(row: usize, file: usize) -> Option<u32>;
    fn coords(bit: u32) -> (usize, usize);
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Diag8;

const G: u64 = 0x1EF;

impl Geometry for Diag8 {
    const N: usize = 8;
    const NSQ: usize = 32;
    const VALID: u64 = G | (G << 9) | (G << 18) | (G << 27);
    const WHITE_PROMO: u64 = 0xF << 32;
    const BLACK_PROMO: u64 = 0xF;
    const FWD_WHITE: &'static [usize] = &[0, 1];
    const FWD_BLACK: &'static [usize] = &[2, 3];
    const NDIRS: usize = 4;

    #[inline(always)]
    fn shift(x: u64, d: usize) -> u64 {
        match d {
            0 => x << 5,
            1 => x << 4,
            2 => x >> 5,
            _ => x >> 4,
        }
    }

    #[inline(always)]
    fn perp(d: usize) -> [usize; 2] {
        [(d + 1) & 3, (d + 3) & 3]
    }

    fn bit(row: usize, file: usize) -> Option<u32> {
        if row >= 8 || file >= 8 || (row + file) % 2 != 0 {
            return None;
        }
        let c = file / 2;
        Some(((row / 2) * 9 + if row % 2 == 0 { c } else { 5 + c }) as u32)
    }

    fn coords(bit: u32) -> (usize, usize) {
        let k = (bit / 9) as usize;
        let r = (bit % 9) as usize;
        if r < 4 {
            (2 * k, 2 * r)
        } else {
            (2 * k + 1, 2 * (r - 5) + 1)
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Diag8M;

impl Geometry for Diag8M {
    const N: usize = 8;
    const NSQ: usize = 32;
    const VALID: u64 = Diag8::VALID;
    const WHITE_PROMO: u64 = Diag8::WHITE_PROMO;
    const BLACK_PROMO: u64 = Diag8::BLACK_PROMO;
    const FWD_WHITE: &'static [usize] = &[0, 1];
    const FWD_BLACK: &'static [usize] = &[2, 3];
    const NDIRS: usize = 4;

    #[inline(always)]
    fn shift(x: u64, d: usize) -> u64 {
        match d {
            0 => x << 4,
            1 => x << 5,
            2 => x >> 4,
            _ => x >> 5,
        }
    }

    #[inline(always)]
    fn perp(d: usize) -> [usize; 2] {
        [(d + 1) & 3, (d + 3) & 3]
    }

    fn bit(row: usize, file: usize) -> Option<u32> {
        if file >= 8 {
            return None;
        }
        Diag8::bit(row, 7 - file)
    }

    fn coords(bit: u32) -> (usize, usize) {
        let (r, f) = Diag8::coords(bit);
        (r, 7 - f)
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Diag10;

const G10: u64 = 0x7DF;

impl Geometry for Diag10 {
    const N: usize = 10;
    const NSQ: usize = 50;
    const VALID: u64 = G10 | (G10 << 11) | (G10 << 22) | (G10 << 33) | (G10 << 44);
    const WHITE_PROMO: u64 = 0x1F << 50;
    const BLACK_PROMO: u64 = 0x1F;
    const FWD_WHITE: &'static [usize] = &[0, 1];
    const FWD_BLACK: &'static [usize] = &[2, 3];
    const NDIRS: usize = 4;

    #[inline(always)]
    fn shift(x: u64, d: usize) -> u64 {
        match d {
            0 => x << 6,
            1 => x << 5,
            2 => x >> 6,
            _ => x >> 5,
        }
    }

    #[inline(always)]
    fn perp(d: usize) -> [usize; 2] {
        [(d + 1) & 3, (d + 3) & 3]
    }

    fn bit(row: usize, file: usize) -> Option<u32> {
        if row >= 10 || file >= 10 || (row + file) % 2 != 0 {
            return None;
        }
        let c = file / 2;
        Some(((row / 2) * 11 + if row % 2 == 0 { c } else { 6 + c }) as u32)
    }

    fn coords(bit: u32) -> (usize, usize) {
        let k = (bit / 11) as usize;
        let r = (bit % 11) as usize;
        if r < 5 {
            (2 * k, 2 * r)
        } else {
            (2 * k + 1, 2 * (r - 6) + 1)
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Ortho8;

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_H: u64 = 0x8080_8080_8080_8080;

impl Geometry for Ortho8 {
    const N: usize = 8;
    const NSQ: usize = 64;
    const VALID: u64 = !0;
    const WHITE_PROMO: u64 = 0xFFu64 << 56;
    const BLACK_PROMO: u64 = 0xFF;
    const FWD_WHITE: &'static [usize] = &[0, 1, 3];
    const FWD_BLACK: &'static [usize] = &[2, 1, 3];
    const NDIRS: usize = 4;

    #[inline(always)]
    fn shift(x: u64, d: usize) -> u64 {
        match d {
            0 => x << 8,
            1 => (x & !FILE_A) >> 1,
            2 => x >> 8,
            _ => (x & !FILE_H) << 1,
        }
    }

    #[inline(always)]
    fn perp(d: usize) -> [usize; 2] {
        [(d + 1) & 3, (d + 3) & 3]
    }

    fn bit(row: usize, file: usize) -> Option<u32> {
        if row >= 8 || file >= 8 {
            return None;
        }
        Some((row * 8 + file) as u32)
    }

    fn coords(bit: u32) -> (usize, usize) {
        ((bit / 8) as usize, (bit % 8) as usize)
    }
}
