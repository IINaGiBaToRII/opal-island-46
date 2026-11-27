
pub const ARITH_VER: u32 = 1;

pub const GRID_HASH: u64 = 0xe0eb_2240_b213_23bf;

#[inline]
pub fn exp_f64(x: f64) -> f64 {
    libm::exp(x)
}

#[inline]
pub fn exp_f32(x: f32) -> f32 {
    libm::exp(x as f64) as f32
}

#[inline]
pub fn ln_f64(x: f64) -> f64 {
    libm::log(x)
}

#[inline]
pub fn cos_f64(x: f64) -> f64 {
    libm::cos(x)
}

#[inline]
pub fn exp2_f64(x: f64) -> f64 {
    libm::exp2(x)
}

#[inline]
pub fn pow_f64(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + exp_f64(-x))
}

#[inline]
pub fn powi_f32(x: f32, n: u32) -> f32 {
    let (mut r, mut b, mut e) = (1.0f32, x, n);
    while e > 0 {
        if e & 1 == 1 {
            r *= b;
        }
        e >>= 1;
        if e > 0 {
            b *= b;
        }
    }
    r
}

#[inline]
pub fn powi_f64(x: f64, n: u32) -> f64 {
    let (mut r, mut b, mut e) = (1.0f64, x, n);
    while e > 0 {
        if e & 1 == 1 {
            r *= b;
        }
        e >>= 1;
        if e > 0 {
            b *= b;
        }
    }
    r
}

#[inline]
fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x100_0000_01B3);
    }
}

pub fn hash_f32(v: &[f32]) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for x in v {
        fnv(&mut h, &x.to_bits().to_le_bytes());
    }
    h
}

pub fn grid_hash() -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for i in 0..(1u32 << 16) {
        let u = i as f64 / 65536.0;
        let vals = [
            exp_f64(-60.0 + 120.0 * u).to_bits(),
            exp_f64(-745.0 + 1454.0 * u).to_bits(),
            ln_f64((i + 1) as f64 * 0.3711).to_bits(),
            ln_f64(1e-300 + u).to_bits(),
            cos_f64(-8.0 + 16.0 * u).to_bits(),
            cos_f64(std::f64::consts::PI * u).to_bits(),
            exp2_f64(-80.0 + 160.0 * u).to_bits(),
            pow_f64(0.01 + 20.0 * u, -2.0 + 4.0 * u).to_bits(),
            sigmoid(-40.0 + 80.0 * u).to_bits(),
            powi_f64(1.0 - 1e-4 * u, i).to_bits(),
            exp_f32((-90.0 + 180.0 * u) as f32).to_bits() as u64,
            powi_f32(0.9, i).to_bits() as u64,
            powi_f32(0.999, i).to_bits() as u64,
        ];
        for v in vals {
            fnv(&mut h, &v.to_le_bytes());
        }
    }
    h
}

pub fn self_check() -> Result<u64, String> {
    static H: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let h = *H.get_or_init(grid_hash);
    if h != GRID_HASH {
        return Err(format!("арифметика обучения не та: хеш сетки detmath {h:#018x}, эталон {GRID_HASH:#018x} (ARITH_VER {ARITH_VER}) — обучение отменено"));
    }
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powi_matches_repeated_multiplication_small() {
        for n in 0..6u32 {
            let mut r = 1.0f32;
            for _ in 0..n {
                r *= 0.9;
            }
            assert!((powi_f32(0.9, n) - r).abs() <= 1e-6, "{n}");
        }
        assert_eq!(powi_f64(2.0, 10), 1024.0);
    }

    #[test]
    fn close_to_std() {
        for i in -200..200 {
            let x = i as f64 * 0.173;
            assert!((exp_f64(x) - x.exp()).abs() <= 4.0 * f64::EPSILON * x.exp());
            assert!((cos_f64(x) - x.cos()).abs() <= 1e-15);
        }
    }
}
