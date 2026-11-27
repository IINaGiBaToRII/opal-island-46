
use crate::geometry::Geometry;
use crate::movegen::Move;
use crate::position::{Position, Side};
use crate::rules::Rules;

pub trait Evaluator<R: Rules>: Send + Sync {
    type State: Clone + Send;
    fn root(&self, pos: &Position<R>) -> Self::State;
    fn child(&self, parent: &Self::State, before: &Position<R>, mv: &Move, after: &Position<R>) -> Self::State;
    fn eval(&self, pos: &Position<R>, st: &mut Self::State) -> i32;
    #[inline(always)]
    fn child_at(&self, stack: &mut [Self::State], ply: usize, before: &Position<R>, mv: &Move, after: &Position<R>) {
        let s = self.child(&stack[ply], before, mv, after);
        stack[ply + 1] = s;
    }
    #[inline(always)]
    fn eval_at(&self, pos: &Position<R>, stack: &mut [Self::State], ply: usize) -> i32 {
        self.eval(pos, &mut stack[ply])
    }
    fn eval_fresh(&self, pos: &Position<R>) -> i32 {
        let mut s = self.root(pos);
        self.eval(pos, &mut s)
    }
}

pub struct SimpleEval {
    man: [[i16; 64]; 2],
    king: [i16; 64],
    tempo: i32,
}

impl SimpleEval {
    pub fn new<R: Rules>() -> Self {
        type G<R> = <R as Rules>::Geo;
        let n = G::<R>::N as i32;
        let mut man = [[0i16; 64]; 2];
        let mut king = [0i16; 64];
        for b in 0..64u32 {
            if G::<R>::VALID >> b & 1 == 0 {
                continue;
            }
            let (r, f) = G::<R>::coords(b);
            let (r, f) = (r as i32, f as i32);
            let dist = (2 * f - (n - 1)).abs().max((2 * r - (n - 1)).abs());
            let center = if dist <= 2 { 4 } else if dist <= 4 { 2 } else { 0 };
            let edge = if f == 0 || f == n - 1 { -3 } else { 0 };
            for c in 0..2 {
                let adv = if c == 0 { r } else { n - 1 - r };
                let a = adv * adv * 24 / ((n - 1) * (n - 1));
                let back = if adv == 0 { 7 } else { 0 };
                man[c][b as usize] = (100 + a + back + center + edge) as i16;
            }
            let diag = if f == r || f + r == n - 1 { 10 } else { 0 };
            let main = if f == r { 6 } else { 0 };
            king[b as usize] = (R::KING_VALUE + diag + main + center) as i16;
        }
        SimpleEval { man, king, tempo: 4 }
    }
}

impl<R: Rules> Evaluator<R> for SimpleEval {
    type State = ();
    #[inline(always)]
    fn root(&self, _pos: &Position<R>) {}
    #[inline(always)]
    fn child(&self, _parent: &(), _before: &Position<R>, _mv: &Move, _after: &Position<R>) {}
    #[inline]
    fn eval(&self, pos: &Position<R>, _st: &mut ()) -> i32 {
        let mut s = 0i32;
        let wk = pos.white & pos.kings;
        let bk = pos.black & pos.kings;
        let mut x = pos.white & !pos.kings;
        while x != 0 {
            s += self.man[0][x.trailing_zeros() as usize] as i32;
            x &= x - 1;
        }
        let mut x = pos.black & !pos.kings;
        while x != 0 {
            s -= self.man[1][x.trailing_zeros() as usize] as i32;
            x &= x - 1;
        }
        let mut x = wk;
        while x != 0 {
            s += self.king[x.trailing_zeros() as usize] as i32;
            x &= x - 1;
        }
        let mut x = bk;
        while x != 0 {
            s -= self.king[x.trailing_zeros() as usize] as i32;
            x &= x - 1;
        }
        if pos.side == Side::Black {
            s = -s;
        }
        s + self.tempo
    }
}
