
use super::format::{EndReason, Game};
use crate::egdb::Egdb;
use crate::geometry::Geometry;
use crate::movegen::{generate, ListSink, Move, MoveList};
use crate::position::{Position, Side};
use crate::rules::Rules;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sample {
    pub result_stm: i8,
    pub score: i16,
    pub depth: u8,
    pub ply: u16,
    pub egdb: Option<i8>,
    pub pieces: u8,
}

pub const MATE_SKIP: i32 = 29000;

pub fn start_position<R: Rules>(g: &Game) -> Result<Position<R>, String> {
    let Some(s) = g.start else { return Ok(Position::<R>::start()) };
    let valid = <R::Geo as Geometry>::VALID;
    let occ = s.white | s.black;
    if s.side > 1 || s.white & s.black != 0 || s.kings & !occ != 0 || occ & !valid != 0 {
        return Err("недопустимая стартовая позиция".into());
    }
    Ok(Position::<R>::new(s.white, s.black, s.kings, if s.side == 0 { Side::White } else { Side::Black }))
}

pub fn legal<R: Rules>(pos: &Position<R>) -> MoveList {
    let mut l = MoveList::new();
    generate(pos, &mut ListSink::<true>(&mut l));
    l
}

pub fn move_index(list: &[Move], mv: &Move) -> Option<u8> {
    let mut idx = 0u32;
    for m in list {
        if m == mv {
            return u8::try_from(idx).ok();
        }
        if m.from == mv.from && m.to == mv.to {
            idx += 1;
        }
    }
    None
}

pub fn replay<R: Rules>(g: &Game, mut f: impl FnMut(usize, &Position<R>, &Move, &super::format::PlyRec)) -> Result<Position<R>, String> {
    let mut pos = start_position::<R>(g)?;
    for (i, p) in g.plies.iter().enumerate() {
        let list = legal::<R>(&pos);
        let mut k = p.idx as u32;
        let mut found = None;
        for m in list.iter() {
            if m.from == p.from && m.to == p.to {
                if k == 0 {
                    found = Some(*m);
                    break;
                }
                k -= 1;
            }
        }
        let mv = found.ok_or_else(|| format!("полуход {i}: нелегальный ход {}-{}#{}", p.from, p.to, p.idx))?;
        f(i, &pos, &mv, p);
        pos = pos.make(&mv);
    }
    if g.reason == EndReason::NoMoves {
        let loser_white = if pos.side == Side::White { 1 } else { -1 };
        if legal::<R>(&pos).len != 0 || g.result != -loser_white {
            return Err("причина «нет ходов» не соответствует конечной позиции".into());
        }
    }
    Ok(pos)
}

pub fn for_each_sample<R: Rules>(
    g: &Game,
    egdb: Option<&Egdb<R>>,
    mut f: impl FnMut(&Position<R>, &Sample),
) -> Result<(), String> {
    replay::<R>(g, |i, pos, _mv, p| {
        if p.random || (p.score as i32).abs() > MATE_SKIP {
            return;
        }
        if R::FORCED && legal::<R>(pos).iter().any(|m| m.is_capture()) {
            return;
        }
        let pieces = pos.occ().count_ones();
        let exact = egdb.and_then(|db| if pieces <= db.max_pieces() { db.probe(pos) } else { None });
        let white_pov = g.result;
        let s = Sample {
            result_stm: if pos.side == Side::White { white_pov } else { -white_pov },
            score: p.score,
            depth: p.depth,
            ply: i.min(u16::MAX as usize) as u16,
            egdb: exact,
            pieces: pieces as u8,
        };
        f(pos, &s);
    })?;
    Ok(())
}
