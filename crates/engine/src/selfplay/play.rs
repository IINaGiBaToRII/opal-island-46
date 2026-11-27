
use super::extract::{legal, move_index};
use super::format::{EndReason, Game, PlyRec, StartPos};
use super::SelfplayParams;
use crate::egdb::Egdb;
use crate::eval::Evaluator;
use crate::position::{Position, Side};
use crate::rules::Rules;
use crate::search::{search_tt_db_recall_ctx, SearchCtx, SearchLimits};
use crate::tt::Tt;
use std::sync::atomic::{AtomicBool, Ordering};

const fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub struct Rng(u64);

impl Rng {
    pub fn for_game(seed: u64, chunk: u32, game: u32) -> Rng {
        Rng(splitmix(splitmix(splitmix(seed) ^ chunk as u64) ^ game as u64))
    }
    pub fn for_game_stream(seed: u64, chunk: u32, game: u32, stream: u64) -> Rng {
        Rng(splitmix(Rng::for_game(seed, chunk, game).0 ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03)))
    }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        splitmix(self.0)
    }
}

pub(crate) fn natural_end<R: Rules>(hist: &[Position<R>], has_moves: bool) -> Option<(i8, EndReason)> {
    natural_end_rules(hist, has_moves, false)
}

pub(crate) fn natural_end_rules<R: Rules>(hist: &[Position<R>], has_moves: bool, minsport: bool) -> Option<(i8, EndReason)> {
    let pos = hist.last()?;
    if R::ONE_V_ONE_DRAW && pos.white.count_ones() == 1 && pos.black.count_ones() == 1 {
        return Some((0, EndReason::RuleDraw));
    }
    if !has_moves {
        return Some((if pos.side == Side::White { -1 } else { 1 }, EndReason::NoMoves));
    }
    if hist.iter().filter(|h| *h == pos).count() >= 3 {
        return Some((0, EndReason::Repetition));
    }
    if minsport && crate::search::ms_active::<R>() {
        let (q, mc) = crate::search::ms_clocks(&hist[..hist.len() - 1], pos);
        return crate::search::ms_draw(pos, q, mc, crate::search::ms_road::<R>()).map(|_| (0, EndReason::RuleDraw));
    }
    if R::DRAW_MOVES > 0 {
        let men = |p: &Position<R>| (p.white | p.black) & !p.kings;
        let cnt = |p: &Position<R>| (p.white | p.black).count_ones();
        let mut plies = 0u32;
        let mut cur = pos;
        for prev in hist[..hist.len() - 1].iter().rev() {
            if cnt(prev) != cnt(cur) || men(prev) != men(cur) {
                break;
            }
            plies += 1;
            cur = prev;
        }
        if plies >= 2 * R::DRAW_MOVES {
            return Some((0, EndReason::RuleDraw));
        }
    }
    None
}

pub fn play_game<R: Rules, E: Evaluator<R>>(
    eval: &E,
    egdb: Option<&Egdb<R>>,
    p: &SelfplayParams,
    chunk: u32,
    game_no: u32,
    stop: &AtomicBool,
) -> Option<Game> {
    let mut rng = Rng::for_game(p.seed, chunk, game_no);
    let (bmin, bmax) = (p.book_min, p.book_max.max(p.book_min));
    let k = bmin + (rng.next() % (bmax - bmin + 1) as u64) as u32;
    let mut mid: Option<(u32, Rng)> = None;
    if p.rnd_permille > 0 {
        let mut r2 = Rng::for_game_stream(p.seed, chunk, game_no, 1);
        if r2.next() % 1000 < p.rnd_permille as u64 {
            let (a, b) = (p.rnd_min, p.rnd_max.max(p.rnd_min));
            let at = a + (r2.next() % (b - a + 1) as u64) as u32;
            if at >= k {
                mid = Some((at, r2));
            }
        }
    }
    let mut start: Option<StartPos> = None;
    let mut first = Position::<R>::start();
    if let Some(ops) = p.openings.as_ref().filter(|o| !o.is_empty()) {
        let i = (Rng::for_game_stream(p.seed, chunk, game_no, 2).next() % ops.len() as u64) as usize;
        if let Ok(pos) = Position::<R>::from_dfen(&ops[i]) {
            start = Some(StartPos { white: pos.white, black: pos.black, kings: pos.kings, side: (pos.side == Side::Black) as u8 });
            first = pos;
        }
    }
    let mut hist: Vec<Position<R>> = vec![first];
    let mut plies: Vec<PlyRec> = Vec::new();
    let mut tt = Tt::new(p.hash_mb);
    let mut ctx = SearchCtx::<E::State>::default();
    let lim = SearchLimits { nodes: Some(p.nodes), hash_mb: p.hash_mb, threads: 1, ..Default::default() };
    let (mut streak, mut sign) = (0u32, 0i32);
    let mut egdb_val = None;
    let db = egdb.filter(|_| p.use_egdb);
    let (result, reason) = loop {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        let pos = *hist.last().unwrap();
        let list = legal::<R>(&pos);
        if let Some(e) = natural_end(&hist, list.len != 0) {
            break e;
        }
        let mut in_db = false;
        if let Some(db) = db {
            if pos.occ().count_ones() <= db.max_pieces() {
                if let Some(v) = db.probe(&pos) {
                    if v == 0 && p.db_play_draws {
                        in_db = true;
                    } else {
                        egdb_val = Some(v);
                        let stm = if v > 0 { 1 } else if v < 0 { -1 } else { 0 };
                        break (if pos.side == Side::White { stm } else { -stm }, EndReason::Egdb);
                    }
                }
            }
        }
        if plies.len() as u32 >= p.max_plies {
            break (0, EndReason::MaxPlies);
        }
        let book = (plies.len() as u32) < k;
        let mid_now = mid.as_ref().is_some_and(|(at, _)| plies.len() as u32 == *at);
        let random = book || mid_now;
        let (mv, score, depth) = if book {
            (list[(rng.next() % list.len as u64) as usize], 0i32, 0u32)
        } else if let Some((_, r2)) = mid.as_mut().filter(|_| mid_now) {
            (list[(r2.next() % list.len as u64) as usize], 0i32, 0u32)
        } else {
            let r = search_tt_db_recall_ctx(&pos, &hist[..hist.len() - 1], eval, &lim, stop, &mut |_| {}, &mut tt, db.filter(|_| !in_db), None, Some(&mut ctx));
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let mv = r.best.filter(|m| list.iter().any(|x| x == m)).unwrap_or(list[0]);
            (mv, r.score, r.depth)
        };
        let idx = move_index(&list, &mv).unwrap_or(0);
        plies.push(PlyRec {
            from: mv.from,
            to: mv.to,
            idx,
            score: score.clamp(-32000, 32000) as i16,
            depth: depth.min(255) as u8,
            random,
            in_db,
        });
        hist.push(pos.make(&mv));
        if !random && p.adj_plies > 0 {
            let w = if pos.side == Side::White { score } else { -score };
            if w.abs() >= p.adj_score {
                let s = w.signum();
                if s == sign {
                    streak += 1;
                } else {
                    sign = s;
                    streak = 1;
                }
            } else {
                streak = 0;
                sign = 0;
            }
            if streak >= p.adj_plies {
                break (sign as i8, EndReason::Adjudicated);
            }
        } else if random {
            streak = 0;
            sign = 0;
        }
    };
    let random_plies = plies.iter().take_while(|x| x.random).count() as u8;
    Some(Game { start, random_plies, plies, result, reason, egdb: egdb_val })
}

#[cfg(test)]
mod draw_rules_tests {
    use super::*;
    use crate::rules::{Brazilian, Russian};

    #[test]
    fn natural_end_minsport() {
        fn game<R: Rules>(n: usize) -> Vec<Position<R>> {
            let sq = ["c1", "e1", "g1", "h2", "f2", "d2", "b2", "a3", "c3", "e3", "g3"];
            (0..n).map(|i| Position::<R>::from_dfen(&crate::facade::imp::fen_to_dfen::<R>(&format!("{}:WK{}:BKh8", if i % 2 == 0 { 'W' } else { 'B' }, sq[i])).unwrap()).unwrap()).collect()
        }
        let h = game::<Russian<true>>(11);
        assert!(natural_end_rules(&h[..10], true, true).is_none());
        assert!(matches!(natural_end_rules(&h, true, true), Some((0, EndReason::RuleDraw))));
        assert!(natural_end_rules(&h, true, false).is_none() && natural_end(&h, true).is_none());
        assert!(matches!(natural_end_rules(&h, false, true), Some((_, EndReason::NoMoves))));
        let b = game::<Brazilian<true>>(11);
        assert!(natural_end_rules(&b, true, true).is_none());
    }
}
