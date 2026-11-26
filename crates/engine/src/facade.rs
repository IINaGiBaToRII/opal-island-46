
use crate::eval::Evaluator;
use crate::learn::select::{with_eval, EvalRun, EvalSpec};
use crate::geometry::Geometry;
use crate::movegen::{generate, perft as perft_gen, ListSink, Move, MoveList, PathMove, PathSink};
use crate::position::{Position, Side};
use crate::rules::{
    American, Brazilian, Czech, International, Italian, Numbering, Rules, Russian, Spanish, Thai, Turkish,
};
use crate::search::{search_tt_db_recall, RootLine, SearchInfo, SearchLimits, MATE, MATE_BOUND};
use crate::tt::Tt;
use serde::{Deserialize, Serialize};
use std::sync::atomic::AtomicBool;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VariantId {
    Russian,
    Brazilian,
    International,
    American,
    Italian,
    Spanish,
    Czech,
    Thai,
    Turkish,
}

impl VariantId {
    pub fn name(&self) -> &'static str {
        match self {
            VariantId::Russian => "russian",
            VariantId::Brazilian => "brazilian",
            VariantId::International => "international",
            VariantId::American => "american",
            VariantId::Italian => "italian",
            VariantId::Spanish => "spanish",
            VariantId::Czech => "czech",
            VariantId::Thai => "thai",
            VariantId::Turkish => "turkish",
        }
    }
    pub fn parse(s: &str) -> Option<VariantId> {
        Some(match s {
            "russian" => VariantId::Russian,
            "brazilian" => VariantId::Brazilian,
            "international" => VariantId::International,
            "american" => VariantId::American,
            "italian" => VariantId::Italian,
            "spanish" => VariantId::Spanish,
            "czech" => VariantId::Czech,
            "thai" => VariantId::Thai,
            "turkish" => VariantId::Turkish,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameConfig {
    pub variant: VariantId,
    pub forced_capture: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoveInfo {
    pub notation: String,
    #[serde(default)]
    pub notation_official: Option<String>,
    pub from: u8,
    pub to: u8,
    pub path: Vec<u8>,
    pub captured: Vec<u8>,
    pub promotes: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoardInfo {
    pub size: usize,
    pub squares: Vec<u8>,
    #[serde(default)]
    pub numbers: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LineOut {
    #[serde(rename = "move")]
    pub notation: String,
    pub from: u8,
    pub to: u8,
    pub path: Vec<u8>,
    pub score: i32,
    pub mate: Option<i32>,
    pub pv: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wdl: Option<[f32; 3]>,
}

pub fn wdl_for_score(cfg: &GameConfig, score: i32) -> Option<[f32; 3]> {
    let a = crate::learn::registry::active(cfg);
    let (w, _d, l) = crate::learn::wdl::probs(a.wdl.as_ref()?, score);
    let (w, l) = (w as f32, l as f32);
    Some([w, (1.0 - w - l).max(0.0), l])
}

pub fn fill_wdl(cfg: &GameConfig, lines: &mut [LineOut], exact_draw: bool) {
    for l in lines.iter_mut() {
        l.wdl = wdl_for_score(cfg, l.score).map(|w| if exact_draw && l.mate.is_none() && l.score == 0 { [0.0, 1.0, 0.0] } else { w });
    }
}

pub fn mate_plies(score: i32) -> Option<i32> {
    if score > MATE_BOUND {
        Some(MATE - score)
    } else if score < -MATE_BOUND {
        Some(-(MATE + score))
    } else {
        None
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchInfoOut {
    pub depth: u32,
    pub score: i32,
    pub nodes: u64,
    pub nps: u64,
    pub time_ms: u64,
    #[serde(default)]
    pub hashfull: u32,
    pub pv: Vec<String>,
    #[serde(default)]
    pub lines: Vec<LineOut>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResultOut {
    pub best_move: Option<String>,
    pub score: i32,
    pub pv: Vec<String>,
    pub depth: u32,
    pub nodes: u64,
    pub time_ms: u64,
    #[serde(default)]
    pub lines: Vec<LineOut>,
    #[serde(default)]
    pub eval_id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub early: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hist_dep: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PositionCheck {
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PosFacts {
    pub canonical: String,
    pub legal: usize,
    pub egdb_exact: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Ongoing,
    WhiteWins,
    BlackWins,
    Draw,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameStatus {
    pub outcome: Outcome,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VariantInfo {
    pub id: VariantId,
    pub name: String,
    pub size: usize,
    pub available: bool,
}

pub fn variants() -> Vec<VariantInfo> {
    use VariantId::*;
    [
        (Russian, "Русские", 8),
        (Brazilian, "Бразильские", 8),
        (International, "Международные", 10),
        (American, "Английские", 8),
        (Italian, "Итальянские", 8),
        (Spanish, "Испанские", 8),
        (Czech, "Чешские", 8),
        (Thai, "Тайские", 8),
        (Turkish, "Турецкие", 8),
    ]
    .into_iter()
    .map(|(id, name, size)| VariantInfo { id, name: name.into(), size, available: variant_available(id) })
    .collect()
}

#[derive(Default)]
pub struct TtHandle(Option<(GameConfig, usize, String, Tt)>);

impl TtHandle {
    pub fn new() -> Self {
        TtHandle(None)
    }
    fn take(&mut self, cfg: &GameConfig, mb: usize, ev: &str) -> Tt {
        if let Some((c, m, e, t)) = self.0.take() {
            if c == *cfg && m == mb && e == ev {
                return t;
            }
            drop(t);
        }
        Tt::new(mb)
    }
    fn put(&mut self, cfg: &GameConfig, mb: usize, ev: String, t: Tt) {
        self.0 = Some((*cfg, mb, ev, t));
    }
}

pub fn eval_tt_key(spec: &crate::learn::select::EvalSpec) -> String {
    use crate::learn::select::EvalSpec;
    let id = match spec {
        EvalSpec::Simple => 0,
        EvalSpec::File(p) => crate::learn::select::load_cached(p).map_or(0, |l| l.eval_id),
    };
    format!("{spec:?}#{id:08x}")
}

pub fn variant_available(v: VariantId) -> bool {
    board_info(&GameConfig { variant: v, forced_capture: true }).is_ok()
}

type G<R> = <R as Rules>::Geo;

pub(crate) mod imp {
    use super::*;

    pub fn parse<R: Rules>(dfen: &str) -> Result<Position<R>, String> {
        Position::<R>::from_dfen(dfen)
    }

    pub fn sq_of_bit<R: Rules>(bit: u8) -> u8 {
        let (r, f) = G::<R>::coords(bit as u32);
        (r * G::<R>::N + f) as u8
    }

    fn bits_to_sqs<R: Rules>(mut x: u64) -> Vec<u8> {
        let mut v = vec![];
        while x != 0 {
            v.push(sq_of_bit::<R>(x.trailing_zeros() as u8));
            x &= x - 1;
        }
        v.sort_unstable();
        v
    }

    pub fn number_of<R: Rules>(bit: u32) -> Option<u32> {
        let n = G::<R>::N;
        let mut k = 0;
        let rows: Vec<usize> = match R::NUMBERING {
            Numbering::None => return None,
            Numbering::TopDown => (0..n).rev().collect(),
            Numbering::BottomUpRightToLeft => (0..n).collect(),
        };
        for r in rows {
            let files: Vec<usize> = if R::NUMBERING == Numbering::TopDown { (0..n).collect() } else { (0..n).rev().collect() };
            for f in files {
                if let Some(b) = G::<R>::bit(r, f) {
                    k += 1;
                    if b == bit {
                        return Some(k);
                    }
                }
            }
        }
        None
    }

    pub fn bit_of_number<R: Rules>(num: u32) -> Option<u32> {
        (0..64u32).find(|b| G::<R>::VALID >> b & 1 != 0 && number_of::<R>(*b) == Some(num))
    }

    pub fn numeric_text<R: Rules>(pm: &PathMove, all: &[PathMove]) -> String {
        let sep = if pm.mv.captured != 0 { "x" } else { "-" };
        let num = |b: u8| number_of::<R>(b as u32).map(|n| n.to_string()).unwrap_or_default();
        let ambiguous = pm.mv.captured != 0
            && all.iter().any(|o| o.mv.from == pm.mv.from && o.mv.to == pm.mv.to && o.mv != pm.mv);
        let mut s = num(pm.mv.from);
        if ambiguous {
            for b in &pm.path {
                s.push_str(sep);
                s.push_str(&num(*b));
            }
        } else {
            s.push_str(sep);
            s.push_str(&num(pm.mv.to));
        }
        s
    }

    pub fn notation<R: Rules>(pm: &PathMove, all: &[PathMove]) -> String {
        if R::NUMERIC_NOTATION {
            return numeric_text::<R>(pm, all);
        }
        let sep = if pm.mv.captured != 0 { ":" } else { "-" };
        let mut s = Position::<R>::square_name(pm.mv.from as u32);
        for b in &pm.path {
            s.push_str(sep);
            s.push_str(&Position::<R>::square_name(*b as u32));
        }
        s
    }

    fn parse_token<R: Rules>(s: &str) -> Result<u32, String> {
        let s = s.trim();
        if s.as_bytes().first().is_some_and(|c| c.is_ascii_digit()) {
            let n: u32 = s.parse().map_err(|_| format!("bad square '{s}'"))?;
            return bit_of_number::<R>(n).ok_or_else(|| format!("bad square number '{s}'"));
        }
        Position::<R>::parse_square(s)
    }

    pub fn egdb_max_pieces<R: Rules>() -> u32 {
        crate::egdb::shared::<R>().map_or(0, |d| d.max_pieces())
    }

    pub fn color_mirror_dfen<R: Rules>(dfen: &str) -> Result<Option<String>, String> {
        let p = parse::<R>(dfen)?;
        Ok(crate::recall::color_mirror(&p).map(|m| m.to_dfen()))
    }

    pub fn all_paths<R: Rules>(pos: &Position<R>) -> Vec<PathMove> {
        let mut s = PathSink::default();
        generate(pos, &mut s);
        s.out
    }

    pub fn move_notation<R: Rules>(pos: &Position<R>, mv: &Move) -> String {
        let all = all_paths(pos);
        all.iter()
            .find(|pm| pm.mv == *mv)
            .map(|pm| notation::<R>(pm, &all))
            .unwrap_or_else(|| "?".to_string())
    }

    pub fn start_position<R: Rules>() -> Result<String, String> {
        Ok(Position::<R>::start().to_dfen())
    }

    pub fn legal_moves<R: Rules>(dfen: &str) -> Result<Vec<MoveInfo>, String> {
        let pos = parse::<R>(dfen)?;
        let all = all_paths(&pos);
        Ok(all
            .iter()
            .map(|pm| MoveInfo {
                notation: notation::<R>(pm, &all),
                notation_official: if R::NUMBERING != Numbering::None { Some(numeric_text::<R>(pm, &all)) } else { None },
                from: sq_of_bit::<R>(pm.mv.from),
                to: sq_of_bit::<R>(pm.mv.to),
                path: pm.path.iter().map(|b| sq_of_bit::<R>(*b)).collect(),
                captured: bits_to_sqs::<R>(pm.mv.captured),
                promotes: pm.mv.promo,
            })
            .collect())
    }

    pub fn find_move<R: Rules>(pos: &Position<R>, text: &str) -> Result<Move, String> {
        let mut sqs = vec![];
        for part in text.trim().split(|c| c == '-' || c == ':' || c == 'x') {
            sqs.push(parse_token::<R>(part)? as u8);
        }
        if sqs.len() < 2 {
            return Err(format!("bad move notation '{text}'"));
        }
        let all = all_paths(pos);
        let full: Vec<&PathMove> = all
            .iter()
            .filter(|pm| pm.mv.from == sqs[0] && pm.path.len() == sqs.len() - 1 && pm.path[..] == sqs[1..])
            .collect();
        if let Some(pm) = full.first() {
            return Ok(pm.mv);
        }
        if sqs.len() == 2 {
            let mut res: Vec<Move> = vec![];
            for pm in all.iter().filter(|pm| pm.mv.from == sqs[0] && pm.mv.to == sqs[1]) {
                if !res.contains(&pm.mv) {
                    res.push(pm.mv);
                }
            }
            return match res.len() {
                1 => Ok(res[0]),
                0 => Err(format!("illegal move '{text}'")),
                _ => Err(format!("ambiguous move '{text}': give the full path")),
            };
        }
        Err(format!("illegal move '{text}'"))
    }

    pub fn apply_move<R: Rules>(dfen: &str, text: &str) -> Result<String, String> {
        let pos = parse::<R>(dfen)?;
        let mv = find_move(&pos, text)?;
        Ok(pos.make(&mv).to_dfen())
    }

    pub fn perft<R: Rules>(dfen: &str, depth: u32, dedup: bool) -> Result<u64, String> {
        let pos = parse::<R>(dfen)?;
        Ok(if dedup { perft_gen::<R, true>(&pos, depth) } else { perft_gen::<R, false>(&pos, depth) })
    }

    pub fn perft_divide<R: Rules>(dfen: &str, depth: u32, dedup: bool) -> Result<Vec<(String, u64)>, String> {
        let pos = parse::<R>(dfen)?;
        let mut out = vec![];
        let mut seen: Vec<Move> = vec![];
        let all = all_paths(&pos);
        for pm in all.iter().cloned() {
            if dedup {
                if seen.contains(&pm.mv) {
                    continue;
                }
                seen.push(pm.mv);
            }
            let child = pos.make(&pm.mv);
            let n = if depth <= 1 {
                1
            } else if dedup {
                perft_gen::<R, true>(&child, depth - 1)
            } else {
                perft_gen::<R, false>(&child, depth - 1)
            };
            out.push((notation::<R>(&pm, &all), n));
        }
        Ok(out)
    }

    pub fn board_info<R: Rules>() -> Result<BoardInfo, String> {
        let n = G::<R>::N;
        let mut squares = vec![];
        let mut numbers = vec![];
        for r in 0..n {
            for f in 0..n {
                if let Some(b) = G::<R>::bit(r, f) {
                    squares.push((r * n + f) as u8);
                    if let Some(k) = number_of::<R>(b) {
                        numbers.push(k as u8);
                    }
                }
            }
        }
        Ok(BoardInfo { size: n, squares, numbers })
    }

    pub fn validate_position<R: Rules>(dfen: &str) -> Result<PositionCheck, String> {
        let pos = parse::<R>(dfen).map_err(|e| format!("Неверная запись позиции: {e}"))?;
        let n = G::<R>::N;
        let st = Position::<R>::start();
        let max = st.white.count_ones().max(st.black.count_ones());
        for (name, set, who) in [("белых", pos.white, "белая"), ("чёрных", pos.black, "чёрная")] {
            if set == 0 {
                return Err(format!("У {name} нет ни одной фигуры."));
            }
            if set.count_ones() > max {
                return Err(format!("У {name} {} фигур — больше максимума ({max}) для этого варианта.", set.count_ones()));
            }
            let promo_row = if who == "белая" { n - 1 } else { 0 };
            let mut men = set & !pos.kings;
            while men != 0 {
                let b = men.trailing_zeros();
                men &= men - 1;
                if G::<R>::coords(b).0 == promo_row {
                    return Err(format!(
                        "Простая фигура ({who}) на поле {} стоит на последнем ряду — поставьте сильную фигуру.",
                        Position::<R>::square_name(b)
                    ));
                }
            }
        }
        pos.check_invariants()?;
        Ok(PositionCheck { warnings: unreachable_warnings::<R>(&pos) })
    }

    fn unreachable_warnings<R: Rules>(pos: &Position<R>) -> Vec<String> {
        let n = G::<R>::N;
        let st = Position::<R>::start();
        let row_mask = |row: u32| -> u64 {
            let mut m = 0u64;
            for f in 0..n {
                if let Some(b) = G::<R>::bit(row as _, f) {
                    m |= 1u64 << b;
                }
            }
            m
        };
        let mut out = vec![];
        let sides = [
            ("белых", "чёрных", pos.white, pos.black, st.black, (n - 1) as u32),
            ("чёрных", "белых", pos.black, pos.white, st.white, 0u32),
        ];
        for (who, opp, own, opp_set, opp_start, promo_row) in sides {
            if own & pos.kings == 0 {
                continue;
            }
            let opp_men = opp_set & !pos.kings;
            if opp_set == opp_start && opp_men == opp_set {
                out.push(format!(
                    "У {who} есть сильная фигура, а лагерь {opp} полностью в начальной расстановке: сильной фигуре неоткуда было взяться."
                ));
                continue;
            }
            let promo = row_mask(promo_row);
            if promo != 0 && promo & opp_men == promo && promo & !opp_start == 0 {
                out.push(format!(
                    "У {who} есть сильная фигура, хотя все поля последнего ряда заняты нетронутыми фигурами {opp} на исходных местах."
                ));
            }
        }
        if all_paths(pos).is_empty() {
            let who = if pos.side == Side::White { "белых" } else { "чёрных" };
            out.push(format!("У {who} нет ходов, а очередь их: в настоящей партии она уже окончена."));
        }
        out
    }

    fn fen_square<R: Rules>(bit: u32) -> (u32, String) {
        if R::NUMERIC_NOTATION {
            let k = number_of::<R>(bit).unwrap_or(0);
            (k, k.to_string())
        } else {
            (bit, Position::<R>::square_name(bit))
        }
    }

    pub fn dfen_to_fen<R: Rules>(dfen: &str) -> Result<String, String> {
        let pos = parse::<R>(dfen)?;
        let list = |set: u64| -> String {
            let mut v: Vec<(u32, String)> = vec![];
            let mut x = set;
            while x != 0 {
                let b = x.trailing_zeros();
                x &= x - 1;
                let (key, name) = fen_square::<R>(b);
                v.push((key, if pos.kings >> b & 1 != 0 { format!("K{name}") } else { name }));
            }
            v.sort();
            v.into_iter().map(|(_, s)| s).collect::<Vec<_>>().join(",")
        };
        Ok(format!(
            "{}:W{}:B{}",
            if pos.side == Side::White { "W" } else { "B" },
            list(pos.white),
            list(pos.black)
        ))
    }

    pub fn fen_to_dfen<R: Rules>(fen: &str) -> Result<String, String> {
        let s = fen.trim().trim_end_matches('.');
        let mut parts = s.split(':');
        let side = match parts.next().map(str::trim) {
            Some("W") | Some("w") => Side::White,
            Some("B") | Some("b") => Side::Black,
            _ => return Err("PDN-FEN: строка должна начинаться с W или B (чья очередь), например W:W31-50:B1-20".into()),
        };
        let (mut white, mut black, mut kings) = (0u64, 0u64, 0u64);
        for seg in parts {
            let seg = seg.trim();
            if seg.is_empty() {
                continue;
            }
            let mut chars = seg.chars();
            let first = chars.next().unwrap_or(' ');
            let c = &seg[..first.len_utf8()];
            let rest = chars.as_str();
            let is_white = match c {
                "W" => true,
                "B" => false,
                _ => return Err(format!("PDN-FEN: ожидался цвет W или B, а не '{c}'")),
            };
            for item in rest.split(',') {
                let item = item.trim();
                if item.is_empty() {
                    continue;
                }
                let (king, tok) = match item.strip_prefix('K') {
                    Some(t) => (true, t),
                    None => (false, item),
                };
                let mut bits = vec![];
                match tok.split_once('-') {
                    Some((a, b)) if a.bytes().all(|c| c.is_ascii_digit()) && b.bytes().all(|c| c.is_ascii_digit()) && !a.is_empty() && !b.is_empty() => {
                        let (a, b): (u32, u32) = (a.parse().map_err(|_| "PDN-FEN: плохой диапазон")?, b.parse().map_err(|_| "PDN-FEN: плохой диапазон")?);
                        for k in a..=b {
                            bits.push(bit_of_number::<R>(k).ok_or_else(|| format!("PDN-FEN: нет поля с номером {k}"))?);
                        }
                    }
                    _ => bits.push(parse_token::<R>(tok).map_err(|e| format!("PDN-FEN: {e}"))?),
                }
                for b in bits {
                    let m = 1u64 << b;
                    if (white | black) & m != 0 {
                        return Err(format!("PDN-FEN: поле {} указано дважды", fen_square::<R>(b).1));
                    }
                    if is_white { white |= m } else { black |= m }
                    if king { kings |= m }
                }
            }
        }
        Ok(Position::<R>::new(white, black, kings, side).to_dfen())
    }

    pub fn game_status<R: Rules>(dfen: &str, history: &[String]) -> Result<GameStatus, String> {
        let pos = parse::<R>(dfen)?;
        if R::ONE_V_ONE_DRAW && pos.white.count_ones() == 1 && pos.black.count_ones() == 1 {
            return Ok(GameStatus { outcome: Outcome::Draw, reason: "one piece each".into() });
        }
        let mut l = MoveList::new();
        generate(&pos, &mut ListSink::<true>(&mut l));
        if l.len == 0 {
            let (outcome, who) = match pos.side {
                Side::White => (Outcome::BlackWins, "white"),
                Side::Black => (Outcome::WhiteWins, "black"),
            };
            return Ok(GameStatus { outcome, reason: format!("{who} has no legal moves") });
        }
        let mut count = 1;
        for h in history {
            if parse::<R>(h)? == pos {
                count += 1;
            }
        }
        if count >= 3 {
            return Ok(GameStatus { outcome: Outcome::Draw, reason: "threefold repetition".into() });
        }
        if R::DRAW_MOVES > 0 {
            let men = |p: &Position<R>| (p.white | p.black) & !p.kings;
            let cnt = |p: &Position<R>| (p.white | p.black).count_ones();
            let mut plies = 0u32;
            let mut cur = pos;
            for h in history.iter().rev() {
                let prev = parse::<R>(h)?;
                if cnt(&prev) != cnt(&cur) || men(&prev) != men(&cur) {
                    break;
                }
                plies += 1;
                cur = prev;
            }
            if plies >= 2 * R::DRAW_MOVES {
                return Ok(GameStatus {
                    outcome: Outcome::Draw,
                    reason: format!("{} moves by each side without capture or man move", R::DRAW_MOVES),
                });
            }
        }
        Ok(GameStatus { outcome: Outcome::Ongoing, reason: String::new() })
    }

    pub fn history_risk<R: Rules>(dfen: &str, history: &[String], notation: Option<&str>) -> Result<HistoryRisk, String> {
        let pos = parse::<R>(dfen)?;
        let child = match notation {
            Some(n) => parse::<R>(&apply_move::<R>(dfen, n)?)?,
            None => pos,
        };
        let mut cur_repeated = false;
        let mut child_repeats = notation.is_some() && child == pos;
        for h in history {
            let p = parse::<R>(h)?;
            cur_repeated |= p == pos;
            child_repeats |= notation.is_some() && p == child;
        }
        let men = |p: &Position<R>| (p.white | p.black) & !p.kings;
        let cnt = |p: &Position<R>| (p.white | p.black).count_ones();
        let mut plies = 0u32;
        let tail;
        {
            let mut c = pos;
            for h in history.iter().rev() {
                let prev = parse::<R>(h)?;
                if cnt(&prev) != cnt(&c) || men(&prev) != men(&c) {
                    break;
                }
                plies += 1;
                c = prev;
            }
            tail = plies > 0;
        }
        plies = if R::DRAW_MOVES == 0 || cnt(&pos) != cnt(&child) || men(&pos) != men(&child) { 0 } else { plies + 1 };
        Ok(HistoryRisk { child_repeats, cur_repeated, plies_after: plies, draw_plies: 2 * R::DRAW_MOVES, kings: child.kings != 0, tail })
    }

    fn pv_notation<R: Rules>(pos: &Position<R>, pv: &[Move]) -> Vec<String> {
        let mut p = *pos;
        let mut out = vec![];
        for m in pv {
            out.push(move_notation::<R>(&p, m));
            p = p.make(m);
        }
        out
    }

    fn line_out<R: Rules>(pos: &Position<R>, all: &[PathMove], l: &RootLine) -> LineOut {
        let pm = all.iter().find(|pm| pm.mv == l.mv);
        LineOut {
            notation: pm.map(|pm| notation::<R>(pm, all)).unwrap_or_else(|| "?".into()),
            from: sq_of_bit::<R>(l.mv.from),
            to: sq_of_bit::<R>(l.mv.to),
            path: pm.map(|pm| pm.path.iter().map(|b| sq_of_bit::<R>(*b)).collect()).unwrap_or_else(|| vec![sq_of_bit::<R>(l.mv.to)]),
            score: l.score,
            mate: mate_plies(l.score),
            pv: pv_notation(pos, &l.pv),
            wdl: None,
        }
    }

    pub fn facts<R: Rules>(dfen: &str) -> Result<PosFacts, String> {
        let pos = parse::<R>(dfen)?;
        let mut l = MoveList::new();
        generate(&pos, &mut ListSink::<true>(&mut l));
        let egdb_exact = crate::egdb::shared::<R>().is_some_and(|d| pos.occ().count_ones() <= d.max_pieces() && d.probe(&pos).is_some());
        Ok(PosFacts { canonical: pos.to_dfen(), legal: l.len, egdb_exact })
    }

    pub fn children<R: Rules>(dfen: &str) -> Result<Vec<(String, String)>, String> {
        let pos = parse::<R>(dfen)?;
        let all = all_paths(&pos);
        Ok(all.iter().map(|pm| (notation::<R>(pm, &all), pos.make(&pm.mv).to_dfen())).collect())
    }

    pub fn pass_turn<R: Rules>(dfen: &str) -> Result<Option<String>, String> {
        let pos = parse::<R>(dfen)?;
        let q = Position::<R>::new(pos.white, pos.black, pos.kings, pos.side.other());
        if q.check_invariants().is_err() {
            return Ok(None);
        }
        let mut l = MoveList::new();
        generate(&q, &mut ListSink::<true>(&mut l));
        Ok(if l.len == 0 { None } else { Some(q.to_dfen()) })
    }

    pub fn search<R: Rules>(
        dfen: &str,
        history: &[String],
        limits: &SearchLimits,
        spec: &EvalSpec,
        stop: &AtomicBool,
        on_info: &mut dyn FnMut(&SearchInfoOut, bool),
        tt: &mut Tt,
        seeds: &[RecallSeed],
    ) -> Result<SearchResultOut, String> {
        let pos = parse::<R>(dfen)?;
        let hist: Vec<Position<R>> = history.iter().map(|h| parse::<R>(h)).collect::<Result<_, _>>()?;
        let all = all_paths(&pos);
        let hinted;
        let limits = match limits.root_hint {
            Some((a, b)) => {
                let bits = all.iter().find(|pm| sq_of_bit::<R>(pm.mv.from) == a && sq_of_bit::<R>(pm.mv.to) == b).map(|pm| (pm.mv.from, pm.mv.to));
                hinted = SearchLimits { root_hint: bits, ..limits.clone() };
                &hinted
            }
            None => limits,
        };
        let mut cb = |i: &SearchInfo| {
            on_info(&SearchInfoOut {
                depth: i.depth,
                score: i.score,
                nodes: i.nodes,
                nps: i.nps,
                time_ms: i.time_ms,
                hashfull: i.hashfull,
                pv: pv_notation(&pos, &i.pv),
                lines: i.lines.iter().map(|l| line_out::<R>(&pos, &all, l)).collect(),
            }, i.hist_dep)
        };
        let db = crate::egdb::shared::<R>();
        let (r, eval_id) = with_eval::<R, _>(spec, SearchRun { pos: &pos, hist: &hist, limits, stop, cb: &mut cb, tt, db: db.as_deref(), seeds })?;
        Ok(SearchResultOut {
            eval_id,
            best_move: r.best.map(|m| move_notation::<R>(&pos, &m)),
            score: r.score,
            pv: pv_notation(&pos, &r.pv),
            depth: r.depth,
            nodes: r.nodes,
            time_ms: r.time_ms,
            lines: r.lines.iter().map(|l| line_out::<R>(&pos, &all, l)).collect(),
            early: r.early.map(String::from),
            hist_dep: r.hist_dep,
        })
    }
}

pub const RECALL_PLIES: u32 = 2;
pub const RECALL_MAX_POSITIONS: usize = 200;
pub const RECALL_MIRROR: bool = true;
pub const RECALL_DRAW_BAND: i32 = 10;
pub const RECALL_DB_MARGIN: u32 = 12;

pub fn recall_draw_sensitive(pieces: u32, db_max: u32, score: i32) -> bool {
    db_max > 0 && pieces <= db_max + RECALL_DB_MARGIN && score.abs() <= RECALL_DRAW_BAND
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecallSeed {
    pub dfen: String,
    pub eval_id: u32,
    pub depth: u32,
    pub score: i32,
    pub best: Option<String>,
    pub mirror: bool,
}

thread_local! {
    static RECALL_LAST: std::cell::Cell<Option<crate::recall::RecallSnapshot>> = const { std::cell::Cell::new(None) };
}

pub fn recall_last_stats() -> Option<crate::recall::RecallSnapshot> {
    RECALL_LAST.with(|c| c.take())
}

struct SearchRun<'a, R: Rules> {
    pos: &'a Position<R>,
    hist: &'a [Position<R>],
    limits: &'a SearchLimits,
    stop: &'a AtomicBool,
    cb: &'a mut dyn FnMut(&SearchInfo),
    tt: &'a mut Tt,
    db: Option<&'a crate::egdb::Egdb<R>>,
    seeds: &'a [RecallSeed],
}

impl<'a, R: Rules> EvalRun<R> for SearchRun<'a, R> {
    type Out = (crate::search::SearchResult, u32);
    fn run<E: Evaluator<R>>(self, eval: &E, eval_id: u32) -> Self::Out {
        let mem = crate::recall::Recall::new(self.seeds.len() + 16);
        for s in self.seeds.iter().filter(|s| s.eval_id == eval_id && s.depth > 0) {
            let Ok(p) = Position::<R>::from_dfen(&s.dfen) else { continue };
            let mv = s.best.as_deref().and_then(|n| {
                let src = if s.mirror { crate::recall::color_mirror(&p)? } else { p };
                let all = imp::all_paths(&src);
                let m = all.iter().find(|pm| imp::notation::<R>(pm, &all) == n).map(|pm| (pm.mv.from, pm.mv.to))?;
                if s.mirror { crate::recall::mirror_move(crate::recall::color_tables::<R>()?, m) } else { Some(m) }
            });
            mem.put(&p, crate::recall::RecallEntry { score: s.score, depth: s.depth.min(255) as u8, bound: crate::tt::BOUND_EXACT, mv, nodes: 0, eval_id });
        }
        let rc = crate::recall::RecallCtx { mem: &mem, eval_id, plies: RECALL_PLIES, max_positions: RECALL_MAX_POSITIONS, record: false, mirror: false };
        let rc = (!mem.is_empty()).then_some(&rc);
        let r = search_tt_db_recall(self.pos, self.hist, eval, self.limits, self.stop, self.cb, self.tt, self.db, rc);
        if let Some(rc) = rc {
            RECALL_LAST.with(|c| c.set(Some(rc.mem.snapshot())));
        }
        (r, eval_id)
    }
}

macro_rules! dispatch {
    ($cfg:expr, $f:ident ( $($a:expr),* )) => {
        match ($cfg.variant, $cfg.forced_capture) {
            (VariantId::Russian, true) => imp::$f::<Russian<true>>($($a),*),
            (VariantId::Russian, false) => imp::$f::<Russian<false>>($($a),*),
            (VariantId::Brazilian, true) => imp::$f::<Brazilian<true>>($($a),*),
            (VariantId::Brazilian, false) => imp::$f::<Brazilian<false>>($($a),*),
            (VariantId::International, true) => imp::$f::<International<true>>($($a),*),
            (VariantId::International, false) => imp::$f::<International<false>>($($a),*),
            (VariantId::American, true) => imp::$f::<American<true>>($($a),*),
            (VariantId::American, false) => imp::$f::<American<false>>($($a),*),
            (VariantId::Italian, true) => imp::$f::<Italian<true>>($($a),*),
            (VariantId::Italian, false) => imp::$f::<Italian<false>>($($a),*),
            (VariantId::Spanish, true) => imp::$f::<Spanish<true>>($($a),*),
            (VariantId::Spanish, false) => imp::$f::<Spanish<false>>($($a),*),
            (VariantId::Czech, true) => imp::$f::<Czech<true>>($($a),*),
            (VariantId::Czech, false) => imp::$f::<Czech<false>>($($a),*),
            (VariantId::Thai, true) => imp::$f::<Thai<true>>($($a),*),
            (VariantId::Thai, false) => imp::$f::<Thai<false>>($($a),*),
            (VariantId::Turkish, true) => imp::$f::<Turkish<true>>($($a),*),
            (VariantId::Turkish, false) => imp::$f::<Turkish<false>>($($a),*),
        }
    };
}

pub fn color_mirror_dfen(cfg: &GameConfig, dfen: &str) -> Result<Option<String>, String> {
    dispatch!(cfg, color_mirror_dfen(dfen))
}

pub fn egdb_max_pieces(cfg: &GameConfig) -> u32 {
    dispatch!(cfg, egdb_max_pieces())
}

pub fn start_position(cfg: &GameConfig) -> Result<String, String> {
    dispatch!(cfg, start_position())
}

pub fn legal_moves(cfg: &GameConfig, dfen: &str) -> Result<Vec<MoveInfo>, String> {
    dispatch!(cfg, legal_moves(dfen))
}

pub fn apply_move(cfg: &GameConfig, dfen: &str, notation: &str) -> Result<String, String> {
    dispatch!(cfg, apply_move(dfen, notation))
}

pub fn search(
    cfg: &GameConfig,
    dfen: &str,
    history: &[String],
    limits: &SearchLimits,
    stop: &AtomicBool,
    on_info: &mut dyn FnMut(&SearchInfoOut),
) -> Result<SearchResultOut, String> {
    search_shared(cfg, dfen, history, limits, stop, on_info, &mut TtHandle::new())
}

pub fn search_shared(
    cfg: &GameConfig,
    dfen: &str,
    history: &[String],
    limits: &SearchLimits,
    stop: &AtomicBool,
    on_info: &mut dyn FnMut(&SearchInfoOut),
    handle: &mut TtHandle,
) -> Result<SearchResultOut, String> {
    search_shared_dep(cfg, dfen, history, limits, stop, &mut |i, _| on_info(i), handle)
}

pub fn search_shared_dep(
    cfg: &GameConfig,
    dfen: &str,
    history: &[String],
    limits: &SearchLimits,
    stop: &AtomicBool,
    on_info: &mut dyn FnMut(&SearchInfoOut, bool),
    handle: &mut TtHandle,
) -> Result<SearchResultOut, String> {
    search_shared_dep_recall(cfg, dfen, history, limits, stop, on_info, handle, &[])
}

#[allow(clippy::too_many_arguments)]
pub fn search_shared_dep_recall(
    cfg: &GameConfig,
    dfen: &str,
    history: &[String],
    limits: &SearchLimits,
    stop: &AtomicBool,
    on_info: &mut dyn FnMut(&SearchInfoOut, bool),
    handle: &mut TtHandle,
    seeds: &[RecallSeed],
) -> Result<SearchResultOut, String> {
    let (spec, rest) = crate::learn::registry::resolve_search(cfg, &limits.opts)?;
    let lim = SearchLimits { opts: rest, ..limits.clone() };
    let ev_key = eval_tt_key(&spec);
    let mut tt = handle.take(cfg, limits.hash_mb, &ev_key);
    let r = dispatch!(cfg, search(dfen, history, &lim, &spec, stop, on_info, &mut tt, seeds));
    handle.put(cfg, limits.hash_mb, ev_key, tt);
    r
}

pub fn position_facts(cfg: &GameConfig, dfen: &str) -> Result<PosFacts, String> {
    dispatch!(cfg, facts(dfen))
}

pub fn children(cfg: &GameConfig, dfen: &str) -> Result<Vec<(String, String)>, String> {
    dispatch!(cfg, children(dfen))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryRisk {
    pub child_repeats: bool,
    pub cur_repeated: bool,
    pub plies_after: u32,
    pub draw_plies: u32,
    pub kings: bool,
    pub tail: bool,
}

impl HistoryRisk {
    pub fn draw_near(&self) -> bool {
        self.draw_plies > 0 && self.kings && self.plies_after * 2 >= self.draw_plies
    }
}

pub fn history_risk(cfg: &GameConfig, dfen: &str, history: &[String], notation: Option<&str>) -> Result<HistoryRisk, String> {
    dispatch!(cfg, history_risk(dfen, history, notation))
}

pub fn pass_turn(cfg: &GameConfig, dfen: &str) -> Result<Option<String>, String> {
    dispatch!(cfg, pass_turn(dfen))
}

pub fn perft(cfg: &GameConfig, dfen: &str, depth: u32, dedup: bool) -> Result<u64, String> {
    dispatch!(cfg, perft(dfen, depth, dedup))
}

pub fn perft_divide(cfg: &GameConfig, dfen: &str, depth: u32, dedup: bool) -> Result<Vec<(String, u64)>, String> {
    dispatch!(cfg, perft_divide(dfen, depth, dedup))
}

pub fn board_info(cfg: &GameConfig) -> Result<BoardInfo, String> {
    dispatch!(cfg, board_info())
}

pub fn validate_position(cfg: &GameConfig, dfen: &str) -> Result<PositionCheck, String> {
    dispatch!(cfg, validate_position(dfen))
}

pub fn dfen_to_fen(cfg: &GameConfig, dfen: &str) -> Result<String, String> {
    dispatch!(cfg, dfen_to_fen(dfen))
}

pub fn fen_to_dfen(cfg: &GameConfig, fen: &str) -> Result<String, String> {
    dispatch!(cfg, fen_to_dfen(fen))
}

pub fn game_status(cfg: &GameConfig, dfen: &str, history: &[String]) -> Result<GameStatus, String> {
    dispatch!(cfg, game_status(dfen, history))
}

#[cfg(test)]
mod validate_tests {
    use super::*;

    fn russian() -> GameConfig {
        GameConfig { variant: VariantId::Russian, forced_capture: true }
    }

    #[test]
    fn validate_start_has_no_warnings() {
        let cfg = russian();
        let st = start_position(&cfg).unwrap();
        assert!(validate_position(&cfg, &st).unwrap().warnings.is_empty());
    }

    #[test]
    fn validate_warns_kings_against_untouched_camp() {
        let cfg = russian();
        let d = "w .b.b.b.b/b.b.b.b./.b.b.b.b/......../.W....../......../......../W.......";
        let r = validate_position(&cfg, d).unwrap();
        assert_eq!(r.warnings.len(), 1, "{:?}", r.warnings);
        assert!(r.warnings[0].contains("сильная фигура") && r.warnings[0].contains("начальной расстановке"), "{:?}", r.warnings);
    }

    #[test]
    fn validate_warns_when_side_to_move_has_no_moves() {
        let cfg = russian();
        let d = "w ......../......../......../......../......../..b...../.b....../w.......";
        let r = validate_position(&cfg, d).unwrap();
        assert!(r.warnings.iter().any(|w| w.contains("нет ходов")), "{:?}", r.warnings);
    }
}
