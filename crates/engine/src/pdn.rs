
use crate::facade::{
    apply_move, dfen_to_fen, fen_to_dfen, legal_moves, start_position, validate_position, GameConfig, MoveInfo, VariantId,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PdnGame {
    pub cfg: Option<GameConfig>,
    #[serde(default)]
    pub start_dfen: Option<String>,
    #[serde(default)]
    pub moves: Vec<String>,
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
    #[serde(default)]
    pub result: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PdnGameOut {
    pub cfg: GameConfig,
    pub start_dfen: String,
    pub positions: Vec<String>,
    pub moves: Vec<String>,
    pub move_infos: Vec<MoveInfo>,
    pub tags: BTreeMap<String, String>,
    pub result: String,
}

fn game_type(v: VariantId) -> (u32, &'static str, &'static str, u32) {
    match v {
        VariantId::International => (20, "W", "N2", 0),
        VariantId::American => (21, "B", "A0", 0),
        VariantId::Italian => (22, "W", "A0", 1),
        VariantId::Spanish => (24, "W", "A0", 1),
        VariantId::Russian => (25, "W", "A0", 0),
        VariantId::Brazilian => (26, "W", "A0", 0),
        VariantId::Czech => (29, "W", "A0", 0),
        VariantId::Turkish => (30, "W", "A0", 0),
        VariantId::Thai => (31, "B", "A0", 0),
    }
}

fn variant_from_game_type(n: u32) -> Option<VariantId> {
    Some(match n {
        20 => VariantId::International,
        21 | 23 => VariantId::American,
        22 => VariantId::Italian,
        24 => VariantId::Spanish,
        25 => VariantId::Russian,
        26 => VariantId::Brazilian,
        29 => VariantId::Czech,
        30 => VariantId::Turkish,
        31 => VariantId::Thai,
        _ => return None,
    })
}

fn board_size(v: VariantId) -> usize {
    if v == VariantId::International {
        10
    } else {
        8
    }
}

fn clean_tag(s: &str) -> String {
    s.replace(['"', '\n', '\r', ']'], "'")
}

fn find_info(cfg: &GameConfig, dfen: &str, text: &str) -> Result<MoveInfo, String> {
    let all = legal_moves(cfg, dfen)?;
    let norm = |s: &str| s.trim().replace(':', "x");
    let t = norm(text);
    if let Some(m) = all.iter().find(|m| norm(&m.notation) == t || m.notation_official.as_deref().map(norm) == Some(t.clone())) {
        return Ok(m.clone());
    }
    let next = apply_move(cfg, dfen, text)?;
    let mut found: Vec<&MoveInfo> = vec![];
    for m in &all {
        if apply_move(cfg, dfen, &m.notation)? == next {
            found.push(m);
        }
    }
    found.first().map(|m| (*m).clone()).ok_or_else(|| format!("недопустимый ход '{text}'"))
}

pub fn game_to_pdn(g: &PdnGame) -> Result<String, String> {
    let cfg = g.cfg.ok_or("не указан вариант партии")?;
    let std_start = start_position(&cfg)?;
    let start = g.start_dfen.clone().unwrap_or_else(|| std_start.clone());
    let mut cur = start.clone();
    let mut texts = vec![];
    for (i, m) in g.moves.iter().enumerate() {
        let info = find_info(&cfg, &cur, m).map_err(|e| format!("ход {} ({m}): {e}", i + 1))?;
        cur = apply_move(&cfg, &cur, &info.notation)?;
        let t = if cfg.variant == VariantId::Russian { info.notation.clone() } else { info.notation.replace(':', "x") };
        texts.push(t);
    }
    let (gt, first, nota, inv) = game_type(cfg.variant);
    let n = board_size(cfg.variant);
    let result = match g.result.as_str() {
        "1-0" | "0-1" | "1/2-1/2" => g.result.clone(),
        _ => "*".to_string(),
    };
    let tag = |k: &str, def: &str| clean_tag(g.tags.get(k).map(String::as_str).filter(|s| !s.is_empty()).unwrap_or(def));
    let mut out = String::new();
    let mut put = |k: &str, v: String| out.push_str(&format!("[{k} \"{v}\"]\n"));
    put("Event", tag("Event", "?"));
    put("Date", tag("Date", "????.??.??"));
    put("White", tag("White", "?"));
    put("Black", tag("Black", "?"));
    put("Result", result.clone());
    put("GameType", format!("{gt},{first},{n},{n},{nota},{inv}"));
    put("Variant", cfg.variant.name().to_string());
    put("CaptureRule", if cfg.forced_capture { "forced" } else { "optional" }.to_string());
    if start != std_start {
        put("FEN", dfen_to_fen(&cfg, &start)?);
    }
    for (k, v) in &g.tags {
        if !["Event", "Date", "White", "Black", "Result", "GameType", "Variant", "CaptureRule", "FEN"].contains(&k.as_str())
            && !k.is_empty()
            && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            put(k, clean_tag(v));
        }
    }
    out.push('\n');
    let mut line = String::new();
    for (i, t) in texts.iter().enumerate() {
        let tok = if i % 2 == 0 { format!("{}. {t}", i / 2 + 1) } else { t.clone() };
        if line.len() + tok.len() + 1 > 78 {
            out.push_str(line.trim_end());
            out.push('\n');
            line.clear();
        }
        line.push_str(&tok);
        line.push(' ');
    }
    line.push_str(&result);
    out.push_str(&line);
    out.push('\n');
    Ok(out)
}

fn parse_tag_line(l: &str) -> Option<(String, String)> {
    let l = l.trim();
    let inner = l.strip_prefix('[')?.strip_suffix(']')?;
    let (k, rest) = inner.split_once(char::is_whitespace)?;
    let v = rest.trim().trim_start_matches('"').trim_end_matches('"');
    Some((k.to_string(), v.to_string()))
}

fn strip_movetext(s: &str) -> String {
    let mut out = String::new();
    let (mut brace, mut paren) = (0u32, 0u32);
    for line in s.lines() {
        let mut chars = line.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '{' => brace += 1,
                '}' => brace = brace.saturating_sub(1),
                '(' if brace == 0 => paren += 1,
                ')' if brace == 0 => paren = paren.saturating_sub(1),
                ';' if brace == 0 && paren == 0 => break,
                _ if brace == 0 && paren == 0 => out.push(c),
                _ => {}
            }
        }
        out.push(' ');
    }
    out
}

pub fn game_from_pdn(text: &str) -> Result<PdnGameOut, String> {
    let mut tags: BTreeMap<String, String> = BTreeMap::new();
    let mut body = String::new();
    let mut in_body = false;
    for line in text.lines() {
        let t = line.trim();
        if !in_body {
            if t.is_empty() {
                continue;
            }
            if t.starts_with('[') {
                if let Some((k, v)) = parse_tag_line(t) {
                    tags.insert(k, v);
                }
                continue;
            }
            in_body = true;
        } else if t.starts_with('[') && parse_tag_line(t).is_some() {
            break;
        }
        body.push_str(line);
        body.push('\n');
    }
    let variant = if let Some(v) = tags.get("Variant").and_then(|s| VariantId::parse(s.trim())) {
        v
    } else if let Some(gt) = tags.get("GameType") {
        let n: u32 = gt.split(',').next().unwrap_or("").trim().parse().map_err(|_| format!("неверный тег GameType '{gt}'"))?;
        variant_from_game_type(n).ok_or_else(|| format!("GameType {n} не поддерживается"))?
    } else {
        return Err("в PDN нет тега Variant или GameType — не удалось определить вариант".into());
    };
    let forced = tags.get("CaptureRule").map(|s| s.trim() != "optional").unwrap_or(true);
    let cfg = GameConfig { variant, forced_capture: forced };
    let start_dfen = match tags.get("FEN") {
        Some(f) if !f.trim().is_empty() => {
            let f = f.trim();
            let d = if f.starts_with("w ") || f.starts_with("b ") { f.to_string() } else { fen_to_dfen(&cfg, f)? };
            validate_position(&cfg, &d)?;
            d
        }
        _ => start_position(&cfg)?,
    };
    let mut result = tags.get("Result").cloned().unwrap_or_else(|| "*".into());
    let mut cur = start_dfen.clone();
    let (mut positions, mut moves, mut infos) = (vec![cur.clone()], vec![], vec![]);
    let clean = strip_movetext(&body);
    for raw in clean.split_whitespace() {
        let mut tok = raw;
        if matches!(tok, "1-0" | "0-1" | "1/2-1/2" | "*" | "2-0" | "0-2" | "1-1" | "0-0") {
            result = match tok {
                "2-0" => "1-0",
                "0-2" => "0-1",
                "1-1" => "1/2-1/2",
                "0-0" => "*",
                t => t,
            }
            .to_string();
            break;
        }
        if tok.starts_with('$') {
            continue;
        }
        let digits = tok.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && tok[digits..].starts_with('.') {
            tok = tok[digits..].trim_start_matches('.');
            if tok.is_empty() {
                continue;
            }
        }
        let tok = tok.trim_end_matches(['!', '?']);
        if tok.is_empty() {
            continue;
        }
        let n = moves.len() + 1;
        let info = find_info(&cfg, &cur, tok).map_err(|e| format!("ход {n} ('{tok}'): {e}"))?;
        cur = apply_move(&cfg, &cur, &info.notation)?;
        moves.push(info.notation.clone());
        infos.push(info);
        positions.push(cur.clone());
    }
    tags.remove("Variant");
    Ok(PdnGameOut { cfg, start_dfen, positions, moves, move_infos: infos, tags, result })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facade::{game_status, Outcome};

    fn random_game(cfg: &GameConfig, start: &str, seed: u64, max: usize) -> (Vec<String>, bool, bool) {
        let mut x = seed | 1;
        let mut cur = start.to_string();
        let (mut hist, mut moves) = (vec![], vec![]);
        let (mut cap, mut multi) = (false, false);
        for _ in 0..max {
            if game_status(cfg, &cur, &hist).unwrap().outcome != Outcome::Ongoing {
                break;
            }
            let ms = legal_moves(cfg, &cur).unwrap();
            if ms.is_empty() {
                break;
            }
            let caps: Vec<&MoveInfo> = ms.iter().filter(|m| !m.captured.is_empty()).collect();
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let m = if !caps.is_empty() && x % 4 != 0 { caps[(x as usize >> 3) % caps.len()] } else { &ms[(x as usize >> 3) % ms.len()] };
            cap |= !m.captured.is_empty();
            multi |= m.captured.len() > 1;
            moves.push(m.notation.clone());
            let nxt = apply_move(cfg, &cur, &m.notation).unwrap();
            hist.push(std::mem::replace(&mut cur, nxt));
        }
        (moves, cap, multi)
    }

    const ALL: [VariantId; 9] = [
        VariantId::Russian,
        VariantId::Brazilian,
        VariantId::International,
        VariantId::American,
        VariantId::Italian,
        VariantId::Spanish,
        VariantId::Czech,
        VariantId::Thai,
        VariantId::Turkish,
    ];

    #[test]
    fn pdn_round_trip_all_variants() {
        for v in ALL {
            for forced in [true, false] {
                let cfg = GameConfig { variant: v, forced_capture: forced };
                let start = start_position(&cfg).unwrap();
                let mut any_cap = false;
                for seed in 1..=6u64 {
                    let (moves, cap, _) = random_game(&cfg, &start, seed * 7919, 120);
                    any_cap |= cap;
                    let mut tags = BTreeMap::new();
                    tags.insert("White".to_string(), "Игрок \"А\"".to_string());
                    tags.insert("Event".to_string(), "Тест".to_string());
                    let g = PdnGame { cfg: Some(cfg), start_dfen: None, moves: moves.clone(), tags, result: "1/2-1/2".into() };
                    let text = game_to_pdn(&g).unwrap();
                    let back = game_from_pdn(&text).unwrap_or_else(|e| panic!("{v:?} forced={forced}: {e}\n{text}"));
                    assert_eq!(back.cfg, cfg, "{text}");
                    assert_eq!(back.start_dfen, start);
                    assert_eq!(back.moves.len(), moves.len(), "{v:?}\n{text}");
                    assert_eq!(back.moves, moves, "{v:?}\n{text}");
                    assert_eq!(back.positions.len(), moves.len() + 1);
                    assert_eq!(back.result, "1/2-1/2");
                    assert_eq!(back.tags.get("Event").map(String::as_str), Some("Тест"));
                    let mut cur = start.clone();
                    for (i, m) in moves.iter().enumerate() {
                        cur = apply_move(&cfg, &cur, m).unwrap();
                        assert_eq!(back.positions[i + 1], cur);
                    }
                    let g2 = PdnGame { cfg: Some(back.cfg), start_dfen: None, moves: back.moves.clone(), tags: back.tags.clone(), result: back.result.clone() };
                    assert_eq!(game_to_pdn(&g2).unwrap(), text);
                }
                assert!(any_cap, "{v:?} forced={forced}: в тестовых партиях не было взятий");
            }
        }
    }

    #[test]
    fn pdn_multi_captures_survive() {
        let cfg = GameConfig { variant: VariantId::Russian, forced_capture: true };
        let fen = "W:Wa1,c3:Bb2,d4,f6";
        let start = fen_to_dfen(&cfg, fen).unwrap();
        let ms = legal_moves(&cfg, &start).unwrap();
        let m = ms.iter().find(|m| m.captured.len() >= 2).expect("серия взятий");
        let g = PdnGame { cfg: Some(cfg), start_dfen: Some(start.clone()), moves: vec![m.notation.clone()], ..Default::default() };
        let text = game_to_pdn(&g).unwrap();
        assert!(text.contains("[FEN \""), "{text}");
        let back = game_from_pdn(&text).unwrap();
        assert_eq!(back.start_dfen, start);
        assert_eq!(back.move_infos[0].captured, m.captured);
        assert_eq!(back.positions[1], apply_move(&cfg, &start, &m.notation).unwrap());
    }

    #[test]
    fn pdn_multi_captures_everywhere() {
        let mut missing = vec![];
        for v in ALL {
            let cfg = GameConfig { variant: v, forced_capture: true };
            let start = start_position(&cfg).unwrap();
            if !(1..=40u64).any(|s| random_game(&cfg, &start, s * 104729, 150).2) {
                missing.push(v);
            }
        }
        assert!(missing.len() <= 3, "нет серий взятий у {missing:?}");
    }

    #[test]
    fn pdn_fen_forms() {
        let cfg = GameConfig { variant: VariantId::International, forced_capture: true };
        let start = start_position(&cfg).unwrap();
        assert_eq!(fen_to_dfen(&cfg, "W:W31-50:B1-20").unwrap(), start);
        assert_eq!(dfen_to_fen(&cfg, &start).unwrap(), "W:W31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50:B1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20");
        let a = fen_to_dfen(&cfg, "B:W6,9,10,K31:BK17,K24").unwrap();
        let b = fen_to_dfen(&cfg, "B:W6,9,10,K31:B,K17,K24").unwrap();
        assert_eq!(a, b);
        assert!(a.starts_with("b "));
        assert_eq!(fen_to_dfen(&cfg, &dfen_to_fen(&cfg, &a).unwrap()).unwrap(), a);
        for v in ALL {
            let cfg = GameConfig { variant: v, forced_capture: true };
            let s = start_position(&cfg).unwrap();
            let f = dfen_to_fen(&cfg, &s).unwrap();
            assert_eq!(fen_to_dfen(&cfg, &f).unwrap(), s, "{v:?} {f}");
        }
        assert!(fen_to_dfen(&cfg, "X:W1:B2").is_err());
    }

    #[test]
    fn pdn_validate_position() {
        let cfg = GameConfig { variant: VariantId::Russian, forced_capture: true };
        let ok = fen_to_dfen(&cfg, "W:Wa1,Kc3:Bb6,Kd8").unwrap();
        assert!(validate_position(&cfg, &ok).is_ok());
        let bad = fen_to_dfen(&cfg, "W:Wa1,b8:Bb6").unwrap();
        let e = validate_position(&cfg, &bad).unwrap_err();
        assert!(e.contains("последнем ряду"), "{e}");
        let none = fen_to_dfen(&cfg, "W:Wa1:B").unwrap();
        assert!(validate_position(&cfg, &none).unwrap_err().contains("нет ни одной"));
        assert!(validate_position(&cfg, "w abc").is_err());
        assert!(fen_to_dfen(&cfg, "W:Wa2:Bb6").is_err());
    }

    #[test]
    fn pdn_game_type_tags() {
        let cfg = GameConfig { variant: VariantId::Thai, forced_capture: true };
        let g = PdnGame { cfg: Some(cfg), ..Default::default() };
        let t = game_to_pdn(&g).unwrap();
        assert!(t.contains("[GameType \"31,B,8,8,A0,0\"]"), "{t}");
        assert!(t.contains("[CaptureRule \"forced\"]"));
        let t2 = "[GameType \"25,W,8,8,A0,0\"]\n\n1. c3-d4 b6-a5 2. d4-c5 d6xb4 3. a3xc5 *\n";
        let r = game_from_pdn(t2).unwrap();
        assert_eq!(r.cfg.variant, VariantId::Russian);
        assert_eq!(r.moves.len(), 5, "{:?}", r.moves);
        assert!(game_from_pdn("[Event \"x\"]\n\n1. a1-b2 *").is_err());
    }
}
