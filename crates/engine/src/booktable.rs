
use crate::facade::GameConfig;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MoveEntry {
    pub mv: String,
    pub value: i32,
    pub reserve: u64,
    pub below: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Entry {
    pub moves: Vec<MoveEntry>,
    pub value: i32,
    pub own_nodes: u64,
    pub ply: u32,
    pub flags: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Accept {
    pub id: String,
    pub content_crc: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Table {
    pub v: u32,
    pub variant: String,
    pub forced: bool,
    pub eval_id: u32,
    pub opts: String,
    pub profile: u32,
    pub profiles: Vec<u32>,
    pub draw_rules: String,
    pub speed_1t: u64,
    pub k: u64,
    pub accept: Option<Accept>,
    pub entries: BTreeMap<String, Entry>,
    #[serde(skip)]
    pub loaded_crc: u32,
}

pub fn profile_label(opts: &str) -> u32 {
    if opts.is_empty() {
        return 0;
    }
    let mut parts: Vec<&str> = opts.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    parts.sort_unstable();
    parts.dedup();
    let canon = parts.join(",");
    let mut h: u32 = 0x811c_9dc5;
    for b in canon.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h.max(1)
}

impl Table {
    pub fn content_crc(&self) -> u32 {
        let mut s = String::new();
        for (p, e) in &self.entries {
            s.push_str(p);
            for m in &e.moves {
                s.push_str(&format!("|{}:{}:{}:{}", m.mv, m.value, m.reserve, m.below));
            }
            s.push_str(&format!("|{}|{}|{}|{}\n", e.value, e.own_nodes, e.ply, e.flags));
        }
        crate::datadir::crc32(s.as_bytes())
    }

    pub fn from_slice(bytes: &[u8]) -> Result<Table, String> {
        let mut t: Table = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if t.v != 2 {
            return Err(format!("версия таблицы {} (нужна 2)", t.v));
        }
        t.loaded_crc = t.content_crc();
        Ok(t)
    }

    pub fn load(path: &Path) -> Result<Table, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Table::from_slice(&b).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn fits(&self, cfg: &GameConfig, eval_id: u32, opts: &str) -> Result<(), String> {
        if self.variant != cfg.variant.name() || self.forced != cfg.forced_capture {
            return Err(format!("таблица для {}:{}, матч — {}:{}", self.variant, self.forced, cfg.variant.name(), cfg.forced_capture));
        }
        if self.eval_id != eval_id {
            return Err(format!("таблица сети {:08x}, у стороны {eval_id:08x}", self.eval_id));
        }
        let prof = profile_label(opts);
        if prof == 0 || !(self.profile == prof || self.profiles.contains(&prof)) {
            return Err(format!("профиль опций стороны {prof:08x} не из принятых таблицей (опции таблицы «{}»)", self.opts));
        }
        let dr = crate::booktask::draw_rules(opts);
        if self.draw_rules != dr {
            return Err(format!("правила ничьей стороны ({dr}) не те, что у таблицы ({})", self.draw_rules));
        }
        if self.k == 0 {
            return Err("в таблице нет кратности запаса k".into());
        }
        Ok(())
    }

    pub fn need_for_time(&self, time_ms: u64, threads: usize) -> u64 {
        (time_ms as u128 * threads.max(1) as u128 * self.speed_1t as u128 / 1000).min(u64::MAX as u128) as u64
    }

    pub fn pick(&self, dfen: &str, seed: u64, need: u64) -> Option<&MoveEntry> {
        let e = self.entries.get(dfen)?;
        if e.flags != 0 || e.own_nodes < need {
            return None;
        }
        let ok: Vec<&MoveEntry> = e.moves.iter().filter(|m| m.reserve >= self.k.saturating_mul(need)).collect();
        if ok.is_empty() || ok[0].mv != e.moves[0].mv {
            return None;
        }
        let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
        for b in dfen.bytes() {
            h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
        h ^= h >> 29;
        ok.get((h % ok.len() as u64) as usize).copied()
    }

    pub fn has(&self, dfen: &str) -> bool {
        self.entries.contains_key(dfen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> Table {
        let mut t = Table { v: 2, variant: "russian".into(), forced: true, eval_id: 7, opts: "rep3=1,dbdraw=10".into(), draw_rules: crate::booktask::draw_rules("rep3=1,dbdraw=10"), speed_1t: 100, k: 8, ..Default::default() };
        t.profile = profile_label(&t.opts);
        t.profiles = vec![t.profile];
        let mv = |m: &str, r: u64| MoveEntry { mv: m.into(), value: 5, reserve: r, below: 4 };
        t.entries.insert("w x".into(), Entry { moves: vec![mv("a", 8000), mv("b", 8000)], value: 5, own_nodes: 1000, ply: 0, flags: 0 });
        t
    }

    #[test]
    fn reader_roundtrip_crc_fits_and_pick() {
        let t = table();
        let bytes = serde_json::to_vec(&serde_json::json!({"v": 2, "variant": "russian", "forced": true, "eval_id": 7, "opts": t.opts, "profile": t.profile, "profiles": t.profiles, "draw_rules": t.draw_rules, "speed_1t": 100, "k": 8, "params": {"max_ply": 3}, "future": 1, "entries": t.entries})).unwrap();
        let l = Table::from_slice(&bytes).unwrap();
        assert_eq!(l.loaded_crc, t.content_crc(), "незнакомые поля (params, future) не мешают, crc — по записям");
        let cfg = GameConfig { variant: crate::VariantId::parse("russian").unwrap(), forced_capture: true };
        assert_eq!(l.fits(&cfg, 7, "dbdraw=10, rep3=1"), Ok(()), "порядок опций метку не меняет");
        assert!(l.fits(&cfg, 8, "rep3=1,dbdraw=10").unwrap_err().contains("сети"));
        assert!(l.fits(&cfg, 7, "rep3=1,dbdraw=10,ext1=1").unwrap_err().contains("профиль"));
        assert!(l.fits(&cfg, 7, "").is_err());
        assert!(l.pick("w x", 1, 1000).is_some());
        assert!(l.pick("w x", 1, 1001).is_none(), "запас меньше k бюджетов — молчит");
        let picks: std::collections::BTreeSet<String> = (0..64).filter_map(|s| l.pick("w x", s, 10).map(|m| m.mv.clone())).collect();
        assert_eq!(picks.len(), 2, "разнообразие среди равных по зерну");
        let mut f = l.clone();
        f.entries.get_mut("w x").unwrap().flags = 4;
        assert!(f.pick("w x", 1, 1).is_none(), "незнакомый признак — молчит");
        assert_eq!(l.need_for_time(1000, 2), 200);
        assert!(Table::from_slice(br#"{"v":1}"#).is_err());
    }
}
