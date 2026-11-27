
use crate::datadir;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"DGM1";
pub const FORMAT: u32 = 1;
const MAX_HEADER: usize = 1 << 20;
const MAX_BODY: u64 = 2 << 30;
pub const ZSTD_LEVEL: i32 = 6;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum EndReason {
    NoMoves = 0,
    RuleDraw = 1,
    Repetition = 2,
    Egdb = 3,
    Adjudicated = 4,
    MaxPlies = 5,
}

impl EndReason {
    pub const ALL: [EndReason; 6] =
        [EndReason::NoMoves, EndReason::RuleDraw, EndReason::Repetition, EndReason::Egdb, EndReason::Adjudicated, EndReason::MaxPlies];
    pub fn from_u8(v: u8) -> Option<EndReason> {
        Self::ALL.get(v as usize).copied()
    }
    pub fn name(self) -> &'static str {
        match self {
            EndReason::NoMoves => "нет ходов",
            EndReason::RuleDraw => "правило ничьей",
            EndReason::Repetition => "повторение",
            EndReason::Egdb => "базы",
            EndReason::Adjudicated => "порог оценки",
            EndReason::MaxPlies => "предел длины",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChunkHeader {
    pub format: u32,
    pub variant: String,
    pub forced: bool,
    pub rules_version: u32,
    pub eval_id: u32,
    pub nodes: u64,
    pub seed: u64,
    pub chunk: u32,
    pub games: u32,
    pub created_unix: u64,
    pub book_min: u32,
    pub book_max: u32,
    pub adj_score: i32,
    pub adj_plies: u32,
    pub max_plies: u32,
    pub hash_mb: u32,
    pub egdb_max_pieces: u32,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub db_play_draws: bool,
    #[serde(skip_serializing_if = "is_zero_u32")]
    pub rnd_permille: u32,
    #[serde(skip_serializing_if = "is_zero_u32")]
    pub rnd_min: u32,
    #[serde(skip_serializing_if = "is_zero_u32")]
    pub rnd_max: u32,
    pub body_len: u64,
    pub body_crc: u32,
}

fn is_zero_u32(x: &u32) -> bool {
    *x == 0
}

impl Default for ChunkHeader {
    fn default() -> Self {
        ChunkHeader {
            format: FORMAT,
            variant: String::new(),
            forced: true,
            rules_version: datadir::RULES_VERSION,
            eval_id: 0,
            nodes: 0,
            seed: 0,
            chunk: 0,
            games: 0,
            created_unix: 0,
            book_min: 0,
            book_max: 0,
            adj_score: 0,
            adj_plies: 0,
            max_plies: 0,
            hash_mb: 0,
            egdb_max_pieces: 0,
            db_play_draws: false,
            rnd_permille: 0,
            rnd_min: 0,
            rnd_max: 0,
            body_len: 0,
            body_crc: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StartPos {
    pub white: u64,
    pub black: u64,
    pub kings: u64,
    pub side: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlyRec {
    pub from: u8,
    pub to: u8,
    pub idx: u8,
    pub score: i16,
    pub depth: u8,
    pub random: bool,
    pub in_db: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Game {
    pub start: Option<StartPos>,
    pub random_plies: u8,
    pub plies: Vec<PlyRec>,
    pub result: i8,
    pub reason: EndReason,
    pub egdb: Option<i8>,
}

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn left(&self) -> usize {
        self.b.len() - self.p
    }
    fn u8(&mut self) -> Result<u8, String> {
        let v = *self.b.get(self.p).ok_or("усечённые данные партии")?;
        self.p += 1;
        Ok(v)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.left() < n {
            return Err("усечённые данные партии".into());
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().map_err(|_| "u64")?))
    }
    fn varint(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        for i in 0..10 {
            let b = self.u8()?;
            v |= ((b & 0x7F) as u64) << (7 * i);
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err("слишком длинное число".into())
    }
}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8 & 0x7F) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

impl Game {
    pub fn encode(&self, out: &mut Vec<u8>) {
        let mut h = self.reason as u8 | (((self.result.clamp(-1, 1) + 1) as u8) << 3);
        if self.egdb.is_some() {
            h |= 1 << 5;
        }
        if self.start.is_some() {
            h |= 1 << 6;
        }
        out.push(h);
        if let Some(s) = &self.start {
            out.extend_from_slice(&s.white.to_le_bytes());
            out.extend_from_slice(&s.black.to_le_bytes());
            out.extend_from_slice(&s.kings.to_le_bytes());
            out.push(s.side);
        }
        if let Some(v) = self.egdb {
            out.push(v as u8);
        }
        out.push(self.random_plies);
        put_varint(out, self.plies.len() as u64);
        for p in &self.plies {
            out.push((p.from & 63) | (p.random as u8) << 6 | (p.in_db as u8) << 7);
            let code = p.idx.min(3);
            out.push((p.to & 63) | code << 6);
            if code == 3 {
                out.push(p.idx);
            }
            if !p.random {
                out.extend_from_slice(&p.score.to_le_bytes());
                out.push(p.depth);
            }
        }
    }

    fn decode(c: &mut Cur) -> Result<Game, String> {
        let h = c.u8()?;
        let reason = EndReason::from_u8(h & 7).ok_or("неизвестная причина окончания")?;
        let r = (h >> 3) & 3;
        if r > 2 {
            return Err("неверный исход".into());
        }
        let result = r as i8 - 1;
        let has_egdb = h & (1 << 5) != 0;
        let explicit = h & (1 << 6) != 0;
        if h & 0x80 != 0 {
            return Err("неизвестные флаги партии".into());
        }
        let start = if explicit {
            let (white, black, kings) = (c.u64()?, c.u64()?, c.u64()?);
            Some(StartPos { white, black, kings, side: c.u8()? })
        } else {
            None
        };
        let egdb = if has_egdb { Some(c.u8()? as i8) } else { None };
        let random_plies = c.u8()?;
        let n = c.varint()? as usize;
        if n > c.left() / 2 {
            return Err("число полуходов больше остатка данных".into());
        }
        let mut plies = Vec::with_capacity(n);
        for _ in 0..n {
            let (b0, b1) = (c.u8()?, c.u8()?);
            let code = b1 >> 6;
            let idx = if code == 3 { c.u8()? } else { code };
            let random = b0 & 64 != 0;
            let (score, depth) = if random {
                (0, 0)
            } else {
                let s = c.take(2)?;
                (i16::from_le_bytes([s[0], s[1]]), c.u8()?)
            };
            plies.push(PlyRec { from: b0 & 63, to: b1 & 63, idx, score, depth, random, in_db: b0 & 128 != 0 });
        }
        let lead = plies.iter().take_while(|p| p.random).count();
        if lead != random_plies as usize {
            return Err("случайные полуходы дебюта не образуют префикс заданной длины".into());
        }
        Ok(Game { start, random_plies, plies, result, reason, egdb })
    }
}

pub fn assemble_chunk(header: &mut ChunkHeader, body: &[u8]) -> Result<Vec<u8>, String> {
    header.body_len = body.len() as u64;
    header.body_crc = datadir::crc32(body);
    let json = serde_json::to_vec(header).map_err(|e| e.to_string())?;
    let z = zstd::bulk::compress(body, ZSTD_LEVEL).map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(8 + json.len() + z.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&json);
    out.extend_from_slice(&z);
    Ok(out)
}

pub fn write_chunk(path: &Path, header: &ChunkHeader, games: &[Game]) -> Result<(), String> {
    let mut body = Vec::new();
    for g in games {
        g.encode(&mut body);
    }
    let mut h = header.clone();
    h.games = games.len() as u32;
    let bytes = assemble_chunk(&mut h, &body)?;
    datadir::write_atomic(path, &bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn parse_header(bytes: &[u8]) -> Result<(ChunkHeader, usize), String> {
    if bytes.len() < 8 || &bytes[0..4] != MAGIC {
        return Err("не файл партий DGM1".into());
    }
    let n = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if n > MAX_HEADER || bytes.len() < 8 + n {
        return Err("усечённый заголовок".into());
    }
    let h: ChunkHeader = serde_json::from_slice(&bytes[8..8 + n]).map_err(|e| format!("заголовок: {e}"))?;
    if h.format != FORMAT {
        return Err(format!("неподдерживаемый формат {}", h.format));
    }
    if h.body_len > MAX_BODY {
        return Err("неправдоподобная длина тела".into());
    }
    Ok((h, 8 + n))
}

static HEADER_READS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn header_reads() -> u64 {
    HEADER_READS.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn read_header(path: &Path) -> Result<ChunkHeader, String> {
    HEADER_READS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let pre = datadir::read_prefix(path, 8).map_err(|e| format!("{}: {e}", path.display()))?;
    if pre.len() < 8 || &pre[0..4] != MAGIC {
        return Err("не файл партий DGM1".into());
    }
    let n = u32::from_le_bytes([pre[4], pre[5], pre[6], pre[7]]) as usize;
    if n > MAX_HEADER {
        return Err("неправдоподобная длина заголовка".into());
    }
    let b = datadir::read_prefix(path, 8 + n).map_err(|e| e.to_string())?;
    parse_header(&b).map(|x| x.0)
}

pub struct Chunk {
    pub header: ChunkHeader,
    pub file_bytes: u64,
    body: Vec<u8>,
}

impl Chunk {
    pub fn read(path: &Path) -> Result<Chunk, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Chunk, String> {
        let (header, off) = parse_header(bytes)?;
        let body = zstd::bulk::decompress(&bytes[off..], header.body_len as usize).map_err(|e| format!("zstd: {e}"))?;
        if body.len() as u64 != header.body_len {
            return Err("длина тела не совпала с заголовком".into());
        }
        if datadir::crc32(&body) != header.body_crc {
            return Err("CRC тела не совпал".into());
        }
        Ok(Chunk { header, file_bytes: bytes.len() as u64, body })
    }

    pub fn body(&self) -> &[u8] {
        &self.body
    }

    pub fn games(&self) -> GameIter<'_> {
        let h = &self.header;
        let rnd = (h.rnd_permille > 0).then_some((h.rnd_min, h.rnd_max));
        GameIter { c: Cur { b: &self.body, p: 0 }, left: self.header.games, failed: false, rnd }
    }
}

pub struct GameIter<'a> {
    c: Cur<'a>,
    left: u32,
    failed: bool,
    rnd: Option<(u32, u32)>,
}

fn check_mid_random(g: &Game, rnd: Option<(u32, u32)>) -> Result<(), String> {
    let mut mids = g.plies.iter().enumerate().skip(g.random_plies as usize).filter(|(_, p)| p.random).map(|(i, _)| i as u32);
    match (mids.next(), rnd) {
        (None, _) => Ok(()),
        (Some(_), None) => Err("случайные полуходы не образуют префикс заданной длины".into()),
        (Some(i), Some((a, b))) => {
            if mids.next().is_some() {
                return Err("больше одного случайного хода в середине партии".into());
            }
            if i < a || i > b {
                return Err(format!("случайный ход в середине партии на полуходе {i} вне [{a}, {b}] из заголовка"));
            }
            Ok(())
        }
    }
}

impl Iterator for GameIter<'_> {
    type Item = Result<Game, String>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        if self.left == 0 {
            if self.c.left() != 0 {
                self.failed = true;
                return Some(Err("лишние данные после последней партии".into()));
            }
            return None;
        }
        self.left -= 1;
        let r = Game::decode(&mut self.c).and_then(|g| check_mid_random(&g, self.rnd).map(|_| g));
        if r.is_err() {
            self.failed = true;
        }
        Some(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(n: usize, idx: u8) -> Game {
        let plies = (0..n)
            .map(|i| PlyRec { from: (i % 64) as u8, to: ((i * 7) % 64) as u8, idx: if i == 3 { idx } else { 0 }, score: if i < 2 { 0 } else { (i as i16 - 5) * 37 }, depth: if i < 2 { 0 } else { 9 }, random: i < 2, in_db: false })
            .collect();
        Game { start: None, random_plies: 2, plies, result: -1, reason: EndReason::Adjudicated, egdb: None }
    }

    #[test]
    fn mid_random_ply_roundtrips_and_prefix_still_checked() {
        let mut a = g(10, 0);
        a.plies[6].random = true;
        a.plies[6].score = 0;
        a.plies[6].depth = 0;
        let mut body = Vec::new();
        a.encode(&mut body);
        let mut h = ChunkHeader { games: 1, variant: "russian".into(), rnd_permille: 250, rnd_min: 5, rnd_max: 7, ..Default::default() };
        let file = assemble_chunk(&mut h, &body).unwrap();
        let ch = Chunk::from_bytes(&file).unwrap();
        assert_eq!((ch.header.rnd_permille, ch.header.rnd_min, ch.header.rnd_max), (250, 5, 7));
        assert_eq!(ch.games().map(|x| x.unwrap()).collect::<Vec<_>>(), vec![a.clone()]);
        let read = |g: &Game, h: ChunkHeader| {
            let mut body = Vec::new();
            g.encode(&mut body);
            let mut h = ChunkHeader { games: 1, variant: "russian".into(), ..h };
            let file = assemble_chunk(&mut h, &body).unwrap();
            Chunk::from_bytes(&file).unwrap().games().next().unwrap()
        };
        let e = read(&a, ChunkHeader::default()).unwrap_err();
        assert!(e.contains("не образуют префикс"), "{e}");
        assert!(read(&a, ChunkHeader { rnd_permille: 250, rnd_min: 7, rnd_max: 9, ..Default::default() }).unwrap_err().contains("вне [7, 9]"));
        let mut two = a.clone();
        two.plies[7].random = true;
        assert!(read(&two, ChunkHeader { rnd_permille: 250, rnd_min: 5, rnd_max: 9, ..Default::default() }).unwrap_err().contains("больше одного"));
        let mut b = a.clone();
        b.random_plies = 3;
        let mut body = Vec::new();
        b.encode(&mut body);
        let mut h = ChunkHeader { games: 1, variant: "russian".into(), ..Default::default() };
        let file = assemble_chunk(&mut h, &body).unwrap();
        assert!(Chunk::from_bytes(&file).unwrap().games().next().unwrap().is_err());
        let js = String::from_utf8(serde_json::to_vec(&ChunkHeader::default()).unwrap()).unwrap();
        assert!(!js.contains("rnd"), "{js}");
    }

    #[test]
    fn roundtrip_with_escape_idx() {
        let games = vec![
            g(10, 0),
            g(10, 2),
            g(10, 3),
            g(10, 200),
            Game { start: Some(StartPos { white: 5, black: 8, kings: 1, side: 1 }), egdb: Some(-3), reason: EndReason::Egdb, result: 0, ..g(4, 0) },
        ];
        let mut body = Vec::new();
        for x in &games {
            x.encode(&mut body);
        }
        let mut h = ChunkHeader { games: games.len() as u32, variant: "russian".into(), ..Default::default() };
        let file = assemble_chunk(&mut h, &body).unwrap();
        let ch = Chunk::from_bytes(&file).unwrap();
        let back: Vec<Game> = ch.games().map(|x| x.unwrap()).collect();
        assert_eq!(back, games);
        for cut in [0, 3, 9, file.len() / 2, file.len() - 1] {
            assert!(Chunk::from_bytes(&file[..cut]).is_err(), "cut {cut}");
        }
    }
}
