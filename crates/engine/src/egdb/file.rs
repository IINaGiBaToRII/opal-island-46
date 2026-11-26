
use super::index::Sig;
use crate::datadir::{self, HeaderCheck};
use rayon::prelude::*;
use std::io::Write;
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"DEGD";
pub const FORMAT: u16 = 1;
pub const BLOCK_SHIFT: u32 = 16;
pub const BLOCK: usize = 1 << BLOCK_SHIFT;
pub const HEADER_LEN: usize = 64;
pub const ZSTD_LEVEL: i32 = 6;

#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub sig: Sig,
    pub forced: bool,
    pub nsq: u8,
    pub n: u64,
    pub nblocks: u64,
    pub index_off: u64,
}

pub fn encode_header(h: &Header) -> [u8; HEADER_LEN] {
    let mut b = [0u8; HEADER_LEN];
    b[0..12].copy_from_slice(&datadir::std_header(MAGIC, FORMAT));
    b[12..16].copy_from_slice(&h.sig);
    b[16] = h.forced as u8;
    b[17] = h.nsq;
    b[18] = BLOCK_SHIFT as u8;
    b[24..32].copy_from_slice(&h.n.to_le_bytes());
    b[32..40].copy_from_slice(&h.nblocks.to_le_bytes());
    b[40..48].copy_from_slice(&h.index_off.to_le_bytes());
    b
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

pub fn decode_header(b: &[u8]) -> Result<Header, HeaderCheck> {
    match datadir::check_header(b, MAGIC, FORMAT) {
        HeaderCheck::Ok => {}
        e => return Err(e),
    }
    if b.len() < HEADER_LEN || b[18] as u32 != BLOCK_SHIFT {
        return Err(HeaderCheck::Stale);
    }
    Ok(Header {
        sig: [b[12], b[13], b[14], b[15]],
        forced: b[16] != 0,
        nsq: b[17],
        n: u64_at(b, 24),
        nblocks: u64_at(b, 32),
        index_off: u64_at(b, 40),
    })
}

pub struct Packed {
    pub header: Header,
    pub blocks: Vec<Vec<u8>>,
}

impl Packed {
    pub fn total_bytes(&self) -> u64 {
        HEADER_LEN as u64 + self.blocks.iter().map(|b| b.len() as u64).sum::<u64>() + 8 * (self.blocks.len() as u64 + 1)
    }
}

pub fn pack(sig: Sig, forced: bool, nsq: usize, n: u64, vals: &[u8]) -> Result<Packed, String> {
    let blocks: Result<Vec<Vec<u8>>, String> = vals
        .par_chunks(BLOCK)
        .map(|ch| zstd::bulk::compress(ch, ZSTD_LEVEL).map_err(|e| e.to_string()))
        .collect();
    let blocks = blocks?;
    let body: u64 = blocks.iter().map(|b| b.len() as u64).sum();
    let header = Header { sig, forced, nsq: nsq as u8, n, nblocks: blocks.len() as u64, index_off: HEADER_LEN as u64 + body };
    Ok(Packed { header, blocks })
}

pub fn write_atomic(path: &Path, p: &Packed) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    {
        let f = std::fs::File::create(&tmp)?;
        let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
        w.write_all(&encode_header(&p.header))?;
        let mut off = HEADER_LEN as u64;
        let mut offs = Vec::with_capacity(p.blocks.len() + 1);
        for b in &p.blocks {
            offs.push(off);
            w.write_all(b)?;
            off += b.len() as u64;
        }
        offs.push(off);
        for o in offs {
            w.write_all(&o.to_le_bytes())?;
        }
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

pub fn validate_len(h: &Header, file_len: u64) -> bool {
    h.index_off >= HEADER_LEN as u64 && h.index_off + 8 * (h.nblocks + 1) == file_len && h.nblocks == (2 * h.n + BLOCK as u64 - 1) / BLOCK as u64
}

pub fn unpack_block(data: &[u8], h: &Header, b: u64) -> Result<Vec<u8>, String> {
    let io = h.index_off as usize + 8 * b as usize;
    let a = u64_at(data, io) as usize;
    let z = u64_at(data, io + 8) as usize;
    if z > data.len() || a > z {
        return Err("повреждённая таблица блоков".into());
    }
    let total = 2 * h.n as usize;
    let want = (total - b as usize * BLOCK).min(BLOCK);
    zstd::bulk::decompress(&data[a..z], want).map_err(|e| e.to_string())
}

pub fn read_all(path: &Path, forced: bool, sig: Sig) -> Result<(Header, Vec<u8>), String> {
    let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let h = decode_header(&data).map_err(|e| format!("{}: заголовок {:?}", path.display(), e))?;
    if h.sig != sig || h.forced != forced || !validate_len(&h, data.len() as u64) {
        return Err(format!("{}: несовпадение заголовка", path.display()));
    }
    let parts: Result<Vec<Vec<u8>>, String> = (0..h.nblocks).into_par_iter().map(|b| unpack_block(&data, &h, b)).collect();
    let mut out = Vec::with_capacity(2 * h.n as usize);
    for p in parts? {
        out.extend_from_slice(&p);
    }
    Ok((h, out))
}
