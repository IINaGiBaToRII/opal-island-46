
use super::arch::*;
use super::view::geometry_name;
use crate::datadir::{write_atomic, RULES_VERSION};
use crate::rules::Rules;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"DWT1";
pub const FORMAT: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Meta {
    pub variant: String,
    pub forced: bool,
    pub rules_version: u32,
}

impl Meta {
    pub fn of<R: Rules>() -> Meta {
        Meta { variant: R::NAME.to_string(), forced: R::FORCED, rules_version: RULES_VERSION }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockInfo {
    pub name: String,
    pub dtype: String,
    pub offset: u64,
    pub len: u64,
}

#[derive(Serialize, Deserialize)]
struct FileJson {
    format: u32,
    variant: String,
    forced: bool,
    rules_version: u32,
    #[serde(flatten)]
    arch: Arch,
    blocks: Vec<BlockInfo>,
    eval_id: u32,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub arch: Arch,
    pub layout: Layout,
    pub meta: Option<Meta>,
    pub wide: Vec<i16>,
    pub globals: Vec<i16>,
    pub deep_v: Vec<i16>,
    pub deep_b: Vec<i16>,
    pub deep_w: Vec<Vec<i8>>,
    pub deep_bias: Vec<Vec<i32>>,
}

pub enum Blk<'a> {
    I16(&'a [i16]),
    I8(&'a [i8]),
    I32(&'a [i32]),
}

struct Fnv(u64);
impl Fnv {
    fn new() -> Fnv {
        Fnv(0xCBF2_9CE4_8422_2325)
    }
    #[inline]
    fn byte(&mut self, b: u8) {
        self.0 = (self.0 ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
    }
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.byte(x);
        }
    }
    fn finish(&self) -> u32 {
        let v = ((self.0 >> 32) ^ self.0) as u32;
        if v == 0 {
            1
        } else {
            v
        }
    }
}

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn align64(x: usize) -> usize {
    (x + 63) & !63
}

impl Model {
    pub fn zeros(arch: Arch) -> Result<Model, String> {
        arch.validate()?;
        let layout = arch.layout()?;
        let (mut deep_w, mut deep_bias) = (vec![], vec![]);
        for &(i, o) in &layout.dims {
            deep_w.push(vec![0i8; i * o]);
            deep_bias.push(vec![0i32; o]);
        }
        let globals_len = if layout.globals.is_empty() { 0 } else { layout.g_row * layout.stages };
        Ok(Model {
            wide: vec![0; layout.wide_len],
            globals: vec![0; globals_len],
            deep_v: vec![0; layout.v_len],
            deep_b: vec![0; arch.deep.as_ref().map(|d| d.width).unwrap_or(0)],
            deep_w,
            deep_bias,
            arch,
            layout,
            meta: None,
        })
    }

    pub fn random(arch: Arch, seed: u64) -> Result<Model, String> {
        let q = arch.quant.wide;
        Self::random_with(arch, seed, 8 * q, 64, 32, 1 << 12)
    }

    pub fn random_extreme(arch: Arch, seed: u64) -> Result<Model, String> {
        Self::random_with(arch, seed, 32767, 32767, 127, 1 << 28)
    }

    fn random_with(arch: Arch, seed: u64, wide: i32, v: i32, w: i32, bias: i32) -> Result<Model, String> {
        let mut m = Model::zeros(arch)?;
        let mut s = seed ^ 0xA5A5_5A5A_1234_8765;
        let mut r = |lim: i32| -> i32 { (splitmix(&mut s) % (2 * lim as u64 + 1)) as i32 - lim };
        for x in m.wide.iter_mut() {
            *x = r(wide) as i16;
        }
        for x in m.globals.iter_mut() {
            *x = r(wide) as i16;
        }
        for x in m.deep_v.iter_mut() {
            *x = r(v) as i16;
        }
        for x in m.deep_b.iter_mut() {
            *x = r(v) as i16;
        }
        for l in m.deep_w.iter_mut() {
            for x in l.iter_mut() {
                *x = r(w) as i8;
            }
        }
        for l in m.deep_bias.iter_mut() {
            for x in l.iter_mut() {
                *x = r(bias);
            }
        }
        Ok(m)
    }

    pub fn blocks(&self) -> Vec<(String, Blk<'_>)> {
        let mut v = vec![];
        if self.arch.wide {
            v.push(("wide".to_string(), Blk::I16(&self.wide)));
        }
        if !self.layout.globals.is_empty() {
            v.push(("globals".to_string(), Blk::I16(&self.globals)));
        }
        if self.arch.deep.is_some() {
            v.push(("deep_acc_w".to_string(), Blk::I16(&self.deep_v)));
            v.push(("deep_acc_b".to_string(), Blk::I16(&self.deep_b)));
            for i in 0..self.deep_w.len() {
                v.push((format!("deep_l{i}_w"), Blk::I8(&self.deep_w[i])));
                v.push((format!("deep_l{i}_b"), Blk::I32(&self.deep_bias[i])));
            }
        }
        v
    }

    pub fn check_shapes(&self) -> Result<(), String> {
        let b = self.blocks();
        if b.len() != self.layout.blocks.len() {
            return Err("число блоков не совпадает с раскладкой".into());
        }
        for ((name, blk), spec) in b.iter().zip(&self.layout.blocks) {
            let len = match blk {
                Blk::I16(x) => x.len(),
                Blk::I8(x) => x.len(),
                Blk::I32(x) => x.len(),
            };
            if *name != spec.name || len != spec.len {
                return Err(format!("блок {name}: длина {len}, ожидалось {}", spec.len));
            }
        }
        Ok(())
    }

    pub fn eval_id(&self) -> u32 {
        let mut h = Fnv::new();
        let mut a = self.arch.clone();
        if a.sigmoid_k.is_finite() {
            if let Ok(c) = format!("{:.6}", a.sigmoid_k).parse::<f64>() {
                a.sigmoid_k = c;
            }
        }
        h.bytes(serde_json::to_string(&a).unwrap_or_default().as_bytes());
        for (name, b) in self.blocks() {
            h.bytes(name.as_bytes());
            match b {
                Blk::I16(x) => x.iter().for_each(|v| h.bytes(&v.to_le_bytes())),
                Blk::I8(x) => x.iter().for_each(|v| h.byte(*v as u8)),
                Blk::I32(x) => x.iter().for_each(|v| h.bytes(&v.to_le_bytes())),
            }
        }
        h.finish()
    }

    fn offsets(layout: &Layout) -> (Vec<usize>, usize) {
        let mut off = 0usize;
        let mut v = vec![];
        for b in &layout.blocks {
            off = align64(off);
            v.push(off);
            off += b.len * b.dtype.size();
        }
        (v, off)
    }

    pub fn to_bytes(&self, meta: &Meta) -> Result<Vec<u8>, String> {
        self.arch.validate()?;
        self.check_shapes()?;
        let (offs, total) = Self::offsets(&self.layout);
        let blocks = self
            .layout
            .blocks
            .iter()
            .zip(&offs)
            .map(|(b, &o)| BlockInfo { name: b.name.clone(), dtype: b.dtype.name().into(), offset: o as u64, len: b.len as u64 })
            .collect();
        let fj = FileJson {
            format: FORMAT,
            variant: meta.variant.clone(),
            forced: meta.forced,
            rules_version: meta.rules_version,
            arch: self.arch.clone(),
            blocks,
            eval_id: self.eval_id(),
        };
        let json = serde_json::to_vec(&fj).map_err(|e| e.to_string())?;
        let data_start = align64(8 + json.len());
        let mut out = Vec::with_capacity(data_start + total);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(&json);
        out.resize(data_start, 0);
        for ((_, b), &o) in self.blocks().iter().zip(&offs) {
            out.resize(data_start + o, 0);
            match b {
                Blk::I16(x) => x.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
                Blk::I8(x) => x.iter().for_each(|v| out.push(*v as u8)),
                Blk::I32(x) => x.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
            }
        }
        Ok(out)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Model, String> {
        if bytes.len() < 8 {
            return Err("файл короче заголовка".into());
        }
        if &bytes[0..4] != MAGIC {
            return Err("не файл весов (магия не DWT1)".into());
        }
        let jl = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let json_end = 8usize.checked_add(jl).filter(|&e| e <= bytes.len()).ok_or("длина JSON выходит за файл")?;
        let fj: FileJson = serde_json::from_slice(&bytes[8..json_end]).map_err(|e| format!("JSON описания: {e}"))?;
        if fj.format != FORMAT {
            return Err(format!("формат {} не поддержан (нужен {FORMAT})", fj.format));
        }
        fj.arch.validate().map_err(|e| format!("описание архитектуры: {e}"))?;
        let mut m = Model::zeros(fj.arch)?;
        let (offs, total) = Self::offsets(&m.layout);
        if fj.blocks.len() != m.layout.blocks.len() {
            return Err("список блоков не совпадает с описанием".into());
        }
        for ((bi, spec), &o) in fj.blocks.iter().zip(&m.layout.blocks).zip(&offs) {
            if bi.name != spec.name || bi.dtype != spec.dtype.name() || bi.len != spec.len as u64 || bi.offset != o as u64 {
                return Err(format!("блок '{}' не совпадает с раскладкой описания", bi.name));
            }
        }
        let data_start = align64(json_end);
        if bytes.len() != data_start + total {
            return Err(format!("размер файла {} != ожидаемого {} (усечён или с мусором)", bytes.len(), data_start + total));
        }
        let data = &bytes[data_start..];
        let blocks = m.layout.blocks.clone();
        let (mut i, mut ws, mut bs) = (0usize, 0usize, 0usize);
        for (bi, spec) in blocks.iter().enumerate() {
            let raw = &data[offs[bi]..offs[bi] + spec.len * spec.dtype.size()];
            let name = spec.name.as_str();
            let rd16 = |raw: &[u8]| -> Vec<i16> { raw.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect() };
            match name {
                "wide" => m.wide = rd16(raw),
                "globals" => m.globals = rd16(raw),
                "deep_acc_w" => m.deep_v = rd16(raw),
                "deep_acc_b" => m.deep_b = rd16(raw),
                _ => {
                    if spec.dtype == Dtype::I8 {
                        m.deep_w[ws] = raw.iter().map(|&b| b as i8).collect();
                        ws += 1;
                    } else {
                        m.deep_bias[bs] = raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                        bs += 1;
                    }
                }
            }
            i += 1;
        }
        let _ = i;
        m.meta = Some(Meta { variant: fj.variant, forced: fj.forced, rules_version: fj.rules_version });
        let id = m.eval_id();
        if id != fj.eval_id {
            return Err(format!("eval_id не сходится ({:#010x} в файле, {:#010x} по содержимому): файл повреждён", fj.eval_id, id));
        }
        Ok(m)
    }

    pub fn save<R: Rules>(&mut self, path: &Path) -> Result<(), String> {
        if geometry_name::<R::Geo>() != self.arch.geometry {
            return Err(format!("геометрия модели {} не подходит правилам {} ({})", self.arch.geometry, R::NAME, geometry_name::<R::Geo>()));
        }
        let meta = Meta::of::<R>();
        let bytes = self.to_bytes(&meta)?;
        write_atomic(path, &bytes).map_err(|e| format!("запись {}: {e}", path.display()))?;
        self.meta = Some(meta);
        Ok(())
    }

    pub fn load_any(path: &Path) -> Result<Model, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("чтение {}: {e}", path.display()))?;
        Model::from_bytes(&bytes)
    }

    pub fn load<R: Rules>(path: &Path) -> Result<Model, String> {
        let m = Model::load_any(path)?;
        m.check_rules::<R>()?;
        Ok(m)
    }

    pub fn check_rules<R: Rules>(&self) -> Result<(), String> {
        if let Some(meta) = &self.meta {
            let want = Meta::of::<R>();
            if meta.variant != want.variant {
                return Err(format!("веса для варианта '{}', а нужен '{}'", meta.variant, want.variant));
            }
            if meta.forced != want.forced {
                return Err(format!("веса для режима взятия {}, а нужен {}", if meta.forced { "обязательно" } else { "по желанию" }, if want.forced { "обязательно" } else { "по желанию" }));
            }
            if meta.rules_version != want.rules_version {
                return Err(format!("версия правил в файле {}, текущая {}", meta.rules_version, want.rules_version));
            }
        }
        let g = geometry_name::<R::Geo>();
        if self.arch.geometry != g {
            return Err(format!("геометрия весов '{}', а нужна '{g}'", self.arch.geometry));
        }
        Ok(())
    }
}
