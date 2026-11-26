
use super::globals::{self, Global};
use super::view::geometry_nsq;
use serde::{Deserialize, Serialize};

pub const PHASE_ONE: i32 = 64;
pub const PHASE_SHIFT: u32 = 6;
pub const ALPHABETS: [u8; 2] = [3, 5];
pub const MAX_TABLE: usize = 4_000_000;
pub const MAX_PATTERNS: usize = 400;
pub const MAX_PATTERN_LEN: usize = 16;
pub const MAX_WEIGHTS: usize = 1 << 27;
pub const MAX_DEEP_DIM: usize = 4096;
pub const EVAL_LIMIT: i32 = 20000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewKind {
    Stm,
    Opp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseKind {
    None,
    Interp,
    Buckets,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseBy {
    Pieces,
    Tempo,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Phase {
    pub kind: PhaseKind,
    pub k: usize,
    pub by: PhaseBy,
    pub norm: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pattern {
    pub name: String,
    pub squares: Vec<u8>,
    pub alphabet: u8,
    pub table: usize,
    pub spec: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Table {
    pub size: usize,
    pub alphabet: u8,
    pub len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Deep {
    pub width: usize,
    pub layers: Vec<usize>,
    pub act: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Quant {
    pub wide: i32,
    pub qa: i32,
    pub qb: i32,
    pub out_div: i32,
}

impl Default for Quant {
    fn default() -> Self {
        Quant { wide: 16, qa: 255, qb: 64, out_div: 64 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Arch {
    pub name: String,
    pub geometry: String,
    pub views: Vec<ViewKind>,
    pub phase: Phase,
    pub patterns: Vec<Pattern>,
    pub tables: Vec<Table>,
    pub globals: Vec<String>,
    pub wide: bool,
    pub deep: Option<Deep>,
    pub quant: Quant,
    pub sigmoid_k: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    I16,
    I8,
    I32,
}

impl Dtype {
    pub fn name(self) -> &'static str {
        match self {
            Dtype::I16 => "i16",
            Dtype::I8 => "i8",
            Dtype::I32 => "i32",
        }
    }
    pub fn size(self) -> usize {
        match self {
            Dtype::I16 => 2,
            Dtype::I8 => 1,
            Dtype::I32 => 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockSpec {
    pub name: String,
    pub dtype: Dtype,
    pub len: usize,
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub nsq: usize,
    pub stages: usize,
    pub nviews: usize,
    pub wide_off: Vec<usize>,
    pub wide_len: usize,
    pub globals: Vec<Global>,
    pub g_off: Vec<usize>,
    pub g_row: usize,
    pub v_off: Vec<usize>,
    pub v_len: usize,
    pub dims: Vec<(usize, usize)>,
    pub blocks: Vec<BlockSpec>,
}

impl Layout {
    pub fn total_elems(&self) -> usize {
        self.blocks.iter().map(|b| b.len).sum()
    }
    pub fn total_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.len * b.dtype.size()).sum()
    }
}

pub fn is_pow2(x: i32) -> bool {
    x > 0 && (x & (x - 1)) == 0
}

impl Arch {
    pub fn nsq(&self) -> Option<usize> {
        geometry_nsq(&self.geometry)
    }

    pub fn stages(&self) -> usize {
        match self.phase.kind {
            PhaseKind::None => 1,
            _ => self.phase.k,
        }
    }

    pub fn global_list(&self) -> Result<Vec<Global>, String> {
        let mut v = vec![];
        for n in &self.globals {
            let g = Global::parse(n).ok_or_else(|| format!("неизвестный глобальный признак '{n}' (есть: {})", globals::names().join(", ")))?;
            if v.contains(&g) {
                return Err(format!("глобальный признак '{n}' повторён"));
            }
            v.push(g);
        }
        Ok(v)
    }

    pub fn validate(&self) -> Result<(), String> {
        let nsq = self.nsq().ok_or_else(|| format!("неизвестная геометрия '{}'", self.geometry))?;
        if self.views.is_empty() || self.views.len() > 2 || self.views[0] != ViewKind::Stm || (self.views.len() == 2 && self.views[1] != ViewKind::Opp) {
            return Err("views: допустимо [stm] или [stm, opp]".into());
        }
        if self.views.len() == 2 && self.deep.is_none() {
            return Err("вид opp нужен только глубокой части: задайте deep".into());
        }
        match self.phase.kind {
            PhaseKind::None => {
                if self.phase.k != 1 {
                    return Err("phase none: k должно быть 1".into());
                }
            }
            PhaseKind::Interp => {
                if self.phase.k < 2 || self.phase.k > 16 {
                    return Err("phase interp: k в 2..=16".into());
                }
            }
            PhaseKind::Buckets => {
                if self.phase.k < 1 || self.phase.k > 64 {
                    return Err("phase buckets: k в 1..=64".into());
                }
            }
        }
        if self.phase.norm == 0 || self.phase.norm > 1_000_000 {
            return Err("phase.norm вне 1..=1000000".into());
        }
        if self.patterns.len() > MAX_PATTERNS {
            return Err(format!("шаблонов {} > {MAX_PATTERNS}", self.patterns.len()));
        }
        for (i, t) in self.tables.iter().enumerate() {
            if !ALPHABETS.contains(&t.alphabet) {
                return Err(format!("таблица {i}: алфавит {} (допустимо 3 или 5)", t.alphabet));
            }
            if t.len == 0 || t.len > MAX_PATTERN_LEN {
                return Err(format!("таблица {i}: длина {} вне 1..={MAX_PATTERN_LEN}", t.len));
            }
            let want = (t.alphabet as usize).checked_pow(t.len as u32).ok_or("переполнение размера таблицы")?;
            if t.size != want {
                return Err(format!("таблица {i}: size {} != {}^{} = {want}", t.size, t.alphabet, t.len));
            }
            if t.size > MAX_TABLE {
                return Err(format!("таблица {i}: размер {} > {MAX_TABLE}", t.size));
            }
        }
        for (i, p) in self.patterns.iter().enumerate() {
            if p.squares.is_empty() || p.squares.len() > MAX_PATTERN_LEN {
                return Err(format!("шаблон {i} ({}): длина {} вне 1..={MAX_PATTERN_LEN}", p.name, p.squares.len()));
            }
            let mut seen = 0u64;
            for &s in &p.squares {
                if s as usize >= nsq {
                    return Err(format!("шаблон {i} ({}): поле {s} вне 0..{nsq}", p.name));
                }
                if seen >> s & 1 != 0 {
                    return Err(format!("шаблон {i} ({}): поле {s} повторено", p.name));
                }
                seen |= 1u64 << s;
            }
            let t = self.tables.get(p.table).ok_or_else(|| format!("шаблон {i} ({}): таблицы {} нет", p.name, p.table))?;
            if t.alphabet != p.alphabet || t.len != p.squares.len() {
                return Err(format!(
                    "шаблон {i} ({}): не согласован с таблицей {} (алфавит {}/{}, длина {}/{})",
                    p.name, p.table, p.alphabet, t.alphabet, p.squares.len(), t.len
                ));
            }
        }
        self.global_list()?;
        if (self.wide || self.deep.is_some()) && self.patterns.is_empty() {
            return Err("wide/deep без шаблонов".into());
        }
        if !self.wide && self.deep.is_none() && self.globals.is_empty() {
            return Err("пустая архитектура: нет ни wide, ни deep, ни globals".into());
        }
        if let Some(d) = &self.deep {
            if d.width == 0 || d.width > MAX_DEEP_DIM {
                return Err("deep.width вне 1..=4096".into());
            }
            if d.layers.is_empty() || *d.layers.last().unwrap() != 1 || d.layers.iter().any(|&w| w == 0 || w > MAX_DEEP_DIM) {
                return Err("deep.layers: непустой список ширин 1..=4096, последняя = 1".into());
            }
            if d.act != "crelu" {
                return Err("deep.act: поддержан только \"crelu\"".into());
            }
        }
        let q = &self.quant;
        if !is_pow2(q.wide) || q.wide > 256 {
            return Err("quant.wide: степень двойки 1..=256".into());
        }
        if q.qa != 255 || q.qb != 64 || q.out_div != 64 {
            return Err("quant: qa=255, qb=64, out_div=64 (зафиксированы формат DWT1, см. docs 3.1)".into());
        }
        if !(self.sigmoid_k.is_finite() && self.sigmoid_k > 0.0) {
            return Err("sigmoid_k должен быть конечным и > 0".into());
        }
        let l = self.layout()?;
        if l.total_elems() > MAX_WEIGHTS {
            return Err(format!("слишком много весов: {} > {MAX_WEIGHTS}", l.total_elems()));
        }
        Ok(())
    }

    pub fn layout(&self) -> Result<Layout, String> {
        let nsq = self.nsq().ok_or_else(|| format!("неизвестная геометрия '{}'", self.geometry))?;
        let stages = self.stages();
        let nviews = self.views.len();
        let mut blocks = vec![];
        let ovf = || "переполнение размеров".to_string();
        let mut wide_off = vec![];
        let mut wide_len = 0usize;
        if self.wide {
            for t in &self.tables {
                wide_off.push(wide_len);
                wide_len = t.size.checked_mul(stages).and_then(|x| x.checked_add(wide_len)).ok_or_else(ovf)?;
            }
            blocks.push(BlockSpec { name: "wide".into(), dtype: Dtype::I16, len: wide_len });
        }
        let globals = self.global_list()?;
        let mut g_off = vec![];
        let mut g_row = 0usize;
        for g in &globals {
            g_off.push(g_row);
            g_row += g.nweights(nsq);
        }
        if !globals.is_empty() {
            blocks.push(BlockSpec { name: "globals".into(), dtype: Dtype::I16, len: g_row * stages });
        }
        let mut v_off = vec![];
        let mut v_len = 0usize;
        let mut dims = vec![];
        if let Some(d) = &self.deep {
            for t in &self.tables {
                v_off.push(v_len);
                v_len = t.size.checked_mul(d.width).and_then(|x| x.checked_add(v_len)).ok_or_else(ovf)?;
            }
            blocks.push(BlockSpec { name: "deep_acc_w".into(), dtype: Dtype::I16, len: v_len });
            blocks.push(BlockSpec { name: "deep_acc_b".into(), dtype: Dtype::I16, len: d.width });
            let mut inp = d.width * nviews;
            for (i, &out) in d.layers.iter().enumerate() {
                dims.push((inp, out));
                blocks.push(BlockSpec { name: format!("deep_l{i}_w"), dtype: Dtype::I8, len: inp.checked_mul(out).ok_or_else(ovf)? });
                blocks.push(BlockSpec { name: format!("deep_l{i}_b"), dtype: Dtype::I32, len: out });
                inp = out;
            }
        }
        Ok(Layout { nsq, stages, nviews, wide_off, wide_len, globals, g_off, g_row, v_off, v_len, dims, blocks })
    }

    pub fn summary(&self) -> Result<String, String> {
        let l = self.layout()?;
        let mut s = String::new();
        let mut lens = std::collections::BTreeMap::new();
        for p in &self.patterns {
            *lens.entry(p.squares.len()).or_insert(0usize) += 1;
        }
        s += &format!("архитектура {} ({}), виды {:?}\n", self.name, self.geometry, self.views);
        s += &format!(
            "  шаблонов {} (длины: {}), таблиц {}, алфавит {}\n",
            self.patterns.len(),
            lens.iter().map(|(k, v)| format!("{k}x{v}")).collect::<Vec<_>>().join(" "),
            self.tables.len(),
            {
                let mut al: Vec<u8> = self.patterns.iter().map(|p| p.alphabet).collect();
                al.sort();
                al.dedup();
                al.iter().map(|a| a.to_string()).collect::<Vec<_>>().join("+")
            }
        );
        s += &format!("  стадии {:?} k={} by {:?} norm {}\n", self.phase.kind, self.phase.k, self.phase.by, self.phase.norm);
        s += &format!("  wide {}, globals [{}] ({} значений на стадию)\n", self.wide, self.globals.join(","), l.g_row);
        if let Some(d) = &self.deep {
            s += &format!("  deep {} -> {:?} ({})\n", d.width * l.nviews, d.layers, d.act);
        }
        s += &format!("  весов {}, байт {} ({:.2} МиБ)\n", l.total_elems(), l.total_bytes(), l.total_bytes() as f64 / 1048576.0);
        Ok(s)
    }
}
