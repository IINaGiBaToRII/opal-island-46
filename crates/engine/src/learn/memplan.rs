
use super::arch::Arch;
use crate::selfplay::format::ChunkHeader;
use std::path::PathBuf;

pub const GIB: u64 = 1 << 30;
pub const DEFAULT_CAP: u64 = 8 * GIB;
pub const PHYS_SHARE: f64 = 0.40;
pub const FALLBACK_LIMIT: u64 = 4 * GIB;
pub const BODY_BYTES_PER_SAMPLE: u64 = 5;
pub const MEM_LIMIT_ERR: &str = "набор данных не помещается в предел памяти обучения";

pub fn physical_memory() -> Option<u64> {
    static M: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        let mac = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<u64>().ok())
            .filter(|&n| n > 0);
        mac.or_else(|| {
            let s = std::fs::read_to_string("/proc/meminfo").ok()?;
            let line = s.lines().find(|l| l.starts_with("MemTotal:"))?;
            let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
            Some(kb * 1024)
        })
    })
}

pub fn default_limit() -> u64 {
    match physical_memory() {
        Some(m) => ((m as f64 * PHYS_SHARE) as u64).min(DEFAULT_CAP),
        None => FALLBACK_LIMIT,
    }
}

pub fn limit_of(explicit: Option<u64>) -> u64 {
    explicit.filter(|&b| b > 0).unwrap_or_else(default_limit)
}

pub fn sample_bytes(arch: &Arch) -> u64 {
    let np = (arch.patterns.len() * arch.views.len()) as u64;
    let g = arch.layout().map(|l| l.g_row as u64).unwrap_or(32);
    let heads = arch.deep.as_ref().map_or(false, |d| d.heads() > 1) as u64;
    let row = np * 4 + 2 + heads + 4 + 4 * g + 16 + 4;
    let tmp = 8 + 4 + 1;
    let map = 16;
    row + tmp + map
}

pub const REF_SAMPLES: f64 = 19.09e6;
pub const REF_UNIQUE: f64 = 7.55e6;
pub const REF_EXP: f64 = 0.604;
pub const SPLIT: f64 = 1.13;
pub const MARGIN: f64 = 1.10;

pub fn unique_of(samples: u64) -> f64 {
    REF_UNIQUE * super::detmath::pow_f64(samples as f64 / REF_SAMPLES, REF_EXP)
}

pub fn model_bytes(samples: u64, sample_bytes: u64) -> f64 {
    MARGIN * SPLIT * sample_bytes as f64 * unique_of(samples)
}

pub fn samples_at(limit: u64, sample_bytes: u64) -> f64 {
    let unique = limit.max(1) as f64 / (MARGIN * SPLIT * sample_bytes.max(1) as f64);
    REF_SAMPLES * super::detmath::pow_f64(unique / REF_UNIQUE, 1.0 / REF_EXP)
}

pub fn per_sample_bytes(limit: u64, sample_bytes: u64) -> f64 {
    limit.max(1) as f64 / samples_at(limit, sample_bytes).max(1.0)
}

pub fn est_bytes(samples: u64, per: f64) -> u64 {
    (samples as f64 * per).ceil() as u64
}

pub fn samples_upper(h: &ChunkHeader) -> u64 {
    h.body_len / BODY_BYTES_PER_SAMPLE
}

#[derive(Clone, Debug)]
pub struct ChunkInfo {
    pub idx: u32,
    pub path: PathBuf,
    pub games: u64,
    pub samples: u64,
    pub est: u64,
    pub imported: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Plan {
    pub files: Vec<PathBuf>,
    pub chunks_total: usize,
    pub games_total: u64,
    pub games_used: u64,
    pub samples_total: u64,
    pub est_total: u64,
    pub est_used: u64,
    pub limit: u64,
    pub limit_nominal: u64,
    pub sampled: bool,
}

impl Plan {
    pub fn note(&self) -> Option<String> {
        self.sampled.then(|| format!("обучение на выборке {} из {} партий: предел памяти {:.1} ГиБ", self.games_used, self.games_total, self.limit_nominal as f64 / GIB as f64))
    }
    pub fn line(&self) -> String {
        format!(
            "набор: {} кусков, {} партий (позиций не больше {}), оценка в памяти {:.2} ГиБ; предел памяти обучения {:.2} ГиБ — {}",
            self.chunks_total,
            self.games_total,
            self.samples_total,
            self.est_total as f64 / GIB as f64,
            self.limit_nominal as f64 / GIB as f64,
            match self.note() {
                Some(n) => format!("{n} (оценка выборки {:.2} ГиБ)", self.est_used as f64 / GIB as f64),
                None => "обучение на всём наборе".into(),
            }
        )
    }
}

fn mix(x: u64) -> u64 {
    super::dataset::splitmix64(x)
}

pub fn plan(chunks: &[ChunkInfo], gen_starts: &[u32], limit: u64, limit_nominal: u64, seed: u64) -> Plan {
    let mut p = Plan {
        chunks_total: chunks.len(),
        games_total: chunks.iter().map(|c| c.games).sum(),
        samples_total: chunks.iter().map(|c| c.samples).sum(),
        est_total: chunks.iter().map(|c| c.est).sum(),
        limit,
        limit_nominal,
        ..Default::default()
    };
    let mut chosen: Vec<&ChunkInfo> = vec![];
    if p.est_total <= limit {
        chosen = chunks.iter().collect();
    } else {
        p.sampled = true;
        let gen_of = |c: &ChunkInfo| if c.imported { 0 } else { gen_starts.iter().filter(|&&s| s <= c.idx).count() };
        let ngen = gen_starts.len();
        let half = limit / 2;
        let mut used = 0u64;
        let mut taken = std::collections::HashSet::new();
        'gens: for g in (1..=ngen).rev() {
            let mut members: Vec<&ChunkInfo> = chunks.iter().filter(|c| gen_of(c) == g).collect();
            members.sort_by_key(|c| std::cmp::Reverse(c.idx));
            let sum: u64 = members.iter().map(|c| c.est).sum();
            if used + sum <= half {
                used += sum;
                for c in members {
                    taken.insert(c.idx);
                    chosen.push(c);
                }
                continue;
            }
            for c in members {
                if used + c.est > half {
                    break 'gens;
                }
                used += c.est;
                taken.insert(c.idx);
                chosen.push(c);
            }
            break;
        }
        let mut rest: Vec<&ChunkInfo> = chunks.iter().filter(|c| !taken.contains(&c.idx)).collect();
        rest.sort_by_key(|c| (mix(seed ^ mix(c.idx as u64 ^ 0x1D2B_3C4D)), c.idx));
        for c in rest {
            if used + c.est <= limit {
                used += c.est;
                chosen.push(c);
            }
        }
    }
    if chosen.is_empty() {
        if let Some(c) = chunks.iter().filter(|c| !c.imported).max_by_key(|c| c.idx).or_else(|| chunks.iter().max_by_key(|c| c.idx)) {
            chosen.push(c);
        }
    }
    let mut files: Vec<(u32, PathBuf, u64, u64)> = chosen.iter().map(|c| (c.idx, c.path.clone(), c.games, c.est)).collect();
    files.sort();
    p.games_used = files.iter().map(|f| f.2).sum();
    p.est_used = files.iter().map(|f| f.3).sum();
    p.files = files.into_iter().map(|f| f.1).collect();
    p
}
