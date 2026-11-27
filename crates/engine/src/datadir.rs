
use std::io::Read;
use std::path::{Path, PathBuf};

pub const RULES_VERSION: u32 = 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Cache,
    Book,
    Egdb,
    Weights,
    Train,
}

impl Kind {
    pub fn dir_name(self) -> &'static str {
        match self {
            Kind::Cache => "cache",
            Kind::Book => "book",
            Kind::Egdb => "egdb",
            Kind::Weights => "weights",
            Kind::Train => "train",
        }
    }
}

pub fn data_root() -> PathBuf {
    if let Some(v) = std::env::var_os("RUNNER_DATA") {
        if !v.is_empty() {
            return PathBuf::from(v);
        }
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join("Library").join("Application Support").join("Runner")
}

pub fn dir_in(root: &Path, variant: &str, forced: bool, kind: Kind) -> PathBuf {
    root.join(variant).join(if forced { "forced" } else { "free" }).join(kind.dir_name())
}

pub fn knowledge_dir(variant: &str, forced: bool, kind: Kind) -> PathBuf {
    dir_in(&data_root(), variant, forced, kind)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HeaderCheck {
    Ok,
    BadMagic,
    Stale,
}

pub fn std_header(magic: &[u8; 4], format: u16) -> [u8; 12] {
    let mut h = [0u8; 12];
    h[0..4].copy_from_slice(magic);
    h[4..6].copy_from_slice(&format.to_le_bytes());
    h[8..12].copy_from_slice(&RULES_VERSION.to_le_bytes());
    h
}

pub fn check_header(bytes: &[u8], magic: &[u8; 4], format: u16) -> HeaderCheck {
    if bytes.len() < 12 || &bytes[0..4] != magic {
        return HeaderCheck::BadMagic;
    }
    let f = u16::from_le_bytes([bytes[4], bytes[5]]);
    let rv = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if f != format || rv != RULES_VERSION {
        HeaderCheck::Stale
    } else {
        HeaderCheck::Ok
    }
}

pub fn read_prefix(path: &Path, n: usize) -> std::io::Result<Vec<u8>> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; n];
    let mut got = 0;
    while got < n {
        let k = f.read(&mut buf[got..])?;
        if k == 0 {
            break;
        }
        got += k;
    }
    buf.truncate(got);
    Ok(buf)
}

pub fn mark_stale(path: &Path) -> std::io::Result<PathBuf> {
    let base = path.as_os_str().to_os_string();
    let mut cand = base.clone();
    cand.push(".stale");
    let mut i = 0;
    let mut target = PathBuf::from(&cand);
    while target.exists() {
        i += 1;
        let mut c = base.clone();
        c.push(format!(".stale.{i}"));
        target = PathBuf::from(c);
    }
    std::fs::rename(path, &target)?;
    touch_now(&target);
    Ok(target)
}

fn touch_now(p: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        if let Ok(c) = std::ffi::CString::new(p.as_os_str().as_bytes()) {
            unsafe {
                libc::utimes(c.as_ptr(), std::ptr::null());
            }
        }
    }
    #[cfg(not(unix))]
    {
        if let Ok(f) = std::fs::File::options().write(true).open(p) {
            let _ = f.set_modified(std::time::SystemTime::now());
        }
    }
}

const fn crc_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}

static CRC_TABLE: [u32; 256] = crc_table();

pub fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c = CRC_TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn mtime_secs(path: &Path) -> u64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(format!(".{}.{}.tmp", std::process::id(), SEQ.fetch_add(1, Ordering::Relaxed)));
    let tmp = PathBuf::from(tmp);
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

pub fn read_json_or_stale<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    read_json_strict(path).ok().flatten()
}

pub fn read_json_strict<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Option<T>, String> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("чтение {}: {e}", path.display())),
    };
    match serde_json::from_slice::<T>(&bytes) {
        Ok(v) => Ok(Some(v)),
        Err(_) => stale_or_err(path).map(|_| None),
    }
}

pub fn stale_or_err(path: &Path) -> Result<PathBuf, String> {
    mark_stale(path).map_err(|e| format!("не удалось отложить {} в .stale: {e}", path.display()))
}

pub fn walk_files(dir: &Path, depth: usize, skip: &[&str], f: &mut dyn FnMut(&Path, &std::fs::Metadata)) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        let Ok(md) = e.metadata() else { continue };
        if md.is_dir() {
            let name = e.file_name().to_string_lossy().to_string();
            if depth > 0 && !skip.contains(&name.as_str()) {
                walk_files(&p, depth - 1, skip, f);
            }
        } else {
            f(&p, &md);
        }
    }
}

fn age_secs(md: &std::fs::Metadata) -> u64 {
    md.modified().ok().and_then(|t| t.elapsed().ok()).map(|d| d.as_secs()).unwrap_or(0)
}

fn strip_tmp_ids(name: &str) -> String {
    let Some(mut base) = name.strip_suffix(".tmp") else { return name.to_string() };
    for _ in 0..2 {
        match base.rsplit_once('.') {
            Some((b, d)) if !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()) => base = b,
            _ => break,
        }
    }
    format!("{base}.tmp")
}

pub fn cleanup_tmp(root: &Path, min_age: std::time::Duration) -> usize {
    let mut n = 0;
    walk_files(root, 4, &["_archive", "library"], &mut |p, md| {
        let raw = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let name = strip_tmp_ids(raw);
        let name = name.as_str();
        let ours = name.ends_with(".egdb.tmp") || name == "manifest.json.tmp" || name.ends_with(".journal.tmp") || name.ends_with(".json.tmp")
            || name.ends_with(".dw.tmp") || name.ends_with(".dgz.tmp") || name.ends_with(".ckpt.tmp") || name.ends_with(".jsonl.tmp");
        if ours && age_secs(md) >= min_age.as_secs() && std::fs::remove_file(p).is_ok() {
            n += 1;
        }
    });
    n
}

pub fn purge_stale(root: &Path, max_age: std::time::Duration) -> usize {
    let mut n = 0;
    walk_files(root, 4, &["_archive", "library"], &mut |p, md| {
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let is_stale = name.ends_with(".stale") || name.rsplit_once(".stale.").is_some_and(|(_, t)| t.chars().all(|c| c.is_ascii_digit()) && !t.is_empty());
        if is_stale && age_secs(md) >= max_age.as_secs() && std::fs::remove_file(p).is_ok() {
            n += 1;
        }
    });
    n
}

pub fn cfg_name(variant: &str, forced: bool) -> String {
    format!("{}:{}", variant, if forced { "forced" } else { "free" })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_atomic_uses_unique_tmp_names_and_concurrent_writers_leave_valid_file() {
        let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("runner-atomic-{}-{}", std::process::id(), n));
        let p = dir.join("x.json");
        let hs: Vec<_> = (0..8)
            .map(|i| {
                let p = p.clone();
                std::thread::spawn(move || {
                    for j in 0..20 {
                        write_atomic(&p, format!("{{\"v\":{}}}", i * 100 + j).as_bytes()).unwrap();
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        assert!(v["v"].is_u64());
        assert!(std::fs::read_dir(&dir).unwrap().flatten().all(|e| !e.file_name().to_string_lossy().ends_with(".tmp")), "временных файлов не осталось");
    }

    #[test]
    fn tmp_names_with_pid_are_normalised_for_cleanup() {
        assert_eq!(strip_tmp_ids("play_profile.json.123.4.tmp"), "play_profile.json.tmp");
        assert_eq!(strip_tmp_ids("w.dw.7.0.tmp"), "w.dw.tmp");
        assert_eq!(strip_tmp_ids("manifest.json.tmp"), "manifest.json.tmp");
        assert_eq!(strip_tmp_ids("a.b"), "a.b");
    }
}
