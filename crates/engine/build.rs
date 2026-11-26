use std::path::Path;
use std::process::Command;

const EXTRA_DIRS: &[&str] = &[];

fn git(args: &[&str]) -> Option<String> {
    let o = Command::new("git").arg("-C").arg("../..").args(args).output().ok().filter(|o| o.status.success())?;
    Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn main() {
    println!("cargo:rerun-if-env-changed=RUNNER_BUILD");
    let extra: Vec<String> = EXTRA_DIRS.iter().map(|d| format!("../../{d}")).collect();
    for p in ["../../BUILD_TAG", "../../.git/HEAD", "../../.git/logs/HEAD", "../../.git/index", "src"].into_iter().chain(extra.iter().map(|s| s.as_str())) {
        if Path::new(p).exists() {
            println!("cargo:rerun-if-changed={p}");
        }
    }
    let tag = std::env::var("RUNNER_BUILD")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s != "unknown")
        .or_else(|| std::fs::read_to_string("../../BUILD_TAG").ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()))
        .or_else(|| {
            let h = git(&["rev-parse", "--short=10", "HEAD"]).filter(|s| !s.is_empty())?;
            let mut st = vec!["status", "--porcelain", "--", "crates"];
            st.extend_from_slice(EXTRA_DIRS);
            st.extend_from_slice(&["Cargo.toml", "Cargo.lock"]);
            let dirty = git(&st).is_some_and(|s| !s.is_empty());
            Some(if dirty { format!("{h}-dirty") } else { h })
        })
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=RUNNER_BUILD_HASH={tag}");
    println!("cargo:rustc-check-cfg=cfg(runner_jobs_embed)");
    jobs_embed();
}

fn jobs_embed() {}
