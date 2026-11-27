
use super::gen::{generate_all, generate_all_ctl, layer_plan, next_slice, GenCtl, GenOpts, GenReport, LayerPlan, SlicePlan};
use super::probe::Egdb;
use super::verify::{verify, VerifyReport};
use super::EgdbInfo;
use crate::datadir::{self, Kind};
use crate::facade::{GameConfig, VariantId};
use crate::position::Position;
use crate::rules::*;
use std::path::{Path, PathBuf};

macro_rules! dispatch_r {
    ($cfg:expr, $f:ident ( $($a:expr),* )) => {
        match ($cfg.variant, $cfg.forced_capture) {
            (VariantId::Russian, true) => $f::<Russian<true>>($($a),*),
            (VariantId::Russian, false) => $f::<Russian<false>>($($a),*),
            (VariantId::Brazilian, true) => $f::<Brazilian<true>>($($a),*),
            (VariantId::Brazilian, false) => $f::<Brazilian<false>>($($a),*),
            (VariantId::International, true) => $f::<International<true>>($($a),*),
            (VariantId::International, false) => $f::<International<false>>($($a),*),
            (VariantId::American, true) => $f::<American<true>>($($a),*),
            (VariantId::American, false) => $f::<American<false>>($($a),*),
            (VariantId::Italian, true) => $f::<Italian<true>>($($a),*),
            (VariantId::Italian, false) => $f::<Italian<false>>($($a),*),
            (VariantId::Spanish, true) => $f::<Spanish<true>>($($a),*),
            (VariantId::Spanish, false) => $f::<Spanish<false>>($($a),*),
            (VariantId::Czech, true) => $f::<Czech<true>>($($a),*),
            (VariantId::Czech, false) => $f::<Czech<false>>($($a),*),
            (VariantId::Thai, true) => $f::<Thai<true>>($($a),*),
            (VariantId::Thai, false) => $f::<Thai<false>>($($a),*),
            (VariantId::Turkish, true) => $f::<Turkish<true>>($($a),*),
            (VariantId::Turkish, false) => $f::<Turkish<false>>($($a),*),
        }
    };
}

pub fn egdb_dir(cfg: &GameConfig) -> PathBuf {
    datadir::knowledge_dir(cfg.variant.name(), cfg.forced_capture, Kind::Egdb)
}

fn info_r<R: Rules>(dir: &Path) -> EgdbInfo {
    Egdb::<R>::open_at(dir).map(|d| d.info()).unwrap_or_default()
}

pub fn egdb_info(cfg: &GameConfig) -> EgdbInfo {
    egdb_info_at(cfg, &egdb_dir(cfg))
}

pub fn egdb_info_at(cfg: &GameConfig, dir: &Path) -> EgdbInfo {
    if !dir.exists() {
        return EgdbInfo::default();
    }
    dispatch_r!(cfg, info_r(dir))
}

fn gen_r<R: Rules>(o: &GenOpts, log: &mut dyn FnMut(&str)) -> Result<GenReport, String> {
    generate_all::<R>(o, log)
}

pub fn egdb_generate(cfg: &GameConfig, o: &GenOpts, log: &mut dyn FnMut(&str)) -> Result<GenReport, String> {
    dispatch_r!(cfg, gen_r(o, log))
}

fn verify_r<R: Rules>(dir: &Path, sample: usize, seed: u64, indep: usize, log: &mut dyn FnMut(&str)) -> Result<VerifyReport, String> {
    verify::<R>(dir, sample, seed, indep, log)
}

pub fn egdb_verify(cfg: &GameConfig, dir: &Path, sample: usize, seed: u64, indep: usize, log: &mut dyn FnMut(&str)) -> Result<VerifyReport, String> {
    dispatch_r!(cfg, verify_r(dir, sample, seed, indep, log))
}

#[derive(Debug)]
pub struct ProbeOut {
    pub value: Option<i8>,
    pub text: String,
    pub moves: Vec<(String, Option<i8>)>,
}

pub fn describe(v: Option<i8>) -> String {
    match v {
        None => "нет в базе".into(),
        Some(0) => "ничья".into(),
        Some(m) if m > 0 => format!("выигрыш за {m} полуходов"),
        Some(m) => format!("проигрыш за {} полуходов", -(m as i32) - 1),
    }
}

fn probe_r<R: Rules>(dir: &Path, cfg: &GameConfig, dfen: &str) -> Result<ProbeOut, String> {
    let db = Egdb::<R>::open_at(dir)?;
    let pos = Position::<R>::from_dfen(dfen)?;
    let value = db.probe(&pos);
    let mut moves = vec![];
    for mi in crate::facade::legal_moves(cfg, dfen)? {
        let nf = crate::facade::apply_move(cfg, dfen, &mi.notation)?;
        let child = Position::<R>::from_dfen(&nf)?;
        moves.push((mi.notation, db.probe(&child)));
    }
    Ok(ProbeOut { value, text: describe(value), moves })
}

pub fn egdb_probe(cfg: &GameConfig, dir: &Path, dfen: &str) -> Result<ProbeOut, String> {
    dispatch_r!(cfg, probe_r(dir, cfg, dfen))
}

fn gen_ctl_r<R: Rules>(o: &GenOpts, ctl: &GenCtl, log: &mut dyn FnMut(&str)) -> Result<GenReport, String> {
    generate_all_ctl::<R>(o, ctl, log)
}

pub fn egdb_generate_ctl(cfg: &GameConfig, o: &GenOpts, ctl: &GenCtl, log: &mut dyn FnMut(&str)) -> Result<GenReport, String> {
    dispatch_r!(cfg, gen_ctl_r(o, ctl, log))
}

fn next_r<R: Rules>(dir: &Path, pieces: u32, ram: u64) -> (Option<SlicePlan>, Option<SlicePlan>) {
    next_slice::<R>(dir, pieces, ram)
}

fn layer_r<R: Rules>(dir: &Path, pieces: u32) -> Result<LayerPlan, String> {
    layer_plan::<R>(dir, pieces)
}

pub fn egdb_layer_plan(cfg: &GameConfig, dir: &Path, pieces: u32) -> Result<LayerPlan, String> {
    dispatch_r!(cfg, layer_r(dir, pieces))
}

pub fn egdb_manifest_rates(dir: &Path, threads_guess: f64) -> (Option<f64>, Option<f64>) {
    let Some(v) = std::fs::read_to_string(dir.join("manifest.json")).ok().and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()) else { return (None, None) };
    let (mut bytes, mut secs, mut pos) = (0.0, 0.0, 0.0);
    for s in v.get("slices").and_then(|x| x.as_array()).into_iter().flatten() {
        let p = s.get("positions").and_then(|x| x.as_f64()).unwrap_or(0.0);
        if p <= 0.0 {
            continue;
        }
        pos += p;
        bytes += s.get("bytes").and_then(|x| x.as_f64()).unwrap_or(0.0);
        secs += s.get("secs").and_then(|x| x.as_f64()).unwrap_or(0.0);
    }
    if pos <= 0.0 {
        return (None, None);
    }
    (Some(bytes / pos).filter(|x| *x > 0.0), Some(secs * threads_guess / pos).filter(|x| *x > 0.0))
}

pub fn egdb_next_slice(cfg: &GameConfig, dir: &Path, pieces: u32, max_ram_bytes: u64) -> (Option<SlicePlan>, Option<SlicePlan>) {
    dispatch_r!(cfg, next_r(dir, pieces, max_ram_bytes))
}
