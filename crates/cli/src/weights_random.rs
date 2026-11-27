
use engine_core::learn::arch::Arch;
use engine_core::learn::weights::{Meta, Model};
use engine_core::VariantId;

pub fn run(get: &dyn Fn(&str) -> Option<String>, flag: &dyn Fn(&str) -> bool) -> Result<(), String> {
    let usage = "usage: runner weights-random --variant V [--free] --arch FORM.json --seed S --name N --out F.dw";
    let vname = get("variant").unwrap_or_else(|| "russian".into());
    let variant = VariantId::parse(&vname).ok_or(format!("unknown variant '{vname}'"))?;
    let form = get("arch").ok_or(usage)?;
    let out = get("out").ok_or(usage)?;
    let name = get("name").ok_or(usage)?;
    let seed: u64 = get("seed").ok_or(usage)?.parse().map_err(|_| "bad value for --seed".to_string())?;
    let text = std::fs::read(&form).map_err(|e| format!("чтение {form}: {e}"))?;
    let mut arch: Arch = serde_json::from_slice(&text).map_err(|e| format!("описание формы {form}: {e}"))?;
    arch.name = name;
    let model = Model::random(arch, seed)?;
    let meta = Meta { variant: variant.name().to_string(), forced: !flag("free"), rules_version: engine_core::datadir::RULES_VERSION };
    let bytes = model.to_bytes(&meta)?;
    let back = Model::from_bytes(&bytes)?;
    std::fs::write(&out, &bytes).map_err(|e| format!("запись {out}: {e}"))?;
    println!("{out}: {} байт, eval_id {:#010x}", bytes.len(), back.eval_id());
    Ok(())
}
