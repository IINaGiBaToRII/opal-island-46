
pub mod arch;
pub mod dataset;
pub mod globals;
pub mod hybrid;
pub mod linear;
pub mod memplan;
pub mod nnfast;
pub mod refeval;
pub mod registry;
pub mod relabel;
pub mod select;
pub mod view;
pub mod wdl;
pub mod weights;

pub use arch::{Arch, Deep, Phase, PhaseBy, PhaseKind, Quant, ViewKind};
pub use select::{default_eval_spec, split_eval_key, EvalSpec};
pub use hybrid::{HybState, HybridEval, HybridEvalInc};
pub use linear::{IncState, LinearEval, LinearEvalInc};
pub use refeval::{Features, RefEval};
pub use view::{color_flip, opp_view, stm_view, View, ViewTables};
pub use weights::Model;

pub fn cmd_weights_info(path: &str) -> Result<String, String> {
    let m = Model::load_any(std::path::Path::new(path))?;
    let meta = m.meta.as_ref().ok_or("нет метаданных")?;
    let mut s = format!(
        "файл {path}\n  вариант {}, взятие {}, версия правил {} (текущая {}), eval_id {:#010x} (проверен по содержимому)\n",
        meta.variant,
        if meta.forced { "обязательно" } else { "по желанию" },
        meta.rules_version,
        crate::datadir::RULES_VERSION,
        m.eval_id()
    );
    s += &m.arch.summary()?;
    for b in &m.layout.blocks {
        s += &format!("  блок {:<12} {} x {}\n", b.name, b.dtype.name(), b.len);
    }
    Ok(s)
}
