
use super::extract::{for_each_sample, replay};
use super::format::{Chunk, EndReason};
use super::dispatch_cfg;
use crate::facade::{GameConfig, VariantId};
use crate::rules::{American, Brazilian, Czech, International, Italian, Rules, Russian, Spanish, Thai, Turkish};
use std::collections::HashSet;
use std::fmt::Write;
use std::path::PathBuf;

#[derive(Default, Debug)]
pub struct Stats {
    pub files: u32,
    pub bad_files: Vec<String>,
    pub other_config: u32,
    pub games: u64,
    pub plies: u64,
    pub white: u64,
    pub draws: u64,
    pub black: u64,
    pub reasons: [u64; 6],
    pub depth_sum: u64,
    pub depth_n: u64,
    pub bytes: u64,
    pub samples: u64,
    pub unique: usize,
    pub by_pieces: Vec<u64>,
    pub exact: u64,
    pub egdb_present: bool,
    pub diversity: bool,
    pub dup_games: u64,
    pub dup_after_book: u64,
    pub mid_random_games: u64,
    pub kings_samples: u64,
    pub imbalance2_samples: u64,
    pub samples_before_mid: u64,
}

impl Stats {
    pub fn report(&self) -> String {
        let mut s = String::new();
        let g = self.games.max(1) as f64;
        let _ = writeln!(s, "файлов {} (повреждённых {}, другой конфигурации {})", self.files, self.bad_files.len(), self.other_config);
        for b in &self.bad_files {
            let _ = writeln!(s, "  ! {b}");
        }
        let _ = writeln!(s, "партий {}  полуходов {}  средняя длина {:.1}", self.games, self.plies, self.plies as f64 / g);
        let _ = writeln!(
            s,
            "белые + {}  ничьи = {}  чёрные - {}  доля ничьих {:.1}%",
            self.white,
            self.draws,
            self.black,
            100.0 * self.draws as f64 / g
        );
        let _ = write!(s, "причины окончания:");
        for r in EndReason::ALL {
            let _ = write!(s, "  {} {} ({:.1}%)", r.name(), self.reasons[r as usize], 100.0 * self.reasons[r as usize] as f64 / g);
        }
        let _ = writeln!(s);
        let _ = writeln!(s, "средняя глубина поиска {:.2}", self.depth_sum as f64 / self.depth_n.max(1) as f64);
        let _ = writeln!(s, "байт на партию (сжато) {:.1}  на полуход {:.2}", self.bytes as f64 / g, self.bytes as f64 / self.plies.max(1) as f64);
        let sm = self.samples.max(1) as f64;
        let _ = writeln!(
            s,
            "обучающих позиций {}  уникальных {} ({:.1}%)  с точным значением баз {} ({:.2}%){}",
            self.samples,
            self.unique,
            100.0 * self.unique as f64 / sm,
            self.exact,
            100.0 * self.exact as f64 / sm,
            if self.egdb_present { "" } else { "  [баз нет или отключены]" }
        );
        let _ = write!(s, "по числу фигур:");
        for (n, c) in self.by_pieces.iter().enumerate() {
            if *c > 0 {
                let _ = write!(s, " {n}:{c}");
            }
        }
        let _ = writeln!(s);
        s
    }

    pub fn diversity_report(&self) -> String {
        let mut s = String::new();
        if !self.diversity {
            return s;
        }
        let g = self.games.max(1) as f64;
        let sm = self.samples.max(1) as f64;
        let _ = writeln!(
            s,
            "разнообразие: повторов партий {} ({:.2}%), повторов позиции после дебюта {} ({:.2}%), партий со случайным ходом в середине {} ({:.1}%)",
            self.dup_games,
            100.0 * self.dup_games as f64 / g,
            self.dup_after_book,
            100.0 * self.dup_after_book as f64 / g,
            self.mid_random_games,
            100.0 * self.mid_random_games as f64 / g
        );
        let _ = writeln!(
            s,
            "позиций обучения на партию {:.2}, уникальных на партию {:.2}; с сильными фигурами {:.2}%; с перевесом ≥2 фигур {:.2}%; средняя длина {:.1}; ничьих {:.1}%; позиций до случайного хода в середине {:.2}%",
            self.samples as f64 / g,
            self.unique as f64 / g,
            100.0 * self.kings_samples as f64 / sm,
            100.0 * self.imbalance2_samples as f64 / sm,
            self.plies as f64 / g,
            100.0 * self.draws as f64 / g,
            100.0 * self.samples_before_mid as f64 / sm
        );
        let _ = writeln!(
            s,
            "TSV\tgames\tdup_games_pct\tdup_after_book_pct\tmid_random_pct\tunique_per_game\tkings_pct\timbalance2_pct\tavg_plies\tdraw_pct\tbefore_mid_pct\n\
             TSV\t{}\t{:.2}\t{:.2}\t{:.1}\t{:.2}\t{:.2}\t{:.2}\t{:.1}\t{:.1}\t{:.2}",
            self.games,
            100.0 * self.dup_games as f64 / g,
            100.0 * self.dup_after_book as f64 / g,
            100.0 * self.mid_random_games as f64 / g,
            self.unique as f64 / g,
            100.0 * self.kings_samples as f64 / sm,
            100.0 * self.imbalance2_samples as f64 / sm,
            self.plies as f64 / g,
            100.0 * self.draws as f64 / g,
            100.0 * self.samples_before_mid as f64 / sm
        );
        s
    }
}

fn game_key(g: &super::format::Game) -> (u64, u64) {
    use std::hash::{Hash, Hasher};
    let mut a = std::collections::hash_map::DefaultHasher::new();
    let mut b = std::collections::hash_map::DefaultHasher::new();
    0x5eed_u64.hash(&mut b);
    for h in [&mut a, &mut b] {
        g.start.map(|s| (s.white, s.black, s.kings, s.side)).hash(h);
        for p in &g.plies {
            (p.from, p.to, p.idx).hash(h);
        }
    }
    (a.finish(), b.finish() ^ g.plies.len() as u64)
}

fn run<R: Rules>(paths: &[PathBuf], use_egdb: bool, st: &mut Stats) {
    let db = if use_egdb { crate::egdb::shared::<R>() } else { None };
    st.egdb_present = db.is_some();
    let mut uniq: HashSet<(u64, u8)> = HashSet::new();
    let (mut seen_games, mut seen_book): (HashSet<(u64, u64)>, HashSet<(u64, u8)>) = (HashSet::new(), HashSet::new());
    st.by_pieces = vec![0; 65];
    for p in paths {
        let c = match Chunk::read(p) {
            Ok(c) => c,
            Err(e) => {
                st.bad_files.push(format!("{}: {e}", p.display()));
                continue;
            }
        };
        if c.header.variant != R::NAME || c.header.forced != R::FORCED {
            st.other_config += 1;
            continue;
        }
        st.files += 1;
        st.bytes += c.file_bytes;
        for g in c.games() {
            let g = match g {
                Ok(g) => g,
                Err(e) => {
                    st.bad_files.push(format!("{}: {e}", p.display()));
                    break;
                }
            };
            st.games += 1;
            st.plies += g.plies.len() as u64;
            match g.result {
                1 => st.white += 1,
                0 => st.draws += 1,
                _ => st.black += 1,
            }
            st.reasons[g.reason as usize] += 1;
            for pl in g.plies.iter().filter(|x| !x.random) {
                st.depth_sum += pl.depth as u64;
                st.depth_n += 1;
            }
            let mut mid_at = None;
            if st.diversity {
                if !seen_games.insert(game_key(&g)) {
                    st.dup_games += 1;
                }
                mid_at = g.plies.iter().enumerate().skip(g.random_plies as usize).find(|(_, p)| p.random).map(|(i, _)| i);
                if mid_at.is_some() {
                    st.mid_random_games += 1;
                }
                let lead = g.random_plies as usize;
                let mut after = None;
                let end = replay::<R>(&g, |i, pos, _, _| {
                    if i == lead {
                        after = Some((pos.hash, pos.side as u8));
                    }
                });
                if let Ok(end) = end {
                    let key = after.unwrap_or((end.hash, end.side as u8));
                    if !seen_book.insert(key) {
                        st.dup_after_book += 1;
                    }
                }
            }
            let div = st.diversity;
            let r = for_each_sample::<R>(&g, db.as_deref(), |pos, s| {
                st.samples += 1;
                uniq.insert((pos.hash, pos.side as u8));
                st.by_pieces[(s.pieces as usize).min(64)] += 1;
                if s.egdb.is_some() {
                    st.exact += 1;
                }
                if div {
                    if pos.kings != 0 {
                        st.kings_samples += 1;
                    }
                    if (pos.white.count_ones() as i32 - pos.black.count_ones() as i32).abs() >= 2 {
                        st.imbalance2_samples += 1;
                    }
                    if mid_at.is_some_and(|m| (s.ply as usize) < m) {
                        st.samples_before_mid += 1;
                    }
                }
            });
            if let Err(e) = r {
                st.bad_files.push(format!("{}: {e}", p.display()));
            }
        }
    }
    st.unique = uniq.len();
}

pub fn stats_for_paths(paths: &[PathBuf], variant: Option<VariantId>, use_egdb: bool) -> Result<(GameConfig, Stats), String> {
    stats_for_paths_with(paths, variant, use_egdb, false)
}

pub fn stats_for_paths_with(paths: &[PathBuf], variant: Option<VariantId>, use_egdb: bool, diversity: bool) -> Result<(GameConfig, Stats), String> {
    let mut cfg = None;
    let mut ok_paths = Vec::new();
    for p in paths {
        let Ok(h) = super::format::read_header(p) else {
            ok_paths.push(p.clone());
            continue;
        };
        let Some(v) = VariantId::parse(&h.variant) else { continue };
        if variant.is_some_and(|f| f != v) {
            continue;
        }
        let c = GameConfig { variant: v, forced_capture: h.forced };
        if cfg.is_none() {
            cfg = Some(c);
        }
        ok_paths.push(p.clone());
    }
    let cfg = cfg.ok_or("нет подходящих файлов партий")?;
    let mut st = Stats { diversity, ..Default::default() };
    dispatch_cfg!(&cfg, run(&ok_paths, use_egdb, &mut st));
    Ok((cfg, st))
}
