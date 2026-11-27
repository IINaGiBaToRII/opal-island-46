
use crate::search::Params;

pub(crate) fn no_new_iteration(p: &Params, ms: u64, hard: u64, last: u64, prev: u64) -> bool {
    if p.nostart > 0 && ms * 100 > hard * p.nostart as u64 {
        return true;
    }
    if p.nopred != 0 && last > 0 {
        let growth = if prev > 0 { (last * 100 / prev).max(150) } else { 150 };
        if ms + last * growth / 100 > hard {
            return true;
        }
    }
    false
}

pub(crate) fn easy_move_stop(p: &Params, ms: u64, hard: u64, stable: u32, score_delta: i32) -> bool {
    p.easy > 0 && stable >= p.easy as u32 && score_delta.abs() <= p.easy_cp && ms * 100 >= hard * p.easy_pct as u64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GameClock {
    pub base_ms: u64,
    pub inc_ms: u64,
    pub moves_to_go: u32,
    pub front_moves: u32,
    pub front_pct: u32,
    pub max_pct: u32,
}

impl GameClock {
    pub fn new(base_ms: u64, inc_ms: u64) -> GameClock {
        GameClock { base_ms, inc_ms, moves_to_go: 30, front_moves: 0, front_pct: 100, max_pct: 50 }
    }

    pub fn move_ms(&self, remaining_ms: u64, own_move: u64) -> u64 {
        let share = remaining_ms / self.moves_to_go.max(1) as u64 + self.inc_ms;
        let share = if own_move < self.front_moves as u64 { share * self.front_pct as u64 / 100 } else { share };
        share.min(remaining_ms * self.max_pct.min(100) as u64 / 100).max(1)
    }

    pub fn after_move(&self, remaining_ms: u64, used_ms: u64) -> u64 {
        remaining_ms.saturating_sub(used_ms) + self.inc_ms
    }
}

pub(crate) struct TimeCtl {
    hard: Option<u64>,
    soft: Option<u64>,
    base_soft: Option<u64>,
    best_share: Option<u32>,
    it_last: u64,
    it_prev: u64,
    stable: u32,
    last_score: Option<i32>,
    score_delta: i32,
}

impl TimeCtl {
    pub(crate) fn new(p: &Params, hard: Option<u64>) -> TimeCtl {
        let soft = hard.map(|h| h * p.soft as u64 / 100);
        TimeCtl { hard, soft, base_soft: soft, best_share: None, it_last: 0, it_prev: 0, stable: 0, last_score: None, score_delta: i32::MAX }
    }

    pub(crate) fn note_iteration(&mut self, changed: bool, score: i32) {
        self.score_delta = self.last_score.map_or(i32::MAX, |s| score - s);
        self.last_score = Some(score);
        self.stable = if changed { 0 } else { self.stable + 1 };
    }

    pub(crate) fn note_best_share(&mut self, share: Option<u32>) {
        self.best_share = share;
    }

    fn wants_extension(&self, p: &Params) -> bool {
        let drop = p.tdrop > 0 && self.score_delta != i32::MAX && self.score_delta <= -p.tdrop;
        let unclear = p.tnodes > 0 && self.best_share.is_some_and(|s| s < p.tnodes as u32);
        drop || unclear
    }

    pub(crate) fn stop_after(&mut self, p: &Params, changed: bool, depth: u32, ms: u64, it_start: u64) -> bool {
        if let (Some(s), Some(h)) = (self.soft, self.hard) {
            if changed && depth > 4 {
                let ext = (s * 3 / 2).min(h * 9 / 10);
                self.soft = Some(if p.softfix != 0 { s.max(ext) } else { ext });
            }
            if depth > 4 && self.wants_extension(p) {
                let ext = (self.base_soft.unwrap_or(s) * p.text.max(100) as u64 / 100).min(h);
                self.soft = Some(self.soft.unwrap().max(ext));
            }
            if ms >= self.soft.unwrap() {
                return true;
            }
        }
        if let Some(h) = self.hard {
            self.it_prev = self.it_last;
            self.it_last = ms.saturating_sub(it_start);
            if no_new_iteration(p, ms, h, self.it_last, self.it_prev) || easy_move_stop(p, ms, h, self.stable, self.score_delta) {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_clock_budget_and_front_skew() {
        let flat = GameClock::new(60_000, 500);
        let skew = GameClock { front_moves: 10, front_pct: 250, ..flat };
        for c in [flat, skew] {
            let (mut rem, mut spent) = (c.base_ms, 0u64);
            let mut lim = Vec::new();
            for k in 0..120u64 {
                let t = c.move_ms(rem, k);
                assert!(t >= 1 && t <= rem.max(1), "ход {k}: {t} при остатке {rem}");
                lim.push(t);
                spent += t;
                rem = c.after_move(rem, t);
            }
            assert!(spent <= c.base_ms + 120 * c.inc_ms, "{spent}");
            assert!(rem >= c.inc_ms);
            if c.front_moves > 0 {
                assert!(lim[0] > flat.move_ms(flat.base_ms, 0) * 2, "перекос на первый ход: {}", lim[0]);
                assert!(lim[9] > lim[10] * 2, "граница перекоса: {} / {}", lim[9], lim[10]);
            } else {
                assert_eq!(lim[0], 60_000 / 30 + 500);
            }
        }
        assert_eq!(flat.move_ms(0, 50), 1);
        assert_eq!(GameClock { moves_to_go: 1, ..flat }.move_ms(10_000, 0), 5_000);
    }

    fn ext_trace(p: &Params, hard: u64, trace: &[(bool, i32, u64, u32)]) -> Option<usize> {
        let mut tc = TimeCtl::new(p, Some(hard));
        let mut ms = 0u64;
        for (i, &(changed, score, dur, share)) in trace.iter().enumerate() {
            let it_start = ms;
            ms += dur;
            tc.note_iteration(changed, score);
            if p.tnodes > 0 {
                tc.note_best_share(Some(share));
            }
            if tc.stop_after(p, changed, i as u32 + 1, ms, it_start) {
                return Some(i);
            }
        }
        None
    }

    #[test]
    fn extension_on_score_drop_and_node_share() {
        let hard = 10_000;
        let trace: Vec<(bool, i32, u64, u32)> =
            (0..12).map(|i| (false, if i >= 4 { -60 } else { 0 }, 1_000, if i >= 4 { 20 } else { 90 })).collect();
        let base = Params::parse("soft=50");
        assert_eq!(ext_trace(&base, hard, &trace), Some(4));
        for off in ["soft=50,tdrop=0,tnodes=0", "soft=50,text=300", "soft=50,tdrop=100", "soft=50,tnodes=10"] {
            assert_eq!(ext_trace(&Params::parse(off), hard, &trace), Some(4), "{off}");
        }
        assert_eq!(ext_trace(&Params::parse("soft=50,tdrop=50"), hard, &trace), Some(7));
        assert_eq!(ext_trace(&Params::parse("soft=50,tnodes=30"), hard, &trace), Some(7));
        assert_eq!(ext_trace(&Params::parse("soft=50,tdrop=50,text=300"), hard, &trace), Some(9));
        let t100 = ext_trace(&Params::parse("tdrop=50,tnodes=30"), hard, &trace);
        assert_eq!(t100, ext_trace(&Params::default(), hard, &trace));
        let p = Params::parse("tdrop=40,tnodes=25,text=180");
        assert_eq!((p.tdrop, p.tnodes, p.text, p.get("tnodes")), (40, 25, 180, Some(25)));
        assert_eq!((Params::default().tdrop, Params::default().tnodes, Params::default().text), (0, 0, 150));
    }

    #[test]
    fn defaults_are_off() {
        let p = Params::default();
        assert!(!no_new_iteration(&p, 9_999, 10_000, 5_000, 1_000));
        assert!(!easy_move_stop(&p, 9_999, 10_000, 100, 0));
    }

    #[test]
    fn nostart_by_fraction() {
        let p = Params::parse("nostart=60");
        assert!(!no_new_iteration(&p, 6_000, 10_000, 100, 50));
        assert!(no_new_iteration(&p, 6_001, 10_000, 100, 50));
    }

    #[test]
    fn nopred_by_growth() {
        let p = Params::parse("nopred=1");
        assert!(!no_new_iteration(&p, 7_900, 10_000, 1_000, 500));
        assert!(no_new_iteration(&p, 8_100, 10_000, 1_000, 500));
        assert!(no_new_iteration(&p, 8_600, 10_000, 1_000, 990));
        assert!(!no_new_iteration(&p, 8_400, 10_000, 1_000, 990));
        assert!(!no_new_iteration(&p, 9_999, 10_000, 0, 0));
    }

    #[test]
    fn easy_move_needs_stability_and_time() {
        let p = Params::parse("easy=3,easy_cp=20,easy_pct=40");
        assert!(!easy_move_stop(&p, 5_000, 10_000, 2, 0));
        assert!(!easy_move_stop(&p, 5_000, 10_000, 3, 25));
        assert!(!easy_move_stop(&p, 3_900, 10_000, 5, -10));
        assert!(easy_move_stop(&p, 4_000, 10_000, 3, -20));
    }

    fn old_no_new_iteration(p: &Params, ms: u64, hard: u64, last: u64, prev: u64) -> bool {
        if p.nostart > 0 && ms * 100 > hard * p.nostart as u64 {
            return true;
        }
        if p.nopred != 0 && last > 0 {
            let growth = if prev > 0 { (last * 100 / prev).max(150) } else { 150 };
            if ms + last * growth / 100 > hard {
                return true;
            }
        }
        false
    }

    fn old_easy_move_stop(p: &Params, ms: u64, hard: u64, stable: u32, score_delta: i32) -> bool {
        p.easy > 0 && stable >= p.easy as u32 && score_delta.abs() <= p.easy_cp && ms * 100 >= hard * p.easy_pct as u64
    }

    type Step = (bool, i32, u64);

    #[allow(unused_assignments)]
    fn old_trace(p: &Params, hard: Option<u64>, trace: &[Step]) -> Option<usize> {
        let mut soft = hard.map(|h| h * p.soft as u64 / 100);
        let (mut it_last, mut it_prev, mut stable, mut last_score) = (0u64, 0u64, 0u32, None::<i32>);
        let mut ms = 0u64;
        for (i, &(changed, score, dur)) in trace.iter().enumerate() {
            let depth = i as u32 + 1;
            let it_start = ms;
            ms += dur;
            let score_delta = last_score.map_or(i32::MAX, |s| score - s);
            last_score = Some(score);
            stable = if changed { 0 } else { stable + 1 };
            if let (Some(s), Some(h)) = (soft, hard) {
                if changed && depth > 4 {
                    let ext = (s * 3 / 2).min(h * 9 / 10);
                    soft = Some(if p.softfix != 0 { s.max(ext) } else { ext });
                }
                if ms >= soft.unwrap() {
                    return Some(i);
                }
            }
            if let Some(h) = hard {
                it_prev = it_last;
                it_last = ms.saturating_sub(it_start);
                if old_no_new_iteration(p, ms, h, it_last, it_prev) || old_easy_move_stop(p, ms, h, stable, score_delta) {
                    return Some(i);
                }
            }
        }
        None
    }

    fn new_trace(p: &Params, hard: Option<u64>, trace: &[Step]) -> Option<usize> {
        let mut tc = TimeCtl::new(p, hard);
        let mut ms = 0u64;
        for (i, &(changed, score, dur)) in trace.iter().enumerate() {
            let it_start = ms;
            ms += dur;
            tc.note_iteration(changed, score);
            if tc.stop_after(p, changed, i as u32 + 1, ms, it_start) {
                return Some(i);
            }
        }
        None
    }

    #[test]
    fn trace_identical_to_old_iterate() {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = |n: u64| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s % n
        };
        let opts = [
            "", "softfix=0", "softfix=1", "soft=60", "soft=60,softfix=1", "soft=150", "nostart=50", "nopred=1",
            "easy=3,easy_cp=20,easy_pct=30", "soft=70,nostart=60,nopred=1,easy=2,easy_cp=10,easy_pct=20,softfix=1",
        ];
        let (mut stops, mut total) = (0u32, 0u32);
        for o in opts {
            let p = Params::parse(o);
            for _ in 0..2000 {
                let hard = match rnd(5) {
                    0 => None,
                    1 => Some(50 + rnd(100)),
                    _ => Some(200 + rnd(20_000)),
                };
                let n = 1 + rnd(30) as usize;
                let mut dur = 1 + rnd(5);
                let mut score = rnd(200) as i32 - 100;
                let trace: Vec<Step> = (0..n)
                    .map(|_| {
                        dur = (dur * (100 + rnd(250)) / 100).max(1);
                        score += rnd(80) as i32 - 40;
                        (rnd(3) == 0, score, dur)
                    })
                    .collect();
                let a = old_trace(&p, hard, &trace);
                assert_eq!(new_trace(&p, hard, &trace), a, "opts {o:?} hard {hard:?} trace {trace:?}");
                total += 1;
                stops += a.is_some() as u32;
            }
        }
        assert!(stops > total / 10 && stops < total, "{stops}/{total}");
    }
}
