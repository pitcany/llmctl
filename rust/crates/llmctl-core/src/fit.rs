//! VRAM fit (M4c item 2): start a workload *beside* what is running when
//! every card it uses has room for it, instead of stopping everything that
//! shares a card. Ported from the Python `vram_fit`.
//!
//! A workload's need is what it held, per card, after it last came up
//! serving (measured, never guessed). The decision fails closed: never
//! measured, free VRAM unreadable, or a systemd `Conflicts=` between the
//! units means the old behaviour — evict what overlaps. "Unknown is never
//! free" stays.
//!
//! An engine that sizes itself from free memory (vLLM's
//! `--gpu-memory-utilization`) measures as nearly the whole card when it
//! starts alone, so it never fits beside anything. That is the safe side.

use std::collections::BTreeMap;

/// Free VRAM a card must keep after the new model loads (CUDA context,
/// spikes). Same figure as the Python fit.
pub const HEADROOM_MIB: u64 = 1024;

/// MiB per card index: `{0: 25400}`.
pub type CardMib = BTreeMap<u32, u64>;

/// True when every card in `need` has that much free plus headroom. An
/// absent or empty side (never measured, unreadable) is never a fit.
pub fn fits(need: Option<&CardMib>, free: Option<&CardMib>) -> bool {
    match (need, free) {
        (Some(need), Some(free)) if !need.is_empty() && !free.is_empty() => need
            .iter()
            .all(|(card, mib)| free.get(card).is_some_and(|f| *f >= mib + HEADROOM_MIB)),
        _ => false,
    }
}

/// Whether a start keeps its neighbours, and why — the reason goes into the
/// plan preview so the owner sees what the decision rested on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitDecision {
    Beside(String),
    Evict(String),
}

impl FitDecision {
    pub fn keeps_neighbours(&self) -> bool {
        matches!(self, FitDecision::Beside(_))
    }

    pub fn reason(&self) -> &str {
        match self {
            FitDecision::Beside(r) | FitDecision::Evict(r) => r,
        }
    }
}

/// `{0: 25400}` → `24.8 GiB on gpu 0`.
pub fn format_mib(per_card: &CardMib) -> String {
    per_card
        .iter()
        .map(|(c, m)| format!("{:.1} GiB on gpu {c}", *m as f64 / 1024.0))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Decide for one start.
///
/// * `need`: the target's last measured holdings, if it was ever measured;
/// * `free`: free MiB per card right now, if the cards could be read;
/// * `own`: what the target itself holds now (a restart gives that back);
/// * `conflicts`: units systemd stops when the target starts (`None` when
///   it could not be asked);
/// * `neighbours`: the units that would otherwise be evicted or unloaded.
pub fn decide(
    need: Option<&CardMib>,
    free: Option<&CardMib>,
    own: &CardMib,
    conflicts: Option<&[String]>,
    neighbours: &[String],
) -> FitDecision {
    let Some(need) = need.filter(|n| !n.is_empty()) else {
        return FitDecision::Evict("never measured, so its VRAM need is unknown".into());
    };
    let Some(free) = free.filter(|f| !f.is_empty()) else {
        return FitDecision::Evict("free VRAM could not be read".into());
    };
    let Some(conflicts) = conflicts else {
        return FitDecision::Evict("the unit's Conflicts= could not be read".into());
    };
    if let Some(hit) = neighbours.iter().find(|n| conflicts.contains(n)) {
        return FitDecision::Evict(format!("systemd Conflicts= would stop {hit} anyway"));
    }
    let mut available = free.clone();
    for (card, mib) in own {
        *available.entry(*card).or_insert(0) += mib;
    }
    if fits(Some(need), Some(&available)) {
        FitDecision::Beside(format!(
            "fits: needs {}, free {} ({} MiB headroom kept)",
            format_mib(need),
            format_mib(&available.iter().filter(|(c, _)| need.contains_key(c)).map(|(c, m)| (*c, *m)).collect()),
            HEADROOM_MIB
        ))
    } else {
        FitDecision::Evict(format!(
            "does not fit: needs {} plus {} MiB headroom, free {}",
            format_mib(need),
            HEADROOM_MIB,
            format_mib(&available.iter().filter(|(c, _)| need.contains_key(c)).map(|(c, m)| (*c, *m)).collect())
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mib(pairs: &[(u32, u64)]) -> CardMib {
        pairs.iter().copied().collect()
    }

    #[test]
    fn fits_needs_headroom_on_every_card_and_fails_closed() {
        let need = mib(&[(0, 6000), (1, 6000)]);
        assert!(fits(Some(&need), Some(&mib(&[(0, 7024), (1, 30000)]))));
        assert!(!fits(Some(&need), Some(&mib(&[(0, 7023), (1, 30000)]))), "one MiB short of headroom");
        assert!(!fits(Some(&need), Some(&mib(&[(0, 30000)]))), "a card that was not read");
        assert!(!fits(None, Some(&mib(&[(0, 30000)]))));
        assert!(!fits(Some(&need), None));
        assert!(!fits(Some(&mib(&[])), Some(&mib(&[(0, 30000)]))));
    }

    #[test]
    fn the_decision_names_what_it_rested_on() {
        let need = mib(&[(0, 6000)]);
        let free = mib(&[(0, 20000), (1, 30000)]);
        let none: [String; 0] = [];
        let coder = ["user/coder.service".to_string()];

        let d = decide(Some(&need), Some(&free), &mib(&[]), Some(&none), &coder);
        assert!(d.keeps_neighbours(), "{d:?}");
        assert!(d.reason().contains("needs 5.9 GiB on gpu 0") && d.reason().contains("free 19.5 GiB on gpu 0"), "{d:?}");

        let tight = mib(&[(0, 6500)]);
        let d = decide(Some(&need), Some(&tight), &mib(&[]), Some(&none), &coder);
        assert!(!d.keeps_neighbours() && d.reason().starts_with("does not fit"), "{d:?}");
        // A restart gives the target's own memory back first.
        let d = decide(Some(&need), Some(&tight), &mib(&[(0, 6000)]), Some(&none), &coder);
        assert!(d.keeps_neighbours(), "{d:?}");

        // Fail closed on every unknown.
        for d in [
            decide(None, Some(&free), &mib(&[]), Some(&none), &coder),
            decide(Some(&need), None, &mib(&[]), Some(&none), &coder),
            decide(Some(&need), Some(&free), &mib(&[]), None, &coder),
            decide(Some(&need), Some(&free), &mib(&[]), Some(&coder), &coder),
        ] {
            assert!(!d.keeps_neighbours(), "{d:?}");
        }
    }
}
