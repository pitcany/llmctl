//! The vendor `load` verb's plan (Rust M5c item 4): one engine-level load
//! of one model onto a *running* vendor service (Ollama today), the mirror
//! of `unload`. What the mirror adds: the model must be in this host's
//! catalog, and the VRAM the load claims is planned before anything runs —
//! a measured need (or the catalog's estimate) against the free VRAM of
//! the service's cards, fail closed on every unknown. `--force` skips the
//! fit only; after a successful load the executor records the measured
//! need (the free-VRAM drop, else the engine's own `size_vram`), so the
//! next load plans with the real figure.
//!
//! [`plan_load`] is pure over its inputs; the API layer calls it for the
//! request check and admission calls it again under the admit lock with
//! fresh observations, the way `unload` re-plans.

use llmctl_core::plan::Plan;
use llmctl_core::{CoreError, HostInventory, Observation, WorkloadName};

use crate::catalog::CatalogStore;
use crate::vram::Store as VramStore;

/// Free VRAM kept on every card the load touches (the M5d router rule).
pub const HEADROOM_MIB: u64 = 1024;

/// Everything [`plan_load`] needs that is not the request itself.
pub struct LoadInputs<'a> {
    pub inventory: &'a HostInventory,
    pub observations: &'a [Observation],
    pub catalog: &'a CatalogStore,
    /// Measured VRAM records (`vram.json`).
    pub vram: &'a VramStore,
    /// Free VRAM per card index right now; `None` = the cards could not
    /// be read, which never fits.
    pub free: Option<&'a llmctl_core::CardMib>,
}

/// Why a load was refused before anything ran.
#[derive(Debug, thiserror::Error)]
pub enum LoadPlanError {
    /// A structural refusal (unknown workload, engine without a load,
    /// not running, the service identity): the core planner's words.
    #[error(transparent)]
    Core(#[from] CoreError),
    /// A load-specific refusal (catalog, need, fit): the message names
    /// the way out. Wire code `load_refused`.
    #[error("{0}")]
    Refused(String),
}

/// Build the load plan: the core planner's structural plan, plus the
/// catalog check and (unless `force`) the VRAM fit.
pub fn plan_load(
    inputs: &LoadInputs<'_>,
    workload: &str,
    model: &str,
    keep_alive_secs: i64,
    force: bool,
) -> Result<Plan, LoadPlanError> {
    let name = WorkloadName::new(workload)?;
    let def = inputs
        .inventory
        .workload(&name)
        .ok_or_else(|| CoreError::UnknownWorkload(workload.to_string()))?;
    let plan = llmctl_core::plan_load(
        inputs.inventory,
        inputs.observations,
        &name,
        model,
        keep_alive_secs,
    )?;
    // The catalog check: this host must know the model, enabled, on this
    // engine. The catalog is what names "which model, on which store".
    let engine = def.engine_kind.to_string();
    let known = inputs.catalog.models.iter().find(|m| {
        m.enabled
            && m.engine == engine
            && (m.source.as_deref() == Some(model.trim()) || m.name == model.trim())
    });
    let record = known.ok_or_else(|| {
        LoadPlanError::Refused(format!(
            "{model:?} is not in this host's catalog (engine {engine}, enabled); \
             `llmctl-next model list` shows what is; `model add` records it"
        ))
    })?;
    if force {
        return Ok(plan);
    }
    // The need: a measured record first, else the catalog's estimate.
    let key = format!("{}|{}", def.name, model.trim());
    let cards: Vec<u32> = def.gpus.iter().map(|g| g.index).collect();
    let per_card: Vec<u64> = match inputs.vram.get(&key) {
        Some(rec) if !rec.per_card.is_empty() => {
            cards.iter().map(|c| rec.per_card.get(c).copied().unwrap_or(0)).collect()
        }
        _ => {
            let estimate = record
                .extra
                .get("estimated_vram_gb")
                .and_then(|v| v.as_f64())
                .filter(|gb| *gb > 0.0);
            match estimate {
                Some(gb) if !cards.is_empty() => {
                    let each = ((gb * 1024.0).ceil() as u64).div_ceil(cards.len() as u64);
                    vec![each; cards.len()]
                }
                _ => {
                    return Err(LoadPlanError::Refused(format!(
                        "{model:?} has no measured VRAM record and the catalog carries no \
                         estimated_vram_gb; `load --force` loads once and records the measured \
                         need, or `model edit {model} --set estimated_vram_gb=<n>` declares it"
                    )))
                }
            }
        }
    };
    let free = inputs.free.ok_or_else(|| {
        LoadPlanError::Refused(format!(
            "the free VRAM of gpu(s) {:?} could not be read; unknown is never free \
             (retry, or --force after a look with `llmctl-next vram`)",
            cards
        ))
    })?;
    let holders = running_neighbours(inputs, &name);
    for (c, need) in cards.iter().zip(per_card.iter()) {
        let have = free.get(c).copied();
        match have {
            Some(have) if need + HEADROOM_MIB <= have => {}
            Some(have) => {
                return Err(LoadPlanError::Refused(format!(
                    "gpu {c} has {:.1} GiB free, the load needs {:.1} GiB plus {:.1} GiB headroom{}; \
                     stop what holds the card or --force",
                    have as f64 / 1024.0,
                    *need as f64 / 1024.0,
                    HEADROOM_MIB as f64 / 1024.0,
                    if holders.is_empty() {
                        String::new()
                    } else {
                        format!(" (running there: {})", holders.join(", "))
                    },
                )))
            }
            None => {
                return Err(LoadPlanError::Refused(format!(
                    "gpu {c} is not among the cards the probe read {:?}; unknown is never free",
                    free.keys().collect::<Vec<_>>()
                )))
            }
        }
    }
    Ok(plan)
}

/// Enrolled workloads observed running that share one of `target`'s cards,
/// name order: who the load would sit beside.
fn running_neighbours(inputs: &LoadInputs<'_>, target: &WorkloadName) -> Vec<String> {
    let Some(def) = inputs.inventory.workload(target) else {
        return Vec::new();
    };
    let mut out: Vec<String> = inputs
        .inventory
        .workloads
        .iter()
        .filter(|w| w.name != *target)
        .filter(|w| w.gpus.iter().any(|g| def.gpus.contains(g)))
        .filter(|w| {
            inputs
                .observations
                .iter()
                .any(|o| o.workload == w.name && o.process == llmctl_core::ProcessState::Running)
        })
        .map(|w| w.name.to_string())
        .collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headroom_is_one_gib() {
        assert_eq!(HEADROOM_MIB, 1024);
    }
}
