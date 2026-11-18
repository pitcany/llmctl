//! Measured VRAM per workload (M4c VRAM fit): what a workload's unit held,
//! per card, the last time it was seen serving. The fit decision
//! ([`llmctl_core::fit`]) compares that need with the free VRAM of the
//! cards; this module measures and remembers.
//!
//! The record is keyed by workload *and* model: a vLLM unit that switched
//! profile needs what the new profile held, not what the old one did.
//! Measurements live in `vram.json` in the data dir, not in the inventory —
//! they change with every start and are not configuration.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use llmctl_core::{CardMib, WorkloadDefinition};
use serde::{Deserialize, Serialize};

use crate::engine::Engine;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VramRecord {
    pub workload: String,
    pub model: String,
    /// MiB per card index.
    pub per_card: BTreeMap<u32, u64>,
    pub measured_at: DateTime<Utc>,
}

pub type Store = BTreeMap<String, VramRecord>;

pub fn store_path(data_dir: &Path) -> PathBuf {
    data_dir.join("vram.json")
}

pub fn key(def: &WorkloadDefinition) -> String {
    format!("{}|{}", def.name, def.model)
}

/// The store, empty when it does not exist or cannot be read: an unknown
/// need is "never measured", which never fits.
pub async fn load(path: &Path) -> Store {
    match tokio::fs::read(path).await {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
        Err(_) => Store::new(),
    }
}

/// What the workload's unit holds on its cards right now. `None` when the
/// cards cannot be read or the unit holds nothing there (not running, or
/// running on the CPU): there is then nothing to record.
pub async fn measure(engine: &dyn Engine, def: &WorkloadDefinition) -> Option<CardMib> {
    let unit = def.unit.to_string();
    let mut out = CardMib::new();
    for occ in engine.gpu_occupancy(&def.gpus).await.ok()? {
        if occ.unit.as_deref() == Some(unit.as_str()) && occ.memory_mib > 0 {
            *out.entry(occ.gpu).or_insert(0) += occ.memory_mib;
        }
    }
    (!out.is_empty()).then_some(out)
}

/// Measure a serving workload and remember it. Best effort: returns what
/// was recorded, or `None` when nothing could be measured or written.
pub async fn remember(
    engine: &dyn Engine,
    data_dir: &Path,
    def: &WorkloadDefinition,
) -> Option<VramRecord> {
    let per_card = measure(engine, def).await?;
    let record = VramRecord {
        workload: def.name.to_string(),
        model: def.model.clone(),
        per_card,
        measured_at: Utc::now(),
    };
    let path = store_path(data_dir);
    let mut store = load(&path).await;
    store.insert(key(def), record.clone());
    crate::engine::write_json_atomic(&path, &store).await.ok()?;
    Some(record)
}
