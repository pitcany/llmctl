//! Version pins: what an install reported when the owner last accepted
//! it. `version <name>` compares a fresh probe with the pin; on systemd
//! the start guard refuses a managed workload whose facts moved off it.
//!
//! A pin is written in two places: at daemon start, for a workload that
//! has a recipe and no pin yet (`app::repin_legacy_versions`), and by
//! `version <name> --repin`, which is how the owner accepts a change (a
//! vendor upgrade, a package installed into the env).

use std::sync::Arc;

use serde_json::{json, Value};

use llmctl_core::{InstallRecipe, Ownership, VersionRef, WorkloadDefinition};

use crate::app::{AppState, InventoryFile};
use crate::ops::OpsError;

/// Recipe kinds whose pin carries a per-kind fact (schema 3).
fn has_per_kind_fact(recipe: Option<&InstallRecipe>) -> bool {
    matches!(
        recipe,
        Some(InstallRecipe::ReleaseArchive(_))
            | Some(InstallRecipe::SourceBuild(_))
            | Some(InstallRecipe::VendorApp(_))
    )
}

/// The pin a workload carries now: next to the definition when llmctl
/// manages it, else the recorded observation.
pub fn pinned_version(def: &WorkloadDefinition) -> Option<&VersionRef> {
    match &def.ownership {
        Ownership::Managed { version, .. } => version.as_ref(),
        Ownership::External => def.observed_version.as_ref(),
    }
}

/// Store `live` as the workload's pin. Returns false when the pin is
/// incomplete: a vendor server that is not answering has no running
/// version, so that fact keeps its previous value (or stays unpinned) and
/// the next daemon start, or the next `--repin` while it serves, fills it.
pub fn apply_pin(w: &mut WorkloadDefinition, live: VersionRef) -> bool {
    let (mut pin, complete) = crate::version::storable(live);
    if !complete {
        pin.fact = pinned_version(w).and_then(|p| p.fact.clone());
    }
    let per_kind = has_per_kind_fact(w.recipe.as_ref());
    match &mut w.ownership {
        Ownership::Managed {
            version,
            definition_schema,
            ..
        } => {
            // Schema 3 says "the per-kind fact is pinned"; without it the
            // startup pin keeps trying.
            *definition_schema = if per_kind && pin.fact.is_some() { 3 } else { 2 };
            *version = Some(pin);
        }
        Ownership::External => w.observed_version = Some(pin),
    }
    complete
}

/// `version <name> --repin`: probe the install and record what it reports
/// as the pin. The answer names every fact that changed, so accepting a
/// change is never silent.
pub async fn repin(state: &AppState, name: &str) -> Result<Value, OpsError> {
    let wl = llmctl_core::WorkloadName::new(name).map_err(|e| OpsError::Invalid(e.to_string()))?;
    let _quiet = state.ops.quiesce().await?;
    let inv = state.ops.inventory().await;
    let Some(def) = inv.workload(&wl) else {
        return Err(OpsError::UnknownWorkload(name.to_string()));
    };
    let Some(recipe) = def.recipe.clone() else {
        return Err(OpsError::Invalid(format!(
            "{name} has no recipe, so there is nothing to probe; add one to its inventory record first"
        )));
    };
    let live = crate::version::probe(&recipe, def)
        .await
        .map_err(|e| OpsError::Engine(format!("probing {name}: {e}")))?;
    let previous = pinned_version(def).cloned();

    let mut next = (*inv).clone();
    let w = next
        .workloads
        .iter_mut()
        .find(|w| w.name == wl)
        .expect("workload exists in the clone");
    let complete = apply_pin(w, live);
    let pinned = pinned_version(w).cloned();
    let changes = match (&previous, &pinned) {
        (Some(old), Some(new)) => {
            let mut diffs = crate::version::version_drift(old, new);
            // `version_drift` skips the fact when the pin has none.
            if old.fact.is_none() && new.fact.is_some() {
                diffs.push(llmctl_core::FieldDiff {
                    field: "version.fact".into(),
                    expected: Value::Null,
                    actual: serde_json::to_value(&new.fact).unwrap_or(Value::Null),
                });
            }
            diffs
        }
        _ => Vec::new(),
    };
    let first_pin = previous.is_none();
    if !first_pin && changes.is_empty() {
        return Ok(json!({
            "workload": name,
            "repinned": false,
            "complete": complete,
            "pinned": pinned,
            "changes": [],
            "inventory_revision": inv.revision,
            "note": "the install reports what the pin says; nothing was written",
        }));
    }
    next.revision += 1;
    next.validate()
        .map_err(|e| OpsError::Invalid(format!("re-pinned inventory invalid: {e}")))?;
    crate::engine::write_json_atomic(
        &state.paths.inventory,
        &InventoryFile {
            inventory: next.clone(),
        },
    )
    .await
    .map_err(|e| OpsError::Engine(e.to_string()))?;
    let revision = next.revision;
    state.ops.update_inventory(Arc::new(next)).await;
    tracing::info!(workload = %name, revision, first_pin, changes = changes.len(), "version pin recorded");
    Ok(json!({
        "workload": name,
        "repinned": true,
        "complete": complete,
        "previous": previous,
        "pinned": pinned,
        "changes": changes,
        "inventory_revision": revision,
        "note": if complete {
            Value::Null
        } else {
            Value::String(
                "the server is not answering, so its running version was not pinned; \
                 re-pin while it serves"
                    .into(),
            )
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmctl_core::{VendorAppRecipe, VendorProbe, VersionFact};

    fn vendor_ref(running: Option<&str>) -> VersionRef {
        VersionRef {
            interpreter: None,
            packages: Vec::new(),
            cuda: None,
            lock_digest: String::new(),
            exec_digests: vec![("/usr/local/bin/app".into(), "sha256:aa".into())],
            fact: Some(VersionFact::VendorApp {
                running: running.map(String::from),
                probe: "http_json /api/version".into(),
            }),
        }
    }

    fn vendor_workload() -> WorkloadDefinition {
        let host = llmctl_core::HostId::new("workstation").unwrap();
        let mut w = crate::app::sample_inventory(&host).workloads.remove(0);
        w.recipe = Some(InstallRecipe::VendorApp(VendorAppRecipe {
            name: "app".into(),
            probe: VendorProbe::HttpJson {
                path: "/api/version".into(),
                field: "version".into(),
            },
            backup_path: None,
            reinstall_cmd: None,
        }));
        w.ownership = Ownership::Managed {
            definition: Box::default(),
            digest: "sha256:00".into(),
            adopted_at: chrono::Utc::now(),
            provenance: vec![],
            service_file: None,
            version: None,
            definition_schema: 1,
            pending_apply: None,
        };
        w
    }

    fn schema_and_fact(w: &WorkloadDefinition) -> (u32, Option<VersionFact>) {
        match &w.ownership {
            Ownership::Managed {
                definition_schema,
                version,
                ..
            } => (*definition_schema, version.as_ref().and_then(|v| v.fact.clone())),
            Ownership::External => (0, w.observed_version.as_ref().and_then(|v| v.fact.clone())),
        }
    }

    #[test]
    fn a_stopped_vendor_server_pins_its_files_and_leaves_the_version_open() {
        let mut w = vendor_workload();
        assert!(!apply_pin(&mut w, vendor_ref(None)), "incomplete");
        let (schema, fact) = schema_and_fact(&w);
        assert_eq!(schema, 2, "below 3, so the next start tries again");
        assert_eq!(fact, None, "no version was observed, so none is pinned");
        assert_eq!(pinned_version(&w).unwrap().exec_digests.len(), 1);

        // Once it serves, the version is pinned and the schema says so.
        assert!(apply_pin(&mut w, vendor_ref(Some("0.33.1"))));
        let (schema, fact) = schema_and_fact(&w);
        assert_eq!(schema, 3);
        assert!(matches!(fact, Some(VersionFact::VendorApp { running: Some(v), .. }) if v == "0.33.1"));
    }

    #[test]
    fn a_repin_while_stopped_keeps_the_last_pinned_version() {
        let mut w = vendor_workload();
        apply_pin(&mut w, vendor_ref(Some("0.33.1")));
        assert!(!apply_pin(&mut w, vendor_ref(None)));
        let (schema, fact) = schema_and_fact(&w);
        assert_eq!(schema, 3);
        assert!(matches!(fact, Some(VersionFact::VendorApp { running: Some(v), .. }) if v == "0.33.1"));
    }

    #[test]
    fn an_unmanaged_workload_keeps_its_pin_on_the_record() {
        let mut w = vendor_workload();
        w.ownership = Ownership::External;
        assert!(apply_pin(&mut w, vendor_ref(Some("1.0"))));
        assert!(w.observed_version.is_some());
        assert_eq!(pinned_version(&w), w.observed_version.as_ref());
    }
    #[test]
    fn a_refusal_over_version_facts_alone_names_the_way_to_accept_them() {
        let diff = |field: &str| llmctl_core::FieldDiff {
            field: field.into(),
            expected: Value::Null,
            actual: Value::Null,
        };
        let version_only = OpsError::DefinitionDrift {
            workload: "vllm-engine".into(),
            diffs: vec![diff("version.lock_digest"), diff("version.exec_digests")],
        }
        .to_string();
        assert!(version_only.contains("llmctl-next version vllm-engine --repin"), "{version_only}");

        // A changed unit definition is not something to wave through.
        let definition = OpsError::DefinitionDrift {
            workload: "vllm-engine".into(),
            diffs: vec![diff("exec_start"), diff("version.lock_digest")],
        }
        .to_string();
        assert!(!definition.contains("--repin"), "{definition}");
    }

}
