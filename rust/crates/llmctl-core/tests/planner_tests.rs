//! Invalid-input and resource-conflict cases first, per the plan's first
//! implementation batch. All tests are pure: fake inventory, fake
//! observations, no host access.

use chrono::{TimeZone, Utc};
use llmctl_core::identity;
use llmctl_core::{
    plan_restart, plan_run, plan_stop, ControlAuthority, CoreError, DesiredIntent, EngineKind,
    GpuId, HostId, HostInventory, Observation, Plan, PlanStep, Platform, ProcessState, Readiness,
    ServedModel, UnitName, UnitScope, WorkloadDefinition, WorkloadName,
};

fn host() -> HostId {
    HostId::new("workstation").unwrap()
}

fn gpu(i: u32) -> GpuId {
    GpuId::new(host(), i)
}

fn workload(name: &str, gpus: &[u32], port: u16) -> WorkloadDefinition {
    WorkloadDefinition {
        name: WorkloadName::new(name).unwrap(),
        engine: "fake".into(),
        engine_kind: EngineKind::FakeEngine,
        model: format!("fake/{name}"),
        gpus: gpus.iter().map(|i| gpu(*i)).collect(),
        port,
        unit: UnitName::systemd(
            identity::UnitScopeMarker::User,
            format!("llmctl-next-fake-{name}.service"),
        )
        .unwrap(),
        authority: ControlAuthority::SystemdUnit {
            scope: UnitScope::User,
            unit: format!("llmctl-next-fake-{name}.service"),
        },
        ownership: Default::default(),
        recovery: None,
        recipe: None,
        readiness_timeout_secs: None,
        gpu_claim: Default::default(),
        hooks: Vec::new(),
        pin_files: Vec::new(),
        observed_version: None,
    }
}

fn inventory() -> HostInventory {
    let i = HostInventory {
        host: host(),
        platform: Platform::Linux,
        gpus: vec![gpu(0), gpu(1)],
        workloads: vec![
            workload("coder", &[0], 8100),
            workload("helper", &[1], 8101),
            workload("big", &[0, 1], 8102),
        ],
        stores: Vec::new(),
        revision: 12,
    };
    i.validate().unwrap();
    i
}

fn obs(name: &str, gpus: &[u32], running: bool) -> Observation {
    Observation {
        workload: WorkloadName::new(name).unwrap(),
        unit: None,
        process: if running {
            ProcessState::Running
        } else {
            ProcessState::Stopped
        },
        readiness: if running {
            Readiness::Ready
        } else {
            Readiness::NotReady
        },
        served_model: if running {
            ServedModel::Verified {
                model: format!("fake/{name}"),
            }
        } else {
            ServedModel::NotServing
        },
        gpus: gpus.iter().map(|i| gpu(*i)).collect(),
        loaded_models: None,
        observed_at: Utc.with_ymd_and_hms(2026, 11, 15, 12, 0, 0).unwrap(),
    }
}

// ---- invalid input -------------------------------------------------------

#[test]
fn run_rejects_unknown_workload() {
    let err = plan_run(&inventory(), &[], &WorkloadName::new("nope").unwrap()).unwrap_err();
    assert!(matches!(err, CoreError::UnknownWorkload(_)), "{err}");
}

#[test]
fn run_rejects_observe_only_authority() {
    let mut inv = inventory();
    inv.workloads[0].authority = ControlAuthority::ObserveOnly {
        endpoint: "http://127.0.0.1:8100".into(),
    };
    let target = WorkloadName::new("coder").unwrap();
    let err = plan_run(&inv, &[], &target).unwrap_err();
    assert!(matches!(err, CoreError::InvalidConfig { .. }), "{err}");
}

#[test]
fn inventory_validation_rejects_empty_gpu_set_and_unknown_gpu() {
    let mut inv = inventory();
    inv.workloads[0].gpus = vec![];
    assert!(matches!(
        inv.validate(),
        Err(CoreError::InvalidConfig { .. })
    ));

    let mut inv2 = inventory();
    let foreign = GpuId::new(HostId::new("mbp2").unwrap(), 9);
    inv2.workloads[0].gpus = vec![foreign];
    assert!(matches!(inv2.validate(), Err(CoreError::UnknownGpu(_))));
}

// ---- resource conflicts --------------------------------------------------

#[test]
fn run_plan_stops_conflicting_workloads_in_name_order() {
    let inv = inventory();
    // helper (gpu 1) and coder (gpu 0) both run; big needs both.
    let observations = vec![obs("coder", &[0], true), obs("helper", &[1], true)];
    let target = WorkloadName::new("big").unwrap();
    let plan = plan_run(&inv, &observations, &target).unwrap();

    let stop_labels: Vec<String> = plan
        .steps
        .iter()
        .filter_map(|s| match s {
            PlanStep::StopUnit { workload, .. } => Some(workload.to_string()),
            _ => None,
        })
        .collect();
    // stable name order: coder, helper
    assert_eq!(stop_labels, vec!["coder", "helper"]);
    assert_eq!(plan.affected_gpus, vec![gpu(0), gpu(1)]);
    assert_eq!(plan.expected_model, "fake/big");
    assert_eq!(plan.revision, 12);
}

#[test]
fn run_plan_ignores_conflicts_that_are_not_running() {
    let inv = inventory();
    let observations = vec![obs("coder", &[0], false), obs("helper", &[1], false)];
    let plan = plan_run(&inv, &observations, &WorkloadName::new("big").unwrap()).unwrap();
    assert!(!plan
        .steps
        .iter()
        .any(|s| matches!(s, PlanStep::StopUnit { .. })));
}

#[test]
fn run_plan_ignores_unknown_process_state_for_conflicts() {
    // Unknown observations must not authorize stopping anything: they are
    // not evidence of "running", and they are not evidence of "free" either.
    let inv = inventory();
    let mut unknown_obs = obs("coder", &[0], true);
    unknown_obs.process = ProcessState::Unknown;
    let observations = vec![unknown_obs, obs("helper", &[1], false)];
    let plan = plan_run(&inv, &observations, &WorkloadName::new("big").unwrap()).unwrap();
    let stops: Vec<&PlanStep> = plan
        .steps
        .iter()
        .filter(|s| matches!(s, PlanStep::StopUnit { .. }))
        .collect();
    // helper isn't running; coder is unknown → nothing is stopped, the plan
    // surfaces the unknown instead of silently evicting.
    assert!(stops.is_empty(), "{stops:?}");
}

#[test]
fn rollback_restarts_evicted_workloads_and_stops_target() {
    let inv = inventory();
    // big (gpu 0+1) evicts coder (gpu 0); rollback restores coder first,
    // then stops the failed target.
    let observations = vec![obs("coder", &[0], true)];
    let target = WorkloadName::new("big").unwrap();
    let plan = plan_run(&inv, &observations, &target).unwrap();
    let rollback: Vec<String> = plan.rollback.iter().map(|s| s.label()).collect();
    assert_eq!(
        rollback,
        vec!["record intent: coder → Running", "start coder", "stop big"]
    );
}

// ---- stop plans ----------------------------------------------------------

#[test]
fn stop_plan_is_manual_and_skips_external_stop_when_not_running() {
    let inv = inventory();
    let plan = plan_stop(&inv, &[], &WorkloadName::new("coder").unwrap()).unwrap();
    assert!(matches!(
        plan.steps.as_slice(),
        [PlanStep::PersistIntent {
            intent: DesiredIntent::Stopped { manual: true },
            ..
        }]
    ));
    assert!(plan.rollback.is_empty());
}

#[test]
fn stop_plan_stops_a_running_workload() {
    let inv = inventory();
    let observations = vec![obs("coder", &[0], true)];
    let plan = plan_stop(&inv, &observations, &WorkloadName::new("coder").unwrap()).unwrap();
    assert_eq!(plan.steps.len(), 2);
    assert!(matches!(plan.steps[1], PlanStep::StopUnit { .. }));
}

// ---- restart plans -------------------------------------------------------

#[test]
fn restart_plan_stops_then_starts_and_reverifies_a_running_target() {
    let inv = inventory();
    let observations = vec![obs("coder", &[0], true)];
    let plan = plan_restart(&inv, &observations, &WorkloadName::new("coder").unwrap()).unwrap();
    assert_eq!(plan.kind, llmctl_core::OpKind::Restart);
    let labels: Vec<String> = plan.steps.iter().map(|s| s.label()).collect();
    assert_eq!(
        labels,
        vec![
            "record intent: coder → Running",
            "stop coder",
            "start coder",
            &plan.steps[3].label(),
            &plan.steps[4].label(),
        ]
    );
    assert!(matches!(plan.steps[3], PlanStep::AwaitReadiness { .. }));
    assert!(matches!(plan.steps[4], PlanStep::VerifyServedModel { .. }));
}

#[test]
fn restart_refuses_a_target_that_is_not_running() {
    let inv = inventory();
    let target = WorkloadName::new("coder").unwrap();
    let err = plan_restart(&inv, &[obs("coder", &[0], false)], &target).unwrap_err();
    assert!(matches!(err, CoreError::NotRunning { .. }), "{err:?}");
    // Unknown process state is not "running" either: never restart blind.
    let mut unknown = obs("coder", &[0], true);
    unknown.process = ProcessState::Unknown;
    let err = plan_restart(&inv, &[unknown], &target).unwrap_err();
    assert!(matches!(err, CoreError::NotRunning { .. }), "{err:?}");
}

// ---- plan identity -------------------------------------------------------

#[test]
fn plans_carry_distinct_ids_and_current_revision() {
    let inv = inventory();
    let a = plan_run(&inv, &[], &WorkloadName::new("coder").unwrap()).unwrap();
    let b = plan_run(&inv, &[], &WorkloadName::new("coder").unwrap()).unwrap();
    assert_ne!(a.id, b.id);
    assert_eq!(a.revision, inv.revision);
}

#[test]
fn run_of_running_target_restarts_it_deterministically() {
    let inv = inventory();
    let observations = vec![obs("coder", &[0], true)];
    let plan = plan_run(&inv, &observations, &WorkloadName::new("coder").unwrap()).unwrap();
    // exactly one stop and one start for the target, start after stop
    let idx = |pred: &dyn Fn(&PlanStep) -> bool| plan.steps.iter().position(pred).unwrap();
    let stop =
        idx(&|s| matches!(s, PlanStep::StopUnit { workload, .. } if workload.as_str() == "coder"));
    let start =
        idx(&|s| matches!(s, PlanStep::StartUnit { workload, .. } if workload.as_str() == "coder"));
    assert!(stop < start);
}

#[test]
fn plan_json_round_trips() {
    let inv = inventory();
    let plan = plan_run(&inv, &[], &WorkloadName::new("coder").unwrap()).unwrap();
    let j = serde_json::to_string(&plan).unwrap();
    let back: Plan = serde_json::from_str(&j).unwrap();
    assert_eq!(back, plan);
}

#[test]
fn readiness_wait_comes_from_the_workload_with_a_default() {
    let readiness = |plan: &Plan| {
        plan.steps
            .iter()
            .find_map(|s| match s {
                PlanStep::AwaitReadiness { timeout_secs, .. } => Some(*timeout_secs),
                _ => None,
            })
            .unwrap()
    };
    let mut inv = inventory();
    let target = WorkloadName::new("coder").unwrap();
    let plan = plan_run(&inv, &[], &target).unwrap();
    assert_eq!(
        readiness(&plan),
        llmctl_core::planner::DEFAULT_READINESS_TIMEOUT_SECS
    );

    // A real model load takes minutes: the inventory says how long.
    inv.workloads[0].readiness_timeout_secs = Some(600);
    let plan = plan_run(&inv, &[], &target).unwrap();
    assert_eq!(readiness(&plan), 600);
    let plan = plan_restart(&inv, &[obs("coder", &[0], true)], &target).unwrap();
    assert_eq!(readiness(&plan), 600);
}

// ---- M4b: engine-level unload ------------------------------------------------

#[test]
fn unload_is_planned_only_for_a_running_vendor_service() {
    let mut inv = inventory();
    // An OpenAI-style engine: the model is the process — refused by kind.
    let err = llmctl_core::plan_unload(&inv, &[obs("coder", &[0], true)], &wn("coder"), "m")
        .unwrap_err();
    assert!(matches!(err, CoreError::InvalidConfig { .. }), "{err:?}");

    inv.workloads[0].engine_kind = EngineKind::Ollama;
    inv.workloads[0].model = "ollama".into();
    // Running Ollama: one external step, no intent change, no rollback.
    let plan =
        llmctl_core::plan_unload(&inv, &[obs("coder", &[0], true)], &wn("coder"), "qwen3:8b")
            .unwrap();
    assert_eq!(plan.kind, llmctl_core::OpKind::Unload);
    assert!(plan.rollback.is_empty() && plan.affected_gpus.is_empty());
    assert!(matches!(
        plan.steps.as_slice(),
        [PlanStep::UnloadModel { model, .. }] if model == "qwen3:8b"
    ));
    // The service identity, an empty name, or a stopped service: refused.
    for bad in ["ollama", "", "  "] {
        assert!(llmctl_core::plan_unload(&inv, &[obs("coder", &[0], true)], &wn("coder"), bad)
            .is_err());
    }
    let err = llmctl_core::plan_unload(&inv, &[obs("coder", &[0], false)], &wn("coder"), "qwen3:8b")
        .unwrap_err();
    assert!(matches!(err, CoreError::NotRunning { .. }), "{err:?}");
}

fn wn(s: &str) -> WorkloadName {
    WorkloadName::new(s).unwrap()
}

// ---- M4c: on-demand GPU claims ------------------------------------------------

#[test]
fn on_demand_services_neither_evict_nor_get_evicted() {
    use llmctl_core::GpuClaim;
    let mut inv = inventory();
    // `helper` (GPU 1) becomes an on-demand vendor service on both cards.
    inv.workloads[1].gpus = vec![gpu(0), gpu(1)];
    inv.workloads[1].gpu_claim = GpuClaim::OnDemand;
    let all_up = [obs("coder", &[0], true), obs("helper", &[0, 1], true), obs("big", &[], false)];

    // Starting the two-card engine stops coder (at_start) but never helper.
    let plan = plan_run(&inv, &all_up, &wn("big")).unwrap();
    let stopped: Vec<String> = plan
        .steps
        .iter()
        .filter_map(|s| match s {
            PlanStep::StopUnit { workload, .. } => Some(workload.to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(stopped, vec!["coder"]);

    // Starting the on-demand service stops nobody, although it shares cards.
    let plan = plan_run(&inv, &[obs("coder", &[0], true), obs("helper", &[], false)], &wn("helper")).unwrap();
    assert!(!plan.steps.iter().any(|s| matches!(s, PlanStep::StopUnit { .. })));
    assert!(plan.rollback.iter().all(|s| s.workload() == &wn("helper")));

    assert_eq!(GpuClaim::default_for(&EngineKind::LmStudio), GpuClaim::OnDemand);
    assert_eq!(GpuClaim::default_for(&EngineKind::Ollama), GpuClaim::OnDemand);
    assert_eq!(GpuClaim::default_for(&EngineKind::Vllm), GpuClaim::AtStart);
}

// ---- M4c: swap plans (unload on-demand models, then evict) ------------------

/// A running observation with an engine-reported loaded list.
fn obs_loaded(name: &str, gpus: &[u32], models: &[&str]) -> Observation {
    let mut o = obs(name, gpus, true);
    o.loaded_models = Some(models.iter().map(|m| m.to_string()).collect());
    o
}

/// `helper` as an on-demand Ollama service that may use both cards.
fn inventory_with_on_demand_helper() -> HostInventory {
    let mut inv = inventory();
    inv.workloads[1].gpus = vec![gpu(0), gpu(1)];
    inv.workloads[1].gpu_claim = llmctl_core::GpuClaim::OnDemand;
    inv.workloads[1].engine_kind = EngineKind::Ollama;
    inv.workloads[1].model = "ollama".into();
    inv.validate().unwrap();
    inv
}

fn labels(steps: &[PlanStep]) -> Vec<String> {
    steps.iter().map(|s| s.label()).collect()
}

#[test]
fn swap_plan_unloads_on_demand_models_before_stopping_conflicts() {
    let inv = inventory_with_on_demand_helper();
    let observations = [
        obs("coder", &[0], true),
        obs_loaded("helper", &[0, 1], &["qwen3:8b", "nomic-embed"]),
        obs("big", &[], false),
    ];
    let plan = plan_run(&inv, &observations, &wn("big")).unwrap();
    // One plan: unload (model-name order), evict, start, verify. The
    // on-demand service itself is never stopped.
    assert_eq!(
        labels(&plan.steps),
        vec![
            "record intent: big → Running",
            "unload nomic-embed from helper",
            "unload qwen3:8b from helper",
            "record intent: coder → Stopped { manual: false }",
            "stop coder",
            "start big",
            "wait for big readiness",
            "verify big serves fake/big",
        ]
    );
    // Rollback restores the evicted at_start set. Unloaded models are not
    // reloaded: an on-demand service loads them again on its next request.
    assert_eq!(
        labels(&plan.rollback),
        vec!["record intent: coder → Running", "start coder", "stop big"]
    );

    // A restart of the running target is the same swap.
    let observations = [
        obs("big", &[0, 1], true),
        obs_loaded("helper", &[0, 1], &["qwen3:8b"]),
    ];
    let plan = plan_restart(&inv, &observations, &wn("big")).unwrap();
    assert_eq!(
        labels(&plan.steps)[..4],
        [
            "record intent: big → Running",
            "unload qwen3:8b from helper",
            "stop big",
            "start big",
        ]
    );
}

#[test]
fn swap_plan_unloads_only_what_it_can_name_and_needs() {
    let unloads = |inv: &HostInventory, observations: &[Observation], target: &str| {
        plan_run(inv, observations, &wn(target))
            .unwrap()
            .steps
            .iter()
            .filter(|s| matches!(s, PlanStep::UnloadModel { .. }))
            .count()
    };
    let inv = inventory_with_on_demand_helper();

    // Loaded list unknown or empty, or the service is down: nothing to name.
    // (The daemon's occupancy guard still refuses when the card is held.)
    assert_eq!(unloads(&inv, &[obs("helper", &[0, 1], true)], "big"), 0);
    assert_eq!(unloads(&inv, &[obs_loaded("helper", &[0, 1], &[])], "big"), 0);
    let mut down = obs_loaded("helper", &[], &["qwen3:8b"]);
    down.process = ProcessState::Stopped;
    assert_eq!(unloads(&inv, &[down], "big"), 0);
    let mut unknown = obs_loaded("helper", &[0, 1], &["qwen3:8b"]);
    unknown.process = ProcessState::Unknown;
    assert_eq!(unloads(&inv, &[unknown], "big"), 0);

    // No shared card: the service keeps its models.
    let mut apart = inv.clone();
    apart.workloads[1].gpus = vec![gpu(1)];
    let loaded = [obs_loaded("helper", &[1], &["qwen3:8b"])];
    assert_eq!(unloads(&apart, &loaded, "coder"), 0);
    assert_eq!(unloads(&apart, &loaded, "big"), 1);

    // An on-demand target claims nothing at start: it unloads nobody, and
    // its own models are not unloaded by its own run.
    let mut both = inv.clone();
    both.workloads[0].gpu_claim = llmctl_core::GpuClaim::OnDemand;
    let loaded = [obs_loaded("helper", &[0, 1], &["qwen3:8b"])];
    assert_eq!(unloads(&both, &loaded, "coder"), 0);
    assert_eq!(unloads(&inv, &loaded, "helper"), 0);

    // No engine-level unload, or no lifecycle authority: never planned.
    let mut no_unload = inv.clone();
    no_unload.workloads[1].engine_kind = EngineKind::FakeEngine;
    assert_eq!(unloads(&no_unload, &loaded, "big"), 0);
    let mut observe_only = inv.clone();
    observe_only.workloads[1].authority = ControlAuthority::ObserveOnly {
        endpoint: "http://127.0.0.1:8101".into(),
    };
    assert_eq!(unloads(&observe_only, &loaded, "big"), 0);
}

#[test]
fn observations_without_a_loaded_list_still_parse() {
    // Journaled plans and older agents carry observations without the field.
    let mut v = serde_json::to_value(obs("coder", &[0], true)).unwrap();
    assert!(v.get("loaded_models").is_none(), "{v}");
    v.as_object_mut().unwrap().remove("loaded_models");
    let back: Observation = serde_json::from_value(v).unwrap();
    assert_eq!(back.loaded_models, None);
}

// ---- M4c: env-file profiles ---------------------------------------------------

fn profile(model: &str) -> llmctl_core::ProfileSwap {
    llmctl_core::ProfileSwap {
        path: "/srv/coder.env".into(),
        source: "/srv/coder.env.fast".into(),
        source_digest: "sha256:new".into(),
        current_digest: "sha256:old".into(),
        backup: "/srv/coder.env.llmctl-previous".into(),
        model: model.into(),
    }
}

#[test]
fn profile_run_swaps_the_env_between_stop_and_start_and_verifies_the_new_model() {
    let inv = inventory();
    // coder serves; helper is an at_start neighbour on another card.
    let observations = [obs("coder", &[0], true), obs("helper", &[1], true)];
    let plan =
        llmctl_core::plan_run_profile(&inv, &observations, &wn("coder"), &profile("fake/coder-fast"))
            .unwrap();
    assert_eq!(plan.kind, llmctl_core::OpKind::Run);
    assert_eq!(plan.expected_model, "fake/coder-fast");
    assert_eq!(
        labels(&plan.steps),
        vec![
            "record intent: coder → Running",
            "stop coder",
            "apply profile coder.env.fast to coder (serves fake/coder-fast)",
            "start coder",
            "wait for coder readiness",
            "verify coder serves fake/coder-fast",
        ]
    );
    // The way back: stop the failed start, put the old env back, and bring
    // the target up again on what it served before, verified.
    assert_eq!(
        labels(&plan.rollback),
        vec![
            "stop coder",
            "restore the previous env of coder (serves fake/coder)",
            "start coder",
            "wait for coder readiness",
            "verify coder serves fake/coder",
        ]
    );
    // The restore names the backup the forward step writes, and expects the
    // bytes that were live when the plan was made.
    let forward = plan.steps.iter().find_map(|s| match s {
        PlanStep::ApplyEnvFile { source, source_digest, backup, path, .. } => {
            Some((source.clone(), source_digest.clone(), backup.clone(), path.clone()))
        }
        _ => None,
    });
    assert_eq!(
        forward,
        Some((
            "/srv/coder.env.fast".into(),
            "sha256:new".into(),
            Some("/srv/coder.env.llmctl-previous".into()),
            "/srv/coder.env".into()
        ))
    );
    let restore = plan.rollback.iter().find_map(|s| match s {
        PlanStep::ApplyEnvFile { source, source_digest, backup, .. } => {
            Some((source.clone(), source_digest.clone(), backup.clone()))
        }
        _ => None,
    });
    assert_eq!(
        restore,
        Some(("/srv/coder.env.llmctl-previous".into(), "sha256:old".into(), None))
    );
}

#[test]
fn profile_run_of_a_stopped_target_evicts_and_restores_like_any_run() {
    let inv = inventory();
    // big (both cards) is down; coder holds card 0.
    let observations = [obs("coder", &[0], true), obs("big", &[], false)];
    let plan =
        llmctl_core::plan_run_profile(&inv, &observations, &wn("big"), &profile("fake/big-awq"))
            .unwrap();
    assert_eq!(
        labels(&plan.steps),
        vec![
            "record intent: big → Running",
            "record intent: coder → Stopped { manual: false }",
            "stop coder",
            "apply profile coder.env.fast to big (serves fake/big-awq)",
            "start big",
            "wait for big readiness",
            "verify big serves fake/big-awq",
        ]
    );
    // The target was not serving before: it stays down, the old env goes
    // back, and the evicted workload returns.
    assert_eq!(
        labels(&plan.rollback),
        vec![
            "stop big",
            "restore the previous env of big (serves fake/big)",
            "record intent: coder → Running",
            "start coder",
        ]
    );
    // The plan still round-trips through the journal.
    let back: Plan = serde_json::from_str(&serde_json::to_string(&plan).unwrap()).unwrap();
    assert_eq!(back, plan);
}

#[test]
fn profile_run_needs_a_model_and_lifecycle_authority() {
    let mut inv = inventory();
    let err = llmctl_core::plan_run_profile(&inv, &[], &wn("coder"), &profile("  ")).unwrap_err();
    assert!(matches!(err, CoreError::InvalidConfig { .. }), "{err:?}");
    let err = llmctl_core::plan_run_profile(&inv, &[], &wn("nope"), &profile("m")).unwrap_err();
    assert!(matches!(err, CoreError::UnknownWorkload(_)), "{err:?}");
    inv.workloads[0].authority = ControlAuthority::ObserveOnly {
        endpoint: "http://127.0.0.1:8100".into(),
    };
    let err = llmctl_core::plan_run_profile(&inv, &[], &wn("coder"), &profile("m")).unwrap_err();
    assert!(matches!(err, CoreError::InvalidConfig { .. }), "{err:?}");
}

// ---- M4c: VRAM fit -------------------------------------------------------------

#[test]
fn a_run_that_fits_keeps_its_neighbours() {
    let inv = inventory_with_on_demand_helper();
    let observations = [
        obs("coder", &[0], true),
        obs_loaded("helper", &[0, 1], &["qwen3:8b"]),
        obs("big", &[], false),
    ];
    // The caller decided there is room: nothing is unloaded or stopped, and
    // a failed start only takes the target down again.
    let plan = llmctl_core::plan_run_with(&inv, &observations, &wn("big"), true).unwrap();
    assert_eq!(
        labels(&plan.steps),
        vec![
            "record intent: big → Running",
            "start big",
            "wait for big readiness",
            "verify big serves fake/big",
        ]
    );
    assert_eq!(labels(&plan.rollback), vec!["stop big"]);

    // No room: the plan evicts as before.
    let plan = llmctl_core::plan_run_with(&inv, &observations, &wn("big"), false).unwrap();
    assert!(plan.steps.iter().any(|s| matches!(s, PlanStep::StopUnit { .. })));
    assert!(plan.steps.iter().any(|s| matches!(s, PlanStep::UnloadModel { .. })));

    // A restart that fits still restarts the target itself, and only it.
    let observations = [obs("coder", &[0], true), obs("big", &[0, 1], true)];
    let plan = llmctl_core::plan_restart_with(&inv, &observations, &wn("big"), true).unwrap();
    assert_eq!(labels(&plan.steps)[..3], ["record intent: big → Running", "stop big", "start big"]);
}

// ---- M4c: companion hooks ------------------------------------------------------

fn hook(name: &str, when: llmctl_core::HookWhen, undo: bool) -> llmctl_core::Hook {
    llmctl_core::Hook {
        name: name.into(),
        when,
        argv: vec!["/usr/bin/true".into(), name.into()],
        undo: if undo { vec!["/usr/bin/true".into(), format!("undo-{name}")] } else { vec![] },
        timeout_secs: 30,
        optional: false,
    }
}

#[test]
fn hooks_wrap_every_start_and_stop_and_their_undo_ends_the_rollback() {
    use llmctl_core::HookWhen::*;
    let mut inv = inventory();
    // coder (evicted): hooks around its stop, and around the start a
    // rollback gives it back. big (the target): hooks around its start.
    inv.workloads[0].hooks = vec![
        hook("drain", PreStop, false),
        hook("detach", PostStop, false),
        hook("adopt", PostStart, false),
    ];
    inv.workloads[2].hooks = vec![hook("free-vram", PreStart, true), hook("announce", PostStart, true)];
    inv.validate().unwrap();
    let plan = plan_run(&inv, &[obs("coder", &[0], true)], &wn("big")).unwrap();
    assert_eq!(
        labels(&plan.steps),
        vec![
            "record intent: big → Running",
            "record intent: coder → Stopped { manual: false }",
            "hook drain (pre_stop of coder)",
            "stop coder",
            "hook detach (post_stop of coder)",
            "hook free-vram (pre_start of big)",
            "start big",
            "wait for big readiness",
            "verify big serves fake/big",
            // post_start waits for the verification of the target.
            "hook announce (post_start of big)",
        ]
    );
    assert_eq!(
        labels(&plan.rollback),
        vec![
            "record intent: coder → Running",
            "start coder",
            // Nothing verifies coder in a rollback: right after its start.
            "hook adopt (post_start of coder)",
            "stop big",
            // The undo of each forward hook, last run first.
            "hook announce (undo of big)",
            "hook free-vram (undo of big)",
        ]
    );
    let undo = plan.rollback.iter().find_map(|s| match s {
        PlanStep::RunHook { name, argv, .. } if name == "free-vram" => Some(argv.clone()),
        _ => None,
    });
    assert_eq!(undo, Some(vec!["/usr/bin/true".to_string(), "undo-free-vram".to_string()]));

    // A stop runs the stop hooks; a profile switch keeps the target's hooks
    // around the swapped start, and in its rollback restart.
    let plan = plan_stop(&inv, &[obs("coder", &[0], true)], &wn("coder")).unwrap();
    assert_eq!(
        labels(&plan.steps)[1..],
        ["hook drain (pre_stop of coder)", "stop coder", "hook detach (post_stop of coder)"]
    );
    let plan =
        llmctl_core::plan_run_profile(&inv, &[obs("big", &[0, 1], true)], &wn("big"), &profile("fake/big-awq"))
            .unwrap();
    let l = labels(&plan.steps);
    let at = |needle: &str| l.iter().position(|x| x.contains(needle)).unwrap_or_else(|| panic!("{needle} in {l:?}"));
    // The pre_start hook stays right before the start, after the env swap.
    assert!(at("apply profile") < at("hook free-vram") && at("hook free-vram") < at("start big"), "{l:?}");
    assert!(at("verify big") < at("hook announce (post_start"), "{l:?}");
    let r = labels(&plan.rollback);
    assert!(r.contains(&"hook free-vram (pre_start of big)".to_string()), "{r:?}");
    assert_eq!(r.last().map(String::as_str), Some("hook free-vram (undo of big)"), "{r:?}");
}

#[test]
fn hook_declarations_are_validated() {
    use llmctl_core::HookWhen::PreStart;
    let with = |f: &dyn Fn(&mut llmctl_core::Hook)| {
        let mut inv = inventory();
        let mut h = hook("h", PreStart, true);
        f(&mut h);
        inv.workloads[0].hooks = vec![h];
        inv.validate()
    };
    assert!(with(&|_| {}).is_ok());
    for (why, bad) in [
        ("relative program", with(&|h| h.argv[0] = "true".into())),
        ("no command", with(&|h| h.argv.clear())),
        ("relative undo", with(&|h| h.undo[0] = "sh".into())),
        ("zero timeout", with(&|h| h.timeout_secs = 0)),
        ("timeout too long", with(&|h| h.timeout_secs = 3601)),
        ("empty name", with(&|h| h.name = " ".into())),
    ] {
        assert!(matches!(bad, Err(CoreError::InvalidConfig { .. })), "{why}: {bad:?}");
    }
    let mut inv = inventory();
    inv.workloads[0].hooks = vec![hook("same", PreStart, false), hook("same", PreStart, false)];
    assert!(inv.validate().is_err(), "duplicate hook names");
    // Records without hooks stay as they were on disk.
    let json = serde_json::to_value(&inventory().workloads[0]).unwrap();
    assert!(json.get("hooks").is_none(), "{json}");
}
