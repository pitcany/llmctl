//! Invalid-input and resource-conflict cases first, per the plan's first
//! implementation batch. All tests are pure: fake inventory, fake
//! observations, no host access.

use chrono::{TimeZone, Utc};
use llmctl_core::identity;
use llmctl_core::{
    plan_restart, plan_run, plan_stop, ControlAuthority, CoreError, DesiredIntent, GpuId, HostId,
    HostInventory, Observation, Plan, PlanStep, Platform, ProcessState, Readiness, ServedModel,
    UnitName, UnitScope, WorkloadDefinition, WorkloadName,
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
