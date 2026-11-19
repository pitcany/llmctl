//! Edits to a llama-swap `matrix:` (Rust M5c item 4, M4 carry-over 2):
//! which lanes may run together on a Mac, changed by verb so the memory
//! budget guards the write (`lanes::plan` refuses a set that would go over
//! it) instead of only reporting.
//!
//! Pure record changes; llama-swap's own rules (var keys 1–8 alphanumeric,
//! known lanes, positive costs, the set DSL) are checked afterwards by
//! `swap::validate`, which the plan runs on the result.

use serde::{Deserialize, Serialize};

use crate::swap::{Matrix, SwapConfig};

/// One change to the matrix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum MatrixEdit {
    /// `vars: {var: lane}` — add or re-point a short id.
    SetVar { var: String, lane: String },
    /// Drop a var (and its eviction cost).
    RemoveVar { var: String },
    /// `sets: {name: expr}` — add or replace a set (`a & b | c`).
    SetSet { name: String, expr: String },
    RemoveSet { name: String },
    /// `evict_costs: {var: cost}`; `None` removes the entry.
    SetEvictCost { var: String, cost: Option<i64> },
}

fn upsert(list: &mut Vec<(String, String)>, k: &str, v: &str) {
    match list.iter_mut().find(|(x, _)| x == k) {
        Some(slot) => slot.1 = v.to_string(),
        None => list.push((k.to_string(), v.to_string())),
    }
}

fn plain(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

/// Apply one edit; returns a one-line summary. `Err` for a request that
/// cannot mean anything (unknown lane, missing entry, a `groups:` config).
pub fn apply(cfg: &mut SwapConfig, e: &MatrixEdit) -> Result<String, String> {
    if cfg.matrix.is_none() {
        if cfg.globals.iter().any(|(k, _)| k == "groups") {
            return Err("this config schedules with `groups:`; a matrix would replace it — edit the file by hand".into());
        }
        if !matches!(e, MatrixEdit::SetVar { .. } | MatrixEdit::SetSet { .. }) {
            return Err("this config has no matrix".into());
        }
        cfg.matrix = Some(Matrix {
            vars: Vec::new(),
            evict_costs: Vec::new(),
            sets: Vec::new(),
            order: vec!["vars".into(), "sets".into()],
            extra: Vec::new(),
        });
        if !cfg.top_order.iter().any(|k| k == "matrix") {
            // Before `models:`, where llama-swap's docs put it.
            let at = cfg.top_order.iter().position(|k| k == "models").unwrap_or(cfg.top_order.len());
            cfg.top_order.insert(at, "matrix".into());
        }
    }
    let lanes: Vec<String> = cfg.lanes.iter().map(|l| l.id.clone()).collect();
    let m = cfg.matrix.as_mut().expect("matrix exists");
    let summary = match e {
        MatrixEdit::SetVar { var, lane } => {
            if !lanes.contains(lane) {
                return Err(format!("no lane {lane:?}"));
            }
            if !plain(var) {
                return Err(format!("var {var:?} is not a plain id"));
            }
            upsert(&mut m.vars, var, lane);
            format!("matrix var {var} = {lane}")
        }
        MatrixEdit::RemoveVar { var } => {
            let before = m.vars.len();
            m.vars.retain(|(k, _)| k != var);
            if m.vars.len() == before {
                return Err(format!("the matrix has no var {var:?}"));
            }
            m.evict_costs.retain(|(k, _)| k != var);
            format!("matrix: remove var {var}")
        }
        MatrixEdit::SetSet { name, expr } => {
            if !plain(name) {
                return Err(format!("set name {name:?} is not a plain id"));
            }
            let expr = expr.trim();
            if expr.is_empty() {
                return Err(format!("set {name}: empty expression"));
            }
            upsert(&mut m.sets, name, expr);
            format!("matrix set {name} = {expr}")
        }
        MatrixEdit::RemoveSet { name } => {
            let before = m.sets.len();
            m.sets.retain(|(k, _)| k != name);
            if m.sets.len() == before {
                return Err(format!("the matrix has no set {name:?}"));
            }
            format!("matrix: remove set {name}")
        }
        MatrixEdit::SetEvictCost { var, cost } => {
            if !m.vars.iter().any(|(k, _)| k == var) {
                return Err(format!("the matrix has no var {var:?}"));
            }
            match cost {
                Some(c) => {
                    match m.evict_costs.iter_mut().find(|(k, _)| k == var) {
                        Some(slot) => slot.1 = serde_yaml::Value::from(*c),
                        None => m.evict_costs.push((var.clone(), serde_yaml::Value::from(*c))),
                    }
                    if !m.order.iter().any(|k| k == "evict_costs") {
                        let at = m.order.iter().position(|k| k == "sets").unwrap_or(m.order.len());
                        m.order.insert(at, "evict_costs".into());
                    }
                    format!("matrix evict cost {var} = {c}")
                }
                None => {
                    m.evict_costs.retain(|(k, _)| k != var);
                    format!("matrix: remove evict cost {var}")
                }
            }
        }
    };
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swap::{self, BudgetReport};

    const CFG: &str = "macros:\n  m: /bin/llama-server\nmatrix:\n  vars:\n    a: lane-a\n    b: lane-b\n  sets:\n    solo: a | b\nmodels:\n  lane-a:\n    cmd: ${m} -m a\n  lane-b:\n    cmd: ${m} -m b\n  lane-c:\n    cmd: ${m} -m c\n";

    fn cfg(text: &str) -> SwapConfig {
        swap::import(text).expect("imports")
    }

    #[test]
    fn edits_change_the_records_and_render() {
        let mut c = cfg(CFG);
        apply(&mut c, &MatrixEdit::SetVar { var: "c".into(), lane: "lane-c".into() }).unwrap();
        apply(&mut c, &MatrixEdit::SetSet { name: "pair".into(), expr: "a & c".into() }).unwrap();
        apply(&mut c, &MatrixEdit::SetEvictCost { var: "c".into(), cost: Some(5) }).unwrap();
        assert!(swap::validate(&c).is_empty(), "{:?}", swap::validate(&c));
        let text = swap::check_render(&c).unwrap();
        assert!(text.contains("    c: lane-c\n"), "{text}");
        assert!(text.contains("  evict_costs:\n    c: 5\n  sets:\n"), "{text}");
        assert!(text.contains("    pair: a & c\n"), "{text}");
        // Remove what was added: the original render comes back.
        apply(&mut c, &MatrixEdit::RemoveSet { name: "pair".into() }).unwrap();
        apply(&mut c, &MatrixEdit::RemoveVar { var: "c".into() }).unwrap();
        assert!(c.matrix.as_ref().unwrap().evict_costs.is_empty(), "the var's cost goes with it");
        assert!(!swap::check_render(&c).unwrap().contains("lane-c\n    c"));

        let err = |e: MatrixEdit| apply(&mut cfg(CFG), &e).unwrap_err();
        assert!(err(MatrixEdit::SetVar { var: "z".into(), lane: "nope".into() }).contains("no lane"));
        assert!(err(MatrixEdit::RemoveSet { name: "x".into() }).contains("no set"));
        assert!(err(MatrixEdit::SetEvictCost { var: "q".into(), cost: Some(1) }).contains("no var"));
        // A var still used by a set is llama-swap's refusal, through validate.
        let mut c = cfg(CFG);
        apply(&mut c, &MatrixEdit::RemoveVar { var: "b".into() }).unwrap();
        assert!(swap::validate(&c).iter().any(|v| v.contains("unknown var")), "{:?}", swap::validate(&c));
    }

    #[test]
    fn a_config_without_matrix_gets_one_and_groups_are_refused() {
        let plain = "models:\n  lane-a:\n    cmd: x\n";
        let mut c = cfg(plain);
        apply(&mut c, &MatrixEdit::SetVar { var: "a".into(), lane: "lane-a".into() }).unwrap();
        apply(&mut c, &MatrixEdit::SetSet { name: "s".into(), expr: "a".into() }).unwrap();
        let text = swap::check_render(&c).unwrap();
        assert!(text.starts_with("matrix:\n  vars:\n    a: lane-a\n  sets:\n    s: a\nmodels:"), "{text}");
        let mut g = cfg("groups:\n  g:\n    members: [lane-a]\nmodels:\n  lane-a:\n    cmd: x\n");
        assert!(apply(&mut g, &MatrixEdit::SetSet { name: "s".into(), expr: "a".into() }).unwrap_err().contains("groups"));
        assert!(apply(&mut cfg(plain), &MatrixEdit::RemoveSet { name: "s".into() }).unwrap_err().contains("no matrix"));
    }

    #[test]
    fn the_budget_sees_a_new_set() {
        let mut c = cfg(CFG);
        let peaks = [("lane-a".to_string(), 40.0), ("lane-b".to_string(), 30.0), ("lane-c".to_string(), 50.0)]
            .into_iter()
            .collect();
        let before: BudgetReport = swap::budget_report(&c, &peaks, 64);
        assert!(before.over.is_empty());
        apply(&mut c, &MatrixEdit::SetVar { var: "c".into(), lane: "lane-c".into() }).unwrap();
        apply(&mut c, &MatrixEdit::SetSet { name: "big".into(), expr: "a & c".into() }).unwrap();
        let after = swap::budget_report(&c, &peaks, 64);
        assert_eq!(after.over, vec![("big".to_string(), 90)]);
        apply(&mut c, &MatrixEdit::SetSet { name: "risky-big".into(), expr: "a & c".into() }).unwrap();
        apply(&mut c, &MatrixEdit::RemoveSet { name: "big".into() }).unwrap();
        assert!(swap::budget_report(&c, &peaks, 64).over.is_empty(), "risky* sets are the owner's call");
    }
}
