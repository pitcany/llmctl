//! The supervisor port: the platform seam between the manager-agnostic
//! adoption / ownership flows and the concrete service manager — systemd
//! (Linux) or launchd (macOS). Everything `adopt` and `own` need from the
//! manager goes through [`Supervisor`], so those flows stay identical
//! across platforms and each platform implements exactly the manager
//! semantics it actually has (RUST-M3-PLAN, A.2).

use std::path::Path;
use std::time::Duration;

use llmctl_core::{
    ControlAuthority, FieldDiff, LaunchDefinition, UnitName, VersionRef, WorkloadName,
};

use crate::engine::{BoxFut, EngineError, UnitProbe};

/// What import learned about one job, manager-neutrally.
#[derive(Debug, Clone)]
pub struct ImportedJob {
    pub load_state: String,
    /// `None` unless the job is loaded and its definition readable.
    pub definition: Option<LaunchDefinition>,
    pub provenance: Vec<llmctl_core::FileProvenance>,
    /// Inline environment names that look like credentials. Their values
    /// are never kept; their presence blocks adoption.
    pub secret_env_names: Vec<String>,
    /// The file backing the job (unit file / plist); empty when the job has
    /// none or runs from manager-internal state.
    pub fragment_path: String,
    /// systemd drop-ins modifying the fragment (launchd: always empty —
    /// there is no merging).
    pub drop_in_paths: Vec<String>,
}

/// The manager-agnostic operations adoption and ownership need.
pub trait Supervisor: Send + Sync {
    /// Inventory engine name routing workloads to this manager
    /// (`systemd` / `launchd`).
    fn engine_name(&self) -> &'static str;

    /// Timeout for manager reads (imports, prints).
    fn show_timeout(&self) -> Duration;

    /// Parse and validate the caller-supplied job reference: a bare user
    /// unit name (`foo.service`) on systemd, a label (`com.example.foo`) on
    /// launchd.
    fn parse_unit(&self, raw: &str) -> Result<UnitName, String>;

    /// The control authority enrolling this job grants.
    fn authority_for(&self, unit: &UnitName) -> ControlAuthority;

    /// Import the job's effective launch definition and file provenance.
    fn import(&self, unit: &UnitName) -> BoxFut<Result<ImportedJob, EngineError>>;

    /// Render the file llmctl would own for `def`. `fault_corrupt_render`
    /// is the test hook that must make the render subtly *wrong* so the
    /// prove step catches it.
    fn render(
        &self,
        def: &LaunchDefinition,
        workload: &WorkloadName,
        fault_corrupt_render: bool,
    ) -> Result<String, String>;

    /// Prove `rendered` loads to the identical effective definition as
    /// `def` (scratch job in the real manager). Empty diffs = proven.
    fn roundtrip(
        &self,
        def: &LaunchDefinition,
        rendered: &str,
    ) -> BoxFut<Result<Vec<FieldDiff>, EngineError>>;

    /// Read-only probe of an unenrolled job (adoption's identity check).
    fn probe_unenrolled(&self, unit: &UnitName, port: u16, model: &str) -> BoxFut<UnitProbe>;

    /// Make the manager re-read files after llmctl wrote one. systemd:
    /// `daemon-reload`. launchd: nothing — a plist is applied by the next
    /// bootstrap, which is exactly what [`Supervisor::deferred_apply`]
    /// records.
    fn reload_manager(&self) -> BoxFut<Result<(), EngineError>>;

    /// Directives/keys set in the current file that rendering `def` would
    /// silently drop.
    fn uncovered(&self, path: &Path, def: &LaunchDefinition) -> Vec<String>;

    /// A section of the source file copied verbatim into the render
    /// (systemd `[Install]`; launchd: none).
    fn carried_verbatim(&self, source: &str) -> Option<String>;

    /// True when writing the owned file does *not* take effect immediately
    /// (launchd: the next admitted restart bootstraps it; decision 6).
    fn deferred_apply(&self) -> bool;

    /// Extra facts pinned at adoption on this platform (launchd: the
    /// program binary's content digest; systemd pins python-env facts
    /// lazily through recipes instead). `None` = nothing to pin.
    fn program_digest_pin(&self, _def: &LaunchDefinition) -> Option<VersionRef> {
        None
    }
}

// ---- Linux: systemd ---------------------------------------------------------

#[cfg(all(unix, target_os = "linux", not(llmctl_macos)))]
pub use linux::SystemdSupervisor;

#[cfg(all(unix, target_os = "linux", not(llmctl_macos)))]
mod linux {
    use super::*;

    use llmctl_core::{ControlAuthority, UnitScope, UnitScopeMarker};

    use crate::engine::EngineError;
    use crate::systemd::SystemdEngine;
    use crate::unitdef;

    impl From<unitdef::ImportedUnit> for ImportedJob {
        fn from(u: unitdef::ImportedUnit) -> Self {
            ImportedJob {
                load_state: u.load_state,
                definition: u.definition,
                provenance: u.provenance,
                secret_env_names: u.secret_env_names,
                fragment_path: u.fragment_path,
                drop_in_paths: u.drop_in_paths,
            }
        }
    }

    #[derive(Clone)]
    pub struct SystemdSupervisor {
        pub engine: SystemdEngine,
    }

    impl SystemdSupervisor {
        pub fn new(engine: SystemdEngine) -> Self {
            Self { engine }
        }
    }

    impl Supervisor for SystemdSupervisor {
        fn engine_name(&self) -> &'static str {
            crate::systemd::ENGINE_NAME
        }

        fn show_timeout(&self) -> Duration {
            self.engine.show_timeout()
        }

        fn parse_unit(&self, raw: &str) -> Result<UnitName, String> {
            if !raw.ends_with(".service") || raw.contains('/') {
                return Err(format!(
                    "unit must be a bare service name like foo.service, got {raw:?}"
                ));
            }
            UnitName::systemd(UnitScopeMarker::User, raw).map_err(|e| e.to_string())
        }

        fn authority_for(&self, unit: &UnitName) -> ControlAuthority {
            ControlAuthority::SystemdUnit {
                scope: UnitScope::User,
                unit: unit.name.clone(),
            }
        }

        fn import(&self, unit: &UnitName) -> BoxFut<Result<ImportedJob, EngineError>> {
            let timeout = self.engine.show_timeout();
            let unit = unit.clone();
            Box::pin(async move {
                unitdef::import_unit(&unit, timeout)
                    .await
                    .map(ImportedJob::from)
            })
        }

        fn render(
            &self,
            def: &LaunchDefinition,
            workload: &WorkloadName,
            fault_corrupt_render: bool,
        ) -> Result<String, String> {
            let header = format!("llmctl-next managed workload {workload}");
            let mut text = unitdef::render_unit(def, &header);
            if fault_corrupt_render {
                text.push_str("Environment=LLMCTL_FAULT_RENDER=1\n");
            }
            Ok(text)
        }

        fn roundtrip(
            &self,
            def: &LaunchDefinition,
            rendered: &str,
        ) -> BoxFut<Result<Vec<FieldDiff>, EngineError>> {
            let timeout = self.engine.show_timeout();
            let def = def.clone();
            let rendered = rendered.to_string();
            Box::pin(async move { unitdef::roundtrip(&def, &rendered, timeout).await })
        }

        fn probe_unenrolled(&self, unit: &UnitName, port: u16, model: &str) -> BoxFut<UnitProbe> {
            let engine = self.engine.clone();
            let unit = unit.clone();
            let model = model.to_string();
            Box::pin(async move { engine.probe_unenrolled(&unit, port, &model).await })
        }

        fn reload_manager(&self) -> BoxFut<Result<(), EngineError>> {
            let timeout = self.engine.show_timeout();
            Box::pin(async move { unitdef::daemon_reload(timeout).await })
        }

        fn uncovered(&self, path: &Path, def: &LaunchDefinition) -> Vec<String> {
            match std::fs::read_to_string(path) {
                Ok(source) => unitdef::uncovered_directives(&source, def),
                Err(_) => Vec::new(),
            }
        }

        fn carried_verbatim(&self, source: &str) -> Option<String> {
            unitdef::install_section(source)
        }

        fn deferred_apply(&self) -> bool {
            false
        }
    }
}

// ---- macOS: launchd ---------------------------------------------------------

#[cfg(any(target_os = "macos", llmctl_macos))]
pub use macos::LaunchdSupervisor;

#[cfg(any(target_os = "macos", llmctl_macos))]
mod macos {
    use super::*;

    use crate::engine::EngineError;
    use crate::launchd::LaunchdEngine;

    #[derive(Clone)]
    pub struct LaunchdSupervisor {
        pub engine: LaunchdEngine,
    }

    impl LaunchdSupervisor {
        pub fn new(engine: LaunchdEngine) -> Self {
            Self { engine }
        }
    }

    impl Supervisor for LaunchdSupervisor {
        fn engine_name(&self) -> &'static str {
            crate::launchd::ENGINE_NAME
        }

        fn show_timeout(&self) -> Duration {
            self.engine.show_timeout()
        }

        fn parse_unit(&self, raw: &str) -> Result<UnitName, String> {
            if raw.contains('/') || raw.is_empty() || raw.len() > 160 {
                return Err(format!(
                    "expected a launchd label like com.example.foo, got {raw:?}"
                ));
            }
            UnitName::launchd(raw).map_err(|e| e.to_string())
        }

        fn authority_for(&self, unit: &UnitName) -> ControlAuthority {
            ControlAuthority::LaunchdService {
                label: unit.name.clone(),
            }
        }

        fn import(&self, unit: &UnitName) -> BoxFut<Result<ImportedJob, EngineError>> {
            let timeout = self.engine.show_timeout();
            let unit = unit.clone();
            Box::pin(async move { crate::plistdef::import_job(&unit, timeout).await })
        }

        fn render(
            &self,
            def: &LaunchDefinition,
            _workload: &WorkloadName,
            fault_corrupt_render: bool,
        ) -> Result<String, String> {
            crate::plistdef::render_plist(def, fault_corrupt_render)
        }

        fn roundtrip(
            &self,
            def: &LaunchDefinition,
            rendered: &str,
        ) -> BoxFut<Result<Vec<FieldDiff>, EngineError>> {
            let timeout = self.engine.show_timeout();
            let def = def.clone();
            let rendered = rendered.to_string();
            Box::pin(async move { crate::plistdef::roundtrip(&def, &rendered, timeout).await })
        }

        fn probe_unenrolled(&self, unit: &UnitName, port: u16, model: &str) -> BoxFut<UnitProbe> {
            let engine = self.engine.clone();
            let unit = unit.clone();
            let model = model.to_string();
            Box::pin(async move { engine.probe_unenrolled(&unit, port, &model).await })
        }

        fn reload_manager(&self) -> BoxFut<Result<(), EngineError>> {
            // launchd applies a plist at bootstrap; there is nothing to
            // reload. (See deferred_apply.)
            Box::pin(async { Ok(()) })
        }

        fn uncovered(&self, path: &Path, def: &LaunchDefinition) -> Vec<String> {
            crate::plistdef::uncovered_keys(path, def)
        }

        fn carried_verbatim(&self, _source: &str) -> Option<String> {
            None
        }

        fn deferred_apply(&self) -> bool {
            true
        }

        fn program_digest_pin(&self, def: &LaunchDefinition) -> Option<VersionRef> {
            crate::launchd::program_digest_pin(def)
        }
    }
}
