//! Domain errors. Distinct, machine-matchable kinds instead of stringly
//! typed failures; callers must be able to branch on "unsupported
//! capability" without parsing messages.

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CoreError {
    #[error("invalid identity: {0}")]
    InvalidIdentity(String),

    #[error("unknown workload: {0}")]
    UnknownWorkload(String),

    #[error("unknown gpu on host: {0}")]
    UnknownGpu(String),

    #[error("invalid workload configuration for {workload}: {reason}")]
    InvalidConfig { workload: String, reason: String },

    #[error("validation failed: {0}")]
    Validation(String),
}

pub type CoreResult<T> = Result<T, CoreError>;
