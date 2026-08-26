//! Generic asexual search over finite reproductive task ecologies.
//!
//! Tasks define observations, actions, transitions, and physical resource
//! events in `task-library`. This crate owns genome expression, lifetime
//! learning, mutation, evaluation scheduling, and reproduction.

pub mod task;

mod co_ecology;
mod resource_ecology;
mod task_adapter;

pub use resource_ecology::{
    run_resource_ecology, AsexualSearchConfig, ResourceEcologyConfig,
    ResourceEcologyGenerationSummary, ResourceEcologyPopulationMember, ResourceEcologyResult,
    ResourceEcologyTermination, ResourceEcologyTerminationStatus, ResourceEcologyWork,
    SelectionScheme, RESOURCE_ECOLOGY_RESULT_SCHEMA_VERSION,
};
pub use co_ecology::{run_co_evolution_next_token, WordForgeGenome};
pub use task::{
    GenomeTask, ResourceEcologyTask, ResourceLifetimeContext, ResourceLifetimeOutcome,
    TaskWorkReport,
};
pub use task_adapter::{
    ActionSelection, AgentEvaluationConfig, LearningNormalization, LearningRule,
    SymbolicEcologyAudit, SymbolicEcologyMetrics, TaskEcology,
};
