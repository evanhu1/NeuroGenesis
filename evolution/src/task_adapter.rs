use crate::{
    GenomeTask, ResourceEcologyTask, ResourceLifetimeContext, ResourceLifetimeOutcome,
    TaskWorkReport,
};
use anyhow::{bail, Result};
use brain::{
    accumulate_synaptic_eligibilities, apply_three_factor_learning, evaluate_brain_state,
    express_genome, reset_episode_state_preserving_weights, store_action_efference_copy,
    ActionLearningSignal, BrainEvalContext, BrainScratch, EligibilityNormalization,
    EligibilityRequest, ThreeFactorLearningRequest,
};
use serde::{Deserialize, Serialize};
use task_library::SymbolicTask;
use types::{BrainState, OrganismGenome, SensoryReceptor, Symbol};

const TRAINING_DOMAIN: u64 = 0x5453_4b45_434f_5452;
const DEVELOPMENT_DOMAIN: u64 = 0x4445_5645_4c4f_504d;
const SEALED_DOMAIN: u64 = 0x5345_414c_4544_5f54;
const ACTION_DOMAIN: u64 = 0x4143_5449_4f4e_4452;

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LearningRule {
    #[default]
    RewardPredictionError,
    CategoricalPredictionError,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ActionSelection {
    Greedy,
    #[default]
    Sampled,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LearningNormalization {
    #[default]
    None,
    Nlms,
}

impl From<LearningNormalization> for EligibilityNormalization {
    fn from(value: LearningNormalization) -> Self {
        match value {
            LearningNormalization::None => Self::None,
            LearningNormalization::Nlms => Self::NormalizedLeastMeanSquares,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvaluationConfig {
    pub training_instances: usize,
    pub development_instances: usize,
    pub sealed_instances: usize,
    pub training_rollouts: usize,
    pub development_rollouts: usize,
    pub sealed_rollouts: usize,
    pub learning_rule: LearningRule,
    pub action_selection: ActionSelection,
    pub exploration_temperature: f32,
    pub learning_normalization: LearningNormalization,
    pub reset_dynamics_at_trial_boundary: bool,
    pub audit_interval: u32,
    /// Opt-in direct feedback alignment: hidden neurons learn from a fixed
    /// random projection of the categorical output error instead of the scalar
    /// reward surprise. Only active on tasks that reveal teaching targets.
    #[serde(default)]
    pub hidden_categorical_feedback: bool,
    /// Use leaky-integrator neuron dynamics with the local e-prop eligibility
    /// state-derivative. When false, neurons are instantaneous and eligibility
    /// is the simpler scalar presynaptic-times-gain trace (the pre-e-prop
    /// learner). Defaults to true.
    #[serde(default = "default_true")]
    pub temporal_credit_leaky: bool,
}

fn default_true() -> bool {
    true
}

impl Default for AgentEvaluationConfig {
    fn default() -> Self {
        Self {
            training_instances: 64,
            development_instances: 64,
            sealed_instances: 64,
            training_rollouts: 1,
            development_rollouts: 1,
            sealed_rollouts: 1,
            learning_rule: LearningRule::RewardPredictionError,
            action_selection: ActionSelection::Sampled,
            exploration_temperature: 1.0,
            learning_normalization: LearningNormalization::None,
            reset_dynamics_at_trial_boundary: true,
            audit_interval: 25,
            hidden_categorical_feedback: false,
            temporal_credit_leaky: true,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskEcologyConfig<C> {
    pub task: C,
    pub agent: AgentEvaluationConfig,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SymbolicEcologyMetrics {
    pub instances: usize,
    pub ticks: u64,
    pub correct: u64,
    pub accuracy: f64,
    pub learning_ticks: u64,
    pub learning_correct: u64,
    pub learning_accuracy: f64,
    pub probe_ticks: u64,
    pub probe_correct: u64,
    pub probe_accuracy: f64,
    pub mean_probe_target_probability: f64,
    pub mean_probe_sequence_probability: f64,
    pub completed_trials: u64,
    pub successful_trials: u64,
    pub trial_success_rate: f64,
    pub resource_units: u64,
    pub resource_throughput_per_1000_ticks: f64,
    pub mean_reward: f64,
    pub mean_absolute_prediction_error: f64,
    pub mean_reward_prediction: f64,
    pub mean_absolute_applied_delta: f64,
    pub clipped_update_count: u64,
    pub edge_evaluation_count: u64,
    pub internal_edge_evaluation_count: u64,
    pub action_edge_evaluation_count: u64,
    pub value_edge_evaluation_count: u64,
    pub nonzero_internal_edge_update_count: u64,
    pub nonzero_action_edge_update_count: u64,
    pub nonzero_value_edge_update_count: u64,
    pub internal_applied_absolute_delta: f64,
    pub action_applied_absolute_delta: f64,
    pub value_applied_absolute_delta: f64,
    pub brain_synapse_operations: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicEcologyAudit {
    pub cohort: String,
    pub primary: SymbolicEcologyMetrics,
}

#[derive(Clone)]
pub struct TaskEcology<T> {
    pub task: T,
    pub agent: AgentEvaluationConfig,
}
impl<T> TaskEcology<T> {
    pub fn new(task: T, agent: AgentEvaluationConfig) -> Self {
        Self { task, agent }
    }
}

pub(crate) struct Instance<S> {
    task: S,
    brain: BrainState,
    sample_seed: u64,
    sample_tick: u64,
}
pub struct TaskEvaluationState<S> {
    instances: Vec<Instance<S>>,
}

impl<S> TaskEvaluationState<S> {
    /// Crate-internal constructors/accessors for sibling modules that compose
    /// panels from multiple sub-evaluations (co-evolutionary forge scoring).
    pub(crate) fn empty() -> Self {
        Self {
            instances: Vec::new(),
        }
    }
    pub(crate) fn instances_mut(&mut self) -> &mut Vec<Instance<S>> {
        &mut self.instances
    }
    /// Move another state's instances into this one (panel composition).
    pub(crate) fn append_from(&mut self, other: &mut Self) {
        self.instances.append(&mut other.instances);
    }
}
impl<T: SymbolicTask> GenomeTask for TaskEcology<T> {
    fn sensor_enabled(&self, _sensor: SensoryReceptor) -> bool {
        self.task.observes_symbols()
    }
    fn action_enabled(&self, symbol: Symbol) -> bool {
        self.task.action_enabled(symbol)
    }
    fn action_feedback_enabled(&self) -> bool {
        true
    }
    fn temporal_credit_enabled(&self) -> bool {
        true
    }
    fn value_prediction_enabled(&self) -> bool {
        self.task.uses_value_critic()
    }
    fn predictive_coding_enabled(&self) -> bool {
        !self.task.uses_value_critic()
    }
    fn lifetime_learning_enabled(&self) -> bool {
        true
    }
}

impl<T: SymbolicTask + Clone> TaskEcology<T> {
    /// Population-wide flattened evaluation: one flat parallel pass over every
    /// (genome, instance) unit in the generation instead of a join per genome.
    /// Integer metric fields merge exactly; per-tick f64 diagnostics may differ
    /// in the last ulp from per-genome sequential accumulation (documented).
    pub(crate) fn evaluate_population_flat(
        &self,
        genomes: &[&OrganismGenome],
        ids: &[u64],
        run_seed: u64,
        generation: u32,
        pool: &rayon::ThreadPool,
    ) -> Result<Vec<PopulationOutcome<SymbolicEcologyMetrics>>> {
        use rayon::prelude::*;
        // Phase A: build every lifetime's panel (expression + task states).
        let mut states = pool.install(|| {
            genomes
                .par_iter()
                .zip(ids)
                .map(|(genome, id)| {
                    self.initialize_lifetime(genome, *id, run_seed, generation)
                })
                .collect::<Result<Vec<_>>>()
        })?;

        // Phase B: drain instances into one flat schedule of evaluation units.
        let mut flat: Vec<(usize, Instance<T::State>)> = Vec::with_capacity(
            states.iter().map(|s| s.instances.len()).sum(),
        );
        for (genome_index, state) in states.iter_mut().enumerate() {
            for instance in state.instances.drain(..) {
                flat.push((genome_index, instance));
            }
        }

        // Phase C: evaluate all units in a single work-stealing pass.
        let partials: Vec<(usize, SymbolicEcologyMetrics, InstanceScalars)> = pool.install(|| {
            flat.par_iter_mut()
                .map(|(genome_index, instance)| {
                    let mut metrics = SymbolicEcologyMetrics::default();
                    let mut scalars = InstanceScalars::default();
                    self.evaluate_one_instance(
                        genomes[*genome_index],
                        instance,
                        &mut metrics,
                        &mut scalars,
                    );
                    (*genome_index, metrics, scalars)
                })
                .collect()
        });

        // Phase D: merge per genome, strictly in genome order.
        let per_genome_units: Vec<Vec<usize>> = {
            let mut acc: Vec<Vec<usize>> = vec![Vec::new(); genomes.len()];
            for (unit_index, (genome_index, _, _)) in partials.iter().enumerate() {
                acc[*genome_index].push(unit_index);
            }
            acc
        };
        let mut outcomes = Vec::with_capacity(genomes.len());
        for unit_indices in per_genome_units { 
            let mut metrics = SymbolicEcologyMetrics {
                instances: unit_indices.len(),
                ..Default::default()
            };
            let mut totals = InstanceScalars::default();
            let mut probe_sum = 0.0_f64;
            for &unit_index in &unit_indices {
                let (_, part, scalars) = &partials[unit_index];
                metrics.brain_synapse_operations += part.brain_synapse_operations;
                metrics.ticks += part.ticks;
                metrics.correct += part.correct;
                metrics.learning_ticks += part.learning_ticks;
                metrics.learning_correct += part.learning_correct;
                metrics.probe_ticks += part.probe_ticks;
                metrics.probe_correct += part.probe_correct;
                metrics.mean_probe_target_probability += part.mean_probe_target_probability;
                metrics.completed_trials += part.completed_trials;
                metrics.successful_trials += part.successful_trials;
                metrics.resource_units += part.resource_units;
                metrics.edge_evaluation_count += part.edge_evaluation_count;
                metrics.internal_edge_evaluation_count += part.internal_edge_evaluation_count;
                metrics.action_edge_evaluation_count += part.action_edge_evaluation_count;
                metrics.value_edge_evaluation_count += part.value_edge_evaluation_count;
                metrics.nonzero_internal_edge_update_count +=
                    part.nonzero_internal_edge_update_count;
                metrics.nonzero_action_edge_update_count +=
                    part.nonzero_action_edge_update_count;
                metrics.nonzero_value_edge_update_count += part.nonzero_value_edge_update_count;
                metrics.internal_applied_absolute_delta += part.internal_applied_absolute_delta;
                metrics.action_applied_absolute_delta += part.action_applied_absolute_delta;
                metrics.value_applied_absolute_delta += part.value_applied_absolute_delta;
                metrics.clipped_update_count += part.clipped_update_count;
                probe_sum += scalars.probe_sequence_probability_sum;
                totals.rewards += scalars.rewards;
                totals.errors += scalars.errors;
                totals.predictions += scalars.predictions;
                totals.deltas += scalars.deltas;
            }
            Self::finalize_metrics(&mut metrics, &mut probe_sum, &mut totals);
            outcomes.push(PopulationOutcome {
                reproductive_tickets: metrics.resource_units,
                work: TaskWorkReport {
                    brain_synapse_operations: metrics.brain_synapse_operations,
                },
                evaluation: metrics,
            });
        }
        Ok(outcomes)
    }
}

impl<T: SymbolicTask + Clone> ResourceEcologyTask for TaskEcology<T> {
    type Config = TaskEcologyConfig<T::Config>;
    type LifetimeState = TaskEvaluationState<T::State>;
    type LifetimeEvaluation = SymbolicEcologyMetrics;
    type AuditEvaluation = SymbolicEcologyAudit;

    fn name(&self) -> &'static str {
        self.task.name()
    }
    fn objective(&self) -> &'static str {
        "finite_task_resource_capture"
    }
    fn config(&self) -> Self::Config {
        TaskEcologyConfig {
            task: self.task.config(),
            agent: self.agent.clone(),
        }
    }
    fn lifetime_ticks(&self) -> usize {
        self.task.max_steps_per_instance() + self.task.probe_steps_per_instance()
    }
    fn evaluation_lifetimes(&self) -> usize {
        self.agent.training_instances * self.agent.training_rollouts
    }
    fn evaluate_population(
        &self,
        genomes: &[&OrganismGenome],
        ids: &[u64],
        run_seed: u64,
        generation: u32,
        pool: &rayon::ThreadPool,
    ) -> Result<Vec<crate::task_adapter::PopulationOutcome<Self::LifetimeEvaluation>>> {
        // The flat schedule pays for itself only when single panels are large
        // enough that per-genome jobs strand workers (measured: population-wide
        // flattening regressed symbolic ti=8 runs by ~5% from instance moves).
        const MIN_FLAT_PANEL: usize = 32;
        if self.agent.training_instances * self.agent.training_rollouts >= MIN_FLAT_PANEL {
            self.evaluate_population_flat(genomes, ids, run_seed, generation, pool)
        } else {
            ResourceEcologyTask::evaluate_population_default(
                self, genomes, ids, run_seed, generation, pool,
            )
        }
    }
    fn validate(&self) -> Result<()> {
        self.task.validate()?;
        if !self.agent.exploration_temperature.is_finite()
            || self.agent.exploration_temperature <= 0.0
        {
            bail!("exploration temperature must be finite and positive");
        }
        if self.agent.audit_interval == 0 {
            bail!("audit interval must be positive");
        }
        if matches!(
            self.agent.learning_rule,
            LearningRule::CategoricalPredictionError
        ) && !self.task.reveals_teaching_targets()
        {
            bail!("categorical prediction learning requires learner-visible teaching targets");
        }
        if self.agent.training_instances == 0
            || self.agent.development_instances == 0
            || self.agent.sealed_instances == 0
            || self.agent.training_rollouts == 0
            || self.agent.development_rollouts == 0
            || self.agent.sealed_rollouts == 0
        {
            bail!("panel instance and rollout counts must be positive");
        }
        Ok(())
    }
    fn initialize_lifetime(
        &self,
        genome: &OrganismGenome,
        _individual_id: u64,
        run_seed: u64,
        _generation: u32,
    ) -> Result<Self::LifetimeState> {
        // The benchmark panel is fixed across generations. Population members
        // still share all stochastic draws within a generation, but evolution
        // cannot be ranked on a moving target distribution.
        let panel_seed = mix64(run_seed ^ TRAINING_DOMAIN);
        // All instances of one genome start from an identical expressed brain:
        // express once, then memcpy-clone per instance (expression involves
        // innovation hashing + validation and dominated small-panel setups).
        let base_brain = express_genome(genome);
        Ok(TaskEvaluationState {
            instances: (0..self.agent.training_instances)
                .flat_map(|index| {
                    (0..self.agent.training_rollouts).map(move |rollout| (index, rollout))
                })
                .map(|(index, rollout)| Instance {
                    task: self.task.start(panel_seed, index),
                    brain: base_brain.clone(),
                    sample_seed: mix64(
                        panel_seed
                            ^ ACTION_DOMAIN
                            ^ (index as u64).rotate_left(17)
                            ^ rollout as u64,
                    ),
                    sample_tick: 0,
                })
                .collect(),
        })
    }
    fn evaluate_lifetime(
        &self,
        genome: &OrganismGenome,
        state: &mut Self::LifetimeState,
        _context: ResourceLifetimeContext,
    ) -> Result<ResourceLifetimeOutcome<Self::LifetimeEvaluation>> {
        let metrics = self.evaluate_instances(genome, &mut state.instances);
        Ok(ResourceLifetimeOutcome {
            reproductive_tickets: metrics.resource_units,
            work: TaskWorkReport {
                brain_synapse_operations: metrics.brain_synapse_operations,
            },
            evaluation: metrics,
        })
    }
    fn audit(
        &self,
        genome: &OrganismGenome,
        cohort: &str,
        audit_seed: u64,
    ) -> Result<Self::AuditEvaluation> {
        let (domain, instance_count, rollout_count) = if cohort == "sealed" {
            (
                SEALED_DOMAIN,
                self.agent.sealed_instances,
                self.agent.sealed_rollouts,
            )
        } else {
            (
                DEVELOPMENT_DOMAIN,
                self.agent.development_instances,
                self.agent.development_rollouts,
            )
        };
        let evaluate = || {
            let panel_seed = mix64(audit_seed ^ domain);
            let mut instances = (0..instance_count)
                .flat_map(|index| (0..rollout_count).map(move |rollout| (index, rollout)))
                .map(|(index, rollout)| Instance {
                    task: self.task.start(panel_seed, index),
                    brain: express_genome(genome),
                    sample_seed: mix64(
                        panel_seed
                            ^ ACTION_DOMAIN
                            ^ (index as u64).rotate_left(17)
                            ^ rollout as u64,
                    ),
                    sample_tick: 0,
                })
                .collect::<Vec<_>>();
            self.evaluate_instances(genome, &mut instances)
        };
        Ok(SymbolicEcologyAudit {
            cohort: cohort.to_owned(),
            primary: evaluate(),
        })
    }
    fn audit_score(&self, audit: &Self::AuditEvaluation) -> f64 {
        audit.primary.accuracy
    }
    fn audit_due(&self, generation: u32, total_generations: u32) -> bool {
        generation + 1 == total_generations
            || (generation + 1).is_multiple_of(self.agent.audit_interval)
    }
}

impl<T: SymbolicTask> TaskEcology<T> {
        /// Serial twin of `evaluate_instances`: identical arithmetic, no rayon.
    /// Used where nested parallelism would be unsafe (e.g. forge scoring runs
    /// while the co-evolution state mutex is held).
    pub(crate) fn evaluate_instances_serial(
        &self,
        genome: &OrganismGenome,
        instances: &mut [Instance<T::State>],
    ) -> SymbolicEcologyMetrics {
        let panel_count = instances.len();
        let mut metrics = SymbolicEcologyMetrics {
            instances: panel_count,
            ..Default::default()
        };
        let mut scalars = InstanceScalars::default();
        for instance in instances {
            self.evaluate_one_instance(genome, instance, &mut metrics, &mut scalars);
        }
        let mut probe_sequence_probability_sum = scalars.probe_sequence_probability_sum;
        Self::finalize_metrics(&mut metrics, &mut probe_sequence_probability_sum, &mut scalars);
        metrics
    }

    /// Shared derivation tail for both evaluation variants.
    fn finalize_metrics(
        metrics: &mut SymbolicEcologyMetrics,
        probe_sequence_probability_sum: &mut f64,
        scalars: &mut InstanceScalars,
    ) {
        let (rewards, errors, predictions, deltas) = (
            scalars.rewards,
            scalars.errors,
            scalars.predictions,
            scalars.deltas,
        );
        if metrics.learning_ticks > 0 {
            metrics.learning_accuracy =
                metrics.learning_correct as f64 / metrics.learning_ticks as f64;
            metrics.mean_reward = rewards / metrics.learning_ticks as f64;
            metrics.mean_absolute_prediction_error = errors / metrics.learning_ticks as f64;
            metrics.mean_reward_prediction = predictions / metrics.learning_ticks as f64;
        }
        if metrics.probe_ticks > 0 {
            metrics.probe_accuracy = metrics.probe_correct as f64 / metrics.probe_ticks as f64;
            metrics.mean_probe_target_probability /= metrics.probe_ticks as f64;
            metrics.mean_probe_sequence_probability =
                *probe_sequence_probability_sum / metrics.instances as f64;
            metrics.accuracy = metrics.probe_accuracy;
        } else if metrics.learning_ticks > 0 {
            metrics.accuracy = metrics.learning_accuracy;
        }
        if metrics.ticks > 0 {
            metrics.resource_throughput_per_1000_ticks =
                metrics.resource_units as f64 * 1000.0 / metrics.ticks as f64;
        }
        if metrics.completed_trials > 0 {
            metrics.trial_success_rate =
                metrics.successful_trials as f64 / metrics.completed_trials as f64;
        }
        if metrics.edge_evaluation_count > 0 {
            metrics.mean_absolute_applied_delta = deltas / metrics.edge_evaluation_count as f64;
        }
    }

    pub(crate) fn evaluate_instances(
        &self,
        genome: &OrganismGenome,
        instances: &mut [Instance<T::State>],
    ) -> SymbolicEcologyMetrics {
        
        // Panels with several instances evaluate them as independent parallel
        // work items: large panels (memory, co-evolution pilots) otherwise
        // strand workers behind one straggler genome. Integer metric fields
        // merge associatively and stay exact; per-tick f64 diagnostic sums
        // (reward/prediction/delta averages) may differ in the last ulp from
        // sequential accumulation when instances >= 2. Selection-relevant
        // quantities (resource_units, correct/tick counts) remain exact.
        const MAX_CHUNKS: usize = 8;
        use rayon::prelude::*;
        let panel_count = instances.len();
        let mut metrics = SymbolicEcologyMetrics {
            instances: panel_count,
            ..Default::default()
        };
        let mut probe_sequence_probability_sum = 0.0_f64;
        let mut totals = InstanceScalars::default();
        let chunk_len = panel_count.div_ceil(MAX_CHUNKS).max(1);
        let partials = instances
            .par_chunks_mut(chunk_len)
            .map(|chunk| {
                let mut part = (SymbolicEcologyMetrics::default(), InstanceScalars::default());
                for instance in chunk {
                    self.evaluate_one_instance(genome, instance, &mut part.0, &mut part.1);
                }
                part
            })
            .collect::<Vec<_>>();
        for (part_metrics, scalars) in partials {
            metrics.brain_synapse_operations += part_metrics.brain_synapse_operations;
            metrics.ticks += part_metrics.ticks;
            metrics.correct += part_metrics.correct;
            metrics.learning_ticks += part_metrics.learning_ticks;
            metrics.learning_correct += part_metrics.learning_correct;
            metrics.probe_ticks += part_metrics.probe_ticks;
            metrics.probe_correct += part_metrics.probe_correct;
            metrics.mean_probe_target_probability +=
                part_metrics.mean_probe_target_probability;
            metrics.completed_trials += part_metrics.completed_trials;
            metrics.successful_trials += part_metrics.successful_trials;
            metrics.resource_units += part_metrics.resource_units;
            metrics.edge_evaluation_count += part_metrics.edge_evaluation_count;
            metrics.internal_edge_evaluation_count +=
                part_metrics.internal_edge_evaluation_count;
            metrics.action_edge_evaluation_count += part_metrics.action_edge_evaluation_count;
            metrics.value_edge_evaluation_count += part_metrics.value_edge_evaluation_count;
            metrics.nonzero_internal_edge_update_count +=
                part_metrics.nonzero_internal_edge_update_count;
            metrics.nonzero_action_edge_update_count +=
                part_metrics.nonzero_action_edge_update_count;
            metrics.nonzero_value_edge_update_count +=
                part_metrics.nonzero_value_edge_update_count;
            metrics.internal_applied_absolute_delta +=
                part_metrics.internal_applied_absolute_delta;
            metrics.action_applied_absolute_delta += part_metrics.action_applied_absolute_delta;
            metrics.value_applied_absolute_delta += part_metrics.value_applied_absolute_delta;
            metrics.clipped_update_count += part_metrics.clipped_update_count;
            probe_sequence_probability_sum += scalars.probe_sequence_probability_sum;
            totals.rewards += scalars.rewards;
            totals.errors += scalars.errors;
            totals.predictions += scalars.predictions;
            totals.deltas += scalars.deltas;
        }
        Self::finalize_metrics(&mut metrics, &mut probe_sequence_probability_sum, &mut totals);
        metrics
    }
}

/// Evaluation result for one population member, decoupled from the task-side
/// lifetime state so a population-wide flattened schedule can produce them in
/// bulk.
#[derive(Debug, Clone)]
pub struct PopulationOutcome<E> {
    pub reproductive_tickets: u64,
    pub work: TaskWorkReport,
    pub evaluation: E,
}

/// Per-instance f64 sample sums carried out of `evaluate_one_instance`.
#[derive(Default)]
struct InstanceScalars {
    rewards: f64,
    errors: f64,
    predictions: f64,
    deltas: f64,
    /// One contribution per instance (its greedy-probe sequence product), so
    /// ordered merging reproduces the historical accumulation exactly.
    probe_sequence_probability_sum: f64,
}

impl<T: SymbolicTask> TaskEcology<T> {
    #[allow(clippy::too_many_arguments)]
    fn evaluate_one_instance(
        &self,
        genome: &OrganismGenome,
        instance: &mut Instance<T::State>,
        metrics: &mut SymbolicEcologyMetrics,
        scalars: &mut InstanceScalars,
    ) {
        let mut scratch = BrainScratch::new();
            for _ in 0..self.task.max_steps_per_instance() {
                apply_observation(&mut instance.brain, self.task.observe(&instance.task));
                let brain_eval = evaluate_brain_state(
                    &mut instance.brain,
                    genome,
                    BrainEvalContext {
                        leaky_neurons_enabled: self.agent.temporal_credit_leaky,
                        action_temperature: 1.0,
                        action_sample: None,
                    },
                    &mut scratch,
                );
                metrics.brain_synapse_operations += brain_eval.synapse_ops;
                let probabilities = action_probabilities(
                    &self.task,
                    brain_eval.action_logits,
                    self.agent.exploration_temperature * genome.plasticity.action_temperature_scale,
                );
                let selected = match self.agent.action_selection {
                    ActionSelection::Greedy => argmax_action(&self.task, brain_eval.action_logits),
                    ActionSelection::Sampled => sample_action(
                        &self.task,
                        probabilities,
                        deterministic_sample(instance.sample_seed, instance.sample_tick),
                    ),
                };
                let transition = self.task.step(&mut instance.task, selected);
                let action_signal = match self.agent.learning_rule {
                    LearningRule::RewardPredictionError => {
                        ActionLearningSignal::RewardPredictionError { selected }
                    }
                    LearningRule::CategoricalPredictionError => {
                        let target = transition.teaching_target.expect(
                            "categorical prediction learning requires a learner-visible target",
                        );
                        ActionLearningSignal::CategoricalPredictionError {
                            target,
                            probabilities,
                        }
                    }
                };
                accumulate_synaptic_eligibilities(
                    &mut instance.brain,
                    &mut scratch,
                    EligibilityRequest {
                        action_signal,
                        value_prediction: brain_eval.value_prediction,
                        normalization: self.agent.learning_normalization.into(),
                        leaky_neurons_enabled: self.agent.temporal_credit_leaky,
                        eligibility_retention: genome.plasticity.eligibility_retention,
                    },
                );
                // Predictive-coding tasks run with no critic: hidden plasticity
                // is driven by the neuromodulatory channels, not reward surprise.
                let predictive_coding = !self.task.uses_value_critic();
                let prediction_error = if predictive_coding {
                    0.0
                } else {
                    transition.reward - brain_eval.value_prediction
                };
                let report = apply_three_factor_learning(
                    &mut instance.brain,
                    ThreeFactorLearningRequest {
                        action_signal,
                        reward_prediction_error: prediction_error,
                        learning_rate: genome.plasticity.initial_learning_rate,
                        eligibility_retention: genome.plasticity.eligibility_retention,
                        fast_weight_retention: genome.plasticity.fast_weight_retention,
                        max_weight_delta: genome.plasticity.max_weight_delta_per_tick,
                        hidden_categorical_feedback: self.agent.hidden_categorical_feedback,
                        predictive_coding,
                    },
                );
                store_action_efference_copy(&mut instance.brain, selected);
                metrics.ticks += 1;
                metrics.learning_ticks += 1;
                metrics.learning_correct += u64::from(transition.correct);
                if self.task.probe_steps_per_instance() == 0 {
                    metrics.correct += u64::from(transition.correct);
                    metrics.resource_units += u64::from(transition.success_events);
                    if let Some(successful) = transition.trial_outcome {
                        metrics.completed_trials += 1;
                        metrics.successful_trials += u64::from(successful);
                    }
                }
                metrics.edge_evaluation_count += report.edge_evaluation_count;
                metrics.internal_edge_evaluation_count += report.internal_edge_evaluation_count;
                metrics.action_edge_evaluation_count += report.action_edge_evaluation_count;
                metrics.value_edge_evaluation_count += report.value_edge_evaluation_count;
                metrics.nonzero_internal_edge_update_count +=
                    report.nonzero_internal_edge_update_count;
                metrics.nonzero_action_edge_update_count += report.nonzero_action_edge_update_count;
                metrics.nonzero_value_edge_update_count += report.nonzero_value_edge_update_count;
                metrics.internal_applied_absolute_delta += report.internal_applied_absolute_delta;
                metrics.action_applied_absolute_delta += report.action_applied_absolute_delta;
                metrics.value_applied_absolute_delta += report.value_applied_absolute_delta;
                metrics.clipped_update_count += report.clipped_update_count;
                scalars.rewards += f64::from(transition.reward);
                scalars.errors += f64::from(prediction_error.abs());
                scalars.predictions += f64::from(brain_eval.value_prediction);
                scalars.deltas += report.applied_absolute_delta;
                instance.sample_tick += 1;
                if transition.trial_outcome.is_some() && self.agent.reset_dynamics_at_trial_boundary
                {
                    reset_episode_state_preserving_weights(&mut instance.brain);
                }
                if transition.done {
                    break;
                }
            }

            let probe_steps = self.task.probe_steps_per_instance();
            if probe_steps > 0 {
                reset_episode_state_preserving_weights(&mut instance.brain);
                self.task.begin_probe(&mut instance.task);
                let mut sequence_probability = 1.0;
                for _ in 0..probe_steps {
                    apply_observation(&mut instance.brain, self.task.probe_observe(&instance.task));
                    let brain_eval = evaluate_brain_state(
                        &mut instance.brain,
                        genome,
                        BrainEvalContext {
                            leaky_neurons_enabled: self.agent.temporal_credit_leaky,
                            action_temperature: 1.0,
                            action_sample: None,
                        },
                        &mut scratch,
                    );
                    metrics.brain_synapse_operations += brain_eval.synapse_ops;
                    let probabilities = action_probabilities(
                        &self.task,
                        brain_eval.action_logits,
                        self.agent.exploration_temperature
                            * genome.plasticity.action_temperature_scale,
                    );
                    let selected = argmax_action(&self.task, brain_eval.action_logits);
                    let transition = self.task.probe_step(&mut instance.task, selected);
                    store_action_efference_copy(&mut instance.brain, selected);
                    if let Some(expected) = transition.expected_action {
                        sequence_probability *= f64::from(probabilities[expected.index()]);
                        metrics.mean_probe_target_probability +=
                            f64::from(probabilities[expected.index()]);
                    }
                    metrics.ticks += 1;
                    metrics.probe_ticks += 1;
                    metrics.probe_correct += u64::from(transition.correct);
                    metrics.correct += u64::from(transition.correct);
                    metrics.resource_units += u64::from(transition.success_events);
                    if let Some(successful) = transition.trial_outcome {
                        metrics.completed_trials += 1;
                        metrics.successful_trials += u64::from(successful);
                    }
                    if transition.done {
                        break;
                    }
                }
                scalars.probe_sequence_probability_sum += sequence_probability;
            }
    }
}

fn apply_observation(brain: &mut BrainState, observation: task_library::Observation) {
    for sensory in &mut brain.sensory {
        sensory.neuron.activation = match (sensory.receptor, observation.symbol) {
            (SensoryReceptor::Symbol { symbol: receptor }, Some(symbol)) => {
                f32::from(receptor == symbol)
            }
            _ => 0.0,
        };
    }
}

fn argmax_action<T: SymbolicTask>(task: &T, logits: [f32; Symbol::COUNT]) -> Symbol {
    Symbol::ALL
        .into_iter()
        .filter(|action| task.action_enabled(*action))
        .max_by(|left, right| {
            logits[left.index()]
                .total_cmp(&logits[right.index()])
                .then_with(|| right.index().cmp(&left.index()))
        })
        .expect("validated task exposes at least one action")
}

fn action_probabilities<T: SymbolicTask>(
    task: &T,
    logits: [f32; Symbol::COUNT],
    temperature: f32,
) -> [f32; Symbol::COUNT] {
    let mut probabilities = [0.0; Symbol::COUNT];
    let max = Symbol::ALL
        .into_iter()
        .filter(|a| task.action_enabled(*a))
        .map(|a| logits[a.index()] / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for action in Symbol::ALL.into_iter().filter(|a| task.action_enabled(*a)) {
        probabilities[action.index()] = (logits[action.index()] / temperature - max).exp();
        total += probabilities[action.index()];
    }
    for probability in &mut probabilities {
        *probability /= total;
    }
    probabilities
}

fn sample_action<T: SymbolicTask>(
    task: &T,
    probabilities: [f32; Symbol::COUNT],
    sample: f32,
) -> Symbol {
    let mut cumulative = 0.0;
    let mut last = Symbol::A;
    for action in Symbol::ALL.into_iter().filter(|a| task.action_enabled(*a)) {
        last = action;
        cumulative += probabilities[action.index()];
        if sample < cumulative {
            return action;
        }
    }
    last
}

fn deterministic_sample(seed: u64, tick: u64) -> f32 {
    let bits = mix64(seed ^ tick);
    ((bits >> 40) as f32 + 0.5) / ((1_u32 << 24) as f32)
}
fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
