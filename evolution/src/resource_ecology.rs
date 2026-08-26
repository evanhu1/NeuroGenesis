use crate::{GenomeTask, ResourceEcologyTask};
use anyhow::{anyhow, bail, Result};
use brain::genome::{
    align_genome_vectors, connection_would_create_cycle, generate_seed_genome, MAX_INTER_NEURONS,
    PLASTICITY_RECEPTOR_MAX, SYNAPSE_PLASTICITY_COEFFICIENT_MAX, SYNAPSE_STRENGTH_MAX,
    SYNAPSE_STRENGTH_MIN,
};
use rand::{
    seq::IndexedMutRandom, seq::IndexedRandom, seq::SliceRandom, Rng, RngCore, SeedableRng,
};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;
use types::{
    action_gene_node_id, action_gene_node_index, connection_innovation_id, is_hidden_gene_node_id,
    is_value_gene_node_id, sensory_gene_node_id, sensory_gene_node_index,
    split_hidden_gene_node_id, value_gene_node_id, ActivationFunction, GeneNodeId, HiddenNodeGene,
    NeuronId, OrganismGenome, SeedGenomeConfig, SensoryReceptor, Symbol, SynapseGene,
    SynapseTiming,
};

const ECOLOGY_REPRODUCTION_DOMAIN: u64 = 0x4543_4f4c_4f47_5952;
const ECOLOGY_AUDIT_DOMAIN: u64 = 0x4543_4f4c_4f47_5941;
const ECOLOGY_DEVELOPMENT_DOMAIN: u64 = 0x4445_5645_4c4f_504d;
const ECOLOGY_SEALED_DOMAIN: u64 = 0x5345_414c_4544_5f52;

pub const RESOURCE_ECOLOGY_RESULT_SCHEMA_VERSION: u32 = 8;

/// Search controls actually used by the asexual ticket ecology. Speciation,
/// crossover, and scalar-fitness selection are deliberately absent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsexualSearchConfig {
    pub population_size: usize,
    pub generations: u32,
    pub evaluation_workers: usize,
    pub mutate_weight_probability: f64,
    pub replace_weight_probability: f64,
    pub weight_perturb_stddev: f32,
    pub mutate_bias_probability: f64,
    pub bias_perturb_stddev: f32,
    pub mutate_time_constant_probability: f64,
    pub time_constant_perturb_stddev: f32,
    pub mutate_learning_rate_probability: f64,
    pub learning_rate_perturb_stddev: f32,
    pub mutate_plasticity_coefficient_probability: f64,
    pub plasticity_coefficient_perturb_stddev: f32,
    pub mutate_plasticity_receptor_probability: f64,
    pub plasticity_receptor_perturb_stddev: f32,
    pub add_connection_probability: f64,
    pub delete_connection_probability: f64,
    pub add_node_probability: f64,
    pub delete_node_probability: f64,
    /// Probability of the temporal-relay operator: insert a copy node that
    /// holds a source's value and immediately projects to an output, so exact
    /// memory can grow one *rewarded* tap at a time. Without it, multi-symbol
    /// memory needs many coordinated mutations whose payoff is deferred until
    /// the whole chain exists, which asexual search never crossed in 1,000
    /// generations even though the substrate could always express it.
    #[serde(default = "default_add_delay_relay_probability")]
    pub add_delay_relay_probability: f64,
    pub mutate_only_active_interface: bool,
    pub recurrent_node_self_connection: bool,
    /// Give every hidden node a plastic self-recurrent (previous-tick) loop at
    /// birth and after every mutation, turning each into a candidate memory
    /// cell. A lever for tasks that require holding a multi-symbol context,
    /// which the feedforward-plus-plastic-readout winner otherwise fails to
    /// form.
    #[serde(default)]
    pub self_recurrent_hidden: bool,
    /// Maintain a dense, plastic previous-tick hidden->hidden recurrent matrix
    /// at birth and after every mutation, mirroring the dense plastic readout.
    /// The canonical architecture pairs a dense feedforward readout with only
    /// sparse recurrence, structurally biasing search toward feedforward
    /// solutions; a full recurrent layer lets evolution and lifetime learning
    /// shape a genuine multi-symbol memory code.
    #[serde(default)]
    pub dense_recurrence: bool,
    /// Initialize founder hidden nodes with a wide spread of leak time
    /// constants instead of the default near-instant response, so the initial
    /// population holds a mix of fast feature detectors and slow integrators.
    /// The feedforward winner otherwise collapses every node to a fast time
    /// constant and never retains a slow memory trace.
    #[serde(default)]
    pub heterogeneous_time_constants: bool,
    /// Number of evolvable neuromodulatory feedback channels for the predictive-
    /// coding learner. Zero means output-only (hidden plasticity off). Only used
    /// by predictive-coding tasks; capped at MAX_FEEDBACK_CHANNELS.
    #[serde(default)]
    pub feedback_channels: usize,
    /// Change-activation operator: probability that one random hidden node's
    /// heritable transfer function is reassigned per offspring. Insert-node
    /// and founder hidden nodes always draw random activations.
    #[serde(default = "default_mutate_activation_probability")]
    pub mutate_activation_probability: f64,
    /// Fraction of possible founder (enabled sensor or seed hidden node) ->
    /// (enabled action or value) pairs enabled at birth. Founders are minimal:
    /// this interface wiring plus the canonical self-recurrent loop is the
    /// entire initial topology, and evolution owns readout wiring from there.
    /// Tasks whose solution is stored in plastic readout weights need a high
    /// fraction (next-token presets 1.0).
    #[serde(default = "default_initial_connection_fraction")]
    pub initial_connection_fraction: f64,
    /// Seed founders with an exact input delay line: `depth` banks of one
    /// node per enabled sensor, bank 0 a current-tick copy of the sensors and
    /// each later bank a previous-tick copy of the one before, built from
    /// weight-one non-plastic edges on saturating-linear, fastest-leak nodes.
    /// Bank nodes join the founder readout wiring, so the last `depth - 1`
    /// symbols are linearly separable at birth. A generic temporal wiring
    /// motif (no task knowledge); evolution owns it after birth. Zero
    /// disables. Motivation: fading self-loop memory plateaus below the
    /// two-character context ceiling and 1,000 generations never discovered a
    /// sharp multi-symbol code on their own.
    #[serde(default)]
    pub input_delay_line_depth: usize,
}

fn default_mutate_activation_probability() -> f64 {
    0.3
}

fn default_add_delay_relay_probability() -> f64 {
    0.05
}

fn default_initial_connection_fraction() -> f64 {
    0.25
}

impl Default for AsexualSearchConfig {
    fn default() -> Self {
        Self {
            population_size: 64,
            generations: 100,
            evaluation_workers: std::thread::available_parallelism()
                .map(|parallelism| parallelism.get())
                .unwrap_or(1),
            mutate_weight_probability: 0.75,
            replace_weight_probability: 0.05,
            weight_perturb_stddev: 0.1,
            mutate_bias_probability: 0.2,
            bias_perturb_stddev: 0.1,
            mutate_time_constant_probability: 0.1,
            time_constant_perturb_stddev: 0.15,
            mutate_learning_rate_probability: 0.2,
            learning_rate_perturb_stddev: 0.04,
            mutate_plasticity_coefficient_probability: 0.15,
            plasticity_coefficient_perturb_stddev: 0.1,
            mutate_plasticity_receptor_probability: 0.15,
            plasticity_receptor_perturb_stddev: 0.1,
            add_connection_probability: 0.25,
            delete_connection_probability: 0.09,
            add_node_probability: 0.2,
            delete_node_probability: 0.05,
            add_delay_relay_probability: default_add_delay_relay_probability(),
            mutate_only_active_interface: true,
            recurrent_node_self_connection: false,
            self_recurrent_hidden: false,
            dense_recurrence: false,
            heterogeneous_time_constants: false,
            feedback_channels: 0,
            mutate_activation_probability: default_mutate_activation_probability(),
            initial_connection_fraction: default_initial_connection_fraction(),
            input_delay_line_depth: 0,
        }
    }
}

impl AsexualSearchConfig {
    pub fn validate(&self) -> Result<()> {
        if self.population_size < 2 || self.generations == 0 || self.evaluation_workers == 0 {
            bail!("population_size must be at least 2; generations and evaluation_workers must be positive");
        }
        for (name, value) in [
            ("mutate_weight_probability", self.mutate_weight_probability),
            (
                "replace_weight_probability",
                self.replace_weight_probability,
            ),
            ("mutate_bias_probability", self.mutate_bias_probability),
            (
                "mutate_time_constant_probability",
                self.mutate_time_constant_probability,
            ),
            (
                "mutate_learning_rate_probability",
                self.mutate_learning_rate_probability,
            ),
            (
                "mutate_plasticity_coefficient_probability",
                self.mutate_plasticity_coefficient_probability,
            ),
            (
                "mutate_plasticity_receptor_probability",
                self.mutate_plasticity_receptor_probability,
            ),
            (
                "add_connection_probability",
                self.add_connection_probability,
            ),
            (
                "delete_connection_probability",
                self.delete_connection_probability,
            ),
            ("add_node_probability", self.add_node_probability),
            ("delete_node_probability", self.delete_node_probability),
            (
                "add_delay_relay_probability",
                self.add_delay_relay_probability,
            ),
            (
                "mutate_activation_probability",
                self.mutate_activation_probability,
            ),
            (
                "initial_connection_fraction",
                self.initial_connection_fraction,
            ),
        ] {
            if !(0.0..=1.0).contains(&value) {
                bail!("{name} must be in [0, 1]");
            }
        }
        if self.add_connection_probability
            + self.delete_connection_probability
            + self.add_node_probability
            + self.delete_node_probability
            + self.add_delay_relay_probability
            > 1.0
        {
            bail!("structural mutation probabilities must sum to at most 1");
        }
        for (name, value) in [
            ("weight_perturb_stddev", self.weight_perturb_stddev),
            ("bias_perturb_stddev", self.bias_perturb_stddev),
            (
                "time_constant_perturb_stddev",
                self.time_constant_perturb_stddev,
            ),
            (
                "learning_rate_perturb_stddev",
                self.learning_rate_perturb_stddev,
            ),
            (
                "plasticity_coefficient_perturb_stddev",
                self.plasticity_coefficient_perturb_stddev,
            ),
            (
                "plasticity_receptor_perturb_stddev",
                self.plasticity_receptor_perturb_stddev,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                bail!("{name} must be finite and nonnegative");
            }
        }
        if self.input_delay_line_depth > 8 {
            bail!("input_delay_line_depth must be at most 8");
        }
        if self.feedback_channels > types::MAX_FEEDBACK_CHANNELS {
            bail!(
                "feedback_channels {} exceeds MAX_FEEDBACK_CHANNELS {}",
                self.feedback_channels,
                types::MAX_FEEDBACK_CHANNELS
            );
        }
        Ok(())
    }
}

/// How the finite offspring slots are allocated across ticket-earning
/// parents. Every scheme ranks only by binary task-ticket counts (the resource
/// currency); none introduces a hand-designed scalar fitness.
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SelectionScheme {
    /// Fixed-K tournament. Selection pressure is invariant to population size,
    /// which also caps how many refinement trials a discovered leader receives
    /// regardless of width.
    #[default]
    Tournament,
    /// (mu, lambda) truncation: the top `truncation_survivors` ranked parents
    /// each receive an equal, deterministic share of the offspring slots. A
    /// fixed absolute survivor count turns added population width directly into
    /// more refinement trials for the leading lineages.
    Truncation,
    /// Ticket-proportional reproduction via stochastic universal sampling. A
    /// lineage that captured twice the resources receives twice the offspring
    /// in expectation, so replicator dynamics amplify a strong lineage
    /// multiplicatively over depth.
    Proportional,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceEcologyConfig {
    /// Highest-ticket incumbents copied exactly into the next generation
    /// before mutated offspring are sampled.
    pub exact_elite_copies: usize,
    /// Reproductive competitors sampled per offspring slot under the tournament
    /// scheme. Ranking uses only binary task-ticket counts; fixed K keeps
    /// selection pressure invariant as population size and absolute ticket
    /// production change.
    pub tournament_size: usize,
    /// Offspring-allocation scheme.
    #[serde(default)]
    pub selection: SelectionScheme,
    /// Number of surviving parents under `Truncation`. Zero requests the
    /// default heuristic (a small fixed fraction of the population).
    #[serde(default)]
    pub truncation_survivors: usize,
}

impl Default for ResourceEcologyConfig {
    fn default() -> Self {
        Self {
            exact_elite_copies: 1,
            tournament_size: 4,
            selection: SelectionScheme::Tournament,
            truncation_survivors: 0,
        }
    }
}

impl ResourceEcologyConfig {
    pub fn resolve_for_population(self, _population_size: usize) -> Self {
        self
    }

    pub fn validate(&self, population_size: usize) -> Result<()> {
        if self.exact_elite_copies > population_size {
            bail!(
                "exact_elite_copies {} exceeds population size {}",
                self.exact_elite_copies,
                population_size
            );
        }
        if self.tournament_size < 2 {
            bail!("tournament_size must be at least 2");
        }
        Ok(())
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct ResourceEcologyWork {
    pub lifetime_evaluations: u64,
    pub audit_evaluations: u64,
    pub offspring_generated: u64,
    pub brain_synapse_operations: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceEcologyPopulationMember<E> {
    pub population_index: usize,
    pub individual_id: u64,
    pub generation_reproductive_tickets: u64,
    pub evaluation: E,
    pub genome: OrganismGenome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceEcologyGenerationSummary<E, A> {
    pub generation: u32,
    pub reproductive_tickets: u64,
    pub offspring_slots: usize,
    pub reproducing_individuals: usize,
    /// Inverse Simpson concentration of ticket-producer shares. Equal ticket
    /// ownership by K producers yields K; concentration in one yields 1.
    pub effective_ticket_producer_count: f64,
    /// Distinct parents selected into the next generation, including elites.
    pub selected_parent_count: usize,
    /// Inverse Simpson concentration of realized offspring shares. Equal
    /// offspring from K selected parents yields K; one lineage yields 1.
    pub effective_selected_parent_count: f64,
    pub maximum_reproductive_tickets: u64,
    pub exact_elite_copies: usize,
    pub leading_population_index: usize,
    pub leading_individual_id: u64,
    pub leading_generation_reproductive_tickets: u64,
    pub leading_evaluation: E,
    pub leading_audit: Option<A>,
    pub leading_hidden_nodes: usize,
    pub leading_enabled_connections: usize,
    pub reproduction_applied: bool,
    pub work: ResourceEcologyWork,
    pub wall_time_seconds: f64,
    pub leading_genome: OrganismGenome,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResourceEcologyTerminationStatus {
    Completed,
    Extinct,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceEcologyTermination {
    pub status: ResourceEcologyTerminationStatus,
    pub configured_generations: u32,
    pub evaluated_generations: u32,
    pub extinction_generation: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceEcologyResult<C, E, A> {
    pub result_schema_version: u32,
    pub algorithm: String,
    pub task: String,
    pub objective: String,
    pub seed: u64,
    pub search_config: AsexualSearchConfig,
    pub ecology_config: ResourceEcologyConfig,
    pub task_config: C,
    pub seed_genome_config: SeedGenomeConfig,
    pub generations: Vec<ResourceEcologyGenerationSummary<E, A>>,
    pub termination: ResourceEcologyTermination,
    pub final_population: Vec<ResourceEcologyPopulationMember<E>>,
    pub final_leading_population_index: usize,
    pub selected_generation: u32,
    pub selected_individual_id: u64,
    pub selected_development_audit: A,
    pub selected_genome: OrganismGenome,
    pub sealed_audit: A,
    pub work: ResourceEcologyWork,
    pub total_wall_time_seconds: f64,
}

struct EcologyIndividual<E> {
    id: u64,
    genome: OrganismGenome,
    evaluation: Option<E>,
    generation_reproductive_tickets: u64,
}

#[derive(Clone, Copy)]
struct OffspringPlan {
    parent_index: usize,
    child_id: u64,
    mutation_seed: u64,
    mutate: bool,
}

pub fn run_resource_ecology<T: ResourceEcologyTask>(
    task: &T,
    search: AsexualSearchConfig,
    ecology: ResourceEcologyConfig,
    seed_genome_config: SeedGenomeConfig,
    seed: u64,
    mut on_generation: impl FnMut(
        &ResourceEcologyGenerationSummary<T::LifetimeEvaluation, T::AuditEvaluation>,
    ),
) -> Result<ResourceEcologyResult<T::Config, T::LifetimeEvaluation, T::AuditEvaluation>> {
    search.validate()?;
    let ecology = ecology.resolve_for_population(search.population_size);
    ecology.validate(search.population_size)?;
    task.validate()?;

    let run_started = Instant::now();
    let mut initialization_rng = ChaCha8Rng::seed_from_u64(seed);
    let mut population = Vec::with_capacity(search.population_size);
    for individual_id in 0..search.population_size as u64 {
        // Population width must widen structural as well as parameter search.
        // Cloning one randomly wired founder makes every initial organism share
        // the same representational bottlenecks and turns additional population
        // into redundant samples of only that topology.
        let mut genome = generate_seed_genome(&seed_genome_config, &mut initialization_rng);
        restrict_genome_to_task_interface(task, &mut genome);
        install_input_delay_lines(task, &mut genome, search.input_delay_line_depth);
        minimalize_founder_wiring(
            task,
            &mut genome,
            search.initial_connection_fraction,
            &mut initialization_rng,
        );
        randomize_parameters(task, &mut genome, &mut initialization_rng);
        ensure_recurrent_substrate(&search, &mut genome, &mut initialization_rng);
        for node in &mut genome.brain.hidden_nodes {
            node.activation_fn = random_activation(&mut initialization_rng);
        }
        if search.heterogeneous_time_constants {
            for node in &mut genome.brain.hidden_nodes {
                node.log_time_constant = initialization_rng
                    .random_range(-std::f32::consts::LN_10..=std::f32::consts::LN_10);
            }
        }
        initialize_feedback_channels(task, &search, &mut genome, &mut initialization_rng);
        canonicalize_delay_memory(&mut genome);
        population.push(EcologyIndividual {
            id: individual_id,
            genome,
            evaluation: None,
            generation_reproductive_tickets: 0,
        });
    }

    let evaluation_pool = EvaluationPool::new(search.evaluation_workers)?;
    let mut next_individual_id = search.population_size as u64;
    let mut generations = Vec::with_capacity(search.generations as usize);
    let mut total_work = ResourceEcologyWork::default();
    let mut final_leading_index = 0_usize;
    let mut termination_status = ResourceEcologyTerminationStatus::Completed;
    let mut extinction_generation = None;
    let mut selected_representative = None::<(u32, u64, f64, T::AuditEvaluation, OrganismGenome)>;

    for generation in 0..search.generations {
        let generation_started = Instant::now();
        // Evaluation is a pure population snapshot: no births, deaths, mating,
        // or replacement can occur until every lifetime has completed.
        // Population-wide flattened evaluation (single flat work-stealing pass
        // over all (genome, instance) units) with the historical per-genome
        // path as the trait default.
        let genomes = population
            .iter()
            .map(|individual| &individual.genome)
            .collect::<Vec<_>>();
        let ids = population
            .iter()
            .map(|individual| individual.id)
            .collect::<Vec<_>>();
        let outcomes =
            task.evaluate_population(&genomes, &ids, seed, generation, &evaluation_pool.pool)?;

        let mut reproductive_tickets = 0_u64;
        let mut reproducing_individuals = BTreeSet::new();
        let mut generation_work = ResourceEcologyWork {
            lifetime_evaluations: (population.len() as u64)
                .saturating_mul(task.evaluation_lifetimes() as u64),
            ..ResourceEcologyWork::default()
        };
        for (index, (individual, outcome)) in population.iter_mut().zip(outcomes).enumerate() {
            individual.generation_reproductive_tickets = outcome.reproductive_tickets;
            reproductive_tickets =
                reproductive_tickets.saturating_add(individual.generation_reproductive_tickets);
            if individual.generation_reproductive_tickets > 0 {
                reproducing_individuals.insert(index);
            }
            individual.evaluation = Some(outcome.evaluation);
            generation_work.brain_synapse_operations = generation_work
                .brain_synapse_operations
                .saturating_add(outcome.work.brain_synapse_operations);
        }

        final_leading_index = leading_index(&population);
        let squared_ticket_total = population
            .iter()
            .map(|individual| {
                let captures = individual.generation_reproductive_tickets as f64;
                captures * captures
            })
            .sum::<f64>();
        let effective_ticket_producer_count = if squared_ticket_total == 0.0 {
            0.0
        } else {
            (reproductive_tickets as f64).powi(2) / squared_ticket_total
        };
        let maximum_reproductive_tickets = population
            .iter()
            .map(|individual| individual.generation_reproductive_tickets)
            .max()
            .unwrap_or(0);
        // Extinction is a valid experimental outcome. Audit its final leader
        // even when it occurs before the next scheduled development audit so
        // the run can still produce a complete, sealed result artifact.
        let leading_audit =
            if reproductive_tickets == 0 || task.audit_due(generation, search.generations) {
                generation_work.audit_evaluations = 1;
                let audit = task.audit(
                    &population[final_leading_index].genome,
                    "development",
                    mix64(seed ^ ECOLOGY_AUDIT_DOMAIN ^ ECOLOGY_DEVELOPMENT_DOMAIN),
                )?;
                let score = task.audit_score(&audit);
                let replace = selected_representative
                    .as_ref()
                    .is_none_or(|(_, _, best_score, _, _)| score > *best_score);
                if replace {
                    selected_representative = Some((
                        generation,
                        population[final_leading_index].id,
                        score,
                        audit.clone(),
                        population[final_leading_index].genome.clone(),
                    ));
                }
                Some(audit)
            } else {
                None
            };
        let is_terminal_generation = generation + 1 == search.generations;
        let reproduction_applied = !is_terminal_generation && reproductive_tickets > 0;
        if reproduction_applied {
            generation_work.offspring_generated = population.len() as u64;
        }
        let plans = if reproduction_applied {
            // Tickets establish eligibility and rank. A fixed-size tournament
            // converts that task-relative ranking into N offspring without
            // making selection pressure depend on population size.
            let mut reproduction_rng = event_rng(
                seed ^ ECOLOGY_REPRODUCTION_DOMAIN,
                generation,
                ECOLOGY_REPRODUCTION_DOMAIN,
            );
            let mut ranked_reproducers =
                reproducing_individuals.iter().copied().collect::<Vec<_>>();
            ranked_reproducers.sort_unstable_by(|&left, &right| {
                population[right]
                    .generation_reproductive_tickets
                    .cmp(&population[left].generation_reproductive_tickets)
                    .then_with(|| population[left].id.cmp(&population[right].id))
            });
            let mut plans = Vec::with_capacity(population.len());
            for &parent_index in ranked_reproducers.iter().take(ecology.exact_elite_copies) {
                plans.push(OffspringPlan {
                    parent_index,
                    child_id: next_individual_id,
                    mutation_seed: reproduction_rng.next_u64(),
                    mutate: false,
                });
                next_individual_id = next_individual_id.saturating_add(1);
            }
            if ecology.selection == SelectionScheme::Tournament {
                // Preserve the exact per-slot draw interleaving of the fixed-K
                // tournament so this default path is byte-for-byte unchanged.
                while plans.len() < population.len() {
                    let mut parent_index = ranked_reproducers
                        [reproduction_rng.random_range(0..ranked_reproducers.len())];
                    for _ in 1..ecology.tournament_size {
                        let contender = ranked_reproducers
                            [reproduction_rng.random_range(0..ranked_reproducers.len())];
                        if population[contender].generation_reproductive_tickets
                            > population[parent_index].generation_reproductive_tickets
                        {
                            parent_index = contender;
                        }
                    }
                    plans.push(OffspringPlan {
                        parent_index,
                        child_id: next_individual_id,
                        mutation_seed: reproduction_rng.next_u64(),
                        mutate: true,
                    });
                    next_individual_id = next_individual_id.saturating_add(1);
                }
            } else {
                let remaining = population.len() - plans.len();
                let mutated_parents = select_mutated_parents(
                    &ecology,
                    &population,
                    &ranked_reproducers,
                    remaining,
                    &mut reproduction_rng,
                );
                for parent_index in mutated_parents {
                    plans.push(OffspringPlan {
                        parent_index,
                        child_id: next_individual_id,
                        mutation_seed: reproduction_rng.next_u64(),
                        mutate: true,
                    });
                    next_individual_id = next_individual_id.saturating_add(1);
                }
            }
            plans
        } else {
            Vec::new()
        };
        let mut offspring_by_parent = BTreeMap::<usize, u64>::new();
        for plan in &plans {
            *offspring_by_parent.entry(plan.parent_index).or_default() += 1;
        }
        let selected_parent_count = offspring_by_parent.len();
        let squared_offspring_total = offspring_by_parent
            .values()
            .map(|&count| (count as f64).powi(2))
            .sum::<f64>();
        let effective_selected_parent_count = if squared_offspring_total == 0.0 {
            0.0
        } else {
            (plans.len() as f64).powi(2) / squared_offspring_total
        };
        let summary = ResourceEcologyGenerationSummary {
            generation,
            reproductive_tickets,
            offspring_slots: population.len(),
            reproducing_individuals: reproducing_individuals.len(),
            effective_ticket_producer_count,
            selected_parent_count,
            effective_selected_parent_count,
            maximum_reproductive_tickets,
            exact_elite_copies: if reproduction_applied {
                ecology
                    .exact_elite_copies
                    .min(reproducing_individuals.len())
            } else {
                0
            },
            leading_population_index: final_leading_index,
            leading_individual_id: population[final_leading_index].id,
            leading_generation_reproductive_tickets: population[final_leading_index]
                .generation_reproductive_tickets,
            leading_evaluation: population[final_leading_index]
                .evaluation
                .clone()
                .expect("generation evaluates every individual"),
            leading_audit,
            leading_hidden_nodes: population[final_leading_index].genome.hidden_node_count(),
            leading_enabled_connections: population[final_leading_index]
                .genome
                .enabled_connection_count(),
            reproduction_applied,
            work: generation_work,
            wall_time_seconds: generation_started.elapsed().as_secs_f64(),
            leading_genome: population[final_leading_index].genome.clone(),
        };
        accumulate_work(&mut total_work, generation_work);
        on_generation(&summary);
        generations.push(summary);

        if is_terminal_generation {
            break;
        }
        if reproductive_tickets == 0 {
            termination_status = ResourceEcologyTerminationStatus::Extinct;
            extinction_generation = Some(generation);
            break;
        }

        let offspring = evaluation_pool.pool.install(|| {
            plans
                .par_iter()
                .map(|plan| {
                    let mut genome = population[plan.parent_index].genome.clone();
                    let mut mutation_rng = ChaCha8Rng::seed_from_u64(plan.mutation_seed);
                    if plan.mutate {
                        mutate(task, &mut genome, &search, &mut mutation_rng);
                    }
                    EcologyIndividual {
                        id: plan.child_id,
                        genome,
                        evaluation: None,
                        generation_reproductive_tickets: 0,
                    }
                })
                .collect::<Vec<_>>()
        });
        population = offspring;
    }

    let evaluated_generations = generations.len() as u32;
    let final_population = population
        .iter()
        .enumerate()
        .map(
            |(population_index, individual)| ResourceEcologyPopulationMember {
                population_index,
                individual_id: individual.id,
                generation_reproductive_tickets: individual.generation_reproductive_tickets,
                evaluation: individual
                    .evaluation
                    .clone()
                    .expect("terminal generation evaluates every individual"),
                genome: individual.genome.clone(),
            },
        )
        .collect::<Vec<_>>();
    let (
        selected_generation,
        selected_individual_id,
        _selected_score,
        selected_development_audit,
        selected_genome,
    ) = selected_representative.expect("terminal generation must receive a development audit");
    let sealed_audit = task.audit(
        &selected_genome,
        "sealed",
        mix64(seed ^ ECOLOGY_AUDIT_DOMAIN ^ ECOLOGY_SEALED_DOMAIN),
    )?;
    total_work.audit_evaluations = total_work.audit_evaluations.saturating_add(1);

    let configured_generations = search.generations;
    Ok(ResourceEcologyResult {
        result_schema_version: RESOURCE_ECOLOGY_RESULT_SCHEMA_VERSION,
        algorithm: "task_ecology_asexual_v1".to_owned(),
        task: task.name().to_owned(),
        objective: task.objective().to_owned(),
        seed,
        search_config: search,
        ecology_config: ecology,
        task_config: task.config(),
        seed_genome_config,
        generations,
        termination: ResourceEcologyTermination {
            status: termination_status,
            configured_generations,
            evaluated_generations,
            extinction_generation,
        },
        final_population,
        final_leading_population_index: final_leading_index,
        selected_generation,
        selected_individual_id,
        selected_development_audit,
        selected_genome,
        sealed_audit,
        work: total_work,
        total_wall_time_seconds: run_started.elapsed().as_secs_f64(),
    })
}

fn accumulate_work(total: &mut ResourceEcologyWork, generation: ResourceEcologyWork) {
    total.lifetime_evaluations = total
        .lifetime_evaluations
        .saturating_add(generation.lifetime_evaluations);
    total.audit_evaluations = total
        .audit_evaluations
        .saturating_add(generation.audit_evaluations);
    total.offspring_generated = total
        .offspring_generated
        .saturating_add(generation.offspring_generated);
    total.brain_synapse_operations = total
        .brain_synapse_operations
        .saturating_add(generation.brain_synapse_operations);
}

fn leading_index<E>(population: &[EcologyIndividual<E>]) -> usize {
    population
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.generation_reproductive_tickets
                .cmp(&right.generation_reproductive_tickets)
                .then_with(|| right.id.cmp(&left.id))
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .expect("resource ecology population is nonempty")
}

/// Choose parents for the mutated (non-elite) offspring slots under a
/// non-tournament scheme. `ranked_reproducers` is sorted best-ticket-first.
fn select_mutated_parents<E>(
    ecology: &ResourceEcologyConfig,
    population: &[EcologyIndividual<E>],
    ranked_reproducers: &[usize],
    count: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<usize> {
    if ranked_reproducers.is_empty() || count == 0 {
        return Vec::new();
    }
    match ecology.selection {
        SelectionScheme::Tournament => (0..count)
            .map(|_| {
                let mut parent = ranked_reproducers[rng.random_range(0..ranked_reproducers.len())];
                for _ in 1..ecology.tournament_size {
                    let contender =
                        ranked_reproducers[rng.random_range(0..ranked_reproducers.len())];
                    if population[contender].generation_reproductive_tickets
                        > population[parent].generation_reproductive_tickets
                    {
                        parent = contender;
                    }
                }
                parent
            })
            .collect(),
        SelectionScheme::Truncation => {
            // A fixed absolute survivor count is what turns added population
            // width into more refinement trials for the leading lineages: each
            // of the top `survivors` parents receives count/survivors slots
            // regardless of how large the population grows.
            let default_survivors = (population.len() / 8).max(2);
            let survivors = if ecology.truncation_survivors > 0 {
                ecology.truncation_survivors
            } else {
                default_survivors
            }
            .clamp(1, ranked_reproducers.len());
            (0..count)
                .map(|slot| ranked_reproducers[slot % survivors])
                .collect()
        }
        SelectionScheme::Proportional => {
            // Stochastic universal sampling by ticket count: one random offset,
            // then evenly spaced pointers across the cumulative ticket mass.
            let tickets = ranked_reproducers
                .iter()
                .map(|&idx| population[idx].generation_reproductive_tickets as f64)
                .collect::<Vec<_>>();
            let total = tickets.iter().sum::<f64>();
            if total <= 0.0 {
                return (0..count)
                    .map(|slot| ranked_reproducers[slot % ranked_reproducers.len()])
                    .collect();
            }
            let step = total / count as f64;
            let start = rng.random::<f64>() * step;
            let mut result = Vec::with_capacity(count);
            let mut cursor = 0usize;
            let mut cumulative = tickets[0];
            for slot in 0..count {
                let pointer = start + slot as f64 * step;
                while pointer > cumulative && cursor + 1 < ranked_reproducers.len() {
                    cursor += 1;
                    cumulative += tickets[cursor];
                }
                result.push(ranked_reproducers[cursor]);
            }
            result
        }
    }
}

struct EvaluationPool {
    pool: ThreadPool,
}

impl EvaluationPool {
    fn new(workers: usize) -> Result<Self> {
        if workers == 0 {
            bail!("evaluation workers must be positive");
        }
        let pool = ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|index| format!("ecology-evaluator-{index}"))
            .build()
            .map_err(|error| anyhow!("failed to build ecology evaluation pool: {error}"))?;
        Ok(Self { pool })
    }
}

fn mutate<T: GenomeTask>(
    task: &T,
    genome: &mut OrganismGenome,
    config: &AsexualSearchConfig,
    rng: &mut ChaCha8Rng,
) {
    if rng.random_bool(config.mutate_weight_probability) {
        let mut eligible = genome
            .brain
            .edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| {
                edge.enabled
                    && (!config.mutate_only_active_interface || edge_interface_enabled(task, edge))
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if rng.random_bool(config.replace_weight_probability) {
            if let Some(&index) = eligible.choose(rng) {
                genome.brain.edges[index].weight = random_weight(rng);
            }
        } else {
            // Search several parameter-space scales without making genome
            // growth either freeze every coordinate (always one) or destroy
            // accumulated subcircuits (always dense). The geometric portfolio
            // favors local moves while retaining progressively rarer broad ones.
            eligible.shuffle(rng);
            let perturbations = multiscale_mutation_count(eligible.len(), rng);
            for index in eligible.into_iter().take(perturbations) {
                let edge = &mut genome.brain.edges[index];
                edge.weight =
                    constrain_weight(edge.weight + normal(rng) * config.weight_perturb_stddev);
            }
        }
    }
    if rng.random_bool(config.mutate_bias_probability) {
        let eligible = Symbol::ALL
            .iter()
            .enumerate()
            .filter(|(_, symbol)| task.action_enabled(**symbol))
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        for index in eligible {
            genome.brain.action_biases[index] = (genome.brain.action_biases[index]
                + normal(rng) * config.bias_perturb_stddev)
                .clamp(-1.0, 1.0);
        }
        for node in &mut genome.brain.hidden_nodes {
            node.bias = (node.bias + normal(rng) * config.bias_perturb_stddev).clamp(-1.0, 1.0);
        }
        if task.value_prediction_enabled() {
            genome.brain.value_bias = (genome.brain.value_bias
                + normal(rng) * config.bias_perturb_stddev)
                .clamp(-1.0, 1.0);
        }
    }
    if rng.random_bool(config.mutate_time_constant_probability) {
        if let Some(node) = genome.brain.hidden_nodes.choose_mut(rng) {
            node.log_time_constant = (node.log_time_constant
                + normal(rng) * config.time_constant_perturb_stddev)
                .clamp(-std::f32::consts::LN_10, std::f32::consts::LN_10);
        }
    }
    if task.lifetime_learning_enabled() && rng.random_bool(config.mutate_learning_rate_probability)
    {
        genome.plasticity.initial_learning_rate = (genome.plasticity.initial_learning_rate
            + normal(rng) * config.learning_rate_perturb_stddev)
            .clamp(0.0, 1.0);
        genome.plasticity.eligibility_retention = (genome.plasticity.eligibility_retention
            + normal(rng) * config.learning_rate_perturb_stddev)
            .clamp(0.0, 1.0);
        genome.plasticity.fast_weight_retention = (genome.plasticity.fast_weight_retention
            + normal(rng) * config.learning_rate_perturb_stddev)
            .clamp(0.0, 1.0);
        genome.plasticity.max_weight_delta_per_tick = (genome.plasticity.max_weight_delta_per_tick
            + normal(rng) * config.learning_rate_perturb_stddev)
            .clamp(0.0, 1.0);
        genome.plasticity.action_temperature_scale =
            (genome.plasticity.action_temperature_scale.ln()
                + normal(rng) * config.learning_rate_perturb_stddev)
                .exp()
                .clamp(0.05, 4.0);
    }
    if task.lifetime_learning_enabled()
        && rng.random_bool(config.mutate_plasticity_coefficient_probability)
    {
        let eligible = genome
            .brain
            .edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| {
                edge.enabled
                    && (!config.mutate_only_active_interface || edge_interface_enabled(task, edge))
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if let Some(&index) = eligible.choose(rng) {
            let edge = &mut genome.brain.edges[index];
            edge.plasticity_coefficient = (edge.plasticity_coefficient
                + normal(rng) * config.plasticity_coefficient_perturb_stddev)
                .clamp(0.0, SYNAPSE_PLASTICITY_COEFFICIENT_MAX);
        }
    }
    let draw = rng.random::<f64>();
    let delete_connection_end =
        config.add_connection_probability + config.delete_connection_probability;
    let add_node_end = delete_connection_end + config.add_node_probability;
    let delete_node_end = add_node_end + config.delete_node_probability;
    if draw < config.add_connection_probability {
        mutate_add_connection(task, genome, rng);
    } else if draw < delete_connection_end {
        mutate_delete_connection(task, genome, config, rng);
    } else if draw < add_node_end {
        mutate_add_node(task, genome, config, rng);
    } else if draw < delete_node_end {
        // The guard lives inside the branch: letting it fail through to the
        // next arm would fire that operator outside its own probability band.
        if genome.brain.hidden_nodes.len() > 1 {
            mutate_delete_node(genome, rng);
        }
    } else if draw < delete_node_end + config.add_delay_relay_probability {
        mutate_add_delay_relay(task, genome, rng);
    }
    ensure_recurrent_substrate(config, genome, rng);
    align_genome_vectors(genome, rng);
    // Receptor expression is an additive search dimension. Keep its random
    // draws after the established parameter and structural operators so
    // enabling this mechanism does not silently change which legacy mutation
    // an offspring receives for the same seed.
    if task.lifetime_learning_enabled()
        && rng.random_bool(config.mutate_plasticity_receptor_probability)
    {
        if let Some(node) = genome.brain.hidden_nodes.choose_mut(rng) {
            node.plasticity_receptor = (node.plasticity_receptor
                + normal(rng) * config.plasticity_receptor_perturb_stddev)
                .clamp(-PLASTICITY_RECEPTOR_MAX, PLASTICITY_RECEPTOR_MAX);
        }
    }
    mutate_feedback_channels(task, genome, config, rng);
    // Change-activation operator: reassign one random hidden node's transfer
    // function uniformly among the other nine.
    if rng.random_bool(config.mutate_activation_probability) {
        if let Some(node) = genome.brain.hidden_nodes.choose_mut(rng) {
            let all = ActivationFunction::ALL;
            let current = all
                .iter()
                .position(|function| *function == node.activation_fn)
                .unwrap_or(0);
            let mut index = rng.random_range(0..all.len() - 1);
            if index >= current {
                index += 1;
            }
            node.activation_fn = all[index];
        }
    }
    canonicalize_delay_memory(genome);
}

fn random_activation(rng: &mut ChaCha8Rng) -> ActivationFunction {
    let all = ActivationFunction::ALL;
    all[rng.random_range(0..all.len())]
}

/// Seed-domain hidden-node index base reserved for delay-line buffer nodes.
/// Canonical founder hidden nodes occupy far smaller indices.
const DELAY_LINE_SEED_INDEX_BASE: u32 = 1_000_000;
const DELAY_RELAY_HASH_DOMAIN: u64 = 0x4445_4c41_595f_5250;

/// Stable identity of the relay that copies `source`. Deriving it from the
/// source makes the operator idempotent per source and lets relays chain:
/// the relay on a relay is a deeper tap of the same signal.
fn delay_relay_node_id(source: GeneNodeId) -> GeneNodeId {
    let span = u64::from(u32::MAX - DELAY_LINE_SEED_INDEX_BASE);
    let offset = (mix64(DELAY_RELAY_HASH_DOMAIN ^ source.0) % span) as u32;
    types::seed_hidden_gene_node_id(DELAY_LINE_SEED_INDEX_BASE + offset)
}

/// Push a canonical copy node plus its weight-one non-plastic copy edge.
fn push_relay_node(
    genome: &mut OrganismGenome,
    source: GeneNodeId,
    node_id: GeneNodeId,
    timing: SynapseTiming,
) {
    genome.brain.hidden_nodes.push(HiddenNodeGene {
        id: node_id,
        bias: 0.0,
        log_time_constant: brain::genome::INTER_LOG_TIME_CONSTANT_MIN,
        activation_fn: ActivationFunction::SaturatingLinear,
        plasticity_receptor: 0.0,
        feedback_receptors: types::zero_feedback_receptors(),
    });
    genome.brain.edges.push(SynapseGene {
        innovation: connection_innovation_id(source, node_id, timing),
        pre_node_id: source,
        post_node_id: node_id,
        timing,
        weight: 1.0,
        plasticity_coefficient: 0.0,
        enabled: true,
    });
}

/// Grow exact temporal memory one *rewarded* tap at a time.
///
/// The operator inserts a relay that copies a source and immediately projects
/// to a random output through a plastic edge, so the new tap is exploitable by
/// lifetime learning in the generation it appears — the property that let
/// single-mutation self-loops be discovered while multi-mutation delay chains
/// never were. Copy edges are weight-one and non-plastic on saturating-linear
/// fastest-leak nodes, so the relay holds its source rather than mixing it.
///
/// A sensor source yields a register (current-tick copy — previous-tick edges
/// may not originate at a sensor) plus one delay tap on top of it, so even the
/// first application produces real one-symbol memory. A hidden source yields a
/// single previous-tick tap; because relays are themselves hidden nodes, later
/// applications chain onto them and each additional tap deepens the memory by
/// one symbol. This installs no task knowledge — it is a temporal wiring
/// motif, and evolution owns it after birth.
fn mutate_add_delay_relay<T: GenomeTask>(
    task: &T,
    genome: &mut OrganismGenome,
    rng: &mut ChaCha8Rng,
) {
    if genome.brain.hidden_nodes.len() + 2 > MAX_INTER_NEURONS as usize {
        return;
    }
    let mut sources = SensoryReceptor::ordered()
        .filter(|sensor| task.sensor_enabled(*sensor))
        .filter_map(SensoryReceptor::neuron_id)
        .map(|id| (sensory_gene_node_id(id.0), true))
        .collect::<Vec<_>>();
    sources.extend(
        genome
            .brain
            .hidden_nodes
            .iter()
            .map(|node| (node.id, false)),
    );
    let Some(&(source, from_sensor)) = sources.choose(rng) else {
        return;
    };
    let mut outputs = Symbol::ALL
        .into_iter()
        .filter(|symbol| task.action_enabled(*symbol))
        .map(|symbol| action_gene_node_id(symbol.index()))
        .collect::<Vec<_>>();
    if task.value_prediction_enabled() {
        outputs.push(value_gene_node_id());
    }
    let Some(&output) = outputs.choose(rng) else {
        return;
    };

    // A sensor cannot originate a previous-tick edge, so its memory needs a
    // current-tick register first; the tap then delays that register.
    let tap_source = if from_sensor {
        let register = delay_relay_node_id(source);
        if !genome
            .brain
            .hidden_nodes
            .iter()
            .any(|node| node.id == register)
        {
            push_relay_node(genome, source, register, SynapseTiming::CurrentTick);
        }
        register
    } else {
        source
    };
    let tap = delay_relay_node_id(tap_source);
    if genome.brain.hidden_nodes.iter().any(|node| node.id == tap) {
        return;
    }
    push_relay_node(genome, tap_source, tap, SynapseTiming::PreviousTick);
    genome.brain.edges.push(SynapseGene {
        innovation: connection_innovation_id(tap, output, SynapseTiming::CurrentTick),
        pre_node_id: tap,
        post_node_id: output,
        timing: SynapseTiming::CurrentTick,
        weight: random_weight(rng),
        plasticity_coefficient: 1.0,
        enabled: true,
    });
}

fn delay_bank_node_id(bank: usize, slot: usize, slots: usize) -> GeneNodeId {
    types::seed_hidden_gene_node_id(DELAY_LINE_SEED_INDEX_BASE + (bank * slots + slot) as u32)
}

fn is_delay_line_node_id(id: GeneNodeId) -> bool {
    id >= types::seed_hidden_gene_node_id(DELAY_LINE_SEED_INDEX_BASE)
        && id <= types::seed_hidden_gene_node_id(u32::MAX)
}

/// Seed the founder with an exact input delay line: bank 0 copies each enabled
/// sensor on the current tick, and each later bank copies the previous bank's
/// node from the previous tick, so bank `b` holds the symbol observed `b`
/// ticks ago. Copy edges are weight-one and non-plastic; buffer nodes are
/// saturating-linear with the fastest leak. Installed before founder readout
/// wiring so the banks are visible to the plastic readout at birth, and
/// restored by `canonicalize_input_delay_lines` after parameter
/// randomization. Evolution owns the motif from birth onward.
fn install_input_delay_lines<T: GenomeTask>(task: &T, genome: &mut OrganismGenome, depth: usize) {
    if depth == 0 {
        return;
    }
    let sensors = SensoryReceptor::ordered()
        .filter(|sensor| task.sensor_enabled(*sensor))
        .filter_map(SensoryReceptor::neuron_id)
        .map(|id| sensory_gene_node_id(id.0))
        .collect::<Vec<_>>();
    if sensors.is_empty() {
        return;
    }
    let slots = sensors.len();
    for bank in 0..depth {
        for (slot, &sensor) in sensors.iter().enumerate() {
            let node_id = delay_bank_node_id(bank, slot, slots);
            genome.brain.hidden_nodes.push(HiddenNodeGene {
                id: node_id,
                bias: 0.0,
                log_time_constant: brain::genome::INTER_LOG_TIME_CONSTANT_MIN,
                activation_fn: ActivationFunction::SaturatingLinear,
                plasticity_receptor: 0.0,
                feedback_receptors: types::zero_feedback_receptors(),
            });
            let (pre, timing) = if bank == 0 {
                (sensor, SynapseTiming::CurrentTick)
            } else {
                (
                    delay_bank_node_id(bank - 1, slot, slots),
                    SynapseTiming::PreviousTick,
                )
            };
            genome.brain.edges.push(SynapseGene {
                innovation: connection_innovation_id(pre, node_id, timing),
                pre_node_id: pre,
                post_node_id: node_id,
                timing,
                weight: 1.0,
                plasticity_coefficient: 0.0,
                enabled: true,
            });
        }
    }
    genome
        .brain
        .hidden_nodes
        .sort_unstable_by_key(|node| node.id);
}

/// Hold delay memory to canonical form: exact-copy nodes and weight-one
/// non-plastic copy edges. Applied after founder randomization and after every
/// mutation, because a memory tap is only worth anything if it faithfully
/// holds its source — parameter mutation and lifetime learning would otherwise
/// scramble copies into ordinary noisy units within a few generations,
/// destroying exactly the property that makes a new tap immediately useful.
///
/// Evolution still owns the memory: it chooses whether a tap exists, what the
/// tap reads out (its readout edge is ordinary and fully plastic), and can
/// delete it. Only the copy itself is fixed, like a wire.
///
/// A copy edge is identified exactly, never guessed: an edge into a delay node
/// from that node's canonical source. Self-loops are excluded so an echo is
/// never converted into a copy.
fn canonicalize_delay_memory(genome: &mut OrganismGenome) {
    for node in &mut genome.brain.hidden_nodes {
        if is_delay_line_node_id(node.id) {
            node.bias = 0.0;
            node.log_time_constant = brain::genome::INTER_LOG_TIME_CONSTANT_MIN;
            node.activation_fn = ActivationFunction::SaturatingLinear;
            node.plasticity_receptor = 0.0;
        }
    }
    for edge in &mut genome.brain.edges {
        if is_delay_line_node_id(edge.post_node_id)
            && edge.pre_node_id != edge.post_node_id
            && (sensory_gene_node_index(edge.pre_node_id).is_some()
                || is_delay_line_node_id(edge.pre_node_id)
                || delay_relay_node_id(edge.pre_node_id) == edge.post_node_id)
        {
            edge.weight = 1.0;
            edge.plasticity_coefficient = 0.0;
        }
    }
}

/// Replace the founder's random dense wiring with WANN-style minimal wiring:
/// keep the seeded self-recurrent loop, drop every other edge, then enable
/// each (enabled sensor or hidden node) -> (enabled action or value) pair with
/// probability `fraction`, forcing at least one edge so no founder is mute.
fn minimalize_founder_wiring<T: GenomeTask>(
    task: &T,
    genome: &mut OrganismGenome,
    fraction: f64,
    rng: &mut ChaCha8Rng,
) {
    genome.brain.edges.retain(|edge| {
        (edge.timing == SynapseTiming::PreviousTick && edge.pre_node_id == edge.post_node_id)
            || is_delay_line_node_id(edge.post_node_id)
    });
    let sources = SensoryReceptor::ordered()
        .filter(|sensor| task.sensor_enabled(*sensor))
        .filter_map(SensoryReceptor::neuron_id)
        .map(|id| sensory_gene_node_id(id.0))
        .chain(genome.brain.hidden_nodes.iter().map(|node| node.id))
        .collect::<Vec<_>>();
    let mut sinks = Symbol::ALL
        .into_iter()
        .filter(|symbol| task.action_enabled(*symbol))
        .map(|symbol| action_gene_node_id(symbol.index()))
        .collect::<Vec<_>>();
    if task.value_prediction_enabled() {
        sinks.push(value_gene_node_id());
    }
    let mut pairs = Vec::with_capacity(sources.len() * sinks.len());
    for &pre in &sources {
        for &post in &sinks {
            pairs.push((pre, post));
        }
    }
    let mut enabled_any = false;
    for &(pre, post) in &pairs {
        if rng.random_bool(fraction) {
            push_interface_edge(genome, pre, post, rng);
            enabled_any = true;
        }
    }
    if !enabled_any && !pairs.is_empty() {
        let (pre, post) = pairs[rng.random_range(0..pairs.len())];
        push_interface_edge(genome, pre, post, rng);
    }
}

fn push_interface_edge(
    genome: &mut OrganismGenome,
    pre: GeneNodeId,
    post: GeneNodeId,
    rng: &mut ChaCha8Rng,
) {
    genome.brain.edges.push(SynapseGene {
        innovation: connection_innovation_id(pre, post, SynapseTiming::CurrentTick),
        pre_node_id: pre,
        post_node_id: post,
        timing: SynapseTiming::CurrentTick,
        weight: random_weight(rng),
        plasticity_coefficient: 1.0,
        enabled: true,
    });
}

fn feedback_channel_count(config: &AsexualSearchConfig) -> usize {
    config.feedback_channels.min(types::MAX_FEEDBACK_CHANNELS)
}

/// Seed a predictive-coding genome's neuromodulatory channels: small-random,
/// evolvable channel projections with neutral (zero) per-neuron receptors.
fn initialize_feedback_channels<T: GenomeTask>(
    task: &T,
    config: &AsexualSearchConfig,
    genome: &mut OrganismGenome,
    rng: &mut ChaCha8Rng,
) {
    let count = feedback_channel_count(config);
    if !task.predictive_coding_enabled() || count == 0 {
        genome.brain.feedback_channels.clear();
        return;
    }
    genome.brain.feedback_channels = (0..count * Symbol::COUNT)
        .map(|_| normal(rng) * 0.2)
        .collect();
    for node in &mut genome.brain.hidden_nodes {
        node.feedback_receptors = types::zero_feedback_receptors();
    }
}

/// Mutate the neuromodulatory machinery for predictive-coding genomes: recruit a
/// per-neuron receptor gain (the targeter) and refine what a channel encodes.
fn mutate_feedback_channels<T: GenomeTask>(
    task: &T,
    genome: &mut OrganismGenome,
    config: &AsexualSearchConfig,
    rng: &mut ChaCha8Rng,
) {
    let count = feedback_channel_count(config);
    if !task.predictive_coding_enabled() || count == 0 {
        return;
    }
    if rng.random_bool(config.mutate_plasticity_receptor_probability) {
        if let Some(node) = genome.brain.hidden_nodes.choose_mut(rng) {
            let channel = rng.random_range(0..count);
            node.feedback_receptors[channel] = (node.feedback_receptors[channel]
                + normal(rng) * config.plasticity_receptor_perturb_stddev)
                .clamp(-PLASTICITY_RECEPTOR_MAX, PLASTICITY_RECEPTOR_MAX);
        }
    }
    if rng.random_bool(config.mutate_plasticity_coefficient_probability)
        && !genome.brain.feedback_channels.is_empty()
    {
        let index = rng.random_range(0..genome.brain.feedback_channels.len());
        genome.brain.feedback_channels[index] = (genome.brain.feedback_channels[index]
            + normal(rng) * config.weight_perturb_stddev)
            .clamp(-SYNAPSE_STRENGTH_MAX, SYNAPSE_STRENGTH_MAX);
    }
}

fn mutate_delete_connection<T: GenomeTask>(
    task: &T,
    genome: &mut OrganismGenome,
    config: &AsexualSearchConfig,
    rng: &mut ChaCha8Rng,
) {
    let candidates = genome
        .brain
        .edges
        .iter()
        .enumerate()
        .filter(|(_, edge)| {
            edge.enabled
                && (!config.mutate_only_active_interface || edge_interface_enabled(task, edge))
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if let Some(&index) = candidates.choose(rng) {
        genome.brain.edges.remove(index);
    }
}

fn mutate_delete_node(genome: &mut OrganismGenome, rng: &mut ChaCha8Rng) {
    let Some(node) = genome.brain.hidden_nodes.choose(rng).copied() else {
        return;
    };
    genome
        .brain
        .hidden_nodes
        .retain(|candidate| candidate.id != node.id);
    genome
        .brain
        .edges
        .retain(|edge| edge.pre_node_id != node.id && edge.post_node_id != node.id);
}

fn mutate_add_connection<T: GenomeTask>(
    task: &T,
    genome: &mut OrganismGenome,
    rng: &mut ChaCha8Rng,
) {
    let sensors = SensoryReceptor::ordered()
        .filter(|sensor| task.sensor_enabled(*sensor))
        .filter_map(SensoryReceptor::neuron_id)
        .map(|id| sensory_gene_node_id(id.0))
        .collect::<Vec<_>>();
    let hidden = genome
        .brain
        .hidden_nodes
        .iter()
        .map(|node| node.id)
        .collect::<Vec<_>>();
    let actions = Symbol::ALL
        .into_iter()
        .filter(|symbol| task.action_enabled(*symbol))
        .map(|symbol| action_gene_node_id(symbol.index()))
        .collect::<Vec<_>>();
    let values = task
        .value_prediction_enabled()
        .then_some(value_gene_node_id())
        .into_iter()
        .collect::<Vec<_>>();
    let mut categories = Vec::new();
    for (pre_nodes, post_nodes, timing) in [
        (
            sensors.as_slice(),
            hidden.as_slice(),
            SynapseTiming::CurrentTick,
        ),
        (
            sensors.as_slice(),
            actions.as_slice(),
            SynapseTiming::CurrentTick,
        ),
        (
            hidden.as_slice(),
            hidden.as_slice(),
            SynapseTiming::CurrentTick,
        ),
        (
            hidden.as_slice(),
            actions.as_slice(),
            SynapseTiming::CurrentTick,
        ),
        (
            sensors.as_slice(),
            values.as_slice(),
            SynapseTiming::CurrentTick,
        ),
        (
            hidden.as_slice(),
            values.as_slice(),
            SynapseTiming::CurrentTick,
        ),
        (
            hidden.as_slice(),
            hidden.as_slice(),
            SynapseTiming::PreviousTick,
        ),
    ] {
        let mut candidates = Vec::new();
        for &pre in pre_nodes {
            for &post in post_nodes {
                collect_connection_candidate(genome, pre, post, timing, &mut candidates);
            }
        }
        if !candidates.is_empty() {
            categories.push(candidates);
        }
    }
    if task.action_feedback_enabled() {
        let mut candidates = Vec::new();
        for &pre in &actions {
            for &post in &hidden {
                collect_connection_candidate(
                    genome,
                    pre,
                    post,
                    SynapseTiming::PreviousTick,
                    &mut candidates,
                );
            }
        }
        if !candidates.is_empty() {
            categories.push(candidates);
        }
    }
    let Some(candidates) = categories.choose(rng) else {
        return;
    };
    let Some(&(pre, post, timing)) = candidates.choose(rng) else {
        return;
    };
    let innovation = connection_innovation_id(pre, post, timing);
    if let Some(edge) = genome
        .brain
        .edges
        .iter_mut()
        .find(|edge| edge.innovation == innovation)
    {
        edge.enabled = true;
        return;
    }
    genome.brain.edges.push(SynapseGene {
        innovation,
        pre_node_id: pre,
        post_node_id: post,
        timing,
        weight: random_weight(rng),
        plasticity_coefficient: 1.0,
        enabled: true,
    });
}

fn collect_connection_candidate(
    genome: &OrganismGenome,
    pre: GeneNodeId,
    post: GeneNodeId,
    timing: SynapseTiming,
    candidates: &mut Vec<(GeneNodeId, GeneNodeId, SynapseTiming)>,
) {
    if (timing == SynapseTiming::CurrentTick && action_gene_node_index(pre).is_some())
        || (timing == SynapseTiming::PreviousTick
            && (!(is_hidden_gene_node_id(pre) || action_gene_node_index(pre).is_some())
                || !is_hidden_gene_node_id(post)))
        || connection_would_create_cycle(genome, pre, post, timing)
    {
        return;
    }
    let innovation = connection_innovation_id(pre, post, timing);
    if !genome
        .brain
        .edges
        .iter()
        .any(|edge| edge.innovation == innovation && edge.enabled)
    {
        candidates.push((pre, post, timing));
    }
}

fn mutate_add_node<T: GenomeTask>(
    task: &T,
    genome: &mut OrganismGenome,
    config: &AsexualSearchConfig,
    rng: &mut ChaCha8Rng,
) {
    if genome.brain.hidden_nodes.len() >= MAX_INTER_NEURONS as usize {
        return;
    }
    let candidates = genome
        .brain
        .edges
        .iter()
        .enumerate()
        .filter(|(_, edge)| {
            edge.enabled
                && edge.timing == SynapseTiming::CurrentTick
                && (!config.mutate_only_active_interface || edge_interface_enabled(task, edge))
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let Some(&edge_index) = candidates.choose(rng) else {
        return;
    };
    let original = genome.brain.edges[edge_index];
    let node_id = split_hidden_gene_node_id(original.innovation);
    if genome
        .brain
        .hidden_nodes
        .iter()
        .any(|node| node.id == node_id)
    {
        return;
    }
    genome.brain.edges[edge_index].enabled = false;
    genome.brain.hidden_nodes.push(HiddenNodeGene {
        id: node_id,
        bias: 0.0,
        log_time_constant: 0.0,
        // Insert-node assigns the new node a random activation so the split
        // explores transfer-function space as well as wiring.
        activation_fn: random_activation(rng),
        plasticity_receptor: 0.0,
        feedback_receptors: types::zero_feedback_receptors(),
    });
    genome.brain.edges.push(SynapseGene {
        innovation: connection_innovation_id(
            original.pre_node_id,
            node_id,
            SynapseTiming::CurrentTick,
        ),
        pre_node_id: original.pre_node_id,
        post_node_id: node_id,
        timing: SynapseTiming::CurrentTick,
        weight: 1.0,
        plasticity_coefficient: original.plasticity_coefficient,
        enabled: true,
    });
    genome.brain.edges.push(SynapseGene {
        innovation: connection_innovation_id(
            node_id,
            original.post_node_id,
            SynapseTiming::CurrentTick,
        ),
        pre_node_id: node_id,
        post_node_id: original.post_node_id,
        timing: SynapseTiming::CurrentTick,
        weight: original.weight,
        plasticity_coefficient: original.plasticity_coefficient,
        enabled: true,
    });
    if config.recurrent_node_self_connection {
        genome.brain.edges.push(SynapseGene {
            innovation: connection_innovation_id(node_id, node_id, SynapseTiming::PreviousTick),
            pre_node_id: node_id,
            post_node_id: node_id,
            timing: SynapseTiming::PreviousTick,
            weight: random_weight(rng),
            plasticity_coefficient: 1.0,
            enabled: true,
        });
    }
}

fn edge_interface_enabled<T: GenomeTask>(task: &T, edge: &SynapseGene) -> bool {
    let pre = if let Some(index) = sensory_gene_node_index(edge.pre_node_id) {
        SensoryReceptor::from_neuron_id(NeuronId(index))
            .is_some_and(|sensor| task.sensor_enabled(sensor))
    } else if let Some(index) = action_gene_node_index(edge.pre_node_id) {
        task.action_feedback_enabled()
            && Symbol::ALL
                .get(index)
                .is_some_and(|symbol| task.action_enabled(*symbol))
    } else {
        true
    };
    let post = if is_value_gene_node_id(edge.post_node_id) {
        task.value_prediction_enabled()
    } else {
        action_gene_node_index(edge.post_node_id).is_none_or(|index| {
            Symbol::ALL
                .get(index)
                .is_some_and(|symbol| task.action_enabled(*symbol))
        })
    };
    pre && post
}

fn restrict_genome_to_task_interface<T: GenomeTask>(task: &T, genome: &mut OrganismGenome) {
    genome
        .brain
        .edges
        .retain(|edge| edge_interface_enabled(task, edge));
}

/// Enable a plastic previous-tick edge from `pre` to `post` (creating it with a
/// small random weight if absent). Previous-tick edges read last tick's state,
/// so they never form a current-tick cycle even for self-loops.
fn ensure_recurrent_edge(
    genome: &mut OrganismGenome,
    pre: GeneNodeId,
    post: GeneNodeId,
    rng: &mut ChaCha8Rng,
) {
    let innovation = connection_innovation_id(pre, post, SynapseTiming::PreviousTick);
    if let Some(edge) = genome
        .brain
        .edges
        .iter_mut()
        .find(|edge| edge.innovation == innovation)
    {
        edge.enabled = true;
        return;
    }
    genome.brain.edges.push(SynapseGene {
        innovation,
        pre_node_id: pre,
        post_node_id: post,
        timing: SynapseTiming::PreviousTick,
        weight: constrain_weight(normal(rng) * 0.2),
        plasticity_coefficient: 1.0,
        enabled: true,
    });
}

/// Give every hidden node a plastic self-recurrent loop (diagonal memory).
/// Delay-line buffer nodes are exempt: a self-echo would corrupt their exact
/// copy semantics.
fn ensure_self_recurrent_loops(genome: &mut OrganismGenome, rng: &mut ChaCha8Rng) {
    let ids = genome
        .brain
        .hidden_nodes
        .iter()
        .map(|node| node.id)
        .filter(|id| !is_delay_line_node_id(*id))
        .collect::<Vec<_>>();
    for id in ids {
        ensure_recurrent_edge(genome, id, id, rng);
    }
}

/// Maintain a dense plastic previous-tick hidden->hidden recurrent matrix,
/// mirroring the dense readout so search is not biased toward feedforward.
fn ensure_dense_recurrence(genome: &mut OrganismGenome, rng: &mut ChaCha8Rng) {
    let ids = genome
        .brain
        .hidden_nodes
        .iter()
        .map(|node| node.id)
        .collect::<Vec<_>>();
    for &pre in &ids {
        for &post in &ids {
            ensure_recurrent_edge(genome, pre, post, rng);
        }
    }
}

/// Apply whichever recurrent-substrate policy the search config requests.
fn ensure_recurrent_substrate(
    config: &AsexualSearchConfig,
    genome: &mut OrganismGenome,
    rng: &mut ChaCha8Rng,
) {
    if config.dense_recurrence {
        ensure_dense_recurrence(genome, rng);
    } else if config.self_recurrent_hidden {
        ensure_self_recurrent_loops(genome, rng);
    }
}

fn randomize_parameters<T: GenomeTask>(
    task: &T,
    genome: &mut OrganismGenome,
    rng: &mut ChaCha8Rng,
) {
    for edge in &mut genome.brain.edges {
        edge.weight = random_weight(rng);
        edge.plasticity_coefficient = rng.random_range(0.0..=SYNAPSE_PLASTICITY_COEFFICIENT_MAX);
    }
    for bias in &mut genome.brain.action_biases {
        *bias = normal(rng).clamp(-1.0, 1.0);
    }
    for node in &mut genome.brain.hidden_nodes {
        node.bias = normal(rng).clamp(-1.0, 1.0);
        // Whole-brain plasticity is an optional extension of the established
        // learner. Neutral initialization preserves that learner exactly;
        // evolution can recruit signed receptor expression by mutation.
        node.plasticity_receptor = 0.0;
    }
    genome.brain.value_bias = normal(rng).clamp(-1.0, 1.0);
    genome.plasticity.initial_learning_rate = rng.random_range(0.0..=0.5);
    genome.plasticity.eligibility_retention = rng.random_range(0.0..=1.0);
    genome.plasticity.fast_weight_retention = rng.random_range(0.0..=1.0);
    genome.plasticity.max_weight_delta_per_tick = rng.random_range(0.0..=1.0);
    genome.plasticity.action_temperature_scale =
        rng.random_range(0.1_f32.ln()..=2.0_f32.ln()).exp();
    let _ = task;
}

fn random_weight(rng: &mut ChaCha8Rng) -> f32 {
    constrain_weight(normal(rng) * 0.5)
}

fn multiscale_mutation_count(parameter_count: usize, rng: &mut ChaCha8Rng) -> usize {
    if parameter_count == 0 {
        return 0;
    }
    let mut count = 1;
    while count < parameter_count && rng.random_bool(0.5) {
        count = count.saturating_mul(2).min(parameter_count);
    }
    count
}
fn constrain_weight(weight: f32) -> f32 {
    if weight == 0.0 {
        SYNAPSE_STRENGTH_MIN
    } else {
        weight.signum()
            * weight
                .abs()
                .clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX)
    }
}
fn normal(rng: &mut ChaCha8Rng) -> f32 {
    StandardNormal.sample(rng)
}
fn event_rng(run_seed: u64, generation: u32, domain: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(mix64(run_seed ^ domain ^ (u64::from(generation) << 32)))
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
