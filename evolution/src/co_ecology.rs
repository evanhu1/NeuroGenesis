//! Adversarial text-forge co-evolution for symbolic word ecologies.
//!
//! Two populations evolve against each other: brains (the standard ticket
//! asexual search) and a *forge* population — parameterizations of the
//! generated-text distribution (per-word sampling weights plus a long-range
//! repetition rate). Each generation, the current champion forge biases half
//! of the brain training panel; forge fitness is the failure rate of the best
//! evolving brains after within-lifetime acquisition attempts on forge text.
//! Brains improve → forges shift toward distributions the brains have not
//! mastered → the difficulty ratchet turns without any hand-authored ladder.
//!
//! Integrity contract: development and sealed audits always evaluate the
//! neutral generator (no bias, no repetition), so every co-evolved run stays
//! directly comparable to static-baseline runs. The other half of each
//! training panel remains neutral too — guaranteed ticket availability, no
//! extinction-by-pathological-forge failure mode.

use crate::{
    run_resource_ecology, AgentEvaluationConfig, ResourceEcologyConfig, ResourceEcologyTask,
    TaskEcology,
};
use anyhow::{bail, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::{Arc, Mutex};
use task_library::next_token_prediction::{NextTokenPredictionConfig, LEXICON};
use types::OrganismGenome;

const FORGE_DOMAIN: u64 = 0x464F_5247_455F_434F; // "FORGE_CO"

/// One adversarial environment genome: word-sampling log-weights over the
/// shared lexicon plus a long-range repetition rate. Repetition injects exact
/// word repeats separated by dozens of characters — a direct memory demand
/// that shallow context filtering cannot satisfy.
#[derive(Debug, Clone)]
pub struct WordForgeGenome {
    pub log_weights: Vec<f32>,
    pub repeat_rate: f32,
}

impl WordForgeGenome {
    fn random(run_seed: u64, forge_index: usize) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(mix64(
            run_seed ^ FORGE_DOMAIN ^ (forge_index as u64).rotate_left(31),
        ));
        Self {
            log_weights: (0..LEXICON.len())
                .map(|_| rng.random_range(-0.5..0.5))
                .collect(),
            repeat_rate: rng.random_range(0.0..0.25),
        }
    }

    fn mutate(&mut self, rng: &mut ChaCha8Rng) {
        let touches = rng.random_range(1..=4);
        for _ in 0..touches {
            let index = rng.random_range(0..self.log_weights.len());
            self.log_weights[index] =
                (self.log_weights[index] + rng.random_range(-0.3..0.3)).clamp(-2.5, 2.5);
        }
        self.repeat_rate = (self.repeat_rate + rng.random_range(-0.05..0.05)).clamp(0.0, 0.6);
    }

    /// Shannon entropy of the induced word distribution (nats) — ratchet
    /// telemetry: collapsing toward a few words shows up as falling entropy.
    fn weight_entropy(&self) -> f32 {
        let max = self.log_weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = self.log_weights.iter().map(|w| (*w - max).exp()).collect();
        let total: f32 = weights.iter().sum();
        weights
            .iter()
            .filter_map(|w| {
                let p = w / total;
                (p > 0.0).then(|| -(p * p.ln()))
            })
            .sum::<f32>()
            .abs()
    }

    fn apply_to(&self, config: &mut NextTokenPredictionConfig) {
        config.lexicon_bias = self.log_weights.clone();
        config.repeat_rate = self.repeat_rate;
    }
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

struct RankedBrain {
    tickets: u64,
    individual_id: u64,
    genome: OrganismGenome,
}

struct ForgePointInternal {
    generation: u32,
    champion_hardness: f32,
    mean_hardness: f32,
    mean_repeat_rate: f32,
    champion_entropy: f32,
}

/// Shared mutable co-evolution state. Interior mutability is required because
/// `ResourceEcologyTask` methods receive `&self`; determinism is preserved
/// because generation advancement fires exactly once per boundary (guarded by
/// `finalized_generation`) and depends only on completed-generation data.
struct ForgeShared {
    run_seed: u64,
    snippet_length: usize,
    forge_snippets: usize,
    learning_passes: usize,
    reference_brains: usize,
    forges: Vec<(u64, WordForgeGenome)>,
    finalized_generation: u32,
    best_brains: Vec<RankedBrain>,
    trajectory: Vec<ForgePointInternal>,
    next_forge_id: u64,
}

impl ForgeShared {
    fn champion(&self) -> WordForgeGenome {
        // Hardest forge from the latest ranking (index 0 after each advance).
        self.forges.first().map(|(_, g)| g.clone()).unwrap_or_else(|| WordForgeGenome {
            log_weights: vec![0.0; LEXICON.len()],
            repeat_rate: 0.0,
        })
    }

    fn advance_if_due(&mut self, generation: u32) -> Result<()> {
        if self.finalized_generation >= generation {
            return Ok(());
        }
        if generation > 0 && !self.best_brains.is_empty() {
            let mut scored: Vec<(f32, u64, &WordForgeGenome)> = Vec::new();
            for (forge_id, forge) in &self.forges {
                let mut accuracy_sum = 0.0_f64;
                for ranked in &self.best_brains {
                    accuracy_sum += score_forge_against_brain(&ForgeScoreRequest {
                        forge,
                        brain_genome: &ranked.genome,
                        run_seed: self.run_seed,
                        generation: generation.saturating_sub(1),
                        forge_id: *forge_id,
                        snippets: self.forge_snippets,
                        snippet_length: self.snippet_length,
                        learning_passes: self.learning_passes,
                    })?;
                }
                let mean_accuracy = accuracy_sum / self.best_brains.len() as f64;
                scored.push(((1.0 - mean_accuracy) as f32, *forge_id, forge));
            }
            // Hardest first; ties broken toward the older (lower-id) lineage.
            scored.sort_by(|l, r| r.0.total_cmp(&l.0).then_with(|| l.1.cmp(&r.1)));
            self.trajectory.push(ForgePointInternal {
                generation: generation - 1,
                champion_hardness: scored[0].0,
                mean_hardness: scored.iter().map(|(h, _, _)| h).sum::<f32>() / scored.len() as f32,
                mean_repeat_rate: scored.iter().map(|(_, _, g)| g.repeat_rate).sum::<f32>()
                    / scored.len() as f32,
                champion_entropy: scored[0].2.weight_entropy(),
            });

            // Reproduce: exact elite + fixed-K tournaments over hardness.
            let mut rng = ChaCha8Rng::seed_from_u64(mix64(
                self.run_seed ^ FORGE_DOMAIN ^ (generation as u64).rotate_left(17),
            ));
            let mut next = Vec::with_capacity(self.forges.len());
            next.push((self.next_forge_id, scored[0].2.clone()));
            self.next_forge_id += 1;
            while next.len() < self.forges.len() {
                let mut parent = scored[rng.random_range(0..scored.len())];
                for _ in 1..4 {
                    let contender = scored[rng.random_range(0..scored.len())];
                    if contender.0 > parent.0 {
                        parent = contender;
                    }
                }
                let mut child = parent.2.clone();
                child.mutate(&mut rng);
                next.push((self.next_forge_id, child));
                self.next_forge_id += 1;
            }
            self.forges = next;
        }
        self.finalized_generation = generation;
        self.best_brains.clear();
        Ok(())
    }
}

pub struct CoEvolutionaryTask {
    inner: TaskEcology<task_library::next_token_prediction::NextTokenPredictionTask>,
    shared: Arc<Mutex<ForgeShared>>,
}

impl CoEvolutionaryTask {
    pub fn new(
        config: NextTokenPredictionConfig,
        agent: AgentEvaluationConfig,
        run_seed: u64,
        forge_population: usize,
        forge_snippets: usize,
    ) -> Self {
        let snippet_length = config.snippet_length;
        let learning_passes = config.learning_passes;
        let forges = (0..forge_population)
            .map(|index| (index as u64, WordForgeGenome::random(run_seed, index)))
            .collect();
        Self {
            inner: TaskEcology::new(
                task_library::next_token_prediction::NextTokenPredictionTask { config },
                agent.clone(),
            ),
            shared: Arc::new(Mutex::new(ForgeShared {
                run_seed,
                snippet_length,
                forge_snippets,
                learning_passes,
                reference_brains: 3,
                forges,
                finalized_generation: 0,
                best_brains: Vec::new(),
                trajectory: Vec::new(),
                next_forge_id: forge_population as u64,
            })),
        }
    }
}

impl crate::GenomeTask for CoEvolutionaryTask {
    fn sensor_enabled(&self, sensor: types::SensoryReceptor) -> bool {
        self.inner.sensor_enabled(sensor)
    }
    fn action_enabled(&self, symbol: types::Symbol) -> bool {
        self.inner.action_enabled(symbol)
    }
}

impl ResourceEcologyTask for CoEvolutionaryTask {
    type Config = <TaskEcology<
        task_library::next_token_prediction::NextTokenPredictionTask,
    > as ResourceEcologyTask>::Config;
    type LifetimeState = <TaskEcology<
        task_library::next_token_prediction::NextTokenPredictionTask,
    > as ResourceEcologyTask>::LifetimeState;
    type LifetimeEvaluation = <TaskEcology<
        task_library::next_token_prediction::NextTokenPredictionTask,
    > as ResourceEcologyTask>::LifetimeEvaluation;
    type AuditEvaluation = <TaskEcology<
        task_library::next_token_prediction::NextTokenPredictionTask,
    > as ResourceEcologyTask>::AuditEvaluation;

    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn objective(&self) -> &'static str {
        "coevolutionary_forged_text_capture"
    }
    fn config(&self) -> Self::Config {
        self.inner.config()
    }
    fn lifetime_ticks(&self) -> usize {
        self.inner.lifetime_ticks()
    }
    fn evaluation_lifetimes(&self) -> usize {
        self.inner.evaluation_lifetimes()
    }
    fn validate(&self) -> Result<()> {
        self.inner.validate()?;
        if !self.inner.task.config.generalize {
            bail!("co-evolution requires --generalize true (forged panels are generated text)");
        }
        let state = self.shared.lock().expect("forge state lock");
        if state.forges.is_empty() {
            bail!("co-evolution requires a non-empty forge population");
        }
        Ok(())
    }

    fn initialize_lifetime(
        &self,
        genome: &OrganismGenome,
        individual_id: u64,
        run_seed: u64,
        generation: u32,
    ) -> Result<Self::LifetimeState> {
        let champion = {
            let mut state = self.shared.lock().expect("forge state lock");
            state.advance_if_due(generation)?;
            state.champion()
        };
        // Panel split: neutral half preserves the static-baseline ticket floor;
        // forged half carries the adversarial pressure.
        let total = self.inner.evaluation_lifetimes();
        let forged_count = total / 2;
        let neutral_count = total - forged_count;
        let mut combined = CombinedState::empty();

        if neutral_count > 0 {
            let mut neutral_agent = self.inner.agent.clone();
            neutral_agent.training_instances = neutral_count;
            let neutral_ecology = TaskEcology::new(self.inner.task.clone(), neutral_agent);
            let mut neutral_state = neutral_ecology.initialize_lifetime(
                genome,
                individual_id,
                run_seed,
                generation,
            )?;
            combined.append_from(&mut neutral_state);
        }
        if forged_count > 0 {
            let mut forged_config = self.inner.task.config.clone();
            champion.apply_to(&mut forged_config);
            let mut forged_agent = self.inner.agent.clone();
            forged_agent.training_instances = forged_count;
            let forged_ecology = TaskEcology::new(
                task_library::next_token_prediction::NextTokenPredictionTask {
                    config: forged_config,
                },
                forged_agent,
            );
            let forged_panel_seed = mix64(run_seed ^ FORGE_DOMAIN ^ (generation as u64));
            let mut forged_state = forged_ecology.initialize_lifetime(
                genome,
                individual_id,
                forged_panel_seed,
                generation,
            )?;
            combined.append_from(&mut forged_state);
        }
        Ok(combined)
    }

    fn evaluate_lifetime(
        &self,
        genome: &OrganismGenome,
        state: &mut Self::LifetimeState,
        context: crate::ResourceLifetimeContext,
    ) -> Result<crate::ResourceLifetimeOutcome<Self::LifetimeEvaluation>> {
        let outcome = self.inner.evaluate_lifetime(genome, state, context)?;
        {
            let mut shared = self.shared.lock().expect("forge state lock");
            let capacity = shared.reference_brains;
            insert_ranked_brain(
                &mut shared.best_brains,
                capacity,
                RankedBrain {
                    tickets: outcome.reproductive_tickets,
                    individual_id: context.individual_id,
                    genome: genome.clone(),
                },
            );
        }
        Ok(outcome)
    }

    fn audit(
        &self,
        genome: &OrganismGenome,
        cohort: &str,
        audit_seed: u64,
    ) -> Result<Self::AuditEvaluation> {
        // Neutral generator: measurement integrity independent of forge state.
        self.inner.audit(genome, cohort, audit_seed)
    }

    fn audit_score(&self, audit: &Self::AuditEvaluation) -> f64 {
        self.inner.audit_score(audit)
    }

    fn audit_due(&self, generation: u32, total_generations: u32) -> bool {
        self.inner.audit_due(generation, total_generations)
    }
}

type CombinedState = <
    TaskEcology<task_library::next_token_prediction::NextTokenPredictionTask>
    as ResourceEcologyTask
>::LifetimeState;

fn insert_ranked_brain(brains: &mut Vec<RankedBrain>, capacity: usize, candidate: RankedBrain) {
    brains.push(candidate);
    brains.sort_by(|left, right| {
        right
            .tickets
            .cmp(&left.tickets)
            .then_with(|| left.individual_id.cmp(&right.individual_id))
    });
    brains.truncate(capacity.max(1));
}

type CoEvolutionResult = crate::ResourceEcologyResult<
    <TaskEcology<task_library::next_token_prediction::NextTokenPredictionTask>
     as ResourceEcologyTask>::Config,
    crate::SymbolicEcologyMetrics,
    crate::SymbolicEcologyAudit,
>;

/// Inputs to one forge-hardness scoring pass.
struct ForgeScoreRequest<'a> {
    forge: &'a WordForgeGenome,
    brain_genome: &'a OrganismGenome,
    run_seed: u64,
    generation: u32,
    forge_id: u64,
    snippets: usize,
    snippet_length: usize,
    learning_passes: usize,
}

/// Accuracy of one brain after within-lifetime acquisition on a forge-biased
/// panel. Reuses the canonical evaluator end-to-end (learning passes included)
/// so hardness reflects genuine acquisition difficulty, not raw transfer.
fn score_forge_against_brain(req: &ForgeScoreRequest) -> Result<f64> {
    let ForgeScoreRequest {
        forge,
        brain_genome,
        run_seed,
        generation,
        forge_id,
        snippets,
        snippet_length,
        learning_passes,
    } = req;
    let mut config = NextTokenPredictionConfig {
        generalize: true,
        snippet_length: *snippet_length,
        learning_passes: (*learning_passes).max(1),
        predictive_coding: true,
        ..Default::default()
    };
    forge.apply_to(&mut config);
    let task = task_library::next_token_prediction::NextTokenPredictionTask { config };
    let agent = AgentEvaluationConfig {
        training_instances: (*snippets).max(1),
        exploration_temperature: 1.0,
        action_selection: crate::ActionSelection::Greedy,
        ..Default::default()
    };
    let ecology = TaskEcology::new(task, agent);
    let scoped_seed = mix64(*run_seed ^ FORGE_DOMAIN ^ forge_id.rotate_left(13));
    let mut state = ecology.initialize_lifetime(brain_genome, 0, scoped_seed, *generation)?;
    let metrics = ecology.evaluate_instances_serial(brain_genome, state.instances_mut());
    Ok(metrics.accuracy)
}


/// Run the dual-population co-evolutionary search. Returns the standard result
/// artifact payload plus the forge-pressure trajectory sidecar payload.
#[allow(clippy::too_many_arguments)]
pub fn run_co_evolution_next_token(
    task_config: NextTokenPredictionConfig,
    search: crate::AsexualSearchConfig,
    ecology: ResourceEcologyConfig,
    seed_genome_config: types::SeedGenomeConfig,
    seed: u64,
    agent: AgentEvaluationConfig,
    forge_population: usize,
    forge_snippets: usize,
    on_generation: impl FnMut(
        &crate::ResourceEcologyGenerationSummary<
            crate::SymbolicEcologyMetrics,
            crate::SymbolicEcologyAudit,
        >,
    ),
) -> Result<(CoEvolutionResult, serde_json::Value)> {
    if forge_population < 2 {
        bail!("forge population must be at least 2");
    }
    if forge_snippets == 0 {
        bail!("forge snippets per scoring round must be positive");
    }
    let task = CoEvolutionaryTask::new(task_config, agent, seed, forge_population, forge_snippets);
    let shared = Arc::clone(&task.shared);
    let result =
        run_resource_ecology(&task, search, ecology, seed_genome_config, seed, on_generation)?;
    let trajectory = {
        let state = shared.lock().expect("forge state lock");
        serde_json::json!({
            "algorithm_extension": "co_evolution_text_forge_v1",
            "forge_population": state.forges.len(),
            "snippet_length": state.snippet_length,
            "points": state.trajectory.iter().map(|p| serde_json::json!({
                "generation": p.generation,
                "champion_hardness": p.champion_hardness,
                "mean_hardness": p.mean_hardness,
                "mean_repeat_rate": p.mean_repeat_rate,
                "champion_weight_entropy": p.champion_entropy,
            })).collect::<Vec<_>>(),
        })
    };
    Ok((result, trajectory))
}
