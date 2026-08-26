use crate::learning::{
    accumulate_synaptic_eligibilities, apply_three_factor_learning, ActionLearningSignal,
    EligibilityNormalization, EligibilityRequest, ThreeFactorLearningRequest,
};
#[cfg(feature = "profiling")]
use crate::profiling::{self, BrainStage};
use crate::topology::refresh_output_synapse_starts_and_count;
use crate::BrainScratch;
#[cfg(feature = "profiling")]
use std::time::Instant;
use types::{BrainState, OrganismState};

const SYNAPSE_PRUNE_INTERVAL_TICKS: u64 = 10;
const PRUNE_ELIGIBILITY_MULTIPLIER: f32 = 2.0;
const WORLD_REWARD_SCALE: f32 = 5.0;

/// Stage the same local sensitivities used by symbolic task ecologies while
/// the world still owns the frozen previous-tick recurrent/action sources.
/// The signed outcome is unavailable until commit, so the common learner
/// applies the third factor later in `apply_runtime_weight_updates`.
pub fn accumulate_runtime_eligibilities(
    organism: &mut OrganismState,
    scratch: &mut BrainScratch,
    selected: types::Symbol,
    leaky_neurons_enabled: bool,
) {
    let effective_eta = organism.genome.plasticity.initial_learning_rate.max(0.0)
        * learning_rate_scale_at_age(&organism.genome, organism.age_turns + 1);
    if effective_eta == 0.0 {
        return;
    }

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    let value_prediction = organism.brain.reward_prediction;
    accumulate_synaptic_eligibilities(
        &mut organism.brain,
        scratch,
        EligibilityRequest {
            action_signal: ActionLearningSignal::RewardPredictionError { selected },
            value_prediction,
            normalization: EligibilityNormalization::None,
            leaky_neurons_enabled,
            eligibility_retention: organism.genome.plasticity.eligibility_retention,
        },
    );
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticityInterTuning, stage_started.elapsed());
}

pub fn apply_runtime_weight_updates(organism: &mut OrganismState) {
    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();

    let eta =
        organism.genome.plasticity.initial_learning_rate.max(0.0) * learning_rate_scale(organism);
    if eta == 0.0 {
        return;
    }
    let normalized_reward = ((organism.energy as f32 - organism.energy_at_last_sensing as f32)
        / WORLD_REWARD_SCALE)
        .clamp(-1.0, 1.0);
    let prediction_error = normalized_reward - organism.brain.reward_prediction;
    apply_three_factor_learning(
        &mut organism.brain,
        ThreeFactorLearningRequest {
            action_signal: ActionLearningSignal::RewardPredictionError {
                selected: organism.last_action_symbol,
            },
            reward_prediction_error: prediction_error,
            learning_rate: eta,
            eligibility_retention: organism.genome.plasticity.eligibility_retention,
            fast_weight_retention: organism.genome.plasticity.fast_weight_retention,
            max_weight_delta: organism.genome.plasticity.max_weight_delta_per_tick,
            hidden_categorical_feedback: false,
            predictive_coding: false,
        },
    );

    if should_prune_synapses(
        organism.age_turns,
        organism.genome.lifecycle.plasticity_maturity_ticks,
    ) {
        prune_low_weight_synapses(
            &mut organism.brain,
            organism.genome.plasticity.synapse_prune_threshold.max(0.0),
        );
    }

    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticityInterTuning, stage_started.elapsed());
}

/// Maturity-dependent learning-rate scale at the organism's current age.
pub fn learning_rate_scale(organism: &OrganismState) -> f32 {
    learning_rate_scale_at_age(&organism.genome, organism.age_turns)
}

fn learning_rate_scale_at_age(genome: &types::OrganismGenome, age_turns: u64) -> f32 {
    let is_mature = age_turns >= u64::from(genome.lifecycle.plasticity_maturity_ticks);
    if is_mature {
        1.0
    } else {
        genome.plasticity.juvenile_eta_scale.max(0.0)
    }
}

fn should_prune_synapses(age_turns: u64, plasticity_maturity_ticks: u32) -> bool {
    let maturity_ticks = u64::from(plasticity_maturity_ticks);
    age_turns >= maturity_ticks && age_turns.is_multiple_of(SYNAPSE_PRUNE_INTERVAL_TICKS)
}

fn prune_low_weight_synapses(brain: &mut BrainState, threshold: f32) -> bool {
    let mut pruned_any = false;
    let mut prune_group = |edges: &mut Vec<types::SynapseEdge>| {
        let before = edges.len();
        edges.retain(|synapse| {
            synapse.weight.abs() >= threshold
                || synapse.eligibility.abs() >= PRUNE_ELIGIBILITY_MULTIPLIER * threshold
        });
        pruned_any |= edges.len() != before;
    };
    for sensory in &mut brain.sensory {
        prune_group(&mut sensory.synapses);
    }
    for hidden in &mut brain.inter {
        prune_group(&mut hidden.synapses);
    }
    prune_group(&mut brain.recurrent_synapses);
    prune_group(&mut brain.action_feedback_synapses);
    if pruned_any {
        refresh_output_synapse_starts_and_count(brain);
    }
    pruned_any
}
