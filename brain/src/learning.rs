use crate::topology::{action_array_index, constrain_weight, is_value_neuron_id, ACTION_COUNT};
use crate::BrainScratch;
use types::{BrainState, Symbol, SynapseEdge, MAX_FEEDBACK_CHANNELS};

const NLMS_EPSILON: f32 = 1.0e-6;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum EligibilityNormalization {
    #[default]
    None,
    NormalizedLeastMeanSquares,
}

/// What event supplies the action neuron's postsynaptic learning signal.
/// Hidden and value targets always use the scalar reward-prediction error.
#[derive(Debug, Clone, Copy)]
pub enum ActionLearningSignal {
    /// Reward-modulated action learning. Only the selected action accumulates
    /// new eligibility, while older traces can retain temporal credit.
    RewardPredictionError { selected: Symbol },
    /// A learner-visible categorical target revealed after prediction. Action
    /// outputs use their exact local error and do not retain cross-example
    /// eligibility; hidden/value synapses still use signed reward surprise.
    CategoricalPredictionError {
        target: Symbol,
        probabilities: [f32; ACTION_COUNT],
    },
}

#[derive(Debug, Clone, Copy)]
pub struct EligibilityRequest {
    pub action_signal: ActionLearningSignal,
    pub value_prediction: f32,
    pub normalization: EligibilityNormalization,
    pub leaky_neurons_enabled: bool,
    pub eligibility_retention: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct ThreeFactorLearningRequest {
    pub action_signal: ActionLearningSignal,
    pub reward_prediction_error: f32,
    pub learning_rate: f32,
    pub eligibility_retention: f32,
    pub fast_weight_retention: f32,
    pub max_weight_delta: f32,
    /// When true and a categorical target is revealed, hidden neurons learn
    /// from a fixed random projection of the output error (direct feedback
    /// alignment) instead of the scalar reward surprise.
    pub hidden_categorical_feedback: bool,
    /// Predictive-coding mode: hidden neurons learn from the categorical
    /// prediction error projected through the evolvable neuromodulatory channel
    /// matrix and read via per-neuron receptors. Overrides the scalar reward
    /// surprise for hidden edges.
    pub predictive_coding: bool,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ThreeFactorLearningReport {
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
    pub clipped_update_count: u64,
    pub requested_absolute_delta: f64,
    pub applied_absolute_delta: f64,
    pub reward_prediction_error: f32,
}

#[derive(Clone, Copy)]
struct LearningParams {
    action_signal: ActionLearningSignal,
    reward_prediction_error: f32,
    eta: f32,
    eligibility_retention: f32,
    fast_weight_retention: f32,
    max_weight_delta: f32,
}

/// Clear episode-local dynamics and causal traces while preserving all runtime
/// weights learned earlier in the same lifetime.
pub fn reset_episode_state_preserving_weights(brain: &mut BrainState) {
    for sensory in &mut brain.sensory {
        sensory.neuron.activation = 0.0;
        clear_edge_traces(&mut sensory.synapses);
    }
    for hidden in &mut brain.inter {
        hidden.neuron.activation = 0.0;
        hidden.state = 0.0;
        clear_edge_traces(&mut hidden.synapses);
    }
    for action in &mut brain.action {
        action.logit = 0.0;
    }
    clear_edge_traces(&mut brain.recurrent_synapses);
    clear_edge_traces(&mut brain.action_feedback_synapses);
    brain.previous_inter_activations.fill(0.0);
    brain.previous_action_activations.fill(0.0);
    brain.reward_prediction = 0.0;
    brain.value_bias_eligibility = 0.0;
    brain.pending_value_bias_eligibility = 0.0;
}

fn clear_edge_traces(edges: &mut [SynapseEdge]) {
    for edge in edges {
        edge.eligibility_state = 0.0;
        edge.eligibility = 0.0;
        edge.pending_eligibility = 0.0;
    }
}

/// Accumulate this decision's local synaptic sensitivities after recurrent
/// evaluation and action selection, but before the selected action overwrites
/// the previous-action efference copy. The outcome-derived third factor is not
/// known yet, so sensitivities are staged on each runtime edge.
pub fn accumulate_synaptic_eligibilities(
    brain: &mut BrainState,
    scratch: &mut BrainScratch,
    request: EligibilityRequest,
) {
    scratch.inter_activations.clear();
    scratch
        .inter_activations
        .extend(brain.inter.iter().map(|hidden| hidden.neuron.activation));
    // One derivative evaluation per hidden neuron per tick: `inter_local_gains`
    // folds in the leak alpha while `inter_activation_gains` keeps the raw gain.
    scratch.inter_local_gains.clear();
    scratch.inter_activation_gains.clear();
    scratch.inter_state_retentions.clear();
    scratch
        .inter_local_gains
        .reserve(brain.inter.len());
    scratch
        .inter_activation_gains
        .reserve(brain.inter.len());
    scratch
        .inter_state_retentions
        .reserve(brain.inter.len());
    for hidden in &brain.inter {
        let activation_gain = crate::activation::derivative(
            hidden.activation_fn,
            hidden.state,
            hidden.neuron.activation,
        );
        if request.leaky_neurons_enabled {
            scratch.inter_local_gains.push(hidden.alpha * activation_gain);
            scratch.inter_state_retentions.push(1.0 - hidden.alpha);
        } else {
            scratch.inter_local_gains.push(activation_gain);
            scratch.inter_state_retentions.push(0.0);
        }
        scratch.inter_activation_gains.push(activation_gain);
    }

    let normalization = match request.normalization {
        EligibilityNormalization::None => 1.0,
        EligibilityNormalization::NormalizedLeastMeanSquares => {
            let source_energy = brain
                .sensory
                .iter()
                .map(|sensory| sensory.neuron.activation.powi(2))
                .chain(
                    scratch
                        .inter_activations
                        .iter()
                        .map(|activation| activation.powi(2)),
                )
                .chain(
                    brain
                        .previous_action_activations
                        .iter()
                        .map(|activation| activation.powi(2)),
                )
                .sum::<f32>();
            1.0 / (NLMS_EPSILON + source_energy)
        }
    };
    let value_gain = 1.0 - request.value_prediction * request.value_prediction;

    for sensory in &mut brain.sensory {
        set_pending_for_edges(
            &mut sensory.synapses,
            sensory.output_synapse_start,
            sensory.neuron.activation * normalization,
            &scratch.inter_local_gains,
            &scratch.inter_activation_gains,
            &scratch.inter_state_retentions,
            value_gain,
            request.action_signal,
            request.eligibility_retention,
            request.leaky_neurons_enabled,
        );
    }
    for (pre_index, hidden) in brain.inter.iter_mut().enumerate() {
        set_pending_for_edges(
            &mut hidden.synapses,
            hidden.output_synapse_start,
            scratch.inter_activations[pre_index] * normalization,
            &scratch.inter_local_gains,
            &scratch.inter_activation_gains,
            &scratch.inter_state_retentions,
            value_gain,
            request.action_signal,
            request.eligibility_retention,
            request.leaky_neurons_enabled,
        );
    }

    debug_assert!(
        brain.recurrent_synapses.is_empty() || scratch.prev_inter.len() == brain.inter.len()
    );
    for edge in &mut brain.recurrent_synapses {
        let pre_index =
            edge.pre_inter_index
                .expect("recurrent edge has a dense presynaptic index") as usize;
        let post_index =
            edge.post_inter_index
                .expect("recurrent edge has a dense postsynaptic index") as usize;
        set_pending_internal(
            edge,
            scratch.prev_inter[pre_index] * normalization,
            post_index,
            &scratch.inter_local_gains,
            &scratch.inter_activation_gains,
            &scratch.inter_state_retentions,
            request.eligibility_retention,
            request.leaky_neurons_enabled,
        );
    }
    for edge in &mut brain.action_feedback_synapses {
        let pre_index =
            edge.pre_action_index
                .expect("action-feedback edge has a dense action index") as usize;
        let post_index =
            edge.post_inter_index
                .expect("action-feedback edge has a dense hidden index") as usize;
        set_pending_internal(
            edge,
            brain.previous_action_activations[pre_index] * normalization,
            post_index,
            &scratch.inter_local_gains,
            &scratch.inter_activation_gains,
            &scratch.inter_state_retentions,
            request.eligibility_retention,
            request.leaky_neurons_enabled,
        );
    }
    brain.pending_value_bias_eligibility = value_gain * normalization;
}

#[allow(clippy::too_many_arguments)]
fn set_pending_for_edges(
    edges: &mut [SynapseEdge],
    output_synapse_start: usize,
    pre: f32,
    inter_local_gains: &[f32],
    inter_activation_gains: &[f32],
    inter_state_retentions: &[f32],
    value_gain: f32,
    action_signal: ActionLearningSignal,
    eligibility_retention: f32,
    leaky_neurons_enabled: bool,
) {
    // Expressed brains store inter-target edges contiguously before
    // output-target edges (see `output_synapse_start`), so the two dispatch
    // families never mix within a sub-slice.
    let (inter_edges, output_edges) = edges.split_at_mut(output_synapse_start);
    for edge in inter_edges {
        let post_index = edge
            .post_inter_index
            .expect("expressed inter-group edge has a dense postsynaptic index")
            as usize;
        set_pending_internal(
            edge,
            pre,
            post_index,
            inter_local_gains,
            inter_activation_gains,
            inter_state_retentions,
            eligibility_retention,
            leaky_neurons_enabled,
        );
    }
    for edge in output_edges {
        if let Some(action_index) = action_array_index(edge.post_neuron_id) {
            edge.pending_eligibility = match action_signal {
                ActionLearningSignal::RewardPredictionError { selected } => {
                    pre * f32::from(action_index == selected.index())
                }
                ActionLearningSignal::CategoricalPredictionError { .. } => pre,
            };
        } else {
            debug_assert!(is_value_neuron_id(edge.post_neuron_id));
            edge.pending_eligibility = pre * value_gain;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn set_pending_internal(
    edge: &mut SynapseEdge,
    pre: f32,
    post_index: usize,
    inter_local_gains: &[f32],
    inter_activation_gains: &[f32],
    inter_state_retentions: &[f32],
    eligibility_retention: f32,
    leaky_neurons_enabled: bool,
) {
    if leaky_neurons_enabled {
        let alpha = if inter_activation_gains[post_index] == 0.0 {
            0.0
        } else {
            inter_local_gains[post_index] / inter_activation_gains[post_index]
        };
        edge.eligibility_state =
            inter_state_retentions[post_index] * edge.eligibility_state + alpha * pre;
        let desired_eligibility = inter_activation_gains[post_index] * edge.eligibility_state;
        edge.pending_eligibility =
            desired_eligibility - eligibility_retention.clamp(0.0, 1.0) * edge.eligibility;
    } else {
        edge.eligibility_state = 0.0;
        edge.pending_eligibility = pre * inter_local_gains[post_index];
    }
}

/// Convert the already-staged local sensitivities into weight changes using
/// one postsynaptic learning-signal rule. Current eligibility is folded before
/// modulation so an immediate outcome credits the circuit that caused it.
pub fn apply_three_factor_learning(
    brain: &mut BrainState,
    request: ThreeFactorLearningRequest,
) -> ThreeFactorLearningReport {
    let params = LearningParams {
        action_signal: request.action_signal,
        reward_prediction_error: request.reward_prediction_error,
        eta: request.learning_rate.clamp(0.0, 1.0),
        eligibility_retention: request.eligibility_retention.clamp(0.0, 1.0),
        fast_weight_retention: request.fast_weight_retention.clamp(0.0, 1.0),
        max_weight_delta: request.max_weight_delta.max(0.0),
    };
    let mut report = ThreeFactorLearningReport {
        reward_prediction_error: request.reward_prediction_error,
        ..ThreeFactorLearningReport::default()
    };

    // When a categorical target is revealed, each hidden neuron can receive a
    // higher-rank internal teaching signal than the scalar reward surprise,
    // computed once per neuron here so the per-edge update can reuse it.
    // Predictive coding projects the output error through the evolvable
    // neuromodulatory channels and per-neuron receptors (the receptor gains are
    // folded in, so it bypasses the scalar receptor). The fixed-random variant
    // (direct feedback alignment) is scaled by the scalar receptor. Reward
    // ticks leave this empty and fall back to reward-prediction error.
    let (hidden_teacher, bypass_receptor) = match params.action_signal {
        ActionLearningSignal::CategoricalPredictionError {
            target,
            probabilities,
        } if request.predictive_coding => (
            neuromodulatory_projection(brain, target, &probabilities),
            true,
        ),
        ActionLearningSignal::CategoricalPredictionError {
            target,
            probabilities,
        } if request.hidden_categorical_feedback => (
            brain
                .inter
                .iter()
                .map(|hidden| {
                    hidden_feedback_projection(hidden.feedback_mask, target, &probabilities)
                })
                .collect::<Vec<_>>(),
            false,
        ),
        _ => (Vec::new(), false),
    };

    let mut counts = BatchedEdgeCounts::default();
    for sensory in &mut brain.sensory {
        apply_edges(
            &mut sensory.synapses,
            sensory.output_synapse_start,
            params,
            &hidden_teacher,
            bypass_receptor,
            &mut report,
            &mut counts,
        );
    }
    for hidden in &mut brain.inter {
        apply_edges(
            &mut hidden.synapses,
            hidden.output_synapse_start,
            params,
            &hidden_teacher,
            bypass_receptor,
            &mut report,
            &mut counts,
        );
    }
    let recurrent_len = brain.recurrent_synapses.len();
    apply_edges(
        &mut brain.recurrent_synapses,
        recurrent_len,
        params,
        &hidden_teacher,
        bypass_receptor,
        &mut report,
        &mut counts,
    );
    let feedback_len = brain.action_feedback_synapses.len();
    apply_edges(
        &mut brain.action_feedback_synapses,
        feedback_len,
        params,
        &hidden_teacher,
        bypass_receptor,
        &mut report,
        &mut counts,
    );
    counts.merge_into(&mut report);

    brain.value_bias = brain.inherited_value_bias
        + params.fast_weight_retention * (brain.value_bias - brain.inherited_value_bias);
    brain.value_bias_eligibility = params.eligibility_retention * brain.value_bias_eligibility
        + brain.pending_value_bias_eligibility;
    brain.pending_value_bias_eligibility = 0.0;
    apply_value_bias_update(brain, params, &mut report);

    report
}

/// Integer per-edge counters batched in registers across a whole
/// `apply_three_factor_learning` call, then merged once. Integer addition is
/// associative, so this is bit-identical to per-edge accumulation; the f64
/// delta accumulators stay on `report` updated edge-by-edge in original order.
#[derive(Default)]
struct BatchedEdgeCounts {
    edge_evaluation_count: u64,
    clipped_update_count: u64,
    internal_edge_evaluation_count: u64,
    nonzero_internal_edge_update_count: u64,
    action_edge_evaluation_count: u64,
    nonzero_action_edge_update_count: u64,
    value_edge_evaluation_count: u64,
    nonzero_value_edge_update_count: u64,
}

impl BatchedEdgeCounts {
    fn merge_into(self, report: &mut ThreeFactorLearningReport) {
        report.edge_evaluation_count += self.edge_evaluation_count;
        report.clipped_update_count += self.clipped_update_count;
        report.internal_edge_evaluation_count += self.internal_edge_evaluation_count;
        report.nonzero_internal_edge_update_count += self.nonzero_internal_edge_update_count;
        report.action_edge_evaluation_count += self.action_edge_evaluation_count;
        report.nonzero_action_edge_update_count += self.nonzero_action_edge_update_count;
        report.value_edge_evaluation_count += self.value_edge_evaluation_count;
        report.nonzero_value_edge_update_count += self.nonzero_value_edge_update_count;
    }
}

fn apply_edges(
    edges: &mut [SynapseEdge],
    output_synapse_start: usize,
    params: LearningParams,
    hidden_teacher: &[f32],
    bypass_receptor: bool,
    report: &mut ThreeFactorLearningReport,
    counts: &mut BatchedEdgeCounts,
) {
    // Expressed brains store inter-target edges contiguously before
    // output-target edges (`output_synapse_start`); recurrent and
    // action-feedback lists are all-internal (start == len).
    let (inter_edges, output_edges) = edges.split_at_mut(output_synapse_start);
    for edge in inter_edges {
        apply_weight_decay(edge, params);
        // Hidden target. Without a precomputed teacher this is the
        // receptor-scaled scalar reward surprise. A precomputed teacher
        // (feedback alignment or the transported-readout diagnostic)
        // supplies a per-neuron signal; the diagnostic bypasses the
        // receptor so the true backprojected error drives learning.
        let post_index = edge
            .post_inter_index
            .expect("expressed inter-group edge has a dense postsynaptic index")
            as usize;
        let internal_teacher = hidden_teacher
            .get(post_index)
            .copied()
            .unwrap_or(params.reward_prediction_error);
        let learning_signal = if bypass_receptor {
            internal_teacher
        } else {
            edge.post_plasticity_receptor * internal_teacher
        };
        apply_eligibility_and_delta(
            edge,
            learning_signal,
            params.eligibility_retention,
            params,
            report,
            counts,
            BatchedClass::Internal,
        );
    }
    for edge in output_edges {
        apply_weight_decay(edge, params);
        if let Some(action_index) = action_array_index(edge.post_neuron_id) {
            let learning_signal = match params.action_signal {
                ActionLearningSignal::RewardPredictionError { .. } => {
                    params.reward_prediction_error
                }
                ActionLearningSignal::CategoricalPredictionError {
                    target,
                    probabilities,
                } => {
                    f32::from(action_index == target.index()) - probabilities[action_index]
                }
            };
            let trace_retention = match params.action_signal {
                ActionLearningSignal::RewardPredictionError { .. } => {
                    params.eligibility_retention
                }
                ActionLearningSignal::CategoricalPredictionError { .. } => 0.0,
            };
            apply_eligibility_and_delta(
                edge,
                learning_signal,
                trace_retention,
                params,
                report,
                counts,
                BatchedClass::Action,
            );
        } else {
            debug_assert!(is_value_neuron_id(edge.post_neuron_id));
            apply_eligibility_and_delta(
                edge,
                params.reward_prediction_error,
                params.eligibility_retention,
                params,
                report,
                counts,
                BatchedClass::Value,
            );
        }
    }
}

#[inline(always)]
fn apply_weight_decay(edge: &mut SynapseEdge, params: LearningParams) {
    edge.weight = edge.inherited_weight
        + params.fast_weight_retention * (edge.weight - edge.inherited_weight);
}

#[inline(always)]
fn apply_eligibility_and_delta(
    edge: &mut SynapseEdge,
    learning_signal: f32,
    trace_retention: f32,
    params: LearningParams,
    report: &mut ThreeFactorLearningReport,
    counts: &mut BatchedEdgeCounts,
    class: BatchedClass,
) {
    edge.eligibility = trace_retention * edge.eligibility + edge.pending_eligibility;
    edge.pending_eligibility = 0.0;
    let requested_delta =
        params.eta * edge.plasticity_coefficient.max(0.0) * learning_signal * edge.eligibility;
    let (applied_delta, clipped) =
        apply_weight_delta(edge, requested_delta, params.max_weight_delta, report);
    counts.edge_evaluation_count += 1;
    counts.clipped_update_count += u64::from(clipped);
    match class {
        BatchedClass::Internal => {
            counts.internal_edge_evaluation_count += 1;
            counts.nonzero_internal_edge_update_count += u64::from(applied_delta != 0.0);
            report.internal_applied_absolute_delta += f64::from(applied_delta.abs());
        }
        BatchedClass::Action => {
            counts.action_edge_evaluation_count += 1;
            counts.nonzero_action_edge_update_count += u64::from(applied_delta != 0.0);
            report.action_applied_absolute_delta += f64::from(applied_delta.abs());
        }
        BatchedClass::Value => {
            counts.value_edge_evaluation_count += 1;
            counts.nonzero_value_edge_update_count += u64::from(applied_delta != 0.0);
            report.value_applied_absolute_delta += f64::from(applied_delta.abs());
        }
    }
}

fn apply_weight_delta(
    edge: &mut SynapseEdge,
    requested_delta: f32,
    max_delta: f32,
    report: &mut ThreeFactorLearningReport,
) -> (f32, bool) {
    let capped_delta = requested_delta.clamp(-max_delta, max_delta);
    let previous_weight = edge.weight;
    let proposed_weight = previous_weight + capped_delta;
    edge.weight = constrain_weight(proposed_weight);
    let applied_delta = edge.weight - previous_weight;
    report.requested_absolute_delta += f64::from(requested_delta.abs());
    report.applied_absolute_delta += f64::from(applied_delta.abs());
    let clipped =
        requested_delta.abs() > max_delta || edge.weight != proposed_weight;
    (applied_delta, clipped)
}

fn apply_value_bias_update(
    brain: &mut BrainState,
    params: LearningParams,
    report: &mut ThreeFactorLearningReport,
) {
    let requested_delta =
        params.eta * params.reward_prediction_error * brain.value_bias_eligibility;
    let capped_delta = requested_delta.clamp(-params.max_weight_delta, params.max_weight_delta);
    let previous_bias = brain.value_bias;
    let proposed_bias = (previous_bias + capped_delta).clamp(-1.0, 1.0);
    brain.value_bias = proposed_bias;
    let applied_delta = proposed_bias - previous_bias;
    report.edge_evaluation_count += 1;
    report.value_edge_evaluation_count += 1;
    report.nonzero_value_edge_update_count += u64::from(applied_delta != 0.0);
    report.value_applied_absolute_delta += f64::from(applied_delta.abs());
    report.clipped_update_count += u64::from(
        requested_delta.abs() > params.max_weight_delta
            || proposed_bias != previous_bias + capped_delta,
    );
    report.requested_absolute_delta += f64::from(requested_delta.abs());
    report.applied_absolute_delta += f64::from(applied_delta.abs());
}

/// Predictive-coding hidden teaching signal via evolvable neuromodulatory
/// channels. Each channel `c` is a global projection of the categorical output
/// error, `modulator_c = sum_k B[c,k] * (onehot(target)_k - prob_k)`; hidden
/// neuron `j` then reads them through its evolvable receptor gains,
/// `m_j = sum_c receptor[j,c] * modulator_c`. This is a rank-`C` factored
/// feedback map (broadcast channels x per-neuron receptors) — no weight
/// transport, no backpropagation. Zero channels yields an empty result (hidden
/// plasticity off, the output-only baseline).
fn neuromodulatory_projection(
    brain: &BrainState,
    target: Symbol,
    probabilities: &[f32; ACTION_COUNT],
) -> Vec<f32> {
    let channel_count = (brain.feedback_channels.len() / ACTION_COUNT).min(MAX_FEEDBACK_CHANNELS);
    if channel_count == 0 {
        return Vec::new();
    }
    let target_index = target.index();
    let mut modulators = [0.0f32; MAX_FEEDBACK_CHANNELS];
    for (channel, modulator) in modulators.iter_mut().enumerate().take(channel_count) {
        let base = channel * ACTION_COUNT;
        let mut sum = 0.0;
        for (action_index, probability) in probabilities.iter().enumerate() {
            let error = f32::from(action_index == target_index) - probability;
            sum += brain.feedback_channels[base + action_index] * error;
        }
        *modulator = sum;
    }
    brain
        .inter
        .iter()
        .map(|hidden| {
            (0..channel_count)
                .map(|channel| hidden.feedback_receptors[channel] * modulators[channel])
                .sum()
        })
        .collect()
}

/// Project the categorical output error through a hidden neuron's fixed sign
/// row (one bit per action). Returns a scalar direct-feedback teaching signal.
fn hidden_feedback_projection(
    feedback_mask: u32,
    target: Symbol,
    probabilities: &[f32; ACTION_COUNT],
) -> f32 {
    let target_index = target.index();
    let mut sum = 0.0;
    for (action_index, probability) in probabilities.iter().enumerate() {
        let error = f32::from(action_index == target_index) - probability;
        let sign = if (feedback_mask >> action_index) & 1 == 1 {
            1.0
        } else {
            -1.0
        };
        sum += sign * error;
    }
    sum
}

pub fn store_action_efference_copy(brain: &mut BrainState, selected: Symbol) {
    brain.previous_action_activations.fill(0.0);
    brain.previous_action_activations[selected.index()] = 1.0;
}

#[derive(Clone, Copy)]
enum BatchedClass {
    Internal,
    Action,
    Value,
}
