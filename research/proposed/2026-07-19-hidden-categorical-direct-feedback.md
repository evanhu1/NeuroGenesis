# Hidden categorical direct-feedback experiment

## Question

Does the harder next-token plateau come from the information bottleneck in the
hidden plasticity signal rather than insufficient mutation pressure?

The unified three-factor baseline gives categorical output synapses the full
28-dimensional `target - probability` error, but every hidden neuron receives
only scalar reward-prediction error through its evolved signed receptor. A
matched tournament-pressure sweep did not improve the generation-100 control:
tournament sizes 4, 16, and 64 reached 83/150, 75/150, and 83/150.
Class-normalized synapse mutation also regressed to 79/150 and was removed.

## Treatment

For learner-visible categorical outcomes, give every stable hidden-node
identity an immutable signed projection of the output error vector:

`m_j = receptor_j * sum_k B[j,k] * (target[k] - probability[k])`

`B[j,k]` is a deterministic `-1/+1` hash of stable hidden identity and action
index. It is not derived from the forward weights and is not task-authored.
Reward-learning tasks retain `m_j = receptor_j * reward_prediction_error`.
Every incoming sensory, current-tick hidden, previous-tick hidden, and action-
feedback synapse retains the same local eligibility-times-modulator update.

This is direct feedback alignment, not backpropagation: there is no backward
graph traversal, transported forward weight, gradient tape, or BPTT. Receptors
still initialize at zero, preserving the established output-only learner as an
exact subspace.

## Matched gates

1. Run the harder 150-target passage at seed 101, population 1,024, generation
   100, tournament size 4. The established matched control is 83/150.
2. Continue to generation 200 only if the treatment improves the early search
   trajectory. The established control is 89/150.
3. If the hard result improves materially, rerun reaction, memory, continual
   learning, and canonical next token at population 1,024 within 200
   generations. Every established task must remain at or above 90% sealed
   primary accuracy.
4. Use one explicit frozen hidden-receptor lesion only for causal attribution
   of a successful hard winner; do not make lesions part of routine runs.

The treatment is rejected and removed if it does not improve the hard task or
if it regresses the established gates. Task success events, finite offspring
allocation, topology, and all other mutation and learner parameters remain
unchanged.
