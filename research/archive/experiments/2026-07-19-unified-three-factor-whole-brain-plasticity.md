# Unified three-factor whole-brain plasticity

Status: completed discovery experiment
Date: 2026-07-19
Slug: 2026-07-19-unified-three-factor-whole-brain-plasticity

## Question

Can one online, local three-factor synaptic rule make the recurrent symbolic
brain learn internal representations during its lifetime without task-specific
state, backpropagation, or separate actor/critic optimizers, while preserving
or improving competence on every established symbolic task?

## Hypothesis

The output-only learner forces evolution to discover a largely inherited
recurrent representation. Extending the same eligibility-times-modulation rule
to every synapse that targets a hidden neuron will let evolution discover
within-lifetime representation learning. A signed evolved receptor on each
hidden neuron will turn the scalar reward-prediction error into heterogeneous
local learning signals without transporting output weights or gradients.

The mechanism is rejected if it cannot reach the established competence gates,
if its gain comes from exposing evaluator-only answers to hidden neurons, or if
greater population/generation budgets systematically reduce performance.

## Architecture contract

Every plastic synapse uses one update law:

```text
eligibility_ij <- eligibility_retention * eligibility_ij
                  + local_sensitivity_ij

delta_weight_ij <- clip(
    learning_rate
    * plasticity_coefficient_ij
    * postsynaptic_learning_signal_j
    * eligibility_ij
)
```

The local sensitivity is computed online from presynaptic activity and the
postsynaptic neuron's local activation gain. Previous-tick hidden and action
sources use their frozen previous activations. There is no backpropagation,
BPTT, transported forward weight, evaluator-authored position, or auxiliary
task memory.

Postsynaptic learning signals are:

- hidden neuron `j`: `plasticity_receptor_j * reward_prediction_error`;
- selected action neuron under reward learning: `reward_prediction_error`;
- scalar reward-prediction neuron: `reward_prediction_error`;
- categorical prediction output: `one_hot(revealed_target) - probability`.

The reward-prediction error is signed immediate surprise:

```text
reward_prediction_error = bounded_reward - predicted_reward
```

The action outputs and scalar reward predictor remain functional neuron roles,
not separately optimized heads. Hidden, action, recurrent, sensory, efference,
and value synapses share the same retention, clipping, eligibility state,
learning rate, and per-edge plasticity coefficient.

The existing hidden-neuron neuromodulatory scalar is cut over to a
`plasticity_receptor`. Prediction error no longer enters ordinary hidden
activation as an additive input. Previous selected action remains an evolved
efference-copy pathway.

Exact categorical error is legal only when the environment explicitly exposes
a learner-visible target, as next-token training does after prediction.
Evaluator-only `expected_action` remains diagnostic and cannot drive learning.

## Temporal contract

One learning step is:

1. apply the environment observation;
2. evaluate current hidden state, action logits, and predicted reward;
3. select the action;
4. accumulate current causal eligibility on every expressed synapse;
5. apply the action and obtain reward or a revealed categorical target;
6. compute the signed postsynaptic learning signals;
7. retain and update each synapse exactly once;
8. store the selected action as next-tick efference copy.

Semantic episode/pass resets clear recurrent dynamics, action copy, all
eligibilities, pending eligibility, and critic eligibility while preserving
learned runtime weights. A frozen probe clears transient state and performs no
updates.

## Experimental contract

- Evolutionary seed for the first matched gate: `101` unless a task's canonical
  saturated record uses a different seed, in which case both canonical and 101
  are reported.
- Population: `1,024`.
- Generations: `200`.
- Evaluation workers: automatic hardware-effective default.
- Founder: `config/seed_genome.toml`.
- All panels, rollouts, task horizons, search operators, and reproductive
  ecology otherwise use each task's canonical defaults.
- Artifact directory:
  `artifacts/research/runs/active/2026-07-19-unified-three-factor-whole-brain-plasticity/`.

Established-task order:

1. basic reaction;
2. basic memory;
3. basic continual learning;
4. canonical basic next-token prediction.

Only after those gates are evaluated will the longer English next-token
passage that defeated the output-only system be run under matched search and
learning-pass budgets.

## Measurements

Primary established-task endpoint: sealed primary accuracy at the selected
development checkpoint. Basic memory additionally reports exact four-symbol
probe success; canonical next-token additionally reports exact 44-token probe
success.

Secondary measurements:

- selected generation and population trajectory;
- learning versus frozen-probe accuracy;
- reward-prediction error and applied update amplitude;
- hidden nodes and enabled edges by connection class;
- internal versus output update counts;
- clipped-update fraction;
- brain synapse operations and wall time;
- a one-time internal-plasticity lesion for causal attribution of this new
  mechanism.

Integrity checks:

- fixed seed and configuration are deterministic across worker counts;
- evaluator-only expected actions never enter the learner;
- every edge receives at most one retention/update operation per tick;
- current eligibility is folded before the outcome modulates it;
- frozen probes cannot alter weights or eligibility;
- increasing population or generations cannot change task semantics.

## Decision rule

The architecture is adopted only if:

- basic reaction, memory, and continual learning each reach at least `90%`
  sealed primary accuracy at population 1,024 within 200 generations;
- canonical next-token prediction reaches at least `90%` frozen character
  accuracy within the same search budget;
- no result depends on evaluator-only target leakage;
- the implementation passes the existing workspace test suite and deterministic
  replay check;
- topology and synapse-operation growth remain commensurate with improved
  competence.

Failure on an established task triggers diagnosis and only general corrections
to the brain, learner, or evolutionary search. Task-authored representations,
position signals, side memories, task-specific learning rates, and shaped
optimizer rewards are prohibited.

If the established gates pass, the harder next-token result determines the
next decision. Improvement over the matched output-only result supports
whole-brain representation learning; no improvement localizes the remaining
bottleneck to the internal credit signal or representational dynamics rather
than search width alone.

## Result

The unified engine passed every preregistered established-task gate after one
general search correction: hidden plasticity receptors now initialize at zero
and mutate only after the established parameter and structural operators. The
first implementation randomized receptors at full scale and consumed its RNG
draw before structural mutation. Merely adding the gene therefore changed all
founder parameter draws and which topology mutation an offspring received. The
neutral formulation preserves the established learner as an exact subspace
while allowing evolution to recruit signed receptor expression.

All matched gate runs used seed 101, automatic 14-worker evaluation, population
1,024, and 200 generations:

| Task | Selected generation | Sealed accuracy | Exact sequence | Hidden | Enabled edges | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| Basic reaction | 49 | 99.724% | n/a | 6 | 213 | 30.38 s |
| Basic memory | 124 | 97.250% | 89.000% | 14 | 149 | 361.60 s |
| Basic continual learning | 174 | 93.219% | n/a | 7 | 78 | 262.60 s |
| Canonical next token | 199 | 95.455% (42/44) | no | 16 | 1,315 | 28.12 s |

Memory had already reached 98.500% sealed character accuracy and 94.000% exact
at population 256 in 100 generations. That calibration exposed and corrected a
task contract bug: memory, continual, and renewable tasks generated only A-H
targets and used an eight-action balanced reward, but admitted all 27
non-terminal outputs. Restricting their legal actions to the declared A-H set
restored the intended 12.5% chance ecology and removed irrelevant readouts.

Canonical next-token scaling after neutral receptor initialization was:

| Population | Generations | Selected generation | Sealed correct | Accuracy | Hidden | Edges |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 200 | 199 | 42/44 | 95.455% | 16 | 1,315 |
| 1,024 | 500 | 199 | 42/44 | 95.455% | 16 | 1,315 |
| 4,096 | 200 | 199 | 43/44 | 97.727% | 8 | 1,066 |
| 8,192 | 200 | 124 | 44/44 | 100.000% | 10 | 1,130 |

The population-8,192 exact learner first appeared in training at generation
112 and was selected at the generation-124 development audit. This is shallower
than the output-only system's first exact population-1,024 solution at
generation 374, but not more genome-evaluation efficient: the wider run spent
more total evaluations to reach that generation.

Corrected sealed telemetry on the exact winner distinguishes edge evaluations
from actual changes. Across 176 learning ticks, 1,632 of 4,928 internal edge
evaluations produced nonzero changes with 7.90 total absolute internal weight
movement. Action edges produced 49,305 nonzero changes and 412.07 movement;
value edges produced 4,753 and 209.28. A one-time frozen lesion setting only
hidden receptors to zero reduced the exact winner from 44/44 to 42/44.

The harder 150-target passage remained unsolved:

```text
the quick brown fox jumps over the lazy dog and walks through a quiet village
where people bake bread read books play music and watch the evening sky
```

| Population | Generations | Selected generation | Sealed correct | Accuracy | Hidden | Edges |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 200 | 199 | 89/150 | 59.333% | 11 | 1,154 |
| 4,096 | 200 | 174 | 80/150 | 53.333% | 7 | 1,031 |

The population-1,024 winner made 5,735 nonzero internal changes across 13,800
internal edge evaluations, totaling 22.26 absolute movement. Its frozen
internal-plasticity lesion fell from 89/150 to 67/150 (44.667%), establishing a
large causal contribution of whole-brain plasticity. Nevertheless, four times
the population regressed by nine tokens, so width did not provide reliable
refinement on the harder task.

## Interpretation and next decision

Adopt the unified three-factor whole-brain engine. It preserves all established
tasks above 90%, recovers exact canonical next-token learning, and its internal
plasticity is causally necessary for both canonical and harder next-token
performance. Keep neutral receptor initialization: optional machinery must not
destroy the established learner or perturb unrelated mutation operators merely
by existing.

Do not claim that this solves basic English learning. The harder passage result
localizes two remaining bottlenecks. First, every hidden neuron receives only a
signed scalar projection of reward surprise. That signal is useful, but it is
low-rank relative to the 28-dimensional categorical error explicitly available
at a supervised next-token boundary. Second, tournament reproduction gives a
top lineage only order-tournament-size expected offspring per generation,
independent of population width. Larger populations broaden independent search
but do not accelerate refinement of a discovered representation, and a single
larger run can regress.

The run telemetry sharpens the optimizer diagnosis. At generation 199, the
population-1,024 run had an effective ticket-producer count of 962.0; the
population-4,096 run had 3,839.6. Thus about 94% of each population remained
effectively reproductive. A unique best organism wins only
`N * (1 - (1 - 1/N)^4)`, approximately four, mutated offspring slots under the
four-way tournament, independent of population width, plus its one exact elite
copy. Quadrupling population therefore quadrupled broad independent search but
did not give a discovered learner more local refinement trials.

The mutation representation compounds that failure. The population-1,024 hard
winner's 1,154 enabled edges comprise only 23 internal representation edges,
versus 1,092 action-readout edges and 39 value-readout edges. This is the
expected dense learnable head, not unexplained biological complexity, but all
enabled edges currently share one uniform inherited-weight and plasticity-
coefficient mutation pool. With the observed genome size, a weight-mutation
event perturbs about 6.06 edges on average, so any particular internal edge is
sampled with probability only about 0.53% per event. A plasticity-coefficient
event mutates exactly one edge, giving the entire internal circuit only
`23/1154` of those trials. Population width does not repair that dilution
because the leading lineage still receives only about four mutated children.

The next experiment should change one bottleneck at a time. The first
optimizer-side correction is class-normalized mutation and reproductive
allocation that turns additional width into additional refinement trials for
promising representations while retaining finite zero-sum offspring slots.
This is general across tasks and does not change their success events. Only
after establishing a scaling optimizer baseline should the learner-side
candidate be tested: generic local feedback projections of learner-visible
categorical error into hidden postsynaptic modulatory signals, retaining the
same eligibility-times-signal update and no BPTT or transported forward
weights. These require separate preregistered experiments.
