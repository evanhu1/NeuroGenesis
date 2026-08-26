# WANN-style minimal topology and activation evolution

## Question

Does weight-agnostic-style architecture search — minimal founders, per-neuron
heritable activation functions, and a change-activation operator — let
evolution tune the architectural inductive bias while three-factor lifetime
plasticity tunes the weights, without losing any established task gate?

The current substrate expresses every hidden neuron as tanh and maintains a
dense (sensory+hidden)->(action+value) readout at birth and after every
mutation. The hard next-token winner carries ~1,180 enabled edges yet behaves
as a feedforward net with a plastic readout; architecture search has no
freedom over transfer functions and no route to a minimal wiring.

## Background: the adopted algorithm

Weight Agnostic Neural Networks (Gaier & Ha 2019, arXiv:1906.04358) searches
topology only: founders wire a fraction of possible inputs directly to outputs
with no hidden nodes; variation uses insert-node (split an edge; the new node
receives a random activation), add-connection (feed-forward), and
change-activation (random reassignment) at a 25/25/50 mix; fitness is measured
with one weight shared across all connections over rollouts at
[-2,-1,-0.5,+0.5,+1,+2]; ranking is NSGA-II on (mean performance,
connection count) 80% of the time and (mean, max) 20%. The activation set is
linear, step, sin(pi x), cos(pi x), Gaussian, tanh, sigmoid, invert, absolute
value, and ReLU.

Adopted here: minimal founders, the heritable activation gene, random
activation on node insertion, and the change-activation operator.

Deliberately not adopted:

- **Shared-weight evaluation.** Per-edge heritable weights stay: lifetime
  plasticity is this substrate's weight-tuning mechanism (the biologically
  plausible analogue of WANN's post-hoc weight tuning), and inherited weights
  remain the Baldwin-effect starting point.
- **Complexity-based multi-objective ranking.** The clean baseline forbids
  scalar fitness, topology rewards, and implicit efficiency metrics. Minimality
  pressure comes only from minimal initialization and the existing
  delete-connection/delete-node operators.

## Treatment

All mechanisms are generic across tasks and gated by search parameters whose
defaults reproduce the legacy substrate byte-for-byte (same-seed RNG streams
included).

1. **Heritable activation functions.** `HiddenNodeGene` gains
   `activation_fn` over the ten WANN functions (default tanh). Expression
   copies it to the runtime neuron; evaluation applies it in place of the
   fixed tanh. Every output lies in [-1, 1]. That bound is a first-class
   invariant of this recurrent substrate, not a tanh legacy: previous-tick
   loops with weight magnitudes up to 1.5 and near-zero leak grow any
   unbounded transfer geometrically across ticks until f32 overflow (the
   paper's networks were strictly feedforward and never faced recurrent
   gain), and one shared range keeps every unit commensurate for recurrent
   mixing, logits, and eligibility. Six functions are naturally bounded; the
   four unbounded originals ship as their saturating forms and are named for
   what they compute (`saturating_linear`, `saturating_inverse`,
   `saturating_abs`, `saturating_relu`). Eligibility gains use each
   function's local derivative with respect to the membrane state; tanh
   keeps the exact legacy `1 - a^2` numerics, and step uses the boxcar
   pseudo-derivative `max(0, 1 - |x|)` (e-prop precedent for
   non-differentiable neurons).
2. **Change-activation operator.** With probability
   `mutate_activation_probability` (default 0), one random hidden node is
   reassigned uniformly among the other nine functions. The draw sits after
   every established operator so legacy streams are unchanged at the default.
   While the operator is enabled, insert-node assigns the new node a random
   activation and founder hidden nodes initialize with random activations.
3. **Minimal founders.** With `wann_minimal_init` (default false), founders
   skip the dense readout entirely. Sources are the enabled sensors plus the
   canonical seed hidden nodes (one node — required because memory and
   continual enable zero sensors); sinks are the enabled actions plus the
   value output when enabled. Each source->sink pair is enabled independently
   with probability `initial_connection_fraction` (default 0.25, at least one
   edge forced), and the canonical self-recurrent loop on the first hidden
   node is kept.
4. **Optional readout guarantee.** `guarantee_dense_readout` (default true)
   gates the dense-readout maintenance at birth and after every mutation.
   WANN mode sets it false so evolution owns readout wiring.

## Matched gates

Population 512, generations 200, seed 101, canonical founder config. Width is
halved relative to the historical population-1,024 baseline for wall time;
the gates below are absolute sealed accuracies and unchanged. WANN arms run
structural rates near the paper's 25/25/50 operator mix rather than the
timid legacy defaults, because minimal founders must complexify within the
generation budget; delete rates keep their defaults as the only minimality
pressure. Arm W1 on every established task, executed one at a time from a
small sign-of-life run upward, tuning to saturate each task before advancing:

```
cli ecology <task> --seed 101 --population 512 --generations 200 \
  --param wann_minimal_init=true --param guarantee_dense_readout=false \
  --param initial_connection_fraction=0.25 \
  --param mutate_activation_probability=0.3 \
  --param add_node_probability=0.2 --param add_connection_probability=0.25 \
  --out-dir artifacts/research/runs/active/wann-minimal-activation-v1
```

1. Every established gate holds under WANN-mode search: reaction >= 99%
   sealed, memory >= 90% sealed character, canonical next-token >= 90%
   sealed, continual >= 90% sealed. (Baselines: 99.724 / 97.250 / 95.455 /
   93.219.)
2. The harder 150-target passage is exploratory only — recorded, no gate.
3. `cargo test --workspace` stays green and default-configuration behavior is
   byte-identical to the pre-change substrate.

Secondary measurements: enabled-connection counts of selected winners versus
the dense baseline (expect a large reduction from ~1,180 on next-token);
activation-function census of champions; a champion lesion rewriting every
activation to tanh and re-running the sealed panel via `ecology TASK evaluate`
to test whether activation diversity is causal.

Diagnostic arms if W1 passes or fails informatively (next-token and continual
only): W2 = activation mutation only (legacy dense init and readout); W3 =
minimal init only (all tanh). These isolate which lever carries the effect.

## Risks

Without the readout guarantee, deletion can sever every output path and a
lineage can go extinct; extinction is a valid outcome and the exact-elite copy
mitigates it. Memory and next-token historically depend on the dense plastic
readout — whether search rediscovers a sufficient sparse readout is exactly
what this experiment tests. Reject and remove the treatment if any
established gate fails across the matched arms.
