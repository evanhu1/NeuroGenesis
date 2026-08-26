# WANN-style minimal topology and activation evolution — adopted

Preregistration:
[proposal](../../proposed/2026-07-20-wann-minimal-topology-activation-evolution.md).
Artifacts: `artifacts/research/runs/completed/wann-minimal-activation-v1/`.

## Question

Does weight-agnostic-style architecture search — minimal founders, per-neuron
heritable activation functions, and a change-activation operator — let
evolution tune the architectural inductive bias while three-factor lifetime
plasticity tunes the weights, without losing any established task gate?

## Method

Mechanisms per the proposal: heritable `activation_fn` over the ten WANN
functions with outputs in [-1, 1] (a recurrent-substrate invariant — unbounded
transfers explode geometrically through previous-tick loops; the four
unbounded originals ship as `saturating_*` forms), per-function eligibility
derivatives (boxcar pseudo-derivative for step), a change-activation operator,
minimal founders (enabled sensors plus the canonical seed hidden node wired to
enabled outputs with probability `initial_connection_fraction`, self-recurrent
loop kept), and the dense-readout guarantee switched off so evolution owns
readout wiring.

All runs: seed 101, population 512 (half the historical baseline width),
canonical founder config, operator rates near the paper's 25/25/50 mix
(`mutate_activation_probability=0.3`, `add_node_probability=0.2`,
`add_connection_probability=0.25`), delete rates at legacy defaults as the
only minimality pressure. Tasks were run one at a time from a sign-of-life
run upward, tuning single levers to saturate each before advancing.

## Results — every established gate passed

| Task | Config | Sealed | Gate | Dense baseline (p1024) | Champion |
|---|---|---:|---:|---:|---|
| reaction | f=0.25, g200 | 99.72% | >=99% | 99.72% | 19 hidden, 68 edges (3 recurrent) |
| continual | f=0.25, g200 | 93.65% | >=90% | 93.22% | 31 hidden, 65 edges (15 recurrent) |
| memory | f=0.25, g400 | 97.75% char / 92.0% exact | >=90% | 98.0% / 92.0% (g250) | 53 hidden, 135 edges (27 recurrent) |
| next-token | f=1.0, g200 | 95.45% | >=90% | 95.46% | 48 hidden, 920 edges (16 recurrent) |

Wall times: 11 s, 221 s, 652 s, 16 s. Memory at g200 scored 83.5% with the
leader still climbing (0.856 -> 0.913 over the final 40 generations); doubling
depth alone saturated it — no parameter change was needed.

Next-token required tuning one preregistered lever. The
`initial_connection_fraction` dose-response at g200 was monotone:

| fraction | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---:|---:|---:|---:|
| sealed | 61.4% | 70.5% | 88.6% | 95.5% |

0.75 also passed at g400 (90.9%). The mechanism is clear: the snippet's
transition table is stored in plastic sensory->action readout weights, so the
fraction of the interface present at birth bounds how much of the table is
learnable, and add-connection (+29 edges over 200 generations) cannot close a
~500-edge gap. Evolution cannot compress a lookup table; where the readout is
the solution, it must be given or grown. Even at f=1.0 the mode differs from
the legacy guarantee — the interface is granted once at birth, prunable, and
never re-densified after mutation — and it costs nothing (95.45 vs 95.46).

## Secondary results

**Activation diversity is causally load-bearing.** Rewriting every champion
hidden node to tanh and re-auditing on a matched sealed panel (seed 777):

| Task | original | all-tanh lesion |
|---|---:|---:|
| reaction | 99.6% | 47.0% |
| continual | 93.6% | 35.3% |
| memory | 97.8% | 38.9% |
| next-token | 95.5% | 65.9% |

This shows the evolved circuits compute through their activation diversity; it
does not show tanh-only search would fail (that is the unrun W3 ablation).
Champions retain 9–10 distinct functions with tanh a minority (2/19 in
reaction, 6/53 in memory). Enrichment looks task-appropriate: `step`
dominates the continual champion (6 nodes; top of the population census) and
the next-token champion (13/48) — latch/switch semantics for tasks about
discrete hidden state.

**Minimality is automatic.** No complexity objective was used; minimal init
plus the existing delete operators produced 65–143-edge champions on three
tasks (>90% wiring reduction vs the forced dense readout) and a 920-edge
champion only where the task's solution genuinely lives in the readout.

**Recurrence is recruited when the readout stops crowding it.** Continual's
champion uses 23% of its edges recurrently and memory's 20%, versus 0.5%
(6/1,180) in the dense hard-next-token winner.

## WANN-repo genome representation and crossover — assessed, not adopted

`prettyNeatWann` stores nodes as a `[3 x N]` matrix (id, type, activation
int) and connections as `[5 x N]` (innovation, source, dest, weight, enabled)
with a global sequential innovation counter threaded through every mutation.
Our typed genes are strictly richer (timing/recurrence, plasticity
coefficient, bias, time constant, receptors) and our content-addressed
innovation IDs (hash of pre/post/timing) are better suited to this substrate:
the same structural change receives the same identity in every lineage with
no shared mutable counter, which is what makes parallel asexual mutation
deterministic and duplicate detection free. The numpy layout's only advantage
is vectorized Python — irrelevant to compiled expression. Not adopted.

Their crossover builds the child from the fitter parent's full topology and,
for connections matching by innovation intersect, takes the weaker parent's
*weight* with probability 0.5; disjoint/excess genes come only from the
fitter parent. Two reasons not to adopt: (1) the WANN paper's own results are
mutation-only — weight-swapping crossover is meaningless under shared
weights, and in our substrate weights are lifetime-tuned, making it a weak
operator; (2) the clean baseline deliberately excludes crossover, and the
intervention sweep showed the hard passage is exploration-limited, not
concentration-limited. Content-addressed innovations keep future crossover
trivially implementable (alignment is a merge by innovation hash) if evidence
ever demands it.

## Decision

Adopt WANN-NEAT as the system: minimal founders, heritable activations, and
evolution-owned readout wiring become the defaults; the dense-readout
guarantee (`ensure_lifetime_learning_readout`) is removed in a clean cutover.
`initial_connection_fraction` stays a per-task tunable (next-token presets to
1.0). Delete operators remain the only minimality pressure.

## Open

Single seed (101); W2/W3 ablations unrun (activation-diversity necessity vs
minimal-init contribution are not separated); the harder 150-target passage
is the next campaign — the f=1.0 + free-recurrence + step-latch combination
is the specific new lever against its multi-symbol memory bottleneck.
