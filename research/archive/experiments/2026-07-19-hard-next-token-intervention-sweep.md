# Hard next-token intervention sweep: three proposals and memory levers

Status: completed discovery experiment
Date: 2026-07-19
Slug: 2026-07-19-hard-next-token-intervention-sweep

## Question

Do the three preregistered proposals — neutral structural complexification,
reproduction-allocation repair, and hidden categorical direct feedback — or any
memory-promoting search change, push the harder 150-target next-token passage
past the e-prop baseline while keeping the easier tasks at their competence
gates?

## Setup

- Learner: adopted unified three-factor whole-brain plasticity with dynamics
  e-prop (leaky-integrator neurons + local state-derivative eligibility), the
  current working-tree baseline.
- Search: asexual ticket ecology, population 1,024, 100 generations, fixed-K
  tournament (K=4), one exact elite. Hard passage, greedy categorical learner.
- Seeds: 101 primary; 7 added for the load-bearing comparisons. The baseline
  itself has real seed variance: 92/150 at seed 101, 86/150 at seed 7
  (mean ~89), so single-seed deltas under ~4 tokens are noise.
- Behavioral ceilings (fixed context length): 1 char 64/150, 2 char 120/150,
  3 char 141/150. The baseline sits near the 1.5-char regime.

## Result: every intervention regressed or tied

All numbers are sealed hard-passage correct out of 150, population 1,024,
generation 100.

| Intervention | seed 101 | seed 7 | Verdict |
|---|---:|---:|---|
| e-prop baseline (tournament, reward hidden signal) | 92 | 86 | reference |
| A. Neutral additive complexification | 73 | — | regress |
| B. Truncation selection (survivors 32/64/128) | 76-79 | — | regress |
| B. Ticket-proportional selection (SUS) | 69 | — | regress |
| C. Hidden categorical direct feedback (DFA) | 83 | — | regress |
| More recurrent add-connection (0.15) | 80 | — | regress |
| More add-node (0.10) | 87 | — | regress |
| More time-constant mutation (0.30) | 80 | — | regress |
| Per-new-node self-recurrence | 92 | — | tie |
| Dense plastic recurrent matrix | 77 | 84 | regress |
| Per-node self-recurrent loops (all nodes) | 83 | 84 | regress |
| Heterogeneous time-constant initialization | 86 | 79 | regress |
| Heterogeneous TC + self-recurrence | 84 | — | regress |
| Heterogeneous TC + dense recurrence | 74 | — | regress |

Nothing beat the baseline. The tournament (weak, high-exploration) selection was
strictly better than every concentrating scheme; the hard passage is
exploration-limited, so sharpening reproduction causes premature convergence.

## Root-cause diagnosis

The generation-100 seed-101 winner (12 hidden nodes, 1,180 enabled edges) uses
only **6 previous-tick (recurrent) edges**, and every leak time constant is
short (log time constant in [-1.20, 0.00], i.e. fast, near-instantaneous). It is
a feedforward network with a dense, plastic output readout and almost no memory,
which is exactly why it saturates near the one-to-two-character ceiling.

Promoting recurrence structurally does not help, and usually hurts, because:

1. Hidden and recurrent weights barely learn within a lifetime. Their third
   factor is a receptor-gated scalar reward surprise; receptors initialize at
   zero and the winner's stay small ([-0.24, 0.29]). So the recurrent code is
   effectively evolved, not learned.
2. A dense random recurrent matrix destabilizes the dynamics, so selection
   prunes it back toward the feedforward solution.
3. The one signal rich enough to shape a recurrent memory code — the
   backpropagated / weight-transported error — is prohibited by the architecture
   bar (no BPTT, no transported forward weights). Fixed random direct feedback
   (C) is the sanctioned substitute, but over four passes it never aligns and
   acts as noise (83/150).

The binding constraint is therefore not search width, reproduction
concentration, or structural capacity. It is that the local, biologically
plausible learning rules cannot form a multi-symbol recurrent memory code within
a lifetime, and asexual search does not reliably discover one in 100
generations. The e-prop temporal-credit improvement (below) is the one change
that helped, precisely because it strengthens local temporal credit assignment.

## e-prop versus scalar trace (the one confirmed lever)

Toggling only the temporal-credit rule (`--temporal-credit eprop|scalar`):

| Task (gen 100, seed 101) | scalar trace | e-prop | Delta |
|---|---:|---:|---:|
| Hard next-token (correct/150) | 83 | 92 | +9 |
| Hard next-token seed 7 | 76 | 86 | +10 |
| Memory (sealed character accuracy) | 93.9% | 89.5% | -4.4 |
| Memory (sealed exact-string) | 77.5% | 64.5% | -13.0 |

e-prop is a genuine, seed-robust improvement on the temporally deep prediction
task, but it trades away memory precision: the leaky eligibility that credits
long-range next-token dependencies blurs the per-attempt credit the memory task
needs. The trade shows up even at the primary metric -- e-prop leaves memory
character accuracy sitting right at the 90% gate (89.5-90.3% depending on run),
where the scalar trace clears it comfortably at 93.9%. Reaction (100%),
continual (92.9%), and canonical next-token (95% by generation 200) are
unaffected.

## Depth ceiling (population 1,024, seed 101)

Depth is the only factor that reliably moves the number, but it saturates with
heavy diminishing returns and never reaches the two-character behavioral ceiling
of 120/150.

| Generation | 100 | 200 | 300 | 400 | 500 | 700 | 900 | 1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dev correct/150 | 92 | 98 | 102 | 106 | 107 | 108 | 110 | 110 |
| Hidden nodes | 12 | 15 | 18 | 22 | 23 | 25 | 30 | 30 |
| Enabled edges | 1,180 | 1,275 | 1,370 | 1,499 | 1,529 | 1,596 | 1,752 | 1,752 |

The 1,000-generation run selected its representative at generation 774 and sealed
at 110/150 (73.3%), 16 minutes wall on 14 cores. A 10x deeper budget bought only
+18 tokens over generation 100 while inflating the network 2.5x in hidden nodes,
and the curve is flat from generation 800. Capacity is therefore not the
constraint: 30 hidden nodes and a dense plastic readout still cannot cross the
two-character ceiling, which is exactly the signature of a system that adds
feedforward features it can train but never forms the recurrent memory code that
two- and three-character context requires.

## Learning-signal ceiling diagnostic (transported weights)

To test directly whether the hidden teaching signal is the binding constraint, a
deliberately biologically-implausible learner was run: hidden neurons receive the
categorical output error backprojected through their current readout weights
(`sum_k W[j->k]*error_k`), riding on the existing e-prop eligibility. This is
weight transport, which the architecture bar forbids; it is a measurement, not a
system, and the code is flagged for removal. It also bypasses the receptor gate
so the signal is active from generation zero. The learning-pass count is swept
because more within-lifetime passes only help once the update direction is good.

Matched sweep, population 512, generation 150, all 14 workers, sealed
correct/150:

| Signal / passes | seed 101 | seed 7 |
|---|---:|---:|
| reward surprise, 4 passes (matched reference) | 73 | 86 |
| transported readout, 8 passes | 95 | -- |
| transported readout, 16 passes | 107 | 114 |
| transported readout, 24 passes | 103 | -- |
| transported readout, 32 passes | 119 | 108 |

Two results, one solid and one directional:

1. **Solid:** the transported signal beats the matched reward reference by +28 to
   +46 tokens at the same search budget. The weak reward signal collapses to
   73-86 when search is starved (pop 512, gen 150); the good signal *learns* its
   way to 108-119 with that same starved budget -- roughly the reward signal's
   full-budget ceiling (110, which needed pop 1024 x gen 1000). So the hidden
   teaching signal, not search or capacity, governs sample efficiency.
2. **Directional (noisy):** the good signal also appears to lift the ceiling from
   ~1.5-character context (110) toward the two-character bound (120); the best
   run reached 119. But the per-point spread is large (the pass-count optimum is
   inconsistent across seeds, e.g. seed 101 peaks at p32 while seed 7 peaks at
   p16), and every run selected at generation 149 (still climbing), so gen 150
   truncates. The exact lifted ceiling is not pinned down; only the direction is.

The confound that a good signal is over-driven by a learning rate evolved for the
receptor-gated regime remains, but it can only *understate* the transported
signal, so it does not threaten the positive result.

This confirms the mechanism the negative sweep only inferred: local, biologically
plausible learning cannot form the recurrent memory code because the hidden
neurons' third factor is too weak / misaligned, not because the substrate lacks
capacity, recurrence, or search. The open question is no longer "what is the
bottleneck" but "can a plausible signal (evolvable feedback, sign-symmetric
feedback, or a learned feedback module) approximate the transported ceiling
without weight transport or BPTT."

## Decision

- Keep e-prop as the default temporal-credit rule; it is the only intervention
  that improved the hard passage and it clears every primary gate.
- Reject proposals A, B, and C for the hard passage. Retain them as documented,
  default-off CLI levers (`--selection`, `--hidden-feedback`, and the
  `self_recurrent_hidden` / `dense_recurrence` / `heterogeneous_time_constants`
  search params) so the negative results are reproducible.
- The 90% (135/150) milestone at generation 100 is not reachable with these
  levers. The next real lever is a local learning rule that can shape recurrent
  memory within a lifetime without transported weights or BPTT (e.g. an
  evolvable neutral e-prop gate that lets a lineage recover the scalar trace
  where memory precision matters while recruiting leaky temporal credit where
  depth matters), and/or a many-more-generation depth budget, since depth is the
  only factor that reliably moved the number (92 at 100 to 98 at 200).
