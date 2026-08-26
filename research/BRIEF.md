# NeuroGenesis research brief

## Goal

**Evolve brains capable of symbolic computation, grounded in English, so that
researcher and organism share a common language.**

Concretely: populations of small recurrent plastic brains whose entire sensor
and motor interface is lowercase English text (`a`–`z`, space, `end`). A brain
must acquire — within its own lifetime, from in-stream demonstrations —
symbol-manipulation operations whose names are English words (`copy`, `reve`,
`rota`, `dupl`, `cyph`), then execute them on fresh queries it has never seen.
The endgame is a lineage that reads English instructions we write and computes
what they mean, with nothing hand-wired between us and it.

### Subgoals (staged milestones)

| Stage | Milestone | Gate |
|---|---|---|
| **S1** | Fixed vocabulary: five English-named string operations taught by demonstration, executed on fresh queries | ≥90% sealed char accuracy on causal ops; reverse gap quantified |
| **S2** | Grounding by demonstration: forge scrambles name↔operation bindings per instance so the English name is arbitrary and meaning must be induced from demos | ≥80% sealed under scrambled bindings |
| **S3** | Open-ended vocabulary: the adversarial forge *invents* new operations; brains must acquire never-before-seen operation kinds within one lifetime | sustained acquisition of forge-invented ops across a run |
| **S4** | Composition: instruction *sequences* (programs) — `reve rota cat tac` — executed as chained transformations | above-chance exact-program rate |

Parallel capability milestone: **working memory**. S1 baselines already
localize the binding constraint to a character register (hold a word across
~6 ticks) plus instruction latching. Every stage above depends on it.

### Constraints (inviolable)

1. **Co-evolutionary EA structure.** Open-endedness comes from the competitive
   environment evolving alongside the brains — a difficulty ratchet that no
   human authors. The forge side must be under selection (fitness derived
   from brain failure/progress), deterministic, and instrumented. Hand-tuned
   difficulty schedules are for calibration only, never the mechanism.
2. **Bottomless learning signal.** Lifetime learning is self-supervised
   prediction error (predictive coding): the learner's own prediction against
   the next observation. No saturating scalar reward, no teacher signal the
   environment did not naturally emit, no BPTT, no transported forward
   weights.
3. **Architecture bar.** Tasks define environment semantics only — no
   task-installed representations, no inspection of genomes or neurons, no
   selection coupling inside evaluation. One generic adapter
   (`evolution::TaskEcology`) is the only bridge. Search remains
   fitness-scalar-free: atomic task success events compete for finite
   offspring slots.
4. **Scaling imperative.** The path to symbolic competence runs through
   massively larger brains than today's ~50-node/900-edge winners. Every
   substrate change must keep open: indirect/region-based encodings (compact
   genomes expressing large connectomes), lifetime structural plasticity
   (prune-and-regrow by local signals), and batch-friendly/GPU-portable
   evaluation. Survey and plan: [scaling beyond NEAT]
   (archive/reports/2026-08-25-scaling-beyond-neat.md).
5. **Determinism.** Fixed config + seed = bit-identical results. All
   tie-breaking by organism ID ordering; all sampling from hash-mixed seeds.
   Optimizations must be verified bit-exact (or documented-exact for
   multi-instance diagnostic floats) before adoption.
6. **The hex-grid world simulator stays.** `world-sim` (with its Axum server
   and web client) is the ecological substrate and long-term home for
   grounding beyond text; the symbolic track is its cognition engine, not a
   replacement for it.

## Current system inventory

- `task-library`: `next_token_prediction` (English acquisition; forge
  substrate) and `symbolic_compute` (north-star task). Legacy
  reaction/memory/continual benchmarks retired 2026-08-25 after their gates
  were re-established; history in `research/`.
- `evolution`: the generic task adapter + asexual ticket ecology
  (`run_resource_ecology`) and adversarial co-evolution
  (`co_ecology`: word-distribution forge population with repetition knob,
  champion-forge panel biasing, hardness telemetry).
- `brain`: WANN-NEAT genomes (heritable activation functions, minimal
  founders), expression, leaky-integrator evaluation, three-factor local
  plasticity (eligibility × postsynaptic signal), predictive-coding mode.
- `cli`: `ecology` (next-token, symbolic, evaluate, analyze, --coevolve) and
  `world` (stateless hex-grid world-as-file tools, TUI, sweeps).
- `world-sim`/`sim-server`/`web-client`/`metrics`/`views`: the deterministic
  hex-grid ecology and its human-facing stack, unchanged by the symbolic track.
- Performance state: ~1.35 G-synops/s aggregate on 14 cores (~80% utilization
  ceiling at current job sizes; rayon join plumbing is the residual); four
  bit-exact brain optimizations in place (−26% wall on E1-scale runs);
  `target-cpu=native` gives an additional −8% and is verified bit-exact.

## Established evidence (what we know)

1. **The learning signal is bottomless.** Self-supervised dose-response on the
   hard 150-target passage is linear through 4× dose: 95→101→105→112 per 150
   at 32/64/96/192 passes (p512/g150, all-time high; scalar reward saturates
   at 110 even at g1000). Frozen champions acquire never-seen English text at
   78% vs 3.57% chance; inherited priors contribute only ~17 points.
2. **Champions already integrate multi-character context.** Exact-task Markov
   ceilings: k=1 → 64/150, k=2 → 120, k=3 → 141. Winners at 112 exceed k=1 via
   leaky-membrane context, without a single recurrent edge. "0% recurrent
   edges" ≠ "no temporal computation."
3. **Open-ended environment pressure works mechanically.** Evolved brains
   acquire arbitrary novel text (63–73% sealed on unseen generated snippets);
   selection rewards acquisition, not memorization.
4. **Co-evolution v1 failed in the informative way.** Forge fitness =
   absolute brain failure saturates at hardness 1.0 the moment difficulty
   exceeds lifetime learnability — every forge ties, gradient dies, and
   co-evolved arms trail static controls 52–60% vs 73.7% (half of every
   lifetime wasted on unlearnable text). V2 spec: relative forge ranking,
   adaptive length knob anchored to demonstrated competence, progress-shaped
   fitness (learnable-with-effort, not impossible).
5. **S1 symbolic baselines: everything fails at ~4–6% vs 3.7% chance —
   including copy.** Not a bug (traced end-to-end). Two structural causes:
   (a) demo streams never repeat verbatim, so readout memorization cannot
   even fit training data — the first direct computation-vs-memorization
   measurement (learning accuracy ~8% on symbolic demos vs 36% on repeating
   text at equal dose); (b) execution requires a character register
   (word-span memory) plus instruction latching, which the substrate lacks
   without recurrent working-memory codes.
6. **Dose-response may not transfer from memorization to computation.** The
   open question separating S1's two failure hypotheses: insufficient
   lifetime data (more passes fix it) vs register-span wall (only structural
   change fixes it). The launched wl2/lp32 arms answer this; they were
   aborted at teardown and must be rerun.

## Active work

1. **S1 diagnosis completion** (rerun ES1b copy@lp32, ES5 copy@wl2, ES6
   mixed@wl2+lp32): separate dose-insufficiency from memory-span wall.
   - wl2 passes & wl4 fails → memory-span localized → re-test
     delay-relay/self-recurrent machinery; push lifetime structural learning.
   - wl2+lp32 still fails → substrate needs structural learning before
     symbolic ops are tractable at all → that becomes the top investment.
2. **Forge fitness v2**: relative ranking + adaptive length knob +
   progress-shaped fitness, per the E3 post-mortem.
3. **Lifetime structural plasticity** (prune-and-regrow by eligibility
   magnitude): the one lever that is simultaneously the missing cognition
   rung (recurrent working-memory codes that lifetime learning can form) and
   the scaling mechanism (sparsity bounds compute as capacity grows).
4. **Deep-dose gate chase** (secondary): hard passage at ~480 passes
   projected to cross 135/150 if linearity holds — a memory-capability proxy
   worth one confirmation run when convenient.

## Records

Experiment index and decision ledger: [INDEX.md](INDEX.md). Complete numerical
history lives in the linked archive records; generated artifacts under
`artifacts/research/runs/` (git-ignored).
