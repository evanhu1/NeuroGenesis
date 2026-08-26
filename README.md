# NeuroGenesis

**Goal:** evolve brains capable of symbolic computation, grounded in English,
so that researcher and organism share a common language. A population of small
recurrent plastic networks whose entire sensor/motor interface is lowercase
English text must acquire — within a single lifetime, from in-stream
demonstrations — string operations named by English words (`copy`, `reve`,
`rota`, `dupl`, `cyph`), then execute them on queries it has never seen. The
endgame is a lineage that reads instructions we write and computes what they
mean, with nothing hand-wired between us and it.

## Method

Three coupled mechanisms, one per axis of open-endedness:

1. **Bottomless learning signal.** Lifetime learning is self-supervised
   predictive coding: the learner's own prediction against the next observed
   character. No saturating scalar reward, no teacher signal the environment
   did not emit, no BPTT, no weight transport. Plasticity is three-factor and
   strictly local (eligibility traces × postsynaptic signal).
2. **Co-evolutionary difficulty ratchet.** The environment evolves under
   selection alongside the brains: an adversarial *forge* population
   (word-distribution genomes over a shared 64-word English lexicon, plus a
   long-range repetition knob that injects exact-repeat memory demands)
   biases each generation's training panels toward text the brains have not
   mastered. Development and sealed audits always evaluate the neutral
   generator, so measurement integrity is preserved. No hand-authored
   curriculum.
3. **Genome complexification under selection.** Search is fitness-scalar-free
   asexual ecology over WANN-NEAT genomes: atomic task success events compete
   for a finite population of offspring slots; exact elites plus fixed-K
   tournaments allocate reproduction; structural mutation (add/delete node and
   connection, delay-relay operator, ten heritable bounded activation
   functions) complexifies topology without any complexity objective.

The hex-grid ecological world simulator (`world-sim`, `sim-server`,
`web-client`) is retained as the long-term grounding substrate beyond text.
Determinism is absolute: fixed config + seed = bit-identical results.

## Achieved to date

- **Bottomless signal, confirmed.** Dose-response of self-supervised passes on
  the hard 150-target English passage is linear through 4× dose: **95 → 101 →
  105 → 112 correct** at 32/64/96/192 passes (p512/g150; all-time high;
  scalar-reward learning saturates at 110 even at 1,000 generations).
- **Open-ended text acquisition.** In generated-text mode (every instance a
  fresh snippet, sealed text never seen during evolution), evolved brains
  acquire novel English within their lifetime at **63–73% sealed accuracy vs
  3.57% chance**; on frozen champions, lifetime learning is causally
  load-bearing (17% inherited-prior floor → 78% at 32 passes).
- **Context-integration ceiling mapped.** Exact-task Markov ceilings on the
  hard passage: k=1 → 64/150, k=2 → 120, k=3 → 141. Champions at 112 already
  exceed the 1-char ceiling via leaky-membrane temporal integration — with
  zero recurrent edges.
- **Co-evolution machinery built; v1 fitness falsified cleanly.** Dual
  population, champion-forge panel biasing, hardness telemetry, verified
  bit-deterministic. V1 fitness (absolute brain failure) saturates at
  hardness 1.0 once difficulty exceeds lifetime learnability — the gradient
  dies and co-evolved arms trail static controls (52–60% vs 73.7%). The
  post-mortem specifies v2: relative forge ranking, adaptive difficulty knob
  anchored to demonstrated competence, progress-shaped fitness.
- **The north-star task localizes its own wall.** S1 symbolic baselines:
  all five operations at ~4–6% vs 3.7% chance — including copy. Not a bug:
  demonstration streams never repeat verbatim, so readout memorization cannot
  fit even the training data (learning accuracy ~8% on non-repeating demos vs
  36% on repeating text at equal dose — the first direct
  computation-vs-memorization measurement on this substrate). The missing
  capabilities are precisely identified: a **character register** (word-span
  working memory) and **instruction latching**.
- **Engine performance.** ~1.35 G-synapse-ops/s aggregate on 14 cores (~80%
  utilization ceiling at current job sizes, cause measured: rayon join
  plumbing between short generation phases); four profile-guided bit-exact
  optimizations worth −26% wall time on full-scale runs; `target-cpu=native`
  verified bit-exact at an additional −8%.

## Next

1. **S1 diagnosis** — separate the two failure hypotheses (insufficient
   lifetime dose vs register-span wall) via copy at 4× passes and copy at
   halved word length.
2. **Forge fitness v2** — relative ranking + adaptive difficulty, per the
   co-evolution post-mortem.
3. **Lifetime structural plasticity** — prune-and-regrow driven by local
   eligibility signals: the single lever that is both the missing cognition
   rung (working-memory codes that lifetime learning can form) and the
   scaling mechanism (sparsity bounds compute as capacity grows).
4. **Milestones** — S2 scrambled name↔operation bindings (grounding earned by
   demonstration), S3 forge-invented operations, S4 compositional programs.
5. **Scale-up** — region-based indirect encodings and batch-friendly
   evaluation per [the scaling plan]
   (research/archive/reports/2026-08-25-scaling-beyond-neat.md).

Full decision ledger and experiment records: [`research/`](research/).
Current objective and constraints: [`research/BRIEF.md`](research/BRIEF.md).

## Structure

- `task-library/`: brain- and optimizer-independent environments —
  `next_token_prediction` (English acquisition; forge substrate) and
  `symbolic_compute` (English-named operations taught by demonstration).
- `brain/`: WANN-NEAT genome encoding/expression, leaky-integrator recurrent
  evaluation, three-factor local plasticity, predictive-coding mode.
- `evolution/`: the sole generic task adapter; asexual ticket ecology;
  adversarial co-evolution (`co_ecology`).
- `types/`, `config/`: shared symbolic/genome/runtime types; canonical
  configuration.
- `cli/`: sole headless research interface (`ecology` + `world` namespaces).
- `world-sim/`, `metrics/`, `views/`, `sim-server/`, `web-client/`: the
  deterministic hex-grid ecological simulator and visualization stack.

## Run

```bash
cargo build -p cli --release

# English text acquisition (open-ended, self-supervised)
./target/release/cli ecology next-token \
  --seed 101 --population 512 --generations 200 \
  --predictive-coding true --learning-passes 32 \
  --generalize true --snippet-length 32 \
  --training-instances 8 --out-dir artifacts/research/runs

# Symbolic operations from demonstration (north star)
./target/release/cli ecology symbolic \
  --seed 7 --population 512 --generations 200 \
  --predictive-coding true --learning-passes 8 \
  --ops copy,reve,rota,dupl,cyph --word-len 4 \
  --training-instances 8 --out-dir artifacts/research/runs

# Co-evolutionary forging (adversarial environment)
./target/release/cli ecology next-token \
  --seed 7 --population 1024 --generations 200 \
  --predictive-coding true --learning-passes 12 \
  --generalize true --snippet-length 48 --training-instances 8 \
  --coevolve true --forge-population 24 --out-dir artifacts/research/runs

# Hex-grid world simulator (stateless, world-as-file)
./target/release/cli world new --seed 7 --out artifacts/worlds/base.bin
./target/release/cli world run-to 500 --in artifacts/worlds/base.bin
```

Read [docs/cli.md](docs/cli.md) before using the CLI. See
[docs/evaluation-tasks.md](docs/evaluation-tasks.md) for the task boundary.

## Development

```bash
cargo check --workspace
cargo test --workspace
make fmt
make lint
cd web-client && npm run typecheck && npm run build
```

Generated outputs belong under `artifacts/`. Durable hypotheses, methods, and
conclusions belong under `research/`.
