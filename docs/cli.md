# `cli` — task ecology and stateless world interface

`cli` is the sole headless research interface. It has two explicit namespaces:

```text
cli ecology <task> ...
cli world <command> ...
```

Output is JSON by default. Task runs stream progress JSON to stderr and write a
compressed result. World commands are stateless one-shot operations over an
explicit world file.

Build a release binary for experiments:

```bash
cargo build -p cli --release
CLI=./target/release/cli
```

## Task ecology

The built-in environments all use the same brain adapter, evaluation
protocol, asexual mutation, and finite reproduction algorithm:

```bash
$CLI ecology next-token [run|plan] [OPTIONS]
$CLI ecology symbolic [run|plan] [OPTIONS]
$CLI ecology TASK evaluate RESULT [OPTIONS] [--lesion internal-plasticity]
$CLI ecology analyze RESULT...
```

`run` is optional. `plan` validates and prints the complete configuration and
maximum task-step budget without evolving. `analyze` reads JSON or `.json.zst`
result artifacts.

Shared options:

- `--seed N`: evolutionary run seed.
- `--population N`: number of genomes and offspring slots.
- `--generations N`: evaluation/reproduction depth.
- `--workers N`: parallel evaluation workers. The default is available
  hardware parallelism.
- `--training-instances N`, `--development-instances N`,
  `--sealed-instances N`: generic panel sizes; these are evaluator settings,
  not task configuration.
- `--training-rollouts N`, `--development-rollouts N`, `--sealed-rollouts N`:
  deterministic rollouts per instance.
- `--seed-config PATH`: founder-genome TOML; defaults to the canonical config.
- `--exact-elites N`: unchanged leading genomes copied between generations.
- `--tournament-size N`: competitors sampled per non-elite offspring slot.
- `--exploration-temperature F`: action-sampling temperature multiplier.
- `--action-selection greedy|sampled`: whether evaluation acts from the
  categorical argmax or deterministic sampling.
- `--learning-rule reward_prediction_error|categorical_prediction_error`:
  generic postsynaptic learning signal. Reward-learning tasks use signed
  immediate reward surprise; next-token prediction uses exact learner-visible
  categorical error at action outputs and reward surprise internally.
- `--temporal-credit eprop|scalar`: hidden eligibility rule. `eprop` (default)
  uses leaky-integrator neuron dynamics with the local e-prop state-derivative
  eligibility; `scalar` uses instantaneous neurons and the simpler presynaptic-
  times-gain trace. e-prop is the confirmed improvement on temporally deep
  prediction; the two coincide when a node's leak time constant is minimal.
- `--hidden-feedback reward|categorical`: hidden-neuron third factor when a
  categorical target is revealed. `reward` (default) modulates by scalar reward
  surprise; `categorical` projects the full output error through a fixed random
  per-neuron sign row (direct feedback alignment, no transported forward
  weights). Experimental: did not beat the reward default on the hard passage.
- `--selection tournament|truncation|proportional`: offspring-allocation scheme
  (default `tournament`). `truncation` gives the top `--truncation-survivors`
  parents equal shares; `proportional` uses ticket-proportional stochastic
  universal sampling. Both concentrate reproduction and, empirically, cause
  premature convergence on exploration-limited tasks.
- `--truncation-survivors N`: surviving-parent count under truncation selection.
- `--learning-normalization none|nlms`: generic plasticity normalization.
- `--reset-dynamics-at-trial-boundary true|false`: adapter policy at semantic
  trial boundaries. Learned weights are retained.
- `--audit-interval N`: development-audit interval.
- `--param key=value`: override an asexual mutation parameter. Valid keys are
  printed by `cli ecology help` and invalid keys fail explicitly. Search is
  WANN-NEAT: founders are minimal (each enabled source→output pair wired with
  probability `initial_connection_fraction`, default 0.25 — next-token
  presets 1.0 because its transition table lives in plastic readout weights —
  plus the canonical self-recurrent loop), every hidden node carries a
  heritable activation function from the ten-function weight-agnostic set
  (random at birth and on insert-node), and `mutate_activation_probability`
  (default 0.3) reassigns one random hidden node's function per offspring.
  There is no dense-readout guarantee; evolution owns readout wiring.
  `add_delay_relay_probability` (default 0.05) is the temporal-memory
  operator: it inserts a copy node holding a source's value that immediately
  projects to an output through a plastic edge, so exact memory grows one
  rewarded tap at a time and relays chain into deeper delays. Relay copies are
  held canonical (weight one, non-plastic, saturating-linear) — evolution owns
  whether a tap exists and what it reads out, not what the copy computes.
  `input_delay_line_depth` (default 0) instead seeds a complete delay line of
  that depth into founders; leave it at zero to require that evolution
  discover the structure itself.
- `--out-dir PATH`: result directory; may appear anywhere after `ecology`.

Search does not use scalar fitness, speciation, crossover, target species,
topology rewards, novelty, or in-evaluation births. Each task success event is
one reproductive ticket. After equal evaluation panels finish, one finite
population-sized set of offspring slots is filled by exact elites and fixed-K
tournaments followed by bounded asexual mutation. A generation with no success
events is extinct.

### Basic next-token prediction

```bash
$CLI ecology next-token plan --population 256 --generations 100
$CLI ecology next-token --seed 101 --population 256 --generations 100 \
  --out-dir artifacts/research/runs
```

The canonical training snippet is `the quick brown fox jumps over the lazy
dog`. Starting from a boundary token, the brain is teacher-forced through the
entire prefix one character at a time and predicts the next character at every
position, including the first character and terminal `end`. Every correct
greedy probe prediction is one success event; exact accuracy requires all 44
targets. The default learner receives four complete supervised passes over the
snippet. Recurrent dynamics reset at each pass boundary while learned weights
persist. After the fourth pass, dynamics reset again and plasticity is frozen
for the scored greedy probe. The common symbol interface contains `a`--`z`,
`space`, and `end`; other tasks expose only their declared subsets.

Task options:

- `--learning-passes N`: supervised passes per lifetime. Four is the calibrated
  default for the canonical snippet; harder passages benefit from 32.
- `--predictive-coding true|false`: self-supervised mode with no critic and no
  reward. The learner's error is its own prediction against the next observed
  character, and each correct in-stream prediction is one reproductive ticket.
- `--generalize true|false` with `--snippet-length N`: instead of one fixed
  string, every panel instance draws a freshly generated snippet. Training,
  development, and sealed panels use different panel seeds, so sealed text is
  never seen during evolution. A genome cannot pass by having memorized one
  sequence — only by carrying an architecture that acquires an arbitrary
  sequence within its own lifetime. Use several `--training-instances` so each
  generation samples multiple sequences.

### Symbolic compute (north star)

```bash
$CLI ecology symbolic plan --population 512 --generations 200
$CLI ecology symbolic --seed 7 --population 512 --generations 200 \
  --predictive-coding true --learning-passes 8 \
  --ops copy,reve,rota,dupl,cyph --word-len 4 \
  --training-instances 8 --out-dir artifacts/research/runs
```

Each instance teaches one English-named string operation (`copy`, `reve`,
`rota`, `dupl`, `cyph`) through demonstration pairs (`reve cat tac`) and probes
with fresh query words. Demo streams never repeat verbatim, so lifetime
memorization cannot fit even the training data — execution requires a
character register plus instruction latching.

Task options: `--ops LIST`, `--word-len N`, `--demos N`, `--queries N`,
`--learning-passes N`, `--predictive-coding true|false`.

Co-evolutionary forging (`next-token` only): `--coevolve true
--forge-population N --forge-snippets S` runs an adversarial forge population
that biases half of each training panel; audits remain on the neutral
generator, and a `<result>.forges.json` trajectory sidecar records hardness
per generation.

### Progress and results

Each generation event reports completed/total generations, progress percent,
leading accuracy and resources, periodic development accuracy, topology size,
elapsed seconds, and ETA. The terminal stdout object reports the result path,
termination, selected generation, development and sealed controls, total work,
and wall time.

Result artifacts contain the complete task, agent, ecology, search, founder,
generation, population, work, development, sealed, and termination contracts.
Audit scores retain a historical representative but never allocate
reproduction. Automatic mechanism lesions were retired after establishing that
plasticity and action efference copy are causal; targeted lesions belong in
explicit experiments rather than every evolution run.

## Explicit world simulator

The simulator is a stateless one-shot CLI. A world is always an explicit file.
Every call loads `--in`, performs one command, and exits. Mutating commands write
`--out`; when omitted, `--out` defaults to `--in` and advances in place.

```bash
$CLI world new --seed 7 --out artifacts/worlds/base.bin
$CLI world run-to 500 --in artifacts/worlds/base.bin
$CLI world brain 0 --in artifacts/worlds/base.bin
```

Do not expect process memory to survive between invocations. Snapshot or fork a
world with `cp`, and fan out independent runs by backgrounding invocations.
Keep worlds under `artifacts/`, not `/tmp`.

### World and metric files

- `--in WORLD`: input world, required except for `new`.
- `--out WORLD`: output world for a mutating command; defaults to `--in`.
- `--metrics PATH`: override the metric sidecar location.
- `--no-metrics`: disable sidecar loading and persistence.

`new` mints `<world>.metrics`. The sidecar follows the output world and is
required by `pillars`, `eco` trajectory, and `timeseries`. Copy both files when
forking a measured world. Raw world state remains readable without the sidecar.

### Mutating commands

```text
new [--config P] [--seed N] [--seed-genome-snapshot P]
    [--set k=v]... [--scale W,POP] [--threads K]
    [--report-every R] --out WORLD
step [N] --in WORLD [--out WORLD]
run-to T --in WORLD [--out WORLD]
watch T [--every E] --in WORLD [--out WORLD]
```

`new` reads canonical TOML by default. `--set` overrides a documented world
configuration key. `--seed-genome-snapshot` loads one bincode
`OrganismGenome`, used for every founder. `step` advances by a relative count;
`run-to` advances to an absolute turn; `watch` emits periodic status while
advancing.

### Read-only commands

```text
turn | state | pillars | eco | lineage | genome --in WORLD
timeseries | inspect | top | hist | find | brain | decide --in WORLD
query --in WORLD
```

Read commands never write a world. `query` batches read commands over one load.
`pillars` returns shared raw windowed metrics—plant/prey consumption rates,
action effectiveness, `mi_sa`, and learning slope—with no implied `[0,1]`
interpretation. Its `granular` field contains the per-report-interval series.

### Throughput and sweeps

```text
bench [N] --in WORLD
sweep --grid k=v,v --seeds N,N --to T [--out-dir D]
```

`bench` measures tick throughput without persisting an advanced world. `sweep`
runs the grid by seed in parallel and writes a result under `--out-dir`
(default `artifacts/runs/`).

### Interactive mode

```bash
$CLI world tui --in artifacts/worlds/base.bin
$CLI world tui --new --seed 7
```

The TUI keeps one resident world and reuses the same world-command dispatch.
Changes remain in memory until `save`; `quit` warns about unsaved changes.

## Artifact policy

Generated worlds, datasets, logs, and rendered outputs belong under
`artifacts/research/runs/`. Durable hypotheses, proposals, conclusions, and the
experiment index belong under `research/`.
