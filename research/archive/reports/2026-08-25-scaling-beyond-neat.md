# Scaling beyond NEAT: walls, frontier survey, and strategy

Date: 2026-08-25. Standing constraint from project direction: this track will
eventually require massively larger neuron/synapse counts, which demands
algorithmic innovation beyond traditional NEAT and compute innovations on our
hardware. This report quantifies current walls, surveys existing frontier work,
and proposes a substrate strategy that keeps our options open.

## 1. Current walls (quantified)

- **Memory**: `SynapseEdge` ≈ 64 B (IDs, timing, weights, eligibility traces).
  A 100 K-edge brain ≈ 6.4 MB; p1024 ≈ 6.5 GB of edges alone before neuron
  state. M4 Max (38 GB) exhausts around ~10× current edge scale for one run.
- **Compute**: evaluation is O(edges × ticks × population × generations). E1a:
  348 G synapse-ops in 257 s ≈ 1.35 G-synops/s aggregate on 14 cores. Linear
  dose-response means each capability step multiplies both edges and ticks:
  ~100× compute per order-of-magnitude capability gain at constant efficiency.
- **NEAT-specific**: explicit edge-list genomes with global innovation counters
  grow O(connections); structural mutation is signal-free (random add/delete);
  published NEAT results top out near 10⁴–10⁵ connections. Random structural
  search flounders beyond that (the "curse of wilderness").

## 2. Frontier survey (existing ideas worth mining)

| Approach | Idea | Fit for us |
|---|---|---|
| HyperNEAT / CPPN indirect encodings | genome = function queried per connection; regularity & scale without genome growth | strong fit conceptually; N² queries need coarse-graining |
| Neural developmental programs (NDPs; Stanley-lineage, 2023–25) | local growth rules expand a seed into a large network | very aligned with "complexification without hand-authoring" |
| ES at scale (OpenAI-ES, CMA-ES, Guided-ES) | rank-based perturbation updates scale to millions of parameters | needs centralized parameter vectors; fits a fixed-topology phase |
| Quality-Diversity (MAP-Elites, CMA-ME) | behavioral niches maintained explicitly | pairs naturally with forge pressure (forges define niches); round-1B novelty failure was descriptor-limited |
| Sparse training (SET, RigL analogues) | prune-and-regrow driven by magnitude/local signals during training | maps onto our prune + eligibility machinery; enables *lifetime structural learning* |
| Regenerative/growing sparse networks (2025 frontier) | regeneration rules grow capacity on demand | same family as NDP; lit check pending |

## 3. The key structural insight for us

Lifetime plasticity owns function; evolution owns architecture-class plus
plasticity hyperparameters. Therefore the evolutionary search space stays small
even when expressed brains become huge. Scaling strategy:

> Grow the expressed brain via **indirect encoding + lifetime structural
> plasticity**, keep genomes compact, and make evaluation batch-friendly.

Concretely ("Forge-scale substrate", v0 sketch):

1. **Region-genome**: R regions (sensory blocks, hidden blocks, output) with
   sizes, an R×R inter-region wiring probability matrix, per-region time
   constant distributions, and per-projection plasticity genes. Expression
   materializes Bernoulli wiring lazily or evaluates block-sparse matrices.
2. **Lifetime structural plasticity** (prune-and-regrow driven by eligibility
   magnitudes — SET-style, no gradient transport). This is simultaneously:
   - the missing ladder rung identified in the hard-passage analysis
     (recurrent codes that lifetime learning can actually form), and
   - the scaling mechanism (sparsity bounds compute as capacity grows).
3. **Layer-relaxed evaluation mode**: region-level topological levels instead
   of exact DAG order → matmul-form evaluation where causality allows.
4. **Population-batched evaluation**: all genomes as one batched tensor stream
   (requires regular layouts — hence the design rule above).

## 4. Hardware path

- M4 Max: Metal/MPS via Rust DL stacks (candle/burn) or MLX bridge;
  block-sparse GEMM; f16 forward passes.
- Decision point: port to GPU **after** v2 ratchet works on CPU (premature now);
  but every substrate change until then must keep the batch-friendly door open.

## 5. New directions beyond existing literature

- **Co-evolved development**: forge pressure selects growth programs —
  environmental complexity drives morphogenetic complexity. The forge side and
  the developmental side complexify against each other.
- **Heritable structure via Hebbian traces**: offspring inherit connection
  usage statistics accumulated during parents' lifetimes (Baldwinian effect
  made mechanistic: eligibility magnitude at death becomes a birth prior).
- **Population-as-ensemble**: maintain architectural diversity as an explicit
  ensemble over niches rather than collapsing to a single champion.

## 6. Immediate design rules adopted today

1. No new substrate irregularity without a batch-evaluation story.
2. Symbolic task streams stay token-uniform (GPU-tileable format later).
3. Structural learning experiments (prune/regrow by eligibility) are scheduled
   after the symbolic S1 baselines — it is both the cognition lever and the
   scaling lever, so it gets priority over cosmetic scale-up.
