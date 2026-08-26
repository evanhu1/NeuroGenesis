# Performance investigation: task-ecology evaluation throughput

Date: 2026-08-25. Scope: `evolution::TaskEcology` + `brain` hot paths that
dominate every `cli ecology` experiment. Machine: M4 Max (14 logical cores),
release profile (`lto=fat`, `codegen-units=1`).

## Method

1. macOS `sample` on a live p256/g40@32p evolution run, 2×10 s windows,
   self-time attribution over the call tree.
2. Worker-count sweep; native-codegen A/B.
3. Code-level optimizations, each verified **bit-exact** by diffing complete
   result JSONs (generation summaries, audits, final population, champion
   genome — everything except wall-time fields) before vs after.

## Profile findings (before)

| Share of CPU | What |
|---|---|
| ~16% | rayon worker threads spinning in `wait_until_cold` between/around parallel phases |
| ~12% | eligibility staging (`accumulate_synaptic_eligibilities` + helpers) |
| ~7% | three-factor weight update (`apply_three_factor_learning` + `apply_edges`) |
| ~6% | forward pass (`evaluate_brain_state`) |
| ~10% | adapter loop itself (softmax, sampling, metrics) |
| <1% | expression, mutation, transcendental libs |

Plasticity passes cost ~3× the forward pass. Two concrete inefficiencies:
activation derivatives computed **twice** per hidden neuron per tick;
per-edge Option-dispatch across three target classes despite the guaranteed
contiguous inter-before-output edge layout (`output_synapse_start`).

## Changes made (all bit-exact)

1. `brain::learning`: compute the activation derivative once per hidden neuron
   per tick (was twice); single pass fills local gains / raw gains / retentions.
2. `brain::learning::set_pending_for_edges` and `apply_edges`: split each
   synapse slice at `output_synapse_start`; internal-group edges skip the
   Option dispatch, output-group edges skip the hidden check.
3. `brain::evaluation::accumulate_inter_inputs`: replace per-edge validated
   `inter_index()` with unchecked inverse mapping `dense_inter_index()`
   (debug_assert retained).
4. Integer report counters batched per call in registers (`BatchedEdgeCounts`)
   and merged once — integer addition is associative so results are identical;
   f64 delta accumulators intentionally left edge-by-edge to preserve exact
   float ordering.

## Measured results

| Workload | Before | After |
|---|---:|---:|
| probe p256/g40@32p | 35.3 s | 33.1 s (−6%) |
| **full E1a p512/g150@32p seed 101** | **346 s** | **257 s (−26%)**, bit-exact vs archived artifact |
| generalize p256/g30@8p ti4 | 5.75 s | 5.69 s |

Worker scaling: 14 workers only ~1.25× faster than 7 → ceiling is not core
count; it is serial-phase boundaries plus spin overhead (~14% still idle).
`target-cpu=native` build: −8% additional, verified bit-exact; adopt for
experiment boxes via `RUSTFLAGS="-C target-cpu=native" cargo build -p cli --release`
(binary becomes machine-specific).

## Larger levers identified, deliberately NOT taken (semantic decisions)

1. **Instance-panel flattening**: evaluate training instances as independent
   rayon tasks (helps multi-instance panels like memory/E2-style runs; tail
   imbalance shrinks). Requires reordering f64 metric accumulation or accepting
   last-ulp drift in diagnostic metrics (selection-relevant integers unaffected).
2. **Report-metric gating**: most per-tick counters feed artifacts only, never
   selection; making them audit-only would remove several ops per edge per tick.
   Changes artifact contents — needs an explicit contract decision.
3. **SoA/SIMD brain layout**: flatten `Vec<SynapseEdge>` per neuron into arena
   with dense cached indices; largest potential (forward+staging), biggest
   refactor, touches world persistence format.

## Verification protocol used

Same seed/config before/after; canonical JSON diff excluding wall-time fields;
`cargo test --workspace --release` all green; `make lint` clean.
