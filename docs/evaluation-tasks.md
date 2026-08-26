# Task-library boundary

`task-library` contains deterministic symbolic environments. A task defines
only domain semantics:

- its serializable configuration and private state;
- legal actions and optional symbolic observations;
- deterministic instance construction from `(panel_seed, instance_index)`;
- rewards, atomic success events, task-relative correctness, trial boundaries,
  trial outcomes, and termination.

It must not import `brain` or `evolution`, inspect a genome, install a neural
representation, invoke a learning rule, assign reproductive tickets, select
parents, mutate offspring, or choose training/audit panel sizes.

## Runtime boundary

```text
task-library::SymbolicTask
  observation -> [evolution::TaskEcology -> brain] -> action
  transition  <- [evolution::TaskEcology]
                         |
                  success events
                         v
             finite reproductive tickets
                         |
             asexual tournament search
```

`evolution::TaskEcology<T>` is the only adapter. It owns genome expression,
sensory encoding, action sampling, learning, frozen probes, panel construction,
agent-state policy at semantic trial boundaries, controls, metrics, and the
conversion from task success events to reproductive tickets.

`evolution::run_resource_ecology` sees only the generic
`ResourceEcologyTask` contract. It owns equal-panel evaluation, finite
reproduction, exact elite retention, tournament parent selection, asexual
mutation, audits, and artifacts. Reproduction never occurs inside evaluation.

## Active tasks

- `next-token`: the English-acquisition substrate. Teacher-forced next-character
  prediction over a fixed snippet or (`--generalize`) freshly generated text;
  predictive-coding mode gives a fully self-supervised bottomless learning
  signal. Sealed panels never repeat training text, so genomes must acquire
  arbitrary novel text within their own lifetimes. Also the substrate for
  adversarial forging (`--coevolve true`): an evolving word-distribution
  population (per-word weights + long-range repetition) biases half of each
  training panel while audits stay on the neutral generator.
- `symbolic`: the north-star task. English instruction words (`copy`, `reve`,
  `rota`, `dupl`, `cyph`) name string operations; each instance teaches one
  operation through demonstration pairs (`reve cat tac`) and probes with fresh
  query words. Demo streams never repeat verbatim, so readout memorization
  cannot fit even the training data — execution requires a character register
  (word-span memory) plus instruction latching.

## Retired tasks (2026-08-25 cleanup)

`reaction`, `memory`, and `continual` were legacy capability benchmarks whose
gates were re-established (99.7% / 92–98% / 93.7% sealed) and which no longer
serve the symbolic-computation-in-English direction. Their evidence and
decision history remain in `research/`; the environments were removed from
the codebase.

## Adding a task

1. Add a module under `task-library/src/` implementing `SymbolicTask`.
2. Validate everything in `validate()`; derive panels deterministically from
   `(panel_seed, instance)`.
3. Register the module in `task-library/src/lib.rs` and wire the CLI dispatch
   (`cli/src/ecology.rs`), presets, help text, and `docs/cli.md`.
4. Success events must be atomic and independent (no prefix privileging).
