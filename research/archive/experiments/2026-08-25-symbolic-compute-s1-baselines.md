# 2026-08-25: symbolic compute S1 baselines (English-grounded operations)

Status: completed (round 1); follow-up dose/register arms running

## Question

Under the new north star (symbolic computation grounded in English), can the
current substrate learn English-named string operations from in-stream
demonstrations, and where does difficulty bite?

## Hypothesis

Instruction-conditioned transformations are learnable by lifetime plasticity +
selection, with a difficulty ladder set by context depth: copy ≈ dupl < rota <
cyph < reve. The `reve` arm quantifies the memory gap that motivates recurrent
codes (exact-task k=W ceiling analysis applies).

## Contract

- Task: `symbolic_compute` v0 (new). Demo episodes `reve cat tac END`;
  probe `reve dog god END`, scored on target chars + trailing END.
- Arms (all p512/g200/lp8/s7, ti=di=si=8, predictive coding):
  - ES1: copy @ word_len 4 (sanity floor)
  - ES2: all five ops mixed @ word_len 4 (instruction conditioning)
  - ES3: cyph alone (induction of per-instance permutation)
  - ES4: reve alone (memory stress)
- Gate (S1): ≥90% sealed char accuracy on causal ops; reve gap quantified.

## Decision rule

- All causal ops ≥90%: S1 gate passed → move to forge-scrambled bindings (S2).
- Mixed ≥ single-op performance − 10 pts: instruction conditioning works.
- reve near zero while others pass: memory-bound wall confirmed → recurrent
  learning rule becomes the priority lever (as anticipated).
- Anything below chance on causal ops after this budget: evaluator/task bug
  hunt before any scientific conclusion.

## Result (round 1, p512/g200/lp8/s7, ti=di=si=8)

| Arm | Sealed acc | Learning acc (demos) | Note |
|---|---:|---:|---|
| copy @ wl4 | 6.25% | 5.4% | chance ≈ 3.7% |
| mixed (5 ops) @ wl4 | 4.03% | 6.3% | |
| cyph @ wl4 | 6.25% | 10.2% | |
| reve @ wl4 | 5.4% | 2.3% | |

**All arms fail at this budget — including copy.** Decision-rule investigation
completed: no task/evaluator bug (traced wl2 example end-to-end). The failures
are structural, in two layers:

1. *Learning* accuracy ~5-10%: demo streams never repeat verbatim (fresh random
   words each demo), so readout memorization cannot even fit the training
   stream — the task demands computation, not lookup. This is the intended
   north-star property, working as designed.
2. *Probe* accuracy ~chance: emitting the transformed word requires holding a
   4-character register across ~6 ticks (word-span memory) plus instruction
   identity across the episode. Substrate lacks recurrent working memory.

Comparison point: hard-passage next-token reached learning accuracy 36% at the
same pass count because its text *repeats* verbatim each pass; symbolic demos
never repeat. The gap between 36% and ~8% at equal dose is the first direct
measurement of "computation vs memorization" difficulty on this substrate.

Follow-up arms launched same day (ES1b/ES5/ES6): copy @ lp32 (dose response),
copy @ wl2 (register span halved), mixed @ wl2+lp32 — to locate the memory
cliff and test whether the linear dose-response transfers from memorization
tasks to computation tasks.
