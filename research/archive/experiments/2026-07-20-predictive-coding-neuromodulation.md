# Predictive coding + evolvable neuromodulatory channels

Status: completed discovery experiment
Date: 2026-07-20
Slug: 2026-07-20-predictive-coding-neuromodulation

## Question

The transported-weight diagnostic showed a perfectly aligned hidden teaching
signal breaks the hard-passage ceiling. Two follow-on questions: (1) can the
supervised categorical error be replaced by a biologically plausible self-
supervised signal, and (2) can a plausible low-rank neuromodulatory pathway
route credit to hidden neurons without weight transport?

## Design

- **Predictive coding:** the network predicts the next symbol; the error
  `e_k = onehot(next_observed)_k − prob_k` is derived from the observation
  stream (the next input), not a revealed teaching target. No critic, no reward;
  correctness is only the outer fitness. Action-readout edges learn from `e_k`
  directly (local to each output).
- **Neuromodulatory channels (the plausible hidden signal):** C global evolvable
  channels `modulator_c = Σ_k B[c,k]·e_k`, read through per-neuron evolvable
  receptors `m_j = Σ_c receptor[j,c]·modulator_c` — a rank-C factorization of a
  feedback map, `F = receptor·B`, with B and receptors evolution-tuned and
  neutral-init (zero receptors → output-only). This is the plausible stand-in
  for the transported `F = W^T`.
- Sweep C at passes 16, population 512, generation 150, seed 101. Probe-based
  fitness (isolates the learned final state; comparable to the transported
  references). Implementation reviewed for correctness (channel math, no-critic,
  neutral init, mutation order all verified).

## Result: channels fail monotonically

| C (channels) | 0 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| Sealed correct/150 | 101 | 95 | 93 | 88 | 85 |

Adding evolvable neuromodulatory channels degrades a strong baseline,
monotonically with C. The alignment problem is the cause: the transported signal
works because `B = W^T` is aligned by construction; evolution cannot discover an
aligned `B` in this budget (no gradient pulls it toward alignment, only weak
downstream selection over a C×28 matrix), and an unaligned channel is direct-
feedback-alignment noise — recruiting a receptor injects misdirected updates into
hidden learning. More channels means more misdirection plus more search dilution.
This mirrors the earlier random-DFA failure (83/150): plausible-but-unaligned
feedback is worse than no feedback.

## The correction: the hidden signal was over-credited

C=0 here is output-only predictive coding — the self-supervised prediction error
at the readout, no hidden learning — and it reaches **101/150** at passes 16 and
**115/150** at passes 32. The passes-32 output-only figure already *beats* the
reward signal's 1,000-generation brute-force ceiling of 110 and approaches the
two-character bound of 120, plausibly and at a fraction of the budget. Against
that, the *perfect* transported hidden signal adds almost nothing: +6 at passes
16 (101 vs 107) and **+4** at passes 32 (115 vs transported 119). The isolated
hidden-signal benefit is ~+4 to +6 — a rounding error next to the output signal.

The earlier "+30 to +46" attributed to the hidden teaching signal was mostly the
**output** signal — categorical prediction error versus scalar reward — plus more
passes; the transported-vs-reward comparison changed all three at once. This
experiment separates them. The output signal is the large, and biologically
plausible, lever: predictive coding computes it locally (prediction vs next
observation, no teacher, no transport). The hidden-feedback refinement is small
and, by every plausible route tried (random DFA, evolvable neuromodulation), not
capturable — only weight transport reaches it.

## Decision

- **Adopt predictive coding as the honest, plausible learner.** Self-supervised
  output error, no critic, no revealed target reaches ~101/150 — the real result
  of this line, and consistent with the architecture bar.
- **Reject the neuromodulatory-channel hidden signal.** It degrades performance
  monotonically. Keep it as a documented, default-off option
  (`--feedback-channels`), not a default.
- Plausible hidden-neuron credit assignment without weight transport remains the
  open problem; this is a clean negative data point on it. The one untried
  variant with a known chance is sign-symmetric feedback (`B = sign(W)`), which
  peeks at forward-weight signs — viable only if the bar admits that gray area.
- Note: continuous-stream fitness (every correct tick a ticket) was specified but
  is currently inert — the adapter still scores selection on the frozen probe.
  For this experiment probe-fitness is preferable (it rewards the learned state,
  not inherited correctness), so the sweep is valid; the continuous variant would
  need the adapter fix and is expected to understate the channel effect.
