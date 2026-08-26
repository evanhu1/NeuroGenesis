# Hard next-token bottleneck localization

## Outcome

The primary learner bottleneck is temporal eligibility. The previous scalar
edge trace did not follow the hidden neuron's recurrent dynamics. Enabling the
existing evolved leak and replacing the internal trace with a local e-prop
state derivative improved the hard 150-target passage from 83 to 92 correct at
generation 100 and from 89 to 98 at generation 200, population 1,024, seed 101.

Rejected matched probes were: tournament sizes 16 and 64; destination-class
normalized mutation; fixed categorical direct feedback; additive neutral node
insertion; a 16-hidden/128-synapse founder; more learning passes; proportional
learning-rate reduction; and frozen eligibility-retention sweeps. Direct
categorical feedback combined with e-prop also regressed to 87/150 at
generation 100 and was removed.

The behavioral ceiling using only the preceding character is 64/150; using two
preceding characters is 120/150 and three is 141/150. The 89/150 baseline was
therefore using context but failed to form a stable three-character code.
Frozen pass sweeps peaked at four passes and then regressed (89, 76, 82, 71
correct at 4, 8, 16, and 32 passes), localizing unstable temporal learning
rather than insufficient exposure.

## E-prop candidate results

| Task | Population / generations | Sealed result |
|---|---:|---:|
| Basic reaction | 1,024 / 200 | 100.000% |
| Basic memory | 1,024 / 200 | 90.750% character; 67.000% exact |
| Basic continual learning | 1,024 / 200 | 93.677% |
| Canonical next token | 1,024 / 200 | 41/44 = 93.182% |
| Hard next token | 1,024 / 100 | 92/150 = 61.333% |
| Hard next token | 1,024 / 200 | 98/150 = 65.333% |

The candidate clears the 90% floor on all established tasks, but it is not yet
a clean replacement: memory regressed from 97.250% character / 89.000% exact,
and canonical next token regressed from 42/44. The old scalar trace must remain
an evolvable neutral subspace before cutover.

Population width also remains independently broken. At generation 100,
population 2,048 with tournament size four reached only 83/150; increasing the
tournament to eight recovered 90/150 but remained below population 1,024's
92/150. Fixed-K reproduction still fails to turn width into proportional local
refinement.
