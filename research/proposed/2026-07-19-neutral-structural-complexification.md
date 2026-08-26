# Neutral structural complexification experiment

## Question

Is harder next-token learning blocked because structural mutation destroys a
parent's behavior before a new representation can be refined?

The scalar hidden-credit baseline reached 89/150 at population 1,024 and 200
generations. Increasing tournament pressure, class-normalizing synapse
mutation, and direct categorical feedback did not break that result. Inspection
then found that adding one hidden neuron created a dense set of randomly
weighted action and value readouts. With 28 actions, one topology mutation
therefore injected 29 immediate output perturbations. The nonlinear NEAT-style
edge split also disabled the parent path, so the mutation was not function
preserving.

## Treatment

Use an additive network morphism:

- retain the selected parent edge;
- add a zero-bias hidden feature driven by the selected edge's source;
- give the new feature equal minimum-magnitude weights to every action output,
  producing a common logit offset that leaves softmax probabilities and argmax
  unchanged;
- give its value edge the same minimum-magnitude initialization;
- retain plasticity coefficient one so ordinary lifetime learning can recruit
  the new feature immediately;
- continue using ordinary add-connection, delete, parameter, and receptor
  mutation afterward.

Founder readouts remain randomized exactly as before. The treatment applies
only to connections introduced after structural mutation. It is independent of
task identity, target symbols, reward, and fitness, and adds no representation.

## Matched gates

1. Hard 150-target passage, seed 101, population 1,024, generation 100,
   tournament size 4. Control: 83/150.
2. Continue to generation 200 only if the treatment improves or materially
   accelerates the control trajectory. Control: 89/150.
3. If successful, sweep population width and re-run all four established tasks
   at population 1,024 within 200 generations. Every established task must
   remain at or above 90% sealed primary accuracy.

Reject and remove the treatment if it does not improve representation growth
or established competence.
