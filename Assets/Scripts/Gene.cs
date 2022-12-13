using System;
using System.Collections.Generic;
using Definitions;
using Random = UnityEngine.Random;

/* A gene describes:
    - InterNeuron
        - Type
        - actionPotentialThreshold
        - restingPotential
        - PotentialDecayRate
    - Synapses
        - fireProbability
        - synapticStrength
        - PreSynapticNeuron
        - PostSynapticNeuron
 */
public class Gene {
    public NeuronType type;
    public int neuronID;
    public int actionPotentialLength;
    public float actionPotentialThreshold;
    public float restingPotential;
    public float potentialDecayRate;
    public float sensorySensitivity;
    
    public Gene(Neuron neuron) {
        neuronID = neuron.NeuronID;
        type = neuron.Type;
        actionPotentialLength = neuron.actionPotentialLength;
        actionPotentialThreshold = neuron.actionPotentialThreshold;
        restingPotential = neuron.restingPotential;
        potentialDecayRate = neuron.PotentialDecayRate;
        if (neuron is SensoryNeuron sensoryNeuron) sensorySensitivity = sensoryNeuron.Receptor.sensorySensitivity;
    }

    public void mutate(float mutationChance, float mutationMagnitude) {
        if (Random.value < mutationChance)
            actionPotentialLength = Math.Max(0, actionPotentialLength + (Random.Range(0, 2) == 0 ? -1 : 1));
        if (Random.value < mutationChance)
            actionPotentialThreshold += Random.Range(0, mutationMagnitude) * (Random.Range(0, 2) == 0 ? -1 : 1);
        if (Random.value < mutationChance)
            restingPotential += Random.Range(0, mutationMagnitude) * (Random.Range(0, 2) == 0 ? -1 : 1);
        if (Random.value < mutationChance)
            potentialDecayRate += Random.Range(0, 0.5f/mutationMagnitude) * (Random.Range(0, 2) == 0 ? -1 : 1);
    }
}
