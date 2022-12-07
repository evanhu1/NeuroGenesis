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
    /// [type, neuronID, fireChance, synapticStrength]
    public List<Tuple<int, int, float, float>> synapses;
    
    public NeuronType type;
    public int neuronID;
    public int actionPotentialLength;
    public float actionPotentialThreshold;
    public float restingPotential;
    public float potentialDecayRate;
    public float sensorySensitivity;
    
    public Gene(Neuron neuron) {
        // Need intermediate Tuple representation for synapse to enable easy mutation.
        synapses = new List<Tuple<int, int, float, float>>();
        
        if (neuron is IOutputNeuron inputNeuron) {
            foreach (Synapse s in inputNeuron.Synapses) {
                synapses.Add(Tuple.Create((int) s.PostSynapticNeuron.Type, s.PostSynapticNeuron.NeuronID, s.FireProbability, s.SynapticStrength));
            }
        }

        neuronID = neuron.NeuronID;
        type = neuron.Type;
        actionPotentialLength = neuron.actionPotentialLength;
        actionPotentialThreshold = neuron.actionPotentialThreshold;
        restingPotential = neuron.restingPotential;
        potentialDecayRate = neuron.PotentialDecayRate;
        if (neuron is SensoryNeuron sensoryNeuron) sensorySensitivity = sensoryNeuron.Receptor.sensorySensitivity;
    }

    public void mutate(float mutationChance, float mutationMagnitude) {
        if (Random.value < mutationChance) mutateSynapses(synapses); 
        if (Random.value < mutationChance)
            actionPotentialLength = Math.Max(0, actionPotentialLength + Random.Range(0, 2) == 0 ? -1 : 1);
        if (Random.value < mutationChance)
            actionPotentialThreshold += Random.Range(0, mutationMagnitude) * Random.Range(0, 2) == 0 ? -1 : 1;
        if (Random.value < mutationChance)
            restingPotential += Random.Range(0, mutationMagnitude) * Random.Range(0, 2) == 0 ? -1 : 1;
        if (Random.value < mutationChance)
            potentialDecayRate += Random.Range(0, 0.5f/mutationMagnitude) * Random.Range(0, 2) == 0 ? -1 : 1;
    }
    
    void mutateSynapses(List<Tuple<int, int, float, float>> sList) {
        // 0 = insertion, 1 = deletion, 2 = substitution 
        int mutationType = Random.Range(0, 3);
        switch (mutationType) {
            case 0:
                sList.Add(Tuple.Create(Random.Range(1, 3), Random.Range(0, int.MaxValue), Random.Range(0.5f, 1.0f), Random.Range(-10.0f, 10.0f)));
                break;
            case 1:
                if (sList.Count == 0) break;
                sList.RemoveAt(Random.Range(0, sList.Count));
                break;
            case 2:
                if (sList.Count == 0) break;
                sList[Random.Range(0, sList.Count)] =
                    Tuple.Create(Random.Range(1, 3), Random.Range(0, int.MaxValue), Random.Range(0.5f, 1.0f), Random.Range(-10.0f, 10.0f));
                break;
        }
    }
}
