using System;
using System.Collections.Generic;
using Definitions;
using Random = UnityEngine.Random;

/* A gene describes:
    - InterNeuron
        - Neurotransmitters
        - Receptors
            - MatchingNeurotransmitters
            - potentialPolarization
        - Type
        - actionPotentialThreshold
        - restingPotential
        - PotentialDecayRate
    - Synapses
        - fireProbability
        - PreSynapticNeuron
        - PostSynapticNeuron
 */
public class Gene {
    public HashSet<NeurotransmitterType> neurotransmitters;
    // [type, neuronID, fireChance]
    public List<Tuple<int, int, float>> synapses;
    public List<Tuple<HashSet<NeurotransmitterType>, float>> receptors;
    public NeuronType type;
    public int neuronID;
    public int actionPotentialLength;
    public float actionPotentialThreshold;
    public float restingPotential;
    public float potentialDecayRate;
    public float sensorySensitivity;
    
    public Gene(Neuron neuron) {
        neurotransmitters = neuron.Neurotransmitters;
        synapses = new List<Tuple<int, int, float>>();
        receptors = new List<Tuple<HashSet<NeurotransmitterType>, float>>();
        if (neuron is IInputNeuron inputNeuron) {
            foreach (Synapse s in inputNeuron.Synapses) {
                synapses.Add(Tuple.Create((int) ((Neuron)s.PostSynapticNeuron).Type, ((Neuron)s.PostSynapticNeuron).NeuronID, s.FireProbability));
            }
        }
        if (neuron is IOutputNeuron outputNeuron) {
            foreach (NeurotransmitterReceptor r in outputNeuron.Receptors) {
                receptors.Add(Tuple.Create(r.MatchingNeurotransmitters, r.PotentialPolarization));
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
        if (Random.value < mutationChance) mutateNeurotransmitters(neurotransmitters);
        if (Random.value < mutationChance) mutateSynapses(synapses); 
        if (Random.value < mutationChance) mutateReceptors(receptors);
        if (Random.value < mutationChance)
            actionPotentialLength = Math.Max(0, actionPotentialLength + Random.Range(0, 2) == 0 ? -1 : 1);
        if (Random.value < mutationChance)
            actionPotentialThreshold += Random.Range(0, mutationMagnitude) * Random.Range(0, 2) == 0 ? -1 : 1;
        if (Random.value < mutationChance)
            restingPotential += Random.Range(0, mutationMagnitude) * Random.Range(0, 2) == 0 ? -1 : 1;
        if (Random.value < mutationChance)
            potentialDecayRate += Random.Range(0, 0.5f/mutationMagnitude) * Random.Range(0, 2) == 0 ? -1 : 1;
    }

    void mutateNeurotransmitters(HashSet<NeurotransmitterType> nSet) {
        // 0 = insertion, 1 = deletion, 2 = substitution 
        int mutationType = Random.Range(0, 3);
        NeurotransmitterType newType = (NeurotransmitterType) Random.Range(0, (int)NeurotransmitterType.NumNeurotransmitters);
        switch (mutationType) {
            case 0:
                if (nSet.Count < (int)NeurotransmitterType.NumNeurotransmitters) {
                    while (nSet.Contains(newType))
                        newType = (NeurotransmitterType)Random.Range(0, (int)NeurotransmitterType.NumNeurotransmitters);
                    nSet.Add(newType);
                }
                break;
            case 1:
                nSet.Remove((NeurotransmitterType) Random.Range(0, nSet.Count));
                break;
            case 2:
                nSet.Remove((NeurotransmitterType) Random.Range(0, nSet.Count));
                while (nSet.Contains(newType)) newType = (NeurotransmitterType) Random.Range(0, (int)NeurotransmitterType.NumNeurotransmitters);
                nSet.Add(newType);
                break;

        }
    }
    
    void mutateSynapses(List<Tuple<int, int, float>> sList) {
        // 0 = insertion, 1 = deletion, 2 = substitution 
        int mutationType = Random.Range(0, 3);
        switch (mutationType) {
            case 0:
                sList.Add(Tuple.Create(Random.Range(1, 3), Random.Range(0, int.MaxValue), Random.Range(0.5f, 1.0f)));
                break;
            case 1:
                if (sList.Count == 0) break;
                sList.RemoveAt(Random.Range(0, sList.Count));
                break;
            case 2:
                if (sList.Count == 0) break;
                sList[Random.Range(0, sList.Count)] =
                    Tuple.Create(Random.Range(1, 3), Random.Range(0, int.MaxValue), Random.Range(0.5f, 1.0f));
                break;
        }
    }
    
    void mutateReceptors(List<Tuple<HashSet<NeurotransmitterType>, float>> rList) {
        // 0 = insertion, 1 = deletion, 2 = substitution 
        int mutationType = Random.Range(0, 3);
        switch (mutationType) {
            case 0:
                rList.Add(Tuple.Create(NeurotransmitterReceptor.generateRandomNeurotransmitters(), Random.Range(-10f, 10f)));
                break;
            case 1:
                if (rList.Count == 0) break;
                rList.RemoveAt(Random.Range(0, rList.Count));
                break;
            case 2:
                if (rList.Count == 0) break;
                rList[Random.Range(0, rList.Count)] =
                    Tuple.Create(NeurotransmitterReceptor.generateRandomNeurotransmitters(), Random.Range(-10f, 10f));
                break;
        }
    }
}
