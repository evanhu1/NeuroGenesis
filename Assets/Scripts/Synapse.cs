using System;
using System.Collections.Generic;
using Definitions;
using Random = UnityEngine.Random;

public class Synapse {
    // Chance that this synapse will fire during an action potential
    public float FireProbability;
    public IInputNeuron PreSynapticNeuron;
    public IOutputNeuron PostSynapticNeuron;

    public Synapse(float releaseProbability, IInputNeuron preSynapticNeuron, IOutputNeuron postSynapticNeuron) {
        FireProbability = releaseProbability;
        PostSynapticNeuron = postSynapticNeuron;
        PreSynapticNeuron = preSynapticNeuron;
    }

    public Synapse(IInputNeuron preSynapticNeuron, IOutputNeuron postSynapticNeuron) {
        FireProbability = Random.Range(0.5f, 1.0f);
        PostSynapticNeuron = postSynapticNeuron;
        PreSynapticNeuron = preSynapticNeuron;
    }

    public void fireSignal() {
        if (Random.value < FireProbability) {
            foreach (NeurotransmitterReceptor receptor in PostSynapticNeuron.Receptors) {
                HashSet<NeurotransmitterType> matchingNeurotransmitters = new (((Neuron) PreSynapticNeuron).Neurotransmitters);
                matchingNeurotransmitters.IntersectWith(receptor.MatchingNeurotransmitters);
                if (matchingNeurotransmitters.Count > 0) {
                    receptor.trigger();
                }
            }
        }
    }
}
