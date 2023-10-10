using System.Collections.Generic;
using Definitions;
using UnityEngine;

public class InterNeuron : Neuron, IOutputNeuron {
    public Dictionary<Neuron, float> Synapses { get; set; }

    public InterNeuron(int NeuronID, bool isInverted = false) : base(NeuronID, NeuronType.InterNeuron, isInverted) {
        Synapses = new Dictionary<Neuron, float>();
    }
    
    public InterNeuron(
        int NeuronID,
        float actionPotentialThreshold,
        float restingPotential,
        int actionPotentialTime,
        float potentialDecayRate,
        Dictionary<Neuron, float> synapses,
        bool isInverted = false
    ) : base(
        NeuronID,
        actionPotentialThreshold,
        restingPotential,
        actionPotentialTime,
        potentialDecayRate,
        NeuronType.InterNeuron,
        isInverted
    ) {
        Synapses = synapses;
    }

    public override void fireActionPotential() {
        foreach (Neuron postSynapticNeuron in Synapses.Keys) {
            postSynapticNeuron.incomingCurrent += Synapses[postSynapticNeuron];
        }

        potential = restingPotential;
    }
}