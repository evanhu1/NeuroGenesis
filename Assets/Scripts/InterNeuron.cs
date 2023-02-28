using System.Collections.Generic;
using Definitions;
using UnityEngine;

public class InterNeuron : Neuron, IOutputNeuron {
    public List<Synapse> Synapses { get; set; }

    public InterNeuron(int NeuronID, bool isInverted = false) : base(NeuronID, NeuronType.InterNeuron, isInverted) {
        Synapses = new List<Synapse>();
    }
    
    public InterNeuron(
        int NeuronID,
        float actionPotentialThreshold,
        float restingPotential,
        int actionPotentialTime,
        float potentialDecayRate,
        List<Synapse> synapses,
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
        foreach (Synapse synapse in Synapses) {
            synapse.fireSignal();
        }
    }
}