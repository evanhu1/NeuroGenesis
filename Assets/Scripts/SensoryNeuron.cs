using System.Collections.Generic;
using Definitions;
using UnityEngine;

public class SensoryNeuron : Neuron, IOutputNeuron {
    public SensoryReceptor Receptor;
    public List<Synapse> Synapses { get; set; }

    public SensoryNeuron(int NeuronID, Organism organism) : base(NeuronID, NeuronType.SensoryNeuron) {
        Synapses = new List<Synapse>();
        Receptor = new SensoryReceptor((SensoryReceptorType) NeuronID, organism);
    }

    public SensoryNeuron(
        int NeuronID,
        float actionPotentialThreshold,
        float restingPotential,
        int actionPotentialTime,
        float potentialDecayRate,
        List<Synapse> synapses,
        SensoryReceptor receptor
    ) : base(
        NeuronID,
        actionPotentialThreshold,
        restingPotential,
        actionPotentialTime,
        potentialDecayRate,
        NeuronType.SensoryNeuron
    ) {
        Synapses = synapses;
        Receptor = receptor;
    }
    
    public override void fireActionPotential() {
        foreach (Synapse synapse in Synapses) {
            synapse.fireSignal();
        }
    }

    public void updateReceptor() {
        Receptor.updateValue();
        incomingCurrent = Receptor.getValue();
    }
}
