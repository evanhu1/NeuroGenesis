using System.Collections.Generic;
using Definitions;
using UnityEngine;

public class InterNeuron : Neuron, IInputNeuron, IOutputNeuron {
    public List<NeurotransmitterReceptor> Receptors { get; set; }
    public List<Synapse> Synapses { get; set; }

    public InterNeuron(int NeuronID) : base(NeuronID, NeuronType.InterNeuron) {
        Receptors = new List<NeurotransmitterReceptor>();
        Synapses = new List<Synapse>();
        
        int numReceptors = Random.Range(1, 4);
        for (int i = 0; i < numReceptors; i++) {
            Receptors.Add(new NeurotransmitterReceptor());
        }
    }
    
    public InterNeuron(
        int NeuronID,
        float actionPotentialThreshold,
        float restingPotential,
        int actionPotentialTime,
        HashSet<NeurotransmitterType> neurotransmitters,
        List<Synapse> synapses,
        List<NeurotransmitterReceptor> receptors
    ) : base(
        NeuronID,
        actionPotentialThreshold,
        restingPotential,
        actionPotentialTime,
        neurotransmitters,
        NeuronType.InterNeuron
    ) {
        Synapses = synapses;
        Receptors = receptors;
    }

    public void createSynapse(IOutputNeuron postSynapticNeuron, float fireChance) {
        if (postSynapticNeuron == this) return;
        foreach (Synapse synapse in Synapses) {
            if (synapse.PostSynapticNeuron == postSynapticNeuron) {
                return;
            }
        }

        Synapses.Add(fireChance < 0
            ? new Synapse(this, postSynapticNeuron)
            : new Synapse(fireChance, this, postSynapticNeuron));
    }

    public override void sumPotentials() {
        foreach (NeurotransmitterReceptor receptor in Receptors) {
            potential += receptor.getValueAndReset();
        }
    }
    
    public override void fireActionPotential() {
        foreach (Synapse synapse in Synapses) {
            synapse.fireSignal();
        }
    }
}