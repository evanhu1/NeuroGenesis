using System.Collections.Generic;
using Definitions;
using UnityEngine;

public class SensoryNeuron : Neuron, IInputNeuron {
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
        HashSet<NeurotransmitterType> neurotransmitters,
        List<Synapse> synapses,
        SensoryReceptor receptor
    ) : base(
        NeuronID,
        actionPotentialThreshold,
        restingPotential,
        actionPotentialTime,
        neurotransmitters,
        NeuronType.SensoryNeuron
    ) {
        Synapses = synapses;
        Receptor = receptor;
    }

    public void createSynapse(IOutputNeuron postSynapticNeuron, float fireChance) {
        if (((Neuron) postSynapticNeuron).NeuronID == NeuronID) return;
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
        potential += Receptor.getValue();
    }
    
    public override void fireActionPotential() {
        foreach (Synapse synapse in Synapses) {
            synapse.fireSignal();
        }
    }

    public void updateReceptor() {
        Receptor.updateValue();
    }
}
