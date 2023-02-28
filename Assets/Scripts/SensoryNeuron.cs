using System.Collections.Generic;
using Definitions;

public class SensoryNeuron : Neuron, IOutputNeuron {
    public SensoryReceptor Receptor;
    public List<Synapse> Synapses { get; set; }

    public SensoryNeuron(int NeuronID, Organism organism) : base(NeuronID, NeuronType.SensoryNeuron, false) {
        Synapses = new List<Synapse>();
        Receptor = new SensoryReceptor((SensoryReceptorType) NeuronID, organism);
        actionPotentialLength = 0;
        restingPotential = 0;
        potential = 0;
        actionPotentialThreshold = 0;
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
        NeuronType.SensoryNeuron,
        false
    ) {
        Synapses = synapses;
        Receptor = receptor;
    }
    
    public override void fireActionPotential() {
        foreach (Synapse synapse in Synapses) {
            synapse.fireSignal();
        }
    }

    public override void sumPotentials() {
        Receptor.updateValue();
        potential += Receptor.getValue();
        foreach (Synapse synapse in Synapses) {
            synapse.SynapticStrength = potential;
        }
    }
}
