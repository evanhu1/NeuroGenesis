using System.Collections.Generic;
using System.Linq;
using Definitions;

public class SensoryNeuron : Neuron, IOutputNeuron {
    public SensoryReceptor Receptor;
    public Dictionary<Neuron, float> Synapses { get; set; }

    public SensoryNeuron(int NeuronID, Organism organism) : base(NeuronID, NeuronType.SensoryNeuron, false) {
        Synapses = new Dictionary<Neuron, float>();
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
        Dictionary<Neuron, float> synapses,
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
        foreach (Neuron postSynapticNeuron in Synapses.Keys) {
            postSynapticNeuron.incomingCurrent += potential;
        }
        
        potential = restingPotential;
    }

    public override void sumPotentials() {
        Receptor.updateValue();
        potential = Receptor.getValue();
    }
}
