using System.Collections.Generic;

namespace Definitions {
    /// <summary>
    /// IOutputNeuron have Synapses and can output signals
    /// </summary>
    public interface IOutputNeuron {
        List<Synapse> Synapses { get; set; }
        
        public void createSynapse(Neuron postSynapticNeuron, float fireProbability, float synapticStrength) {
            // Forbid autapses
            if (postSynapticNeuron == this) return;
            foreach (Synapse synapse in Synapses) {
                if (synapse.PostSynapticNeuron == postSynapticNeuron) {
                    return;
                }
            }

            Synapses.Add(new Synapse(this, postSynapticNeuron, fireProbability, synapticStrength));
        }
    }
}