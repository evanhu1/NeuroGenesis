using System.Collections.Generic;

namespace Definitions {
    /// <summary>
    /// IInputNeurons have Synapses and can output signals
    /// </summary>
    public interface IInputNeuron {
        List<Synapse> Synapses { get; set; }
    
        public abstract void createSynapse(IOutputNeuron postSynapticNeuron, float fireChance);
    }
}