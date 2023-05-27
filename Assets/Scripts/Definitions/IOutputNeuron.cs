using System.Collections.Generic;
using UnityEngine;

namespace Definitions {
    /// <summary>
    /// IOutputNeuron have Synapses and can output signals
    /// </summary>
    public interface IOutputNeuron {
        Dictionary<Neuron, float> Synapses { get; set; }
        const float synapticStrengthMax = 8f;
        
        public void createSynapse(Neuron postSynapticNeuron, float synapticStrength) {
            // Forbid autapses/duplicate Synapses
            if (postSynapticNeuron == this || postSynapticNeuron.parentNeurons.Contains(this)) return;
            
            postSynapticNeuron.parentNeurons.Add(this);
            Synapses[postSynapticNeuron] = synapticStrength < 0 ? Random.Range(-synapticStrengthMax, synapticStrengthMax) : synapticStrength;
        }
    }
}
