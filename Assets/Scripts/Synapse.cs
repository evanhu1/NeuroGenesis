using Definitions;
using Random = UnityEngine.Random;

public class Synapse {
    /// <summary>
    /// The additive impact (Amperes) this synapse has on the post-synaptic neuron.
    /// </summary>
    public float SynapticStrength;
    
    public IOutputNeuron PreSynapticNeuron;
    public Neuron PostSynapticNeuron;
    
    /// <summary>
    /// Randomly initializes RELEASE_PROBABILITY and SYNAPTIC_STRENGTH if negative values are passed in for either.
    /// </summary>
    public Synapse(IOutputNeuron preSynapticNeuron, Neuron postSynapticNeuron, float synapticStrength) {
        PreSynapticNeuron = preSynapticNeuron;
        PostSynapticNeuron = postSynapticNeuron;
        SynapticStrength = synapticStrength < 0 ? Random.Range(-10f, 10f) : synapticStrength;
    }

    public void fireSignal() {
        PostSynapticNeuron.incomingCurrent += SynapticStrength;
    }
}
