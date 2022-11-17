using System.Collections.Generic;

namespace Definitions {
    /// <summary>
    /// IOutputNeuron have Receptors and can receive signals
    /// </summary>
    public interface IOutputNeuron {
        List<NeurotransmitterReceptor> Receptors { get; set; }
    }
}