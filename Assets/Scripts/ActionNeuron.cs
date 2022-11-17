using System.Collections.Generic;
using Definitions;
using UnityEngine;

public class ActionNeuron : Neuron, IOutputNeuron {
    public ActionType ActionType;
    public bool IsActive;
    public List<NeurotransmitterReceptor> Receptors { get; set; }

    public ActionNeuron(
        int NeuronID,
        float actionPotentialThreshold,
        float restingPotential,
        int actionPotentialLength,
        HashSet<NeurotransmitterType> neurotransmitters,
        List<NeurotransmitterReceptor> receptors,
        ActionType actionType
    ) : base(
        NeuronID,
        actionPotentialThreshold,
        restingPotential,
        actionPotentialLength,
        neurotransmitters,
        NeuronType.ActionNeuron
    ) {
        ActionType = actionType;
        Receptors = receptors;
    }

    public ActionNeuron(int NeuronID, ActionType actionType) : base(NeuronID, NeuronType.ActionNeuron) {
        ActionType = actionType;
        Receptors = new List<NeurotransmitterReceptor>();

        int numReceptors = Random.Range(1, 4);
        for (int i = 0; i < numReceptors; i++) {
            Receptors.Add(new NeurotransmitterReceptor());
        }
    }

    public override void sumPotentials() {
        foreach (NeurotransmitterReceptor receptor in Receptors) {
            potential += receptor.getValueAndReset();
        }
    }

    public override void fireActionPotential() {
        IsActive = potential > actionPotentialThreshold;
    }
}
