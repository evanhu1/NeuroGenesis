using System.Collections.Generic;
using Definitions;
using UnityEngine;

public class ActionNeuron : Neuron {
    public ActionType ActionType;
    public bool IsActive;

    public ActionNeuron(
        int NeuronID,
        float actionPotentialThreshold,
        float restingPotential,
        int actionPotentialLength,
        float potentialDecayRate,
        ActionType actionType
    ) : base(
        NeuronID,
        actionPotentialThreshold,
        restingPotential,
        actionPotentialLength,
        potentialDecayRate,
        NeuronType.ActionNeuron
    ) {
        ActionType = actionType;
    }

    public ActionNeuron(int NeuronID, ActionType actionType) : base(NeuronID, NeuronType.ActionNeuron) {
        ActionType = actionType;
    }

    public override void fireActionPotential() {
        IsActive = potential > actionPotentialThreshold;
    }
}
