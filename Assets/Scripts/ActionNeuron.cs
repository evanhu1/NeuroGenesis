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
        ActionType actionType,
        bool isInverted = false
    ) : base(
        NeuronID,
        actionPotentialThreshold,
        restingPotential,
        actionPotentialLength,
        potentialDecayRate,
        NeuronType.ActionNeuron,
        isInverted
    ) {
        ActionType = actionType;
    }

    public ActionNeuron(int NeuronID, ActionType actionType, bool isInverted = false) : base(NeuronID, NeuronType.ActionNeuron, isInverted) {
        ActionType = actionType;
    }

    public override void fireActionPotential() {
        IsActive = potential > actionPotentialThreshold;
        potential = restingPotential;
    }
}
