using System;
using System.Collections.Generic;
using Definitions;
using UnityEngine.Assertions;
using Random = UnityEngine.Random;

public class Brain {
    int numActions = (int) ActionType.NumActions;
    int numSensors = (int) SensoryReceptorType.NumTypes;
    int numNeurons;
    public int numSynapses;
    public List<InterNeuron> InterNeurons;
    public List<SensoryNeuron> SensoryNeurons;
    public List<ActionNeuron> ActionNeurons;
    Organism organism;

    public Brain(Organism organism, int numNeurons, int numSynapses) {
        Assert.IsTrue(numSynapses <=
                      (numNeurons + numActions + numSensors) * (numNeurons + numActions + numSensors));
        
        this.numNeurons = numNeurons;
        this.numSynapses = numSynapses;
        this.organism = organism;
        InterNeurons = new List<InterNeuron>();
        SensoryNeurons = new List<SensoryNeuron>();
        ActionNeurons = new List<ActionNeuron>();
        generateSensoryNeurons();
        generateInterNeurons();
        generateActionNeurons();
        generateSynapses();
    }

    public Brain(Organism organism, int numSynapses, List<InterNeuron> interNeurons, List<SensoryNeuron> sensoryNeurons, List<ActionNeuron> actionNeurons) {
        this.organism = organism;
        this.numSynapses = numSynapses;
        numNeurons = interNeurons.Count;
        SensoryNeurons = sensoryNeurons;
        InterNeurons = interNeurons;
        ActionNeurons = actionNeurons;
    }

    /// <summary>
    /// Randomly creates UP TO numSynapses synapses. Chance of a synapse being made decreases
    /// as network grows more dense to encourage sparsity.
    /// </summary>
    void generateSynapses() {
        for (int i = 0; i < numSynapses; i++) {
            createSynapse();
        }
    }

    /// <summary>
    /// Randomly creates a synapse from {sensory neuron, inter neuron} to {inter neuron, action neuron}.
    /// If a duplicate synapse would be created, the operation is aborted.
    /// </summary>
    void createSynapse() {
        int preNeuronIndex = Random.Range(0, InterNeurons.Count + numSensors);
        IOutputNeuron preSynapticNeuron = preNeuronIndex < InterNeurons.Count
            ? InterNeurons[preNeuronIndex]
            : SensoryNeurons[preNeuronIndex - InterNeurons.Count];
        int postNeuronIndex = Random.Range(0, InterNeurons.Count + numActions);
        Neuron postSynapticNeuron = postNeuronIndex < InterNeurons.Count
            ? InterNeurons[postNeuronIndex]
            : ActionNeurons[postNeuronIndex - InterNeurons.Count];
        preSynapticNeuron.createSynapse(postSynapticNeuron, -1, -1);
    }

    void generateSensoryNeurons() {
        for (int i = 0; i < numSensors; i++) {
            SensoryNeurons.Add(
                new SensoryNeuron(i, organism)
            );
        }
    }

    void generateInterNeurons() {
        for (int i = 0; i < numNeurons; i++) {
            InterNeurons.Add(new InterNeuron(i));
        }
    }

    void generateActionNeurons() {
        for (int i = 0; i < numActions; i++) {
            ActionNeurons.Add(
                new ActionNeuron(i, (ActionType) i)
            );
        }
    }

    // Simulates one time step. Each time step the following occurs in order:
    // 1. Sensory receptors are evaluated
    // 2. All potential summations are computed
    // 3. Action potentials are initiated
    // 4. Axon neurotransmitters are released
    // 5. Actions are taken if action neurons experience action potential
    // 6. Decay/propagation is incremented
    public bool[] simulateStep() {
        bool[] actionOutcomes = new bool[numActions];

        // Update all sensory receptors
        foreach (SensoryNeuron sensoryNeuron in SensoryNeurons) {
            sensoryNeuron.updateReceptor();
        }
        
        // Sum potentials for all neurons
        foreach (InterNeuron neuron in InterNeurons) {
            neuron.sumPotentials();
        }
        foreach (SensoryNeuron sensoryNeuron in SensoryNeurons) {
            sensoryNeuron.sumPotentials();
        }
        foreach (ActionNeuron actionNeuron in ActionNeurons) {
            actionNeuron.sumPotentials();
        }
        
        // Initiate action potentials and fire synapses
        foreach (InterNeuron neuron in InterNeurons) {
            neuron.initiateActionPotential();
            neuron.incrementActionPotential();
        }
        foreach (SensoryNeuron sensoryNeuron in SensoryNeurons) {
            sensoryNeuron.initiateActionPotential();
            sensoryNeuron.incrementActionPotential();
        }
        foreach (ActionNeuron actionNeuron in ActionNeurons) {
            actionNeuron.initiateActionPotential();
            actionNeuron.incrementActionPotential();
        }
        
        // Detect action signals
        foreach (ActionNeuron actionNeuron in ActionNeurons) {
            if (actionNeuron.IsActive) {
                actionOutcomes[(int)actionNeuron.ActionType] = true;
            }
        }
        
        // Decay neuron potentials
        foreach (InterNeuron neuron in InterNeurons) {
            neuron.decayPotential();
        }
        foreach (SensoryNeuron sensoryNeuron in SensoryNeurons) {
            sensoryNeuron.decayPotential();
        }
        foreach (ActionNeuron actionNeuron in ActionNeurons) {
            actionNeuron.decayPotential();
        }

        return actionOutcomes;
    }
}
