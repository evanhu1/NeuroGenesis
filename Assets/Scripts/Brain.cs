using System;
using System.Collections.Generic;
using Definitions;
using UnityEngine.Assertions;
using Random = UnityEngine.Random;

public class Brain {
    int numActions = (int) ActionType.NumActions;
    int numSensors = (int) SensoryReceptorType.NumTypes;
    int numNeurons;
    int numSynapses;
    public List<InterNeuron> Neurons;
    public List<SensoryNeuron> SensoryNeurons;
    public List<ActionNeuron> ActionNeurons;
    Organism organism;

    public Brain(Organism organism, int numNeurons, int numSynapses) {
        Assert.IsTrue(numSynapses <=
                      (numNeurons + numActions + numSensors) * (numNeurons + numActions + numSensors));
        
        this.numNeurons = numNeurons;
        this.numSynapses = numSynapses;
        this.organism = organism;
        Neurons = new List<InterNeuron>();
        SensoryNeurons = new List<SensoryNeuron>();
        ActionNeurons = new List<ActionNeuron>();
        generateSensoryNeurons();
        generateActionNeurons();
        generateNeurons();
        generateSynapses();
        // printBrain();
    }

    public Brain(Organism organism, int numSynapses, List<InterNeuron> neurons, List<SensoryNeuron> sensoryNeurons, List<ActionNeuron> actionNeurons) {
        this.organism = organism;
        this.numSynapses = numSynapses;
        numNeurons = neurons.Count;
        SensoryNeurons = sensoryNeurons;
        Neurons = neurons;
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
        int preNeuronIndex = Random.Range(0, Neurons.Count + numSensors);
        IOutputNeuron preSynapticNeuron = preNeuronIndex < Neurons.Count
            ? Neurons[preNeuronIndex]
            : SensoryNeurons[preNeuronIndex - Neurons.Count];
        int postNeuronIndex = Random.Range(0, Neurons.Count + numActions);
        Neuron postSynapticNeuron = postNeuronIndex < Neurons.Count
            ? Neurons[postNeuronIndex]
            : ActionNeurons[postNeuronIndex - Neurons.Count];
        preSynapticNeuron.createSynapse(postSynapticNeuron, -1, -1);
    }

    void generateNeurons() {
        for (int i = 0; i < numNeurons; i++) {
            Neurons.Add(new InterNeuron(i));
        }
    }

    void generateSensoryNeurons() {
        for (int i = 0; i < numSensors; i++) {
            SensoryNeurons.Add(
                new SensoryNeuron(i, organism)
            );
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
        foreach (InterNeuron neuron in Neurons) {
            neuron.sumPotentials();
        }
        foreach (SensoryNeuron sensoryNeuron in SensoryNeurons) {
            sensoryNeuron.sumPotentials();
        }
        foreach (ActionNeuron actionNeuron in ActionNeurons) {
            actionNeuron.sumPotentials();
        }
        
        // Initiate action potentials and fire synapses
        foreach (InterNeuron neuron in Neurons) {
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
        foreach (InterNeuron neuron in Neurons) {
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

    public void printSensors() {
        // foreach (SensoryNeuron sensoryNeuron in sensoryNeurons) {
        //     Debug.Log(sensoryNeuron.Receptor.Type);
        //     Debug.Log(sensoryNeuron.Receptor.getValue());
        // }
    }
}
