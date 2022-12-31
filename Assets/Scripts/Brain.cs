using System;
using System.Collections.Generic;
using System.Linq;
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

    public Brain(Organism organism, int numSynapses, List<SensoryNeuron> sensoryNeurons, List<InterNeuron> interNeurons, List<ActionNeuron> actionNeurons) {
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

    /// <summary>
    /// Deep clones a Brain object using DFS traversal of synapses starting from NEURON for organism ORG. Uses NEURON_MAP to keep map neurons to cloned equivalents { neuron -> clone }.
    /// </summary>
    /// <param name="neuron"></param>
    /// <param name="neuronMap"></param>
    /// <param name="org"></param>
    Neuron replicateBrainDFS(Neuron neuron, Dictionary<Neuron, Neuron> neuronMap, Organism org) {
        if (neuronMap.ContainsKey(neuron)) {
            return neuronMap[neuron];
        }

        Neuron newNeuron = null;
        
        switch (neuron.Type) {
            case NeuronType.SensoryNeuron:
                newNeuron = new SensoryNeuron(
                    neuron.NeuronID,
                    neuron.actionPotentialThreshold,
                    neuron.restingPotential,
                    neuron.actionPotentialLength,
                    neuron.PotentialDecayRate,
                    new List<Synapse>(),
                    new SensoryReceptor(((SensoryNeuron)neuron).Receptor.Type, org)
                );
                break;
            case NeuronType.InterNeuron:
                newNeuron = new InterNeuron(
                    neuron.NeuronID,
                    neuron.actionPotentialThreshold,
                    neuron.restingPotential,
                    neuron.actionPotentialLength,
                    neuron.PotentialDecayRate,
                    new List<Synapse>()
                );
                break;
            case NeuronType.ActionNeuron:
                newNeuron = new ActionNeuron(
                    neuron.NeuronID,
                    neuron.actionPotentialThreshold,
                    neuron.restingPotential,
                    neuron.actionPotentialLength,
                    neuron.PotentialDecayRate,
                    ((ActionNeuron)neuron).ActionType
                );
                break;
        }
        neuronMap[neuron] = newNeuron;

        if (neuron is IOutputNeuron outputNeuron) {
            foreach (Synapse synapse in outputNeuron.Synapses) {
                replicateBrainDFS(synapse.PostSynapticNeuron, neuronMap, org);
                ((IOutputNeuron)neuronMap[neuron]).createSynapse(neuronMap[synapse.PostSynapticNeuron], synapse.FireProbability, synapse.SynapticStrength);
            }
        }

        return newNeuron;
    }
    
    public Brain replicateAndMutate(Organism newOrganism, float mutationChance, float mutationMagnitude) {
        List<SensoryNeuron> newSensoryNeurons = new List<SensoryNeuron>();
        List<InterNeuron> newInterNeurons = new List<InterNeuron>();
        List<ActionNeuron> newActionNeurons = new List<ActionNeuron>();
        Dictionary<Neuron, Neuron> neuronMap = new Dictionary<Neuron, Neuron>();
        
        foreach (SensoryNeuron sensoryNeuron in SensoryNeurons) {
            newSensoryNeurons.Add((SensoryNeuron) replicateBrainDFS(sensoryNeuron, neuronMap, newOrganism));
        }

        foreach (InterNeuron neuron in InterNeurons) {
            newInterNeurons.Add((InterNeuron) replicateBrainDFS(neuron, neuronMap, newOrganism));
        }
        
        foreach (ActionNeuron actionNeuron in ActionNeurons) {
            newActionNeurons.Add((ActionNeuron) replicateBrainDFS(actionNeuron, neuronMap, newOrganism));
        }
        
        Brain newBrain = new Brain(newOrganism, numSynapses, newSensoryNeurons, newInterNeurons, newActionNeurons);
        
        return newBrain;
    }
}
