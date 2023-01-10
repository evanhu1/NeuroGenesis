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
            SensoryNeuron newNeuron = new SensoryNeuron(
                sensoryNeuron.NeuronID,
                sensoryNeuron.actionPotentialThreshold,
                sensoryNeuron.restingPotential,
                sensoryNeuron.actionPotentialLength,
                sensoryNeuron.PotentialDecayRate,
                new List<Synapse>(),
                new SensoryReceptor(sensoryNeuron.Receptor.Type, newOrganism));
            neuronMap[sensoryNeuron] = newNeuron;
            newSensoryNeurons.Add(newNeuron);
        }

        foreach (InterNeuron neuron in InterNeurons) {
            InterNeuron newNeuron = new InterNeuron(
                neuron.NeuronID,
                neuron.actionPotentialThreshold,
                neuron.restingPotential,
                neuron.actionPotentialLength,
                neuron.PotentialDecayRate,
                new List<Synapse>());
            neuronMap[neuron] = newNeuron;
            newInterNeurons.Add(newNeuron);
        }
        
        foreach (ActionNeuron actionNeuron in ActionNeurons) {
            ActionNeuron newNeuron = new ActionNeuron(
                actionNeuron.NeuronID,
                actionNeuron.actionPotentialThreshold,
                actionNeuron.restingPotential,
                actionNeuron.actionPotentialLength,
                actionNeuron.PotentialDecayRate,
                actionNeuron.ActionType);
            neuronMap[actionNeuron] = newNeuron;
            newActionNeurons.Add(newNeuron);
        }
        
        // Create all synapses for new brain
        foreach (SensoryNeuron sensoryNeuron in SensoryNeurons) {
            foreach (Synapse synapse in sensoryNeuron.Synapses) {
                ((IOutputNeuron)neuronMap[(Neuron)synapse.PreSynapticNeuron]).createSynapse(neuronMap[synapse.PostSynapticNeuron], synapse.FireProbability, synapse.SynapticStrength);
            }
        }
        
        foreach (InterNeuron interNeuron in InterNeurons) {
            foreach (Synapse synapse in interNeuron.Synapses) {
                ((IOutputNeuron)neuronMap[(Neuron)synapse.PreSynapticNeuron]).createSynapse(neuronMap[synapse.PostSynapticNeuron], synapse.FireProbability, synapse.SynapticStrength);
            }
        }

        Brain newBrain = new Brain(newOrganism, numSynapses, newSensoryNeurons, newInterNeurons, newActionNeurons);
        
        mutateNeurons((int) mutationMagnitude, newBrain);
        mutateSynapses((int) mutationMagnitude, newBrain);

        return newBrain;
    }
    
    void mutateNeurons(int mutationMagnitude, Brain brain) {
        List<InterNeuron> interNeurons = brain.InterNeurons;

        // 0 = insertion, 1 = deletion, 2 = substitution 
        for (int i = 0; i < mutationMagnitude; i++) {
            int mutationType = Random.Range(0, 3);
            switch (mutationType) {
                case 0:
                    interNeurons.Add(
                        new InterNeuron(interNeurons[^1].NeuronID + 1));
                    break;
                case 1:
                    if (interNeurons.Count == 0) break;
                    interNeurons.RemoveAt(Random.Range(0, interNeurons.Count));
                    break;
                case 2:
                    if (interNeurons.Count == 0) break;
                    int randomIndex = Random.Range(0, interNeurons.Count);
                    interNeurons[randomIndex] =
                        new InterNeuron(interNeurons[randomIndex].NeuronID);
                    break;
            }
        }
    }
    
    void mutateSynapses(int mutationMagnitude, Brain brain) {
        // 0 = insertion, 1 = deletion
        for (int i = 0; i < mutationMagnitude; i++) {
            // 0 = Sensory Neuron, 1 = Interneuron
            int neuronTypeToMutate = Random.Range(0, 2);
            IOutputNeuron neuronToMutate = (neuronTypeToMutate == 0) 
                ? brain.SensoryNeurons[Random.Range(0, brain.SensoryNeurons.Count)]
                : brain.InterNeurons[Random.Range(0, brain.InterNeurons.Count)];
            int mutationType = Random.Range(0, 2);
            switch (mutationType) {
                case 0:
                    brain.createSynapse();
                    break;
                case 1:
                    if (neuronToMutate.Synapses.Count == 0) break;
                    neuronToMutate.Synapses.RemoveAt(Random.Range(0, neuronToMutate.Synapses.Count));
                    break;
            }
        }
    }
}
