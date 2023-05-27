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
    
    float invertedNeuronRate = 0.5f;

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
        preSynapticNeuron.createSynapse(postSynapticNeuron, -1);
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
            InterNeurons.Add(new InterNeuron(i, Random.value < invertedNeuronRate));
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
    // 1. Decay/propagation is incremented
    // 2. All potential summations are computed
    // 3. Action potentials are initiated
    // 4. Actions are taken if action neurons experience action potential
    public bool[] simulateStep() {
        bool[] actionOutcomes = new bool[numActions];
        IEnumerable<Neuron> allNeurons = InterNeurons.Concat<Neuron>(SensoryNeurons).Concat(ActionNeurons);
        
        // Decay neuron potentials
        foreach (Neuron neuron in allNeurons) {
            neuron.decayPotential();
        }

        // Sum potentials for all neurons
        foreach (Neuron neuron in allNeurons) {
            neuron.sumPotentials();
        }
        
        // Initiate action potentials and fire synapses
        foreach (Neuron neuron in allNeurons) {
            neuron.actionPotential();
        }

        // Detect action signals
        foreach (ActionNeuron actionNeuron in ActionNeurons) {
            if (actionNeuron.IsActive) {
                actionOutcomes[(int)actionNeuron.ActionType] = true;
            }
        }

        return actionOutcomes;
    }
    
    /// <summary>
    /// Deep copies this Brain instance and mutates the Neurons and Synapses. Returns the new Brain object.
    /// </summary>
    public Brain copyAndMutate(Organism newOrganism, float mutationChance, float mutationMagnitude) {
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
                new Dictionary<Neuron, float>(),
                new SensoryReceptor(sensoryNeuron.Receptor.Type, newOrganism));
            neuronMap[sensoryNeuron] = newNeuron;
            newSensoryNeurons.Add(newNeuron);
        }

        foreach (InterNeuron neuron in InterNeurons) {
            if (neuron.parentNeurons.Count == 0) continue;
            InterNeuron newNeuron = new InterNeuron(
                neuron.NeuronID,
                neuron.actionPotentialThreshold,
                neuron.restingPotential,
                neuron.actionPotentialLength,
                neuron.PotentialDecayRate,
                new Dictionary<Neuron, float>());
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

        Brain newBrain = new Brain(newOrganism, numSynapses, newSensoryNeurons, newInterNeurons, newActionNeurons);
        
        // mutateNeurons((int) mutationMagnitude, newBrain);
        // mutateSynapses((int) mutationMagnitude, newBrain);
        
        // Copies all synapses in new brain
        foreach (SensoryNeuron sensoryNeuron in SensoryNeurons) {
            foreach ((Neuron postSynapticNeuron, float synapticStrength) in sensoryNeuron.Synapses) {
                if (neuronMap.ContainsKey(postSynapticNeuron))
                    ((IOutputNeuron)neuronMap[sensoryNeuron]).createSynapse(neuronMap[postSynapticNeuron], synapticStrength);
            }
        }
        
        foreach (InterNeuron interNeuron in InterNeurons) {
            if (interNeuron.parentNeurons.Count == 0) continue;
            foreach ((Neuron postSynapticNeuron, float synapticStrength) in interNeuron.Synapses) {
                if (neuronMap.ContainsKey(postSynapticNeuron))
                    ((IOutputNeuron)neuronMap[interNeuron]).createSynapse(neuronMap[postSynapticNeuron], synapticStrength);
            }
        }

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
                        new InterNeuron(interNeurons.Count == 0 ? 0 : interNeurons[^1].NeuronID + 1, Random.value < invertedNeuronRate));
                    break;
                case 1:
                    if (interNeurons.Count == 0 || interNeurons.Count == Grid.Instance.world.maxNumNeurons) break;
                    int randIndex = Random.Range(0, interNeurons.Count);
                    foreach (IOutputNeuron parent in interNeurons[randIndex].parentNeurons) {
                        parent.Synapses.Remove(interNeurons[randIndex]);
                    }
                    interNeurons.RemoveAt(randIndex);
                    // Shifting subsequent neuronID's to account for deletion
                    // for (int j = randIndex; j < interNeurons.Count; j++) {
                    //     interNeurons[j].NeuronID--;
                    // }
                    break;
                case 2:
                    if (interNeurons.Count == 0) break;
                    int randomIndex = Random.Range(0, interNeurons.Count);

                    // Substitute random InterNeuron with new InterNeuron while preserving existing Synapses
                    InterNeuron newInter = new InterNeuron(interNeurons[randomIndex].NeuronID, Random.value < invertedNeuronRate)
                        {
                            Synapses = interNeurons[randomIndex].Synapses
                        };

                    interNeurons[randomIndex] = newInter;
                    break;
            }
        }
    }
    
    void mutateSynapses(int mutationMagnitude, Brain brain) {
        // 0 = insertion, 1 = deletion
        if (brain.InterNeurons.Count == 0) return;
        
        for (int i = 0; i < mutationMagnitude; i++) {
            // 0 = Sensory Neuron, 1 = Interneuron
            int neuronTypeToMutate = Random.Range(0, 2);
            IOutputNeuron neuronToMutate = (neuronTypeToMutate == 0)
                ? brain.SensoryNeurons[Random.Range(0, brain.SensoryNeurons.Count)]
                : brain.InterNeurons[Random.Range(0, brain.InterNeurons.Count)];
            switch (neuronTypeToMutate) {
                case 0:
                    brain.createSynapse();
                    break;
                case 1:
                    if (neuronToMutate.Synapses.Count == 0) break;
                    neuronToMutate.Synapses.Remove(neuronToMutate.Synapses.Keys.ElementAt(Random.Range(0, neuronToMutate.Synapses.Keys.Count)));
                    break;
            }
        }
    }

    public void ResetState() {
        foreach (InterNeuron InterNeuron in InterNeurons) {
            InterNeuron.potential = InterNeuron.restingPotential;
            InterNeuron.incomingCurrent = 0f;
        }
        foreach (ActionNeuron actionNeuron in ActionNeurons) {
            actionNeuron.potential = actionNeuron.restingPotential;
            actionNeuron.incomingCurrent = 0f;
        }
    }
}
