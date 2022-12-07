using System;
using System.Collections.Generic;
using Definitions;

public class Genome {
    List<Gene> genes;
    int numSynapses;
    
    /// Hashes new Neurons by a tuple of (Type, NeuronID)
    Dictionary<Tuple<int, int>, Neuron> neuronMap;
    
    // Creates a Genome (blueprint) object for "brain"
    public Genome(Brain brain) {
        genes = new List<Gene>();
        foreach (InterNeuron neuron in brain.Neurons) {
            genes.Add(new Gene(neuron));
        }
        foreach (SensoryNeuron sensoryNeuron in brain.SensoryNeurons) {
            genes.Add(new Gene(sensoryNeuron));
        }
        foreach (ActionNeuron actionNeuron in brain.ActionNeurons) {
            genes.Add(new Gene(actionNeuron));
        }
    }

    // Creates a new Brain object according to the genome blueprint, and passes "newOrganism" to the new brain.
    public Brain constructBrain(Organism newOrganism, float mutateChance, float mutationMagnitude) {
        List<SensoryNeuron> sensoryNeurons = new List<SensoryNeuron>();
        List<InterNeuron> neurons = new List<InterNeuron>();
        List<ActionNeuron> actionNeurons = new List<ActionNeuron>();
        neuronMap = new Dictionary<Tuple<int, int>, Neuron>();
        
        foreach (Gene gene in genes) {
            gene.mutate(mutateChance, mutationMagnitude);
            
            switch (gene.type) {
                case NeuronType.SensoryNeuron:
                    SensoryNeuron newSensory = new SensoryNeuron(
                        gene.neuronID,
                        gene.actionPotentialThreshold,
                        gene.restingPotential,
                        gene.actionPotentialLength,
                        gene.potentialDecayRate,
                        new List<Synapse>(),
                        new SensoryReceptor((SensoryReceptorType)gene.neuronID, newOrganism, gene.sensorySensitivity)
                    );
                    sensoryNeurons.Add(newSensory);
                    neuronMap[Tuple.Create((int) gene.type, gene.neuronID)] = newSensory;
                    break;
                case NeuronType.InterNeuron:
                    InterNeuron newInter = new InterNeuron(
                        gene.neuronID,
                        gene.actionPotentialThreshold,
                        gene.restingPotential,
                        gene.actionPotentialLength,
                        gene.potentialDecayRate,
                        new List<Synapse>()
                    );
                    neurons.Add(newInter);
                    neuronMap[Tuple.Create((int) gene.type, gene.neuronID)] = newInter;
                    break;
                case NeuronType.ActionNeuron:
                    ActionNeuron newAction = new ActionNeuron(
                        gene.neuronID,
                        gene.actionPotentialThreshold,
                        gene.restingPotential,
                        gene.actionPotentialLength,
                        gene.potentialDecayRate,
                        (ActionType)gene.neuronID
                    );
                    actionNeurons.Add(newAction);
                    neuronMap[Tuple.Create((int) gene.type, gene.neuronID)] = newAction;
                    break;
            }
        }
        // Iterate through all synapses and use hashmap to connect the new neurons together accordingly, same with receptors
        numSynapses = 0;
        foreach (Gene gene in genes) {
            if (gene.type is NeuronType.SensoryNeuron or NeuronType.InterNeuron) {
                foreach (Tuple<int, int, float, float> synapse in gene.synapses) {
                    numSynapses++;
                    int postSynapticType = synapse.Item1;
                    int postSynapticID = synapse.Item2;
                    float fireChance = synapse.Item3;
                    float synapticStrength = synapse.Item4;
                    
                    // postSynapticID needs to be modulated (modulo'd) because mutations generate a random int value for ID. 
                    int postSynapticNeuronID = (NeuronType) postSynapticType == NeuronType.InterNeuron
                        ? postSynapticID % neurons.Count
                        : postSynapticID % actionNeurons.Count;
                    
                    IOutputNeuron preSynapticNeuron = (IOutputNeuron) neuronMap[Tuple.Create((int)gene.type, gene.neuronID)];
                    Neuron postSynapticNeuron = neuronMap[Tuple.Create(postSynapticType, postSynapticNeuronID)];
                    
                    preSynapticNeuron.createSynapse(postSynapticNeuron, fireChance, synapticStrength);
                }
            }
        }

        return new Brain(newOrganism, numSynapses, neurons, sensoryNeurons, actionNeurons);
    }
}