using System;
using System.Collections.Generic;
using Definitions;
using UnityEngine;

public class Genome {
    List<Gene> genes;
    int numSynapses;
    
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
        // Hashes new Neurons by a tuple of (Type, NeuronID)
        Dictionary<Tuple<int, int>, Neuron> neuronMap = new Dictionary<Tuple<int, int>, Neuron>();
        
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
                        gene.neurotransmitters,
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
                        gene.neurotransmitters,
                        new List<Synapse>(),
                        new List<NeurotransmitterReceptor>()
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
                        gene.neurotransmitters,
                        new List<NeurotransmitterReceptor>(),
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
                foreach (Tuple<int, int, float> synapse in gene.synapses) {
                    numSynapses++;
                    int id = synapse.Item1 == 1 ? synapse.Item2 % neurons.Count : synapse.Item2 % actionNeurons.Count;
                    IInputNeuron inputNeuron = (IInputNeuron) neuronMap[Tuple.Create((int)gene.type, gene.neuronID)];
                    IOutputNeuron outputNeuron = (IOutputNeuron) neuronMap[Tuple.Create(synapse.Item1, id)];
                    inputNeuron.createSynapse(outputNeuron, synapse.Item3);
                }
            }
            if (gene.type is NeuronType.InterNeuron or NeuronType.ActionNeuron) {
                foreach (Tuple<HashSet<NeurotransmitterType>, float> receptor in gene.receptors) {
                    IOutputNeuron outputNeuron = (IOutputNeuron) neuronMap[Tuple.Create((int)gene.type, gene.neuronID)];
                    outputNeuron.Receptors.Add(new NeurotransmitterReceptor(receptor.Item2, receptor.Item1));
                }
            }
        }

        return new Brain(newOrganism, numSynapses, neurons, sensoryNeurons, actionNeurons);
    }
}