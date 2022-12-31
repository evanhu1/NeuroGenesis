using System;
using System.Collections.Generic;
using Definitions;
using Random = UnityEngine.Random;

public class Genome {
    List<Gene> genes;
    List<SensoryNeuron> sensoryNeurons;
    List<InterNeuron> interNeurons;
    List<ActionNeuron> actionNeurons;
    
    /// [preSynapticType, preSynapticID, postSynapticType, postSynapticID, fireChance, synapticStrength]
    /// Need intermediate Tuple representation for synapse to enable easy mutation.
    List<Tuple<int, int, int, int, float, float>> synapses;
    
    /// Hashes new Neurons by a tuple of (Type, NeuronID)
    Dictionary<Tuple<int, int>, Neuron> neuronMap;
    
    // Creates a Genome (blueprint) object for "brain"
    public Genome(Brain brain) {
        synapses = new List<Tuple<int, int, int, int, float, float>>();
        genes = new List<Gene>();
        
        foreach (SensoryNeuron sensoryNeuron in brain.SensoryNeurons) {
            genes.Add(new Gene(sensoryNeuron));
            foreach (Synapse s in sensoryNeuron.Synapses) {
                synapses.Add(Tuple.Create(
                    (int)((Neuron)s.PreSynapticNeuron).Type,
                    ((Neuron)s.PreSynapticNeuron).NeuronID, 
                    (int)s.PostSynapticNeuron.Type,
                    s.PostSynapticNeuron.NeuronID, 
                    s.FireProbability, 
                    s.SynapticStrength));
            }
        }

        foreach (InterNeuron neuron in brain.InterNeurons) {
            genes.Add(new Gene(neuron));
            foreach (Synapse s in neuron.Synapses) {
                synapses.Add(Tuple.Create(
                    (int)((Neuron)s.PreSynapticNeuron).Type,
                    ((Neuron)s.PreSynapticNeuron).NeuronID, 
                    (int)s.PostSynapticNeuron.Type,
                    s.PostSynapticNeuron.NeuronID, 
                    s.FireProbability, 
                    s.SynapticStrength));
            }
        }
        
        foreach (ActionNeuron actionNeuron in brain.ActionNeurons) {
            genes.Add(new Gene(actionNeuron));
        }
    }

    // Creates a new Brain object according to the genome blueprint, and passes "newOrganism" to the new brain.
    public Brain constructBrain(Organism newOrganism, float mutationChance, float mutationMagnitude) {
        sensoryNeurons = new List<SensoryNeuron>();
        interNeurons = new List<InterNeuron>();
        actionNeurons = new List<ActionNeuron>();
        neuronMap = new Dictionary<Tuple<int, int>, Neuron>();
        
        foreach (Gene gene in genes) {
            gene.mutate(mutationChance, mutationMagnitude);
            
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
                    interNeurons.Add(newInter);
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
        
        if (Random.value < mutationChance) mutateNeurons(Random.Range(1, (int)mutationMagnitude)); 
        if (Random.value < mutationChance) mutateSynapses(Random.Range(1, (int)mutationMagnitude)); 
        
        // Iterate through all synapses and use hashmap to connect the new neurons together accordingly
        foreach ((int preSynapticType, int preSynapticID, int postSynapticType, int postSynapticID, float fireChance,
                     float synapticStrength) in synapses) {
            // postSynapticID needs to be modulated (modulo'd) because mutations generate a random int value for ID. 
            int postSynapticNeuronID = (NeuronType)postSynapticType == NeuronType.InterNeuron
                ? postSynapticID % interNeurons.Count
                : postSynapticID % actionNeurons.Count;
        
            if (!neuronMap.ContainsKey(Tuple.Create(preSynapticType, preSynapticID)) ||
                !neuronMap.ContainsKey(Tuple.Create(postSynapticType, postSynapticNeuronID))) continue;
            
            IOutputNeuron preSynapticNeuron = (IOutputNeuron) neuronMap[Tuple.Create(preSynapticType, preSynapticID)];
            Neuron postSynapticNeuron = neuronMap[Tuple.Create(postSynapticType, postSynapticNeuronID)];
            
            preSynapticNeuron.createSynapse(postSynapticNeuron, fireChance, synapticStrength);
        }

        return new Brain(newOrganism, synapses.Count, sensoryNeurons, interNeurons, actionNeurons);
    }
    void mutateNeurons(int mutationMagnitude) {
        // 0 = insertion, 1 = deletion, 2 = substitution 
        for (int i = 0; i < mutationMagnitude; i++) {
            int mutationType = Random.Range(0, 3);
            switch (mutationType) {
                case 0:
                    interNeurons.Add(
                        new InterNeuron(interNeurons[^1].NeuronID));
                    break;
                case 1:
                    if (interNeurons.Count == 0) break;
                    interNeurons.RemoveAt(Random.Range(0, interNeurons.Count));
                    break;
                case 2:
                    if (interNeurons.Count == 0) break;
                    interNeurons.Add(
                        new InterNeuron(interNeurons[^1].NeuronID));
                    break;
            }
        }
    }
    
    void mutateSynapses(int mutationMagnitude) {
        // 0 = insertion, 1 = deletion, 2 = substitution 
        for (int i = 0; i < mutationMagnitude; i++) {
            int mutationType = Random.Range(0, 3);
            switch (mutationType) {
                case 0:
                    synapses.Add(
                        Tuple.Create(
                            Random.Range(0, 2), 
                            Random.Range(0, int.MaxValue), 
                            Random.Range(1, 3),
                            Random.Range(0, int.MaxValue), 
                            Random.Range(0.5f, 1.0f), 
                            Random.Range(-10.0f, 10.0f)));
                    break;
                case 1:
                    if (synapses.Count == 0) break;
                    synapses.RemoveAt(Random.Range(0, synapses.Count));
                    break;
                case 2:
                    if (synapses.Count == 0) break;
                    synapses[Random.Range(0, synapses.Count)] =
                        Tuple.Create(
                            Random.Range(0, 2), 
                            Random.Range(0, int.MaxValue), 
                            Random.Range(1, 3),
                            Random.Range(0, int.MaxValue), 
                            Random.Range(0.5f, 1.0f), 
                            Random.Range(-10.0f, 10.0f));
                    break;
            }
        }
    }
}
