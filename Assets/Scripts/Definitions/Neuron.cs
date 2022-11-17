using System.Collections.Generic;
using UnityEngine;

namespace Definitions {
    public abstract class Neuron {
        public HashSet<NeurotransmitterType> Neurotransmitters;
        public NeuronType Type;
        public float actionPotentialThreshold;
        public float restingPotential;
        public float potential;
        public float PotentialDecayRate;
        public readonly int NeuronID;
        // Number of simulation steps for an action potential to fire once initiated
        public int actionPotentialLength;
        // Number of simulation steps since the last action potential was initiated,
        // Value is set to -1 when no actionPotential is in progress
        protected int actionPotentialTime;
    
        protected Neuron(
            int NeuronID,
            float actionPotentialThreshold,
            float restingPotential,
            int actionPotentialLength,
            HashSet<NeurotransmitterType> neurotransmitters,
            NeuronType type
        ) {
            this.NeuronID = NeuronID;
            this.actionPotentialThreshold = actionPotentialThreshold;
            this.restingPotential = restingPotential;
            this.actionPotentialLength = actionPotentialLength;
            potential = restingPotential;
            Type = type;
            Neurotransmitters = neurotransmitters;
        }
        
        /// <summary>
        /// Initializes ID, type, and generates a random set of matching neurotransmitters
        /// </summary>
        /// <param name="NeuronID"></param>
        /// <param name="type"></param>
        protected Neuron(int NeuronID, NeuronType type) : this() {
            this.NeuronID = NeuronID;
            Type = type;
            Neurotransmitters = NeurotransmitterReceptor.generateRandomNeurotransmitters();
        }

        protected Neuron() {
            actionPotentialThreshold = -55f;
            restingPotential = -70f;
            PotentialDecayRate = Random.Range(0.7f, 0.9f);
            potential = restingPotential;
            actionPotentialLength = Random.Range(0, 3);
            actionPotentialTime = -1;
        }
    
        /// <summary>
        /// Decays a neuron's current potential by multiplying it's distance towards restingPotential by PotentialDecayRate
        /// </summary>
        public void decayPotential() {
            potential = restingPotential + (potential - restingPotential) * PotentialDecayRate;
        }

        public abstract void fireActionPotential();

        public void initiateActionPotential() {
            if (potential > actionPotentialThreshold && actionPotentialTime < 0) actionPotentialTime = 0;
        }

        public void incrementActionPotential() {
            if (actionPotentialTime == actionPotentialLength) {
                fireActionPotential();
                actionPotentialTime = -1;
            }
            else if (actionPotentialTime >= 0) {
                actionPotentialTime++;
            }
        }

        public abstract void sumPotentials();

        public bool thresholdReached() => potential > actionPotentialThreshold;
    }
}