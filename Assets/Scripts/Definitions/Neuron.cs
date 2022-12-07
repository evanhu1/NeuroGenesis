using System.Collections.Generic;
using UnityEngine;

namespace Definitions {
    public abstract class Neuron {
        public NeuronType Type;
        public float actionPotentialThreshold;
        public float restingPotential;
        public float potential;
        public float PotentialDecayRate;
        public readonly int NeuronID;
        
        /// <summary>
        /// Number of simulation steps for an action potential to fire once initiated
        /// </summary>
        public int actionPotentialLength;
        
        /// <summary>
        ///  Number of simulation steps since the last action potential was initiated.
        ///  Value is set to -1 when no actionPotential is in progress.
        /// </summary>
        protected int actionPotentialTime;
        
        /// <summary>
        ///  The total incoming current (Amperes) accumulated from the action potentials of other connected neurons
        ///  (or the sensory receptor for SensoryNeurons).
        /// </summary>
        public float incomingCurrent;
    
        protected Neuron(
            int NeuronID,
            float actionPotentialThreshold,
            float restingPotential,
            int actionPotentialLength,
            float potentialDecayRate,
            NeuronType type
        ) {
            this.NeuronID = NeuronID;
            this.actionPotentialThreshold = actionPotentialThreshold;
            this.restingPotential = restingPotential;
            this.actionPotentialLength = actionPotentialLength;
            PotentialDecayRate = potentialDecayRate;
            potential = restingPotential;
            Type = type;
        }
        
        /// <summary>
        /// Initializes ID and type from input, and randomly sets all other parameters.
        /// </summary>
        /// <param name="NeuronID"></param>
        /// <param name="type"></param>
        protected Neuron(int NeuronID, NeuronType type) : this() {
            this.NeuronID = NeuronID;
            Type = type;
        }

        protected Neuron() {
            actionPotentialThreshold = -55f;
            restingPotential = -70f;
            PotentialDecayRate = Random.Range(0.5f, 0.9f);
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

        public void sumPotentials() {
            potential += incomingCurrent;
            incomingCurrent = 0;
        }

        public bool thresholdReached() => potential > actionPotentialThreshold;
    }
}