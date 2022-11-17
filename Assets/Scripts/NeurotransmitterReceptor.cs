using System.Collections.Generic;
using Definitions;
using UnityEngine;
using Random = UnityEngine.Random;

public class NeurotransmitterReceptor {
    public HashSet<NeurotransmitterType> MatchingNeurotransmitters;
    bool isActive;
    // How much receptor changes the potential of the neuron when activated
    public float PotentialPolarization;

    public NeurotransmitterReceptor(float potentialPolarization, HashSet<NeurotransmitterType> matchingNeurotransmitters) {
        isActive = false;
        MatchingNeurotransmitters = matchingNeurotransmitters;
        PotentialPolarization = potentialPolarization;
    }

    public NeurotransmitterReceptor() {
        isActive = false;
        PotentialPolarization = Random.Range(-10f, 10f);
        MatchingNeurotransmitters = generateRandomNeurotransmitters();
    }

    public static HashSet<NeurotransmitterType> generateRandomNeurotransmitters() {
        HashSet<NeurotransmitterType> set = new HashSet<NeurotransmitterType>();
        for (int i = 0; i < (int)NeurotransmitterType.NumNeurotransmitters; i++) {
            if (Random.value < 0.5) {
                set.Add((NeurotransmitterType)i);
            }
        }

        return set;
    }
    
    public float getValueAndReset() {
        if (!isActive) return 0f;
        isActive = false;
        return PotentialPolarization;
    }

    public void trigger() {
        isActive = true;
    }
}