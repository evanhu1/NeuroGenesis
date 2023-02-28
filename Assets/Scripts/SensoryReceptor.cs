using System;
using System.Collections.Generic;
using Definitions;
using Random = UnityEngine.Random;

public class SensoryReceptor {
    Organism organism;
    public SensoryReceptorType Type;
    const int VisionDepth = 3;
    float value;

    public SensoryReceptor(SensoryReceptorType type, Organism organism) {
        this.organism = organism;
        Type = type;
    }
    
    // Argument bounds format: [left, right, bottom, top]
    int look(IReadOnlyList<int> bounds) {
        int signalStrength = 0;
        for (int y = bounds[2]; y < bounds[3]; y++) {
            for (int x = bounds[0]; x < bounds[1]; x++) {
                if (Grid.Instance.checkOutOfBounds(x, y) || Grid.Instance.checkOrganismExistsAt(x, y)) {
                    signalStrength++;
                }
            }
        }

        return signalStrength;
    }

    public void updateValue() {
        value = Type switch {
            SensoryReceptorType.LookLeft => look(new int[] {organism.x - VisionDepth, organism.x, organism.y - VisionDepth, organism.y + VisionDepth}),
            SensoryReceptorType.LookRight => look(new int[] {organism.x + 1, organism.x + VisionDepth, organism.y - VisionDepth, organism.y + VisionDepth}),
            SensoryReceptorType.LookUp => look(new int[] {organism.x - VisionDepth, organism.x + VisionDepth, organism.y + 1, organism.y + VisionDepth}),
            SensoryReceptorType.LookDown => look(new int[] {organism.x - VisionDepth, organism.x + VisionDepth, organism.y - VisionDepth, organism.y}),
            SensoryReceptorType.X => organism.x,
            SensoryReceptorType.Y => organism.y,
            _ => value
        };
    }

    public float getValue() {
        return value;
    }
}
