using System;
using System.Collections.Generic;
using Definitions;
using Random = UnityEngine.Random;

public class SensoryReceptor {
    Organism organism;
    public SensoryReceptorType Type;
    const int VisionDepth = 0;
    float value;
    
    // All sensory values are normalized to be between [0, normalizedSensoryBound]
    float normalizedSensoryBound = 8f;

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

    public float lookDirection(SensoryReceptorType type) {
        if (VisionDepth == 0) return 0f;
        return type switch {
            SensoryReceptorType.LookLeft =>
                look(new int[]
                    { organism.x - VisionDepth, organism.x, organism.y - VisionDepth, organism.y + VisionDepth + 1 })
                / ((VisionDepth * 2f + 1) * VisionDepth) * normalizedSensoryBound,
            SensoryReceptorType.LookRight =>
                look(new int[] {
                    organism.x + 1, organism.x + VisionDepth + 1, organism.y - VisionDepth, organism.y + VisionDepth + 1
                })
                / ((VisionDepth * 2f + 1) * VisionDepth) * normalizedSensoryBound,
            SensoryReceptorType.LookUp =>
                look(new int[] {
                    organism.x - VisionDepth, organism.x + VisionDepth + 1, organism.y + 1, organism.y + VisionDepth + 1
                })
                / ((VisionDepth * 2f + 1) * VisionDepth) * normalizedSensoryBound,
            SensoryReceptorType.LookDown =>
                look(new int[]
                    { organism.x - VisionDepth, organism.x + VisionDepth + 1, organism.y - VisionDepth, organism.y })                
                / ((VisionDepth * 2f + 1) * VisionDepth) * normalizedSensoryBound,
            _ => 0
        };
    }

    public void updateValue() {
        value = Type switch {
            SensoryReceptorType.LookDown => lookDirection(Type),
            SensoryReceptorType.LookLeft => lookDirection(Type),
            SensoryReceptorType.LookRight => lookDirection(Type),
            SensoryReceptorType.LookUp => lookDirection(Type),
            SensoryReceptorType.X => 1f * organism.x / Grid.Instance.columns * normalizedSensoryBound,
            SensoryReceptorType.Y => 1f * organism.y / Grid.Instance.rows * normalizedSensoryBound,
            _ => value
        };
    }

    public float getValue() {
        return value;
    }
}
