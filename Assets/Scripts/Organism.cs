using System;
using System.Collections.Generic;
using Definitions;
using UI;
using UnityEngine;

public class Organism : MonoBehaviour {
    public int x, y;
    Brain brain;
    public Genome Genome;

    public void Init(int newX, int newY, int numNeurons, int numSynapses) {
        x = newX;
        y = newY;
        brain = new Brain(this, numNeurons, numSynapses);
        Genome = new Genome(brain, this);
    }
    
    public void InitWithBrain(int newX, int newY, Brain b) {
        x = newX;
        y = newY;
        brain = b;
        Genome = new Genome(brain, this);
    }

    public void simulateStep() {
        bool[] actionOutcomes = brain.simulateStep();
        for (int i = 0; i < (int)ActionType.NumActions; i++) {
            if (actionOutcomes[i]) {
                switch ((ActionType)i) {
                    case ActionType.MoveUp:
                        move(x, y + 1);
                        break;
                    case ActionType.MoveDown:
                        move(x, y - 1);
                        break;
                    case ActionType.MoveLeft:
                        move(x - 1, y);
                        break;
                    case ActionType.MoveRight:
                        move(x + 1, y);
                        break;
                }
            }
        }
    }

    void move(int newX, int newY) {
        if (Grid.Instance.checkOutOfBounds(newX, newY)) return;
        Grid.Instance.moveOrganism(this, newX, newY);
        x = newX;
        y = newY;
        transform.position = Grid.Instance.getPosition(x, y);
    }

    void OnMouseUp() {
        // brain.printSensors();
        DrawBrain.Instance.drawBrain(brain);
        Grid.Instance.world.changeFocus(this);
    }
}
