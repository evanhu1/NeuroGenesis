using System;
using System.Collections.Generic;
using System.Linq;
using Definitions;
using UnityEngine;

namespace UI {
    public class DrawBrain : MonoBehaviour {
        public static DrawBrain Instance { get; private set; }
        public LineRenderer lineRenderer;
        public NeuronSprite neuron;
        int neuronsPerColumn = 5;
        float spacingOffset = 0.5f;
        float neuronWidth;
        float neuronHeight;
        public Canvas canvas;
        List<GameObject> clones;
        
        void Start() {
            clones = new List<GameObject>();
            Vector2 size = neuron.spriteRenderer.size;
            neuronWidth = size.x + spacingOffset;
            neuronHeight = size.y + spacingOffset;
            Instance = this;
            transform.position = Grid.Instance.getCenter() + new Vector2(Grid.Instance.getWidth() / 2 + neuronWidth, neuronHeight * neuronsPerColumn / 2f);
        }

        public void drawBrain(Brain brain) {
            foreach (GameObject g in clones) Destroy(g);
            drawNeurons(brain.SensoryNeurons, brain);
            drawNeurons(brain.InterNeurons, brain);
            drawNeurons(brain.ActionNeurons, brain);
            drawSynapses(brain);
        }
        
        void drawNeurons(IEnumerable<Neuron> neurons, Brain brain) {
            foreach (Neuron n in neurons) {
                NeuronSprite s = Instantiate(neuron, getNeuronPosition(n.NeuronID, n.Type, brain), Quaternion.identity);
                s.neuron = n;
                s.canvas = canvas;
                clones.Add(s.gameObject);
            }
        }

        Vector3 getNeuronPosition(int NeuronID, NeuronType type, Brain brain) {
            switch (type) {
                case NeuronType.SensoryNeuron:
                    return transform.position + new Vector3(0, -NeuronID * neuronHeight);
                case NeuronType.InterNeuron:
                    return transform.position + new Vector3((1 + NeuronID / neuronsPerColumn) * (neuronWidth),
                        -(NeuronID % neuronsPerColumn) * neuronHeight);
                default: {
                    int numInterNeuronColumns = (brain.InterNeurons[^1].NeuronID + neuronsPerColumn - 1) / neuronsPerColumn;
                    return transform.position + new Vector3((1 + numInterNeuronColumns) * neuronWidth,
                        -NeuronID * neuronHeight);
                }
            }
        }

        void drawSynapse(IOutputNeuron outputNeuron, Brain brain) {
            foreach (Neuron postSynapticNeuron in outputNeuron.Synapses.Keys) {
                Neuron preSynapticNeuron = (Neuron)outputNeuron;
                LineRenderer drawer = Instantiate(lineRenderer, transform.position, Quaternion.identity);
                drawer.positionCount = 2;
                drawer.SetPosition(0, getNeuronPosition(
                    preSynapticNeuron.NeuronID, preSynapticNeuron.Type, brain));
                drawer.SetPosition(1, getNeuronPosition(
                    postSynapticNeuron.NeuronID, postSynapticNeuron.Type, brain));
                clones.Add(drawer.gameObject);
            }
        }

        void drawSynapses(Brain brain) {
            foreach (SensoryNeuron inputNeuron in brain.SensoryNeurons) {
                drawSynapse(inputNeuron, brain);
            }
            foreach (InterNeuron inputNeuron in brain.InterNeurons) {
                drawSynapse(inputNeuron, brain);
            }
        }
    }
}
