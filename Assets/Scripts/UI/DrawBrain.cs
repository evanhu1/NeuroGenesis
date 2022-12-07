using System;
using System.Collections.Generic;
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
            neuronWidth = neuron.spriteRenderer.size.x + spacingOffset;
            neuronHeight = neuron.spriteRenderer.size.y + spacingOffset;
            Instance = this;
            transform.position = Grid.Instance.getCenter() + new Vector2(Grid.Instance.getWidth() / 2 + neuronWidth, neuronHeight * neuronsPerColumn / 2f);
        }

        public void drawBrain(Brain brain) {
            foreach (GameObject g in clones) Destroy(g);
            drawSensoryNeurons(brain);
            drawInterNeurons(brain);
            drawActionNeurons(brain);
            drawSynapses(brain);
        }
        
        void drawSensoryNeurons(Brain brain) {
            for (int i = 0; i < brain.SensoryNeurons.Count; i++) {
                NeuronSprite n = Instantiate(neuron, transform.position + new Vector3(0, -i *
                    (neuronHeight)), Quaternion.identity);
                n.neuron = brain.SensoryNeurons[i];
                n.canvas = canvas;
                clones.Add(n.gameObject);
            }
        }
        
        void drawInterNeurons(Brain brain) {
            Vector3 startPos = transform.position + new Vector3(neuronWidth, 0);
            int numColumns = (brain.Neurons.Count + neuronsPerColumn - 1) / neuronsPerColumn;

            // Iterate once for each column
            for (int i = 0; i < numColumns; i++) {
                for (int j = 0; j < Math.Min(brain.Neurons.Count - i * neuronsPerColumn, neuronsPerColumn); j++) {
                    NeuronSprite n = Instantiate(neuron, startPos + new Vector3(i * (neuronWidth), -j *
                        (neuronHeight)), Quaternion.identity);
                    n.neuron = brain.Neurons[i * neuronsPerColumn + j];
                    n.canvas = canvas;
                    clones.Add(n.gameObject);
                }
            }
        }
        
        void drawActionNeurons(Brain brain) {
            int numInterNeuronColumns = (brain.Neurons.Count + neuronsPerColumn - 1) / neuronsPerColumn;
            // Offset SensoryNeuron column, and InterNeuron columns
            Vector3 startPos = transform.position + new Vector3((1 + numInterNeuronColumns) * neuronWidth, 0);

            for (int i = 0; i < brain.ActionNeurons.Count; i++) {
                NeuronSprite n = Instantiate(neuron, startPos + new Vector3(0, -i * neuronHeight), Quaternion.identity);
                n.neuron = brain.ActionNeurons[i];
                n.canvas = canvas;
                clones.Add(n.gameObject);
            }
        }

        Vector3 getNeuronPosition(int NeuronID, NeuronType type, Brain brain) {
            if (type == NeuronType.SensoryNeuron) {
                return transform.position + new Vector3(0, -NeuronID * neuronHeight);
            } 
            if (type == NeuronType.InterNeuron) {
                return transform.position + new Vector3((1 + NeuronID / neuronsPerColumn) * (neuronWidth), 
                    -(NeuronID % neuronsPerColumn) * neuronHeight);
            } else {
                int numInterNeuronColumns = (brain.Neurons.Count + neuronsPerColumn - 1) / neuronsPerColumn;
                return transform.position + new Vector3((1 + numInterNeuronColumns) * neuronWidth,
                    -NeuronID * neuronHeight);
            }
        }

        void drawSynapse(IOutputNeuron outputNeuron, Brain brain) {
            foreach (Synapse synapse in outputNeuron.Synapses) {
                Neuron preSynapticNeuron = (Neuron)synapse.PreSynapticNeuron;
                Neuron postSynapticNeuron = (Neuron)synapse.PostSynapticNeuron;
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
            foreach (InterNeuron inputNeuron in brain.Neurons) {
                drawSynapse(inputNeuron, brain);
            }
        }
    }
}
