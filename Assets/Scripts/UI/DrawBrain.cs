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
            
            // Used to map NeuronID to index of Neuron in its own list, since NeuronIDs are not equivalent to index.
            Dictionary<int, int> idMap = new Dictionary<int, int>();
            
            drawNeurons(brain.SensoryNeurons, brain, idMap);
            drawNeurons(brain.InterNeurons, brain, idMap);
            drawNeurons(brain.ActionNeurons, brain, idMap);
            drawSynapses(brain, idMap);
        }
        
        void drawNeurons(IEnumerable<Neuron> neurons, Brain brain, Dictionary<int, int> idMap) {
            int count = 0;
            foreach (Neuron n in neurons) {
                NeuronSprite s = Instantiate(neuron, getNeuronPosition(count, n.Type, brain, idMap), Quaternion.identity);
                s.neuron = n;
                s.canvas = canvas;
                clones.Add(s.gameObject);
                idMap[n.NeuronID] = count;
                count++;
            }
        }
        
        Vector3 getNeuronPosition(int NeuronID, NeuronType type, Brain brain, Dictionary<int, int> idMap) {
            switch (type) {
                case NeuronType.SensoryNeuron:
                    return transform.position + new Vector3(0, -NeuronID * neuronHeight);
                case NeuronType.InterNeuron:
                    return transform.position + new Vector3((1 + NeuronID / neuronsPerColumn) * (neuronWidth),
                        -(NeuronID % neuronsPerColumn) * neuronHeight);
                default: {
                    int numInterNeuronColumns = (idMap[brain.InterNeurons[^1].NeuronID] + neuronsPerColumn - 1) / neuronsPerColumn;
                    return transform.position + new Vector3((1 + numInterNeuronColumns) * neuronWidth,
                        -NeuronID * neuronHeight);
                }
            }
        }
        
        void drawSynapse(IOutputNeuron outputNeuron, Brain brain, Dictionary<int, int> idMap) {
            foreach (Neuron postSynapticNeuron in outputNeuron.Synapses.Keys) {
                Neuron preSynapticNeuron = (Neuron)outputNeuron;
                float synapticStrength = outputNeuron.Synapses[postSynapticNeuron];
                LineRenderer drawer = Instantiate(lineRenderer, transform.position, Quaternion.identity);
                
                if (synapticStrength < 0) drawer.endColor = Color.black;
                drawer.widthMultiplier = Math.Abs(synapticStrength) / IOutputNeuron.synapticStrengthMax * 2;
                drawer.positionCount = 2;
                drawer.SetPosition(0, getNeuronPosition(
                    idMap[preSynapticNeuron.NeuronID],
                    preSynapticNeuron.Type, 
                    brain,
                    idMap));
                drawer.SetPosition(1, getNeuronPosition(
                    idMap[postSynapticNeuron.NeuronID],
                    postSynapticNeuron.Type,
                    brain,
                    idMap));
                clones.Add(drawer.gameObject);
            }
        }
        
        void drawSynapses(Brain brain, Dictionary<int, int> idMap) {
            foreach (SensoryNeuron inputNeuron in brain.SensoryNeurons) {
                drawSynapse(inputNeuron, brain, idMap);
            }
            foreach (InterNeuron inputNeuron in brain.InterNeurons) {
                drawSynapse(inputNeuron, brain, idMap);
            }
        }
    }
}
