using System;
using System.Globalization;
using Definitions;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace UI {
    public class NeuronSprite : MonoBehaviour {
        public SpriteRenderer spriteRenderer;
        public Neuron neuron;
        public TextMeshProUGUI text;
        TextMeshProUGUI label;
        TextMeshProUGUI id;
        Color color;
        public Canvas canvas;
        
        void Start() {
            color = neuron.Type == NeuronType.SensoryNeuron 
                ? Color.blue
                : neuron.Type == NeuronType.InterNeuron
                ? Color.cyan
                : Color.green;
            label = Instantiate(text, transform.position + new Vector3(0, spriteRenderer.size.y),
                Quaternion.identity);
            id = Instantiate(text, transform.position + new Vector3(0, -spriteRenderer.size.y),
                Quaternion.identity);
            label.transform.SetParent(canvas.transform);
            label.enabled = false;
            id.transform.SetParent(canvas.transform);
            id.enabled = false;
            id.text = neuron.Type switch {
                NeuronType.SensoryNeuron => ((SensoryNeuron) neuron).Receptor.Type.ToString(),
                NeuronType.ActionNeuron => ((ActionNeuron) neuron).ActionType.ToString(),
                NeuronType.InterNeuron => neuron.NeuronID.ToString(),
                _ => null
            };
        }
        void Update() {
            spriteRenderer.color = neuron.thresholdReached() ? Color.white : neuron.isInvertedNeuron ? Color.gray : color;
            if (label.enabled) label.text = neuron.potential.ToString(CultureInfo.CurrentCulture);
        }

        void OnMouseEnter() {
            label.enabled = true;
            id.enabled = true;
        }

        void OnMouseExit() {
            label.enabled = false;
            id.enabled = false;

        }

        void OnDestroy() {
            if (label != null) Destroy(label.gameObject);
            if (id != null) Destroy(id.gameObject);
        }
    }
}