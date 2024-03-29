using System;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using Button = UnityEngine.UI.Button;

namespace UI {
    public class GameUIManager : MonoBehaviour {
        public Button animateSingle, skipSingle, skipTen, skipX, scatterOrganisms, benchmarkEpoch, killUnfit;
        public TMP_InputField customEpochs;
        public World world;
        public TextMeshProUGUI epochLabel, survivingCount;
        public static GameUIManager Instance { get; private set; }
        
        // Start is called before the first frame update
        void Start() {
            Instance = this;
            animateSingle.onClick.AddListener(() => StartCoroutine(world.simulateEpochs(1, true)));
            skipSingle.onClick.AddListener(() => StartCoroutine(world.simulateEpochs(1, false)));
            skipTen.onClick.AddListener(() => StartCoroutine(world.simulateEpochs(10, false)));
            skipX.onClick.AddListener(() => StartCoroutine(world.simulateEpochs(Int32.Parse(customEpochs.text), false)));
            scatterOrganisms.onClick.AddListener(() => world.scatterOrganisms());
            benchmarkEpoch.onClick.AddListener(() => print(world.benchmarkEpoch()));
            killUnfit.onClick.AddListener(() => world.processSurvivingOrganisms());
        }

        // Update is called once per frame
        void Update()
        {
            
        }
    }
}
