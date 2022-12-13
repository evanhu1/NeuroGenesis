using UnityEngine;
using Button = UnityEngine.UI.Button;

namespace UI {
    public class GameUIManager : MonoBehaviour {
        public Button animateSingle, skipSingle, skipTen, scatterOrganisms, benchmarkEpoch;
        public World world;
        
        // Start is called before the first frame update
        void Start()
        {
            animateSingle.onClick.AddListener(() => StartCoroutine(world.simulateEpochs(1, true)));
            skipSingle.onClick.AddListener(() => StartCoroutine(world.simulateEpochs(1, false)));
            skipTen.onClick.AddListener(() => StartCoroutine(world.simulateEpochs(10, false)));
            scatterOrganisms.onClick.AddListener(() => world.scatterOrganisms());
            benchmarkEpoch.onClick.AddListener(() => print(world.benchmarkEpoch()));
        }

        // Update is called once per frame
        void Update()
        {
            
        }
    }
}
