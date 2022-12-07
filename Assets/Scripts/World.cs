using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;
using Random = UnityEngine.Random;

public class World : MonoBehaviour {
    public OrganismManagerScriptableObject manager;
    public Grid grid;
    public int simulationStepsPerEpoch = 20;
    public int simulationStepsPerSecond = 5;
    public int numOrganisms = 5;
    public int numNeurons = 0;
    public int numSynapses = 10;
    public float mutationChance;
    public float mutationMagnitude;
    Organism focusedOrganism;

    void Start() {
        Assert.IsTrue(numOrganisms < grid.columns * grid.rows);

        manager.organismList = new List<Organism>();
        manager.OrganismDict = new Dictionary<Tuple<int, int>, HashSet<Organism>>();

        grid.Init();

        for (int i = 0; i < numOrganisms; i++) {
            while (true) {
                int x = Random.Range(0, grid.columns);
                int y = Random.Range(0, grid.rows);
                if (manager.OrganismDict.ContainsKey(Tuple.Create(x, y))) continue;
                createOrganism(i, x, y);
                break;
            }
        }
    }

    public void changeFocus(Organism organism) {
        if (focusedOrganism != null) focusedOrganism.GetComponent<SpriteRenderer>().color = Color.green;
        focusedOrganism = organism;
        focusedOrganism.GetComponent<SpriteRenderer>().color = Color.blue;
    }

    void createOrganism(int id, int x, int y) {
        Vector3 pos = grid.getPosition(x, y);
        Organism newOrganism = Instantiate(manager.organism, pos, Quaternion.identity);
        newOrganism.Init(id, x, y, numNeurons, numSynapses);
        manager.addOrganism(newOrganism, x, y);
    }

    void spawnOffspring(Organism parent) {
        int x = parent.x;
        int y = parent.y;
        Vector3 pos = grid.getPosition(x, y);
        Organism child = Instantiate(manager.organism, pos, Quaternion.identity);
        child.InitWithBrain(manager.organismList.Count, x, y,
            parent.Genome.constructBrain(child, mutationChance, mutationMagnitude));
        manager.addOrganism(child, x, y);
    }

    void killOrganism(Organism organism) {
        manager.deleteOrganism(organism);
        Destroy(organism.gameObject);
    }

    void executeOrganismActions() {
        foreach (Organism organism in manager.organismList) {
            organism.simulateStep();
        }
    }

    public bool survivalCheck(Organism organism) {
        // return organism.y > Grid.Instance.rows / 2;
        // return true;
        return (Grid.Instance.columns / 4) < organism.x 
               && organism.x <= (Grid.Instance.columns * 3 / 4)
               && (Grid.Instance.rows / 4) < organism.y 
               && organism.y <= (Grid.Instance.rows * 3 / 4);
    }

    IEnumerator simulateEpochs(int epochs, bool wait) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            scatterOrganisms();
            for (int i = 0; i < simulationStepsPerEpoch; i++) {
                executeOrganismActions();
                if (wait) yield return new WaitForSeconds(1.0f / simulationStepsPerSecond);
            }

            foreach (Organism organism in manager.organismList.ToList()) {
                bool isSurviving = survivalCheck(organism);

                // Randomly preserve 2% of the unfit population / kill 5% of fit population
                if ((!isSurviving && Random.value < 0.98) || (isSurviving && Random.value < 0.05)) {
                    killOrganism(organism);
                }
            }

            // Fill in 80% of lacking population by cloning survivors, and the remaining 20% by creating new Organisms
            if (manager.organismList.Count > 0) {
                int originalCount = manager.organismList.Count;
                while (manager.organismList.Count < numOrganisms) {
                    if (Random.value < 0.8) spawnOffspring(manager.organismList[Random.Range(0, originalCount)]);
                    else createOrganism(manager.organismList.Count, 0, 0);
                }
            }
            else {
                for (int i = 0; i < numOrganisms; i++) {
                    createOrganism(i, 0, 0);
                }
            }
        }
    }
    
    // Places all existing organisms randomly on the grid
    void scatterOrganisms() {
        foreach (Organism org in manager.organismList) {
            int newX = Random.Range(0, Grid.Instance.columns);
            int newY = Random.Range(0, Grid.Instance.rows);
            org.move(newX, newY);
        }
    }
    
    // Update is called once per frame.
    void Update() {
        if (Input.GetKeyUp("space")) {
            StartCoroutine(simulateEpochs(1, true));
        }
        if (Input.GetKeyUp("s")) {
            StartCoroutine(simulateEpochs(1, false));
        }
        if (Input.GetKeyUp("f")) {
            StartCoroutine(simulateEpochs(10, false));
        }
        if (Input.GetKeyUp("n")) {
            executeOrganismActions();
        }
        if (Input.GetKeyUp("r")) {
            scatterOrganisms();
        }
    }
}