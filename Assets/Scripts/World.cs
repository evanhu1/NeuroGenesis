using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UI;
using UnityEngine;
using UnityEngine.Assertions;
using Random = UnityEngine.Random;

public class World : MonoBehaviour {
    public OrganismManagerScriptableObject manager;
    public Grid grid;
    public int stepsPerEpoch = 20;
    public int stepsPerSecond = 5;
    public int numOrganisms = 5;
    public int numNeurons = 0;
    public int maxNumNeurons = 100;
    public int numSynapses = 10;
    public float mutationChance;
    public float mutationMagnitude;
    Organism focusedOrganism;
    public int CurrentEpoch;

    void Start() {
        manager.organismList = new List<Organism>();
        manager.OrganismDict = new Dictionary<Tuple<int, int>, HashSet<Organism>>();

        grid.Init();

        for (int i = 0; i < numOrganisms; i++) {
            int x = Random.Range(0, grid.columns);
            int y = Random.Range(0, grid.rows);
            createOrganism(i, x, y);
        }

        CurrentEpoch = 0;
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

    /// <summary>
    /// Instantiates new Organism object with a Brain inherited from organism PARENT; the brain contains mutations of its own.
    /// </summary>
    void spawnOffspring(Organism parent) {
        int x = parent.x;
        int y = parent.y;
        Vector3 pos = grid.getPosition(x, y);
        Organism child = Instantiate(manager.organism, pos, Quaternion.identity);
        // child.InitWithBrain(manager.organismList.Count, x, y,
        //     parent.Genome.constructBrain(child, mutationChance, mutationMagnitude));
        child.InitWithBrain(manager.organismList.Count, x, y,
            parent.brain.copyAndMutate(child, mutationChance, mutationMagnitude));
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

    static bool survivalCheck(Organism organism) {
        // return organism.y == Grid.Instance.rows - 1 && organism.x == Grid.Instance.columns - 1;
        // return organism.y > Grid.Instance.rows / 2;
        return ((Grid.Instance.columns / 4) < organism.x
                && organism.x < (Grid.Instance.columns * 3 / 4)
                && (Grid.Instance.rows / 4) < organism.y
                && organism.y < (Grid.Instance.rows * 3 / 4));
    }

    public int processSurvivingOrganisms() {
        foreach (Organism organism in manager.organismList.ToList()) {
            bool isSurviving = survivalCheck(organism);

            // Randomly preserve 5% of the unfit population: && Random.value < 0.95
            if (!isSurviving) {
                killOrganism(organism);
            }
        }
        int survivingCount = manager.organismList.Count;

        // Fill in 70% of missing population by cloning survivors, and the remaining 10% by creating new Organisms.
        // If no survivors then just creates a new generation of organisms.
        if (manager.organismList.Count > 0) {
            for (int i = 0; i < (int)(0.7f * (numOrganisms - survivingCount)); i++) spawnOffspring(manager.organismList[Random.Range(0, survivingCount)]);
            for (int i = 0; i < numOrganisms - manager.organismList.Count; i++) createOrganism(manager.organismList.Count, 0, 0);
        }
        else {
            for (int i = 0; i < numOrganisms; i++) {
                createOrganism(i, 0, 0);
            }
        }

        return survivingCount;
    }
    
    public IEnumerator simulateEpochs(int epochs, bool wait) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            scatterOrganisms();
            for (int i = 0; i < stepsPerEpoch; i++) {
                executeOrganismActions();
                if (wait) yield return new WaitForSeconds(1.0f / stepsPerSecond);
            }
            
            int survivingCount = processSurvivingOrganisms();
            CurrentEpoch++;
            GameUIManager.Instance.epochLabel.text = "Epoch: " + CurrentEpoch;
            GameUIManager.Instance.survivingCount.text = survivingCount + " Surviving Organisms";
            yield return new WaitForEndOfFrame();
        }
    }

    void simulateEpoch() {
        scatterOrganisms();
        for (int i = 0; i < stepsPerEpoch; i++) {
            executeOrganismActions();
        }
        
        int survivingCount = processSurvivingOrganisms();
        CurrentEpoch++;
        GameUIManager.Instance.epochLabel.text = "Epoch: " + CurrentEpoch;
        GameUIManager.Instance.survivingCount.text = survivingCount + " Surviving Organisms";
    }
    
    // Places all existing organisms randomly on the grid such that none meet survival criteria
    public void scatterOrganisms() {
        foreach (Organism org in manager.organismList) {
            do {
                int newX = Random.Range(0, Grid.Instance.columns);
                int newY = Random.Range(0, Grid.Instance.rows);
                org.brain.ResetState();
                org.move(newX, newY);
            } while (survivalCheck(org));
        }
    }

    /// Finds the average normalized time of 10 epochs
    public double benchmarkEpoch() {
        double average = 0;
        Stopwatch sw = new Stopwatch();
        for (int i = 0; i < 10; i++) {
            sw.Start();
            simulateEpoch();
            sw.Stop();
            average += sw.ElapsedMilliseconds * 1000.0 / (stepsPerEpoch * (numNeurons + numSynapses) * numOrganisms);
        }

        return average / 10.0;
    }
    
    void Update() {
        if (Input.GetKeyUp("n")) {
            executeOrganismActions();
        }
        if (Input.GetKeyUp("t")) {
            foreach (Organism org in manager.organismList.ToList()) {
                spawnOffspring(org);
            }
        }
    }
}