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
    Coroutine simulationLoop;
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
                createOrganism(x, y, null);
                break;
            }
        }
    }

    public void changeFocus(Organism organism) {
        if (focusedOrganism != null) focusedOrganism.GetComponent<SpriteRenderer>().color = Color.green;
        focusedOrganism = organism;
        focusedOrganism.GetComponent<SpriteRenderer>().color = Color.blue;
    }

    void createOrganism(int x, int y, Brain brain) {
        Vector3 pos = grid.getPosition(x, y);
        Organism newOrganism = Instantiate(manager.organism, pos, Quaternion.identity);
        if (brain == null) newOrganism.Init(x, y, numNeurons, numSynapses);
        else newOrganism.InitWithBrain(x, y, brain);
        manager.addOrganism(newOrganism, x, y);
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

    bool survivalCheck(Organism organism) {
        return true;
        // return organism.x < 10 && organism.y < 10;
    }

    IEnumerator simulateEpoch() {
        for (int i = 0; i < simulationStepsPerEpoch; i++) {
            executeOrganismActions();
            yield return new WaitForSeconds(1.0f / simulationStepsPerSecond);
        }

        foreach (Organism organism in manager.organismList.ToList()) {
            if (survivalCheck(organism)) {
                createOrganism(organism.x + 1, organism.y, organism.Genome.constructBrain(mutationChance, mutationMagnitude));
            }
            else {
                killOrganism(organism);
            }
        }
    }

    // Update is called once per frame
    void Update() {
        if (Input.GetKeyUp("space")) {
            simulationLoop = StartCoroutine(simulateEpoch());
        }
        if (Input.GetKeyUp("n")) {
            executeOrganismActions();
        }
    }
}