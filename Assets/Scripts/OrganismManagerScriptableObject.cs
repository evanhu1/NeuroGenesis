using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(menuName = "ScriptableObjects/OrganismManagerScriptableObject")]
public class OrganismManagerScriptableObject : ScriptableObject {
    public Organism organism;
    public List<Organism> organismList;
    public Dictionary<Tuple<int, int>, HashSet<Organism>> OrganismDict;

    public void addOrganism(Organism newOrganism, int x, int y) {
        OrganismDict[Tuple.Create(x, y)] = new HashSet<Organism> { newOrganism };
        organismList.Add(newOrganism);
    }
    
    public void moveOrganism(Organism org, int x, int y) {
        OrganismDict[Tuple.Create(org.x, org.y)].Remove(org);
        if (!OrganismDict.ContainsKey(Tuple.Create(x, y))) OrganismDict[Tuple.Create(x, y)] = new HashSet<Organism>();
        OrganismDict[Tuple.Create(x, y)].Add(org);
    }

    public void deleteOrganism(Organism org) {
        organismList.Remove(org);
        OrganismDict[Tuple.Create(org.x, org.y)].Remove(org);
    }
}