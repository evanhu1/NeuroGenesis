using System;
using UnityEngine;

public class Grid : MonoBehaviour {
    public static Grid Instance { get; private set; }
    public Cell cell;
    public int columns = 8;
    public int rows = 8;
    public new Camera camera;
    public World world;
    public void Awake() {
        Instance = this;
    }
    
    public void Init() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                createCell(i, j);
            }
        }

        camera.transform.position = (Vector3) getCenter() + new Vector3(0, 0, -10);
        camera.orthographicSize = rows / 2f;
    }

    public Vector2 getPosition(int x, int y) {
        return new Vector2(x * cell.width, y * cell.height);
    }

    public Vector2 getCenter() {
        return rows % 2 != 0 ? 
            getPosition((columns + 1) / 2, (rows + 1) / 2)
            : Vector3.Lerp(
                getPosition((columns + 1) / 2, (rows + 1) / 2), 
                getPosition(columns / 2 - 1, rows / 2 - 1), 0.5f);
    }

    public float getWidth() => columns * cell.width;
    public float getHeight() => rows * cell.height;
    
    void createCell(int x, int y) {
        Cell c = Instantiate(cell, getPosition(x, y), Quaternion.identity);
        c.x = x;
        c.y = y;
    }

    public bool checkOutOfBounds(int x, int y) {
        return x < 0 || y < 0 || x >= columns || y >= rows;
    }

    public bool checkOrganismExistsAt(int x, int y) {
        return world.manager.OrganismDict.ContainsKey(Tuple.Create(x, y));
    }

    public void moveOrganism(Organism organism, int x, int y) => world.manager.moveOrganism(organism, x, y);

    // Update is called once per frame
    void Update() {
    }
}