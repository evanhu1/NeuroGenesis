using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Cell : MonoBehaviour {
    SpriteRenderer sprite;
    public int x;
    public int y;
    public float width = 1, height = 1;

    // Start is called before the first frame update
    void Start() {
        sprite = GetComponent<SpriteRenderer>();
        sprite.size = new Vector2(width, height);
        if ((transform.position.x + transform.position.y) % 2 == 0) {
            sprite.color = new Color(0.8f, 0.8f, 0.8f, 1);
        }
    }

    void OnMouseEnter() {
        sprite.color -= new Color(0.1f, 0.1f, 0.1f, 0f);
    }

    void OnMouseExit() {
        sprite.color += new Color(0.1f, 0.1f, 0.1f, 0f);
    }
}