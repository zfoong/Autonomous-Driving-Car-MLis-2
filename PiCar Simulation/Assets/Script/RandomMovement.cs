using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RandomMovement : MonoBehaviour
{

    public bool rotateFlag = false;
    public bool translateFlag = false;
    private int randomFactor = 1;
    private Vector3 startPosition;
    private int xTranslate = 0;
    private int yTranslate = 0;
    private int zTranslate = 0;

    // Start is called before the first frame update
    void Start()
    {
        randomFactor = Random.Range(5, 15);
        xTranslate = Random.Range(0, 2);
        yTranslate = Random.Range(0, 2);
        zTranslate = Random.Range(0, 2);

        startPosition = transform.position;
    }

    // Update is called once per frame
    void Update()
    {
        float sin = Mathf.Sin(Time.time);

        if (rotateFlag)
            transform.RotateAround(transform.position, Vector3.up, randomFactor * Time.deltaTime);
        if(translateFlag)
            transform.position = startPosition + new Vector3(sin * xTranslate, sin * yTranslate, sin * zTranslate);
    }
}
