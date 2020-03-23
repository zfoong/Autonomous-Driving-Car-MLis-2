using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DragObject : MonoBehaviour
{

    private Vector3 offset;
    private Vector3 screenPoint;
    private float offsetY = 8;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {

    }

    void OnMouseDown()
    {
        screenPoint = Camera.main.WorldToScreenPoint(Input.mousePosition);
        offset = transform.position - Camera.main.ScreenToWorldPoint(new Vector3(Input.mousePosition.x, Input.mousePosition.y, transform.position.z));
    }

    void OnMouseDrag()
    {
        Vector3 curScreenPoint = new Vector3(Input.mousePosition.x, Input.mousePosition.y, transform.position.z);
        Vector3 curPosition = Camera.main.ScreenToWorldPoint(curScreenPoint) + offset;
        Vector3 newPos = new Vector3(curPosition.x, offsetY, curPosition.z);
        transform.position = Vector3.Lerp(curPosition, newPos, 0.001f);
        this.GetComponent<Rigidbody>().isKinematic = true;
    }

    void OnMouseUp()
    {
        this.GetComponent<Rigidbody>().isKinematic = false;
    }
}
