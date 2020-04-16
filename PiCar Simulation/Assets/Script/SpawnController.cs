using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpawnController : MonoBehaviour
{
    private Vector3 offset;
    private Vector3 screenPoint;
    private float offsetY = 0.2f;
    private Transform currentTranform;

    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            if (Camera.main == null)
                return;

            RaycastHit hit;
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);

            if (Physics.Raycast(ray, out hit))
            {
                if(hit.transform.tag == "Grabable")
                {
                    currentTranform = hit.transform;
                    screenPoint = Camera.main.WorldToScreenPoint(Input.mousePosition);
                    offset = currentTranform.position - Camera.main.ScreenToWorldPoint(new Vector3(Input.mousePosition.x, Input.mousePosition.y, currentTranform.position.z));
                }
            }
        }

        if (Input.GetMouseButton(0))
        {
            if (currentTranform != null)
            {
                Vector3 curScreenPoint = new Vector3(Input.mousePosition.x, Input.mousePosition.y, currentTranform.position.z);
                Vector3 curPosition = Camera.main.ScreenToWorldPoint(curScreenPoint) + offset;
                Vector3 oldPos = new Vector3(curPosition.x, currentTranform.position.y, curPosition.z);
                Vector3 newPos = new Vector3(curPosition.x, offsetY, curPosition.z);
                currentTranform.position = Vector3.Lerp(oldPos, newPos, 0.1f);
                currentTranform.GetComponent<Rigidbody>().isKinematic = true;
            }
        }

        if (Input.GetMouseButtonUp(0))
        {
            if (currentTranform != null)
            {
                currentTranform.GetComponent<Rigidbody>().isKinematic = false;
                currentTranform = null;
            }
        }

        if (Input.GetKey(KeyCode.Q))
        {
            if(currentTranform != null)
            {
                currentTranform.RotateAround(currentTranform.position, Vector3.down, 2);
            }
        }
    }

    public void Spawn(GameObject obj)
    {
        Instantiate(obj, this.transform.position, obj.transform.rotation);
    }
}
