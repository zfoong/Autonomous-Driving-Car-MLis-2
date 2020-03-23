using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TrackManager : MonoBehaviour
{

    public List<Texture> trackImageList;
    private int currentIndex = 0;
    private MeshRenderer meshRenderer;

    // Start is called before the first frame update
    void Start()
    {
        meshRenderer = this.gameObject.GetComponent<MeshRenderer>();
    }

    public void NextTrack()
    {
        currentIndex = currentIndex >= trackImageList.Count-1 ? 0 : currentIndex+1;
        meshRenderer.material.mainTexture = trackImageList[currentIndex];
    }
}
