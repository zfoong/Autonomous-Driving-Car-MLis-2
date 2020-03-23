using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class ViewStreamManager : MonoBehaviour
{
    public Camera cam;
    public CarControllerScript car;
    public bool recordFlag = false;
    public float timer = 0.0f;
    public float timeGap = 1f;
    public GameObject recIcon;
    private string dataFolderName = "Image Data"; 

    // Start is called before the first frame update
    void Start()
    {
        cam = GetComponent<Camera>();
        string fullPath = Path.GetFullPath(Application.dataPath);
        fullPath = Path.Combine(fullPath, dataFolderName);
        try
        {
            if (!Directory.Exists(fullPath))
            {
                Directory.CreateDirectory(fullPath);
            }

        }
        catch (IOException ex)
        {
            Console.WriteLine(ex.Message);
        }
    }

    private void LateUpdate()
    {
        if (Input.GetKeyDown(KeyCode.R))
        {
            recordFlag = !recordFlag;
            if (recordFlag)
            {
                Debug.Log("Start recording");
                recIcon.SetActive(true);
            }else {
                Debug.Log("Stop recording");
                recIcon.SetActive(false);
            }
        }

        if (recordFlag)
        {
            timer += Time.deltaTime;
            if (timer >= timeGap)
            {
                CamCapture(car.steeringAngle, (int)car.currentSpeed);
                timer = timer % timeGap;
            }
        }
    }

    void CamCapture(int steerAngle, int speed)
    {
        long msecNow = DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond;

        cam = GetComponent<Camera>();

        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = cam.targetTexture;

        cam.Render();

        Texture2D img = new Texture2D(cam.targetTexture.width, cam.targetTexture.height);
        img.ReadPixels(new Rect(0, 0, cam.targetTexture.width, cam.targetTexture.height), 0, 0);
        img.Apply();
        RenderTexture.active = currentRT;

        var bytes = img.EncodeToPNG();
        Destroy(img);

        steerAngle += 90;
        
        string imageName = msecNow + "_" + steerAngle + "_" + speed + ".png";
        Debug.Log(Application.dataPath + "/Image Data/" + imageName);
        File.WriteAllBytes(Application.dataPath + "/Image Data/" + imageName, bytes);
    }
}
