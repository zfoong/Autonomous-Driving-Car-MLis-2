using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

public class ViewStreamManager : MonoBehaviour
{
    public Camera cam;
    public CarControllerScript car;

    public bool recordFlag = false;
    public bool selfDriveFlag = false;

    public float timer = 0.0f;
    public float timeGap = 0.5f;
    public GameObject recIcon;
    private string dataFolderName = "Image Data";
    private string liveStreamFolderName = "Live Stream";

    // Start is called before the first frame update
    void Start()
    {
        cam = GetComponent<Camera>();
        string fullPath = Path.GetFullPath(Application.dataPath);
        string fullRecordPath = Path.Combine(fullPath, dataFolderName);
        string fullLiveStreamPath = Path.Combine(fullPath, liveStreamFolderName);
        try
        {
            if (!Directory.Exists(fullRecordPath))
            {
                Directory.CreateDirectory(fullRecordPath);
            }

            if (!Directory.Exists(fullLiveStreamPath))
            {
                Directory.CreateDirectory(fullLiveStreamPath);
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

        if (timer >= timeGap)
        {
            if (recordFlag)
            {
                byte[] data = CamCapture();
                int steerAngle = car.steeringAngle + 90;
                long msecNow = DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond;
                string imageName = msecNow + "_" + steerAngle + "_" + (int)car.currentSpeed + ".png";
                SaveImage(data, dataFolderName, imageName);
            }
            if (selfDriveFlag)
            {
                byte[] data = CamCapture();
                string imageName = "LiveSteamOutput.png";
                SaveImage(data, liveStreamFolderName, imageName);
            }
        }

        if (selfDriveFlag || recordFlag)
        {
            if (timer >= timeGap)
                timer = timer % timeGap;
            timer += Time.deltaTime;
        }
    }

    byte[] CamCapture()
    {
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

        return bytes;
    }

    private void SaveImage(byte[] bytes, string folderPathName, string imageName)
    {
        string path = Path.Combine(Application.dataPath, folderPathName);
        path = Path.Combine(path, imageName);
        File.WriteAllBytes(path, bytes);
    }

    public void setTimeGap(float gap)
    {
        timeGap = gap;
    }

    public void setTimeGap(Slider slider)
    {
        setTimeGap(slider.value);
    }

    public void ToggleSelfDriveFlag()
    {
        selfDriveFlag = !selfDriveFlag;
    }
}
