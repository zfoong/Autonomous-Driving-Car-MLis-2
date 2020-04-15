using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using System.Security.Permissions;

public class AutoDrivingController : MonoBehaviour
{
    string predictionDirPath = "PyScript";
    string predictionFileName = "prediction.json";
    public int speed = 0;
    public int angle = 90;

    // Start is called before the first frame update
    void Start()
    {
        string fullPredictionDirPath = Path.Combine(Application.dataPath, predictionDirPath);
        try
        {
            if (!Directory.Exists(fullPredictionDirPath))
            {
                Directory.CreateDirectory(fullPredictionDirPath);
            }
        }
        catch (IOException ex)
        {
            Console.WriteLine(ex.Message);
        }

        if (Directory.Exists(fullPredictionDirPath))
        {
            FileSystemWatcher watcher = new FileSystemWatcher();
            watcher.Path = fullPredictionDirPath;
            watcher.Filter = predictionFileName;

            watcher.Changed += PredictionFileUpdated;
            watcher.EnableRaisingEvents = true;
        }
        else
        {
            Debug.LogError("Path not exists: " + fullPredictionDirPath);
        }
    }

    private void PredictionFileUpdated(object sender, FileSystemEventArgs args)
    {
        string jsonString = File.ReadAllText(args.FullPath);
        Prediction pred = Prediction.CreateFromJSON(jsonString);
        speed = pred.speed;
        angle = pred.angle;
        Debug.Log(string.Format("Reading from input...speed is {0} and angle is {1}", speed, angle));
    }
}

[System.Serializable]
public class Prediction
{
    public int speed = 0;
    public int angle = 90;

    public static Prediction CreateFromJSON(string jsonString)
    {
        return JsonUtility.FromJson<Prediction>(jsonString);
    }
}
