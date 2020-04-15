using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class UIController : MonoBehaviour
{
    public Text timeIntervalText;
    public Text cameraAngleText;
    public Camera[] cameraArr;
    public Text currentSpeedText;
    public Text currentAngleText;

    public Slider cameraAngleSlider;
    public Slider TimeIntervalSlider;

    void Start()
    {
        setTimeIntervalText(TimeIntervalSlider);
        setCameraAngle(cameraAngleSlider);
    }

    public void setTimeIntervalText(Slider slider)
    {
        timeIntervalText.text = "Stream Interval : " + slider.value.ToString("0.0") +"s";
    }

    public void setCameraAngle(Slider slider)
    {
        foreach (Camera c in cameraArr)
        {
            c.transform.localEulerAngles = new Vector3(slider.value, 0, 0);
        }
        cameraAngleText.text = "Camera Angle : " + slider.value.ToString("0.0");
    }

    public void setCurrentSpeedText(int speed)
    {
        currentSpeedText.text = "Current Speed : " + speed.ToString();
    }

    public void setCurrentAngleText(int angle)
    {
        currentAngleText.text = "Current Steering Angle : " + angle.ToString();
    }
}
