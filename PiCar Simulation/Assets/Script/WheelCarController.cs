using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WheelCarController : MonoBehaviour
{
    public enum Speed
    {
        Back = -35,
        Stop = 0,
        Slow = 35,
        Fast = 75
    }

    public WheelCollider frontDriverW, frontPassengerW;
    public WheelCollider rearDriverW, rearPassengerW;
    public Transform frontDriverT, frontPassengerT;
    public Transform rearDriverT, rearPassengerT;

    public float rpm = 0;

    public const int MAX_ANGLE = 40;
    public const int MIN_ANGLE = -40;
    public const int ANGLE_GAP = 5;
    private List<Speed> speedList;

    public Speed currentSpeed;
    public int steeringAngle = 0;

    void Start()
    {
        currentSpeed = Speed.Stop;
        speedList = new List<Speed> { Speed.Back, Speed.Stop, Speed.Slow, Speed.Fast };
    }

    private void UpdateWheelPoses()
    {
        UpdateWheelPose(frontDriverW, frontDriverT);
        UpdateWheelPose(frontPassengerW, frontPassengerT);
        UpdateWheelPose(rearDriverW, rearDriverT);
        UpdateWheelPose(rearPassengerW, rearPassengerT);
    }

    private void UpdateWheelPose(WheelCollider _collider, Transform _transform)
    {
        Vector3 _pos = _transform.position;
        Quaternion _quat = _transform.rotation;

       _collider.GetWorldPose(out _pos, out _quat);

        _transform.position = _pos;
        _transform.rotation = _quat;
    }

    private Speed prevSpeed(Speed _currentSpeed, List<Speed> _speedList)
    {
        int index = _speedList.IndexOf(_currentSpeed);
        if (index > 0) { return _speedList[index - 1]; }
        else { return _currentSpeed; }
    }

    private Speed nextSpeed(Speed _currentSpeed, List<Speed> _speedList)
    {
        int index = _speedList.IndexOf(_currentSpeed);
        if (index < _speedList.Count - 1) { return _speedList[index + 1]; }
        else { return _currentSpeed; }
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.A))
        {
            steeringAngle = Mathf.Max(MIN_ANGLE, steeringAngle -= ANGLE_GAP);
        }
        else if (Input.GetKeyDown(KeyCode.D))
        {
            steeringAngle = Mathf.Min(MAX_ANGLE, steeringAngle += ANGLE_GAP);
        }   
        frontPassengerW.steerAngle = (float)steeringAngle;
        frontDriverW.steerAngle = (float)steeringAngle;

        UpdateWheelPoses();
    }   
}
