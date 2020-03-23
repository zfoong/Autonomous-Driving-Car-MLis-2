using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarControllerScript : MonoBehaviour {

    public int steeringAngle = 0;
    public float REAL_WORLD_FACTOR = 0.001f;

    public const int MAX_ANGLE = 40;
    public const int MIN_ANGLE = -40;
    public const int ANGLE_GAP = 5;
    public const int ANGLE = 5;

    public enum Speed
    {
        Back = -35,
        Stop = 0,
        Slow = 35,
        Fast = 75
    }

    public Speed currentSpeed;
    private List<Speed> speedList;

    public Rigidbody rigidbody;

    public WheelCollider frontDriverW, frontPassengerW;
    public WheelCollider rearDriverW, rearPassengerW;
    public Transform frontDriverT, frontPassengerT;
    public Transform rearDriverT, rearPassengerT;

    // Use this for initialization
    void Start () {
        currentSpeed = Speed.Stop;
        speedList = new List<Speed> { Speed.Back, Speed.Stop, Speed.Slow, Speed.Fast };
        rigidbody = GetComponent<Rigidbody>();
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

    // Update is called once per frame
    void Update () {
        // Get input
        if (Input.GetKeyDown(KeyCode.W))
        {
            currentSpeed = nextSpeed(currentSpeed, speedList);
            Debug.Log("Current Speed is " + currentSpeed);
        }
        else if (Input.GetKeyDown(KeyCode.S))
        {
            currentSpeed = prevSpeed(currentSpeed, speedList);
            Debug.Log("Current Speed is " + currentSpeed);
        }

        if (Input.GetKeyDown(KeyCode.A))
        {
            steeringAngle = Mathf.Max(MIN_ANGLE, steeringAngle -= ANGLE_GAP);
        }
        else if (Input.GetKeyDown(KeyCode.D))
        {
            steeringAngle = Mathf.Min(MAX_ANGLE, steeringAngle += ANGLE_GAP);
        }

        //Steer
        frontPassengerW.steerAngle = (float)steeringAngle;
        frontDriverW.steerAngle = (float)steeringAngle;

        // Movement
        Vector3 forwardForce = transform.forward * (float)currentSpeed * REAL_WORLD_FACTOR;
        Vector3 gravityForce = new Vector3(0, rigidbody.velocity.y, 0);
        rigidbody.velocity = forwardForce + gravityForce;
        //rigidbody.AddForce(transform.forward * (float)currentSpeed * REAL_WORLD_FACTOR, ForceMode.VelocityChange);

        UpdateWheelPoses();
    }

    private Speed prevSpeed(Speed _currentSpeed, List<Speed> _speedList)
    {
        int index = _speedList.IndexOf(_currentSpeed);
        if(index > 0) { return _speedList[index - 1];}
        else{return _currentSpeed;}
    }

    private Speed nextSpeed(Speed _currentSpeed, List<Speed> _speedList)
    {
        int index = _speedList.IndexOf(_currentSpeed);
        if (index < _speedList.Count - 1) { return _speedList[index + 1]; }
        else { return _currentSpeed; }
    }
}
