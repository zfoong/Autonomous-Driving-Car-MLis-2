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

    public bool selfDriving = false;
    public AutoDrivingController adController;

    public UIController uiController;

    public enum Speed
    {
        Back = -35,
        Stop = 0,
        Slow = 35,
        Fast = 50
    }

    public Speed currentSpeed;
    private List<Speed> speedList;

    public Rigidbody rigidbody;

    public WheelCollider frontDriverW, frontPassengerW;
    public WheelCollider rearDriverW, rearPassengerW;
    public Transform frontDriverT, frontPassengerT;
    public Transform rearDriverT, rearPassengerT;

    public List<GameObject> raycastObjectList;

    public float objDis = 0;
    public Vector3 objSize = new Vector3();
    

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

    private void GetInput()
    {
        if (Input.GetKeyDown(KeyCode.W))
            currentSpeed = nextSpeed(currentSpeed, speedList);
        else if (Input.GetKeyDown(KeyCode.S))
            currentSpeed = prevSpeed(currentSpeed, speedList);

        if (Input.GetKeyDown(KeyCode.A))
            steeringAngle = Mathf.Max(MIN_ANGLE, steeringAngle -= ANGLE_GAP);
        else if (Input.GetKeyDown(KeyCode.D))
            steeringAngle = Mathf.Min(MAX_ANGLE, steeringAngle += ANGLE_GAP);
    }

    private void ReadInput()
    {
        currentSpeed = (Speed)adController.speed;
        steeringAngle = adController.angle - 90;
    }

    // Update is called once per frame
    void Update () {

        if (selfDriving)
            ReadInput();
        else
            GetInput();

        //Steer
        frontPassengerW.steerAngle = steeringAngle;
        frontDriverW.steerAngle = steeringAngle;

        // Movement
        Vector3 forwardForce = transform.forward * (float)currentSpeed * REAL_WORLD_FACTOR;
        Vector3 gravityForce = new Vector3(0, rigidbody.velocity.y, 0);
        rigidbody.velocity = forwardForce + gravityForce;
        //rigidbody.AddForce(transform.forward * (float)currentSpeed * REAL_WORLD_FACTOR, ForceMode.VelocityChange);

        UpdateWheelPoses();
        updateUIController();

        CheckForHit();
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

    public void ToggleSelfDriving()
    {
        selfDriving = !selfDriving;
        currentSpeed = Speed.Stop;
        steeringAngle = 0;
    }

    public void updateUIController()
    {
        uiController.setCurrentAngleText(steeringAngle + 90);
        uiController.setCurrentSpeedText((int)currentSpeed);
    }

    void CheckForHit()
    {
        RaycastHit objectHit;
        Transform tf = null;
        float minDis = Mathf.Infinity;
        foreach (GameObject go in raycastObjectList)
        {
            Transform raycastTransform = go.transform;
            float distance = 1;
            if (Physics.Raycast(raycastTransform.position, raycastTransform.forward, out objectHit, distance))
            {
                if (objectHit.transform.tag == "Grabable")
                {
                    if(objectHit.distance < minDis)
                    {
                        minDis = objectHit.distance;
                        tf = objectHit.transform;
                    }
                }
            }
            Debug.DrawRay(raycastTransform.position, raycastTransform.forward * distance, Color.green);
        }

        if (tf != null)
        {
            Vector3 co = tf.GetComponent<BoxCollider>().size;
            Vector3 size = new Vector3(co.x * tf.localScale.x, co.y * tf.localScale.y, co.z * tf.localScale.z);
            objDis = minDis;
            objSize = size;
        }
        else
        {
            objDis = 0;
            objSize = new Vector3();
        }
    }
}
