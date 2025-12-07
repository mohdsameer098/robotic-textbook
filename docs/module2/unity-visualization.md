# Unity for Visualization and Human-Robot Interaction

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand Unity's role in robotics simulation
- Set up Unity Robotics Hub for ROS 2 integration
- Create photorealistic environments for robot testing
- Implement human-robot interaction scenarios
- Design natural interaction interfaces
- Optimize Unity simulations for performance
- Export data from Unity for analysis

## Why Unity for Robotics?

While Gazebo excels at physics simulation, **Unity** brings different strengths:

### Unity's Advantages

1. **Photorealism**: AAA game engine graphics for realistic visualization
2. **Human Models**: Rich ecosystem of human animations and behaviors
3. **UI/UX**: Easy to create interactive interfaces and visualizations
4. **Asset Store**: Thousands of pre-made environments, characters, and props
5. **Cross-Platform**: Deploy to desktop, mobile, VR/AR
6. **Performance**: Highly optimized rendering pipeline

### Gazebo vs Unity: When to Use Each

| Use Case | Best Tool | Reason |
|----------|-----------|---------|
| Physics testing | Gazebo | More accurate physics, easier ROS integration |
| Visual perception | Unity | Photorealistic rendering, better camera simulation |
| Human-robot interaction | Unity | Human models, animation, natural environments |
| Navigation algorithms | Gazebo | Faster, physics-focused |
| Marketing demos | Unity | Beautiful visuals |
| Multi-sensor fusion | Gazebo | Better sensor plugin ecosystem |

**Best practice**: Use **both**! Develop in Gazebo, showcase in Unity.

## Unity Robotics Hub

**Unity Robotics Hub** is Unity's official toolkit for robotics:

- **ROS-TCP-Connector**: Bidirectional communication with ROS/ROS 2
- **URDF Importer**: Import robot models directly
- **Articulation Body**: Unity's advanced physics for robots
- **Computer Vision**: Tools for generating training data

### Architecture

```
ROS 2 System ←→ ROS-TCP-Endpoint ←→ Unity (ROS-TCP-Connector)
```

Unity runs separately from ROS but communicates via TCP/IP.

## Setting Up Unity Robotics Hub

### Prerequisites

- **Unity Hub**: Download from unity.com
- **Unity Editor**: Version 2020.3 LTS or newer (recommended: 2021.3 LTS)
- **ROS 2 Humble**: Already installed from Module 1

### Step 1: Install Unity

```bash
# Download Unity Hub from https://unity.com/download

# Install Unity Editor (2021.3 LTS recommended)
# Select Linux Build Support module
```

### Step 2: Create Unity Project

1. Open Unity Hub
2. Click "New Project"
3. Select **3D** template
4. Name: "HumanoidRoboticsProject"
5. Click "Create"

### Step 3: Install Unity Robotics Hub Packages

In Unity Editor:

1. **Window** → **Package Manager**
2. Click **+** → **Add package from git URL**
3. Add these packages:

```
https://github.com/Unity-Technologies/ROS-TCP-Connector.git?path=/com.unity.robotics.ros-tcp-connector

https://github.com/Unity-Technologies/URDF-Importer.git?path=/com.unity.robotics.urdf-importer
```

### Step 4: Install ROS-TCP-Endpoint (ROS side)

```bash
# In your ROS 2 workspace
cd ~/ros2_ws/src

# Clone the endpoint package
git clone https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git

# Build
cd ~/ros2_ws
colcon build --packages-select ros_tcp_endpoint

# Source
source install/setup.bash
```

### Step 5: Test Connection

**Terminal 1** (ROS):
```bash
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=127.0.0.1
```

**Unity Editor**:
1. **Robotics** → **ROS Settings**
2. Set **ROS IP Address**: 127.0.0.1
3. Set **ROS Port**: 10000
4. Click **Connect**

You should see "Connected" status!

## Importing Robot URDF into Unity

### Method 1: URDF Importer

1. **Assets** → **Import Robot from URDF**
2. Select your URDF file
3. Configure import settings:
   - **Axis Type**: Y-Axis
   - **Override Mesh Origin**: Check if needed
4. Click **Import**

Unity creates a GameObject with:
- All links as child objects
- ArticulationBody components for joints
- Colliders and visual meshes

### Method 2: Manual Setup

For better control, set up manually:

```csharp
using UnityEngine;
using Unity.Robotics.UrdfImporter;

public class RobotSpawner : MonoBehaviour
{
    public string urdfPath = "Assets/URDF/robot.urdf";
    
    void Start()
    {
        // Import URDF at runtime
        UrdfRobot robot = UrdfRobotExtensions.Create(urdfPath);
        robot.transform.position = Vector3.zero;
    }
}
```

### Configuring Articulation Bodies

Unity's **ArticulationBody** simulates robot joints:

```csharp
using UnityEngine;

public class JointController : MonoBehaviour
{
    private ArticulationBody articulationBody;
    
    void Start()
    {
        articulationBody = GetComponent<ArticulationBody>();
        
        // Configure joint
        if (articulationBody.jointType == ArticulationJointType.RevoluteJoint)
        {
            // Set joint limits
            var drive = articulationBody.xDrive;
            drive.lowerLimit = -90f;  // degrees
            drive.upperLimit = 90f;
            drive.stiffness = 10000f;
            drive.damping = 100f;
            articulationBody.xDrive = drive;
        }
    }
    
    public void SetJointPosition(float targetAngle)
    {
        var drive = articulationBody.xDrive;
        drive.target = targetAngle;
        articulationBody.xDrive = drive;
    }
}
```

## Creating Realistic Environments

### Using Unity Asset Store

Unity Asset Store offers thousands of free and paid assets:

1. **Window** → **Asset Store**
2. Search for:
   - "Office Interior"
   - "Home Furniture"
   - "Warehouse"
   - "Hospital Environment"
3. Download and import

### Building Custom Environments

Create a simple room:

1. **GameObject** → **3D Object** → **Cube** (for walls)
2. Scale: `(10, 3, 0.1)` for a wall
3. Duplicate and rotate to create 4 walls
4. Add floor: Cube scaled to `(10, 0.1, 10)`
5. Add ceiling: Cube scaled to `(10, 0.1, 10)`

Add realistic materials:

```csharp
using UnityEngine;

public class MaterialSetup : MonoBehaviour
{
    void Start()
    {
        // Get renderer
        Renderer renderer = GetComponent<Renderer>();
        
        // Create material
        Material wallMaterial = new Material(Shader.Find("Standard"));
        wallMaterial.color = new Color(0.9f, 0.9f, 0.85f);
        wallMaterial.SetFloat("_Smoothness", 0.3f);
        
        renderer.material = wallMaterial;
    }
}
```

### Lighting for Realism

Good lighting is crucial for photorealism:

```csharp
using UnityEngine;

public class LightingSetup : MonoBehaviour
{
    void Start()
    {
        // Directional light (sun)
        GameObject sun = new GameObject("Sun");
        Light sunLight = sun.AddComponent<Light>();
        sunLight.type = LightType.Directional;
        sunLight.intensity = 1.0f;
        sunLight.color = new Color(1.0f, 0.95f, 0.9f);
        sun.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        
        // Ambient lighting
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Skybox;
        RenderSettings.ambientIntensity = 0.5f;
    }
}
```

Enable **Global Illumination**:
1. **Window** → **Rendering** → **Lighting**
2. Check **Baked Global Illumination**
3. Click **Generate Lighting**

## ROS 2 Communication from Unity

### Publishing to ROS 2 Topics

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

public class RobotStatusPublisher : MonoBehaviour
{
    private ROSConnection ros;
    private string topicName = "/robot/status";
    private float publishRate = 10f;  // Hz
    
    void Start()
    {
        // Get ROS connection
        ros = ROSConnection.GetOrCreateInstance();
        
        // Register publisher
        ros.RegisterPublisher<StringMsg>(topicName);
        
        // Start publishing
        InvokeRepeating("PublishStatus", 0f, 1f / publishRate);
    }
    
    void PublishStatus()
    {
        StringMsg msg = new StringMsg
        {
            data = $"Robot status: Active at {Time.time}"
        };
        
        ros.Publish(topicName, msg);
    }
}
```

### Subscribing to ROS 2 Topics

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class VelocitySubscriber : MonoBehaviour
{
    private ROSConnection ros;
    private string topicName = "/cmd_vel";
    
    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<TwistMsg>(topicName, OnVelocityReceived);
    }
    
    void OnVelocityReceived(TwistMsg msg)
    {
        // Extract linear and angular velocity
        float linearX = (float)msg.linear.x;
        float angularZ = (float)msg.angular.z;
        
        // Apply to robot (implementation depends on robot type)
        Debug.Log($"Velocity: linear={linearX}, angular={angularZ}");
    }
}
```

### Calling ROS 2 Services

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Example;

public class ServiceCaller : MonoBehaviour
{
    private ROSConnection ros;
    
    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
    }
    
    public void CallAddService(int a, int b)
    {
        AddTwoIntsRequest request = new AddTwoIntsRequest
        {
            a = a,
            b = b
        };
        
        ros.SendServiceMessage<AddTwoIntsResponse>(
            "/add_two_ints",
            request,
            OnServiceResponse
        );
    }
    
    void OnServiceResponse(AddTwoIntsResponse response)
    {
        Debug.Log($"Service result: {response.sum}");
    }
}
```

## Humanoid Animation and Interaction

### Importing Human Models

Unity's **Humanoid Rig** system:

1. Import character from Asset Store
2. Select character model
3. **Inspector** → **Rig** → **Animation Type**: Humanoid
4. Click **Apply**

### Animating Humans

```csharp
using UnityEngine;

public class HumanController : MonoBehaviour
{
    private Animator animator;
    
    void Start()
    {
        animator = GetComponent<Animator>();
    }
    
    public void Walk()
    {
        animator.SetBool("isWalking", true);
    }
    
    public void Stop()
    {
        animator.SetBool("isWalking", false);
    }
    
    public void Wave()
    {
        animator.SetTrigger("wave");
    }
}
```

### Human-Robot Interaction Scenarios

**Scenario 1: Handshake**

```csharp
using UnityEngine;

public class HandshakeInteraction : MonoBehaviour
{
    public Transform robotHand;
    public Transform humanHand;
    public float approachSpeed = 0.5f;
    
    void Update()
    {
        // Move robot hand toward human hand
        robotHand.position = Vector3.MoveTowards(
            robotHand.position,
            humanHand.position,
            approachSpeed * Time.deltaTime
        );
        
        // Check if close enough
        if (Vector3.Distance(robotHand.position, humanHand.position) < 0.05f)
        {
            Debug.Log("Handshake successful!");
        }
    }
}
```

**Scenario 2: Object Handoff**

```csharp
using UnityEngine;

public class ObjectHandoff : MonoBehaviour
{
    public GameObject objectToHand;
    public Transform humanHand;
    public Transform robotHand;
    
    private bool objectWithHuman = true;
    
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            HandoffObject();
        }
    }
    
    void HandoffObject()
    {
        if (objectWithHuman)
        {
            // Transfer to robot
            objectToHand.transform.SetParent(robotHand);
            objectToHand.transform.localPosition = Vector3.zero;
            objectWithHuman = false;
        }
        else
        {
            // Transfer to human
            objectToHand.transform.SetParent(humanHand);
            objectToHand.transform.localPosition = Vector3.zero;
            objectWithHuman = true;
        }
    }
}
```

## Camera Systems for Perception

### RGB Camera Simulation

```csharp
using UnityEngine;

public class RGBCamera : MonoBehaviour
{
    public Camera cameraComponent;
    public int width = 640;
    public int height = 480;
    public float captureRate = 30f;  // Hz
    
    private RenderTexture renderTexture;
    private Texture2D texture;
    
    void Start()
    {
        // Create render texture
        renderTexture = new RenderTexture(width, height, 24);
        cameraComponent.targetTexture = renderTexture;
        
        // Create texture for reading
        texture = new Texture2D(width, height, TextureFormat.RGB24, false);
        
        // Start capture loop
        InvokeRepeating("CaptureImage", 0f, 1f / captureRate);
    }
    
    void CaptureImage()
    {
        RenderTexture.active = renderTexture;
        texture.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        texture.Apply();
        
        // Get pixel data
        byte[] bytes = texture.EncodeToJPG();
        
        // TODO: Publish to ROS topic
    }
}
```

### Depth Camera Simulation

```csharp
using UnityEngine;

public class DepthCamera : MonoBehaviour
{
    public Camera cameraComponent;
    
    void Start()
    {
        // Enable depth texture mode
        cameraComponent.depthTextureMode = DepthTextureMode.Depth;
    }
    
    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        // Apply depth shader
        Graphics.Blit(source, destination);
        
        // Read depth data
        // TODO: Convert to depth values and publish
    }
}
```

## Performance Optimization

### Tips for Smooth Simulation

1. **Occlusion Culling**: Don't render what's not visible
2. **LOD (Level of Detail)**: Use simpler models at distance
3. **Bake Lighting**: Pre-calculate lighting (faster than real-time)
4. **Reduce Shadows**: Limit shadow distance and quality
5. **Object Pooling**: Reuse GameObjects instead of Instantiate/Destroy

### Occlusion Culling Setup

1. **Window** → **Rendering** → **Occlusion Culling**
2. Mark static objects as **Occluder Static**
3. Click **Bake**

### LOD Groups

```csharp
using UnityEngine;

public class LODSetup : MonoBehaviour
{
    void Start()
    {
        LODGroup lodGroup = gameObject.AddComponent<LODGroup>();
        
        LOD[] lods = new LOD[3];
        
        // High detail (0-50% screen)
        Renderer[] highRenderers = new Renderer[] { /* high detail meshes */ };
        lods[0] = new LOD(0.5f, highRenderers);
        
        // Medium detail (50-25% screen)
        Renderer[] medRenderers = new Renderer[] { /* medium detail meshes */ };
        lods[1] = new LOD(0.25f, medRenderers);
        
        // Low detail (25-10% screen)
        Renderer[] lowRenderers = new Renderer[] { /* low detail meshes */ };
        lods[2] = new LOD(0.1f, lowRenderers);
        
        lodGroup.SetLODs(lods);
    }
}
```

## Data Collection for Machine Learning

Unity is excellent for generating training data:

### Synthetic Data Generation

```csharp
using UnityEngine;
using System.IO;

public class DatasetGenerator : MonoBehaviour
{
    public Camera[] cameras;
    public int imagesPerCamera = 1000;
    public string outputPath = "Assets/Dataset";
    
    void Start()
    {
        StartCoroutine(GenerateDataset());
    }
    
    System.Collections.IEnumerator GenerateDataset()
    {
        for (int i = 0; i < imagesPerCamera; i++)
        {
            // Randomize scene
            RandomizeObjects();
            RandomizeLighting();
            
            yield return new WaitForEndOfFrame();
            
            // Capture from each camera
            foreach (var cam in cameras)
            {
                CaptureAndSave(cam, i);
            }
        }
    }
    
    void CaptureAndSave(Camera cam, int index)
    {
        RenderTexture rt = new RenderTexture(640, 480, 24);
        cam.targetTexture = rt;
        cam.Render();
        
        RenderTexture.active = rt;
        Texture2D image = new Texture2D(640, 480, TextureFormat.RGB24, false);
        image.ReadPixels(new Rect(0, 0, 640, 480), 0, 0);
        image.Apply();
        
        byte[] bytes = image.EncodeToPNG();
        File.WriteAllBytes($"{outputPath}/image_{index}.png", bytes);
        
        cam.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);
    }
    
    void RandomizeObjects()
    {
        // Randomize object positions, rotations, scales
    }
    
    void RandomizeLighting()
    {
        // Randomize light colors, intensities, positions
    }
}
```

## Key Takeaways

✅ **Unity** provides photorealistic visualization for robotics  
✅ **Unity Robotics Hub** enables ROS 2 integration  
✅ **URDF Importer** brings robots from ROS to Unity  
✅ **Humanoid animation** enables realistic HRI scenarios  
✅ **Camera simulation** generates training data for perception  
✅ **Performance optimization** ensures smooth real-time simulation

## What's Next?

In the next chapter, we'll dive into **sensor simulation** in detail, covering LiDAR, depth cameras, IMUs, and sensor fusion techniques.

Continue to: [Sensor Simulation →](/docs/module2/sensor-simulation)

## Exercises

### Exercise 1: Import Your Robot
Import the humanoid URDF you created in Module 1 into Unity. Configure all joints.

### Exercise 2: Interaction Scene
Create a scene where a human hands an object to your robot. Implement the handoff logic.

### Exercise 3: Camera Capture
Set up an RGB camera on your robot and publish images to ROS 2 at 30 Hz.

### Exercise 4: Dataset Generation
Generate 100 images of objects on a table from random viewpoints for training an object detector.

### Exercise 5: Performance Test
Create a scene with 10 humanoid robots. Optimize to maintain 60 FPS.

---

**Navigation**: [← Previous: Gazebo Physics](/docs/module2/gazebo-physics) | [Next: Sensor Simulation →](/docs/module2/sensor-simulation)