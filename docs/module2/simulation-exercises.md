# Simulation Exercises

## Overview

Congratulations on completing Module 2! You've learned about physics simulation in Gazebo, photorealistic visualization in Unity, and comprehensive sensor simulation. Now it's time to apply these concepts through hands-on exercises.

This chapter contains:
- **Practical simulation exercises** (beginner to advanced)
- **Integration projects** combining Gazebo, Unity, and ROS 2
- **Performance optimization challenges**
- **Assessment questions**

## Exercise Set 1: Gazebo World Building

### Exercise 1.1: Indoor Environment

**Difficulty**: Beginner  
**Time**: 1 hour  
**Concepts**: World files, static models, lighting

**Task**: Create a realistic indoor environment for humanoid robot testing.

**Requirements**:
1. Create a 10m × 10m room with:
   - Four walls (3m height)
   - Floor and ceiling
   - Two doorways (0.9m wide)
   - Windows with natural lighting

2. Add furniture:
   - Table (0.8m height)
   - 4 chairs
   - Bookshelf
   - Couch

3. Place 5-10 small objects on surfaces (books, cups, etc.)

4. Configure realistic lighting:
   - Ambient light (soft)
   - Directional light (sunlight through windows)
   - Point lights (ceiling lamps)

**Deliverables**:
- `indoor_environment.world` file
- Screenshot from multiple angles
- Launch file to start Gazebo with your world

**Hints**:
```xml
<model name="wall_north">
  <pose>5 0 1.5 0 0 0</pose>
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry>
        <box><size>10 0.2 3</size></box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box><size>10 0.2 3</size></box>
      </geometry>
      <material>
        <ambient>0.8 0.8 0.7 1</ambient>
        <diffuse>0.8 0.8 0.7 1</diffuse>
      </material>
    </visual>
  </link>
</model>
```

---

### Exercise 1.2: Dynamic Obstacles

**Difficulty**: Intermediate  
**Time**: 1.5 hours  
**Concepts**: Model plugins, dynamic objects

**Task**: Create a world with moving obstacles to test robot navigation.

**Requirements**:
1. Base world with corridors and rooms
2. Add 3 moving obstacles:
   - Cylinder rolling back and forth
   - Box sliding side to side
   - Sphere bouncing

3. Use model plugins to control motion
4. Add static obstacles as well

**Code Example** (Model Plugin):
```xml
<model name="moving_box">
  <pose>0 0 0.5 0 0 0</pose>
  <link name="link">
    <collision name="collision">
      <geometry><box><size>1 1 1</size></box></geometry>
    </collision>
    <visual name="visual">
      <geometry><box><size>1 1 1</size></box></geometry>
    </visual>
  </link>
  
  <plugin name="box_mover" filename="libgazebo_ros_planar_move.so">
    <ros>
      <namespace>/moving_box</namespace>
      <remapping>cmd_vel:=cmd_vel</remapping>
    </ros>
    <update_rate>10</update_rate>
    <publish_odom>true</publish_odom>
    <publish_odom_tf>true</publish_odom_tf>
  </plugin>
</model>
```

Create a node to control the obstacles:
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math

class ObstacleController(Node):
    def __init__(self):
        super().__init__('obstacle_controller')
        self.pub = self.create_publisher(Twist, '/moving_box/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.move_obstacle)
        self.t = 0.0
        
    def move_obstacle(self):
        msg = Twist()
        # Sinusoidal motion
        msg.linear.x = 0.5 * math.sin(self.t)
        self.pub.publish(msg)
        self.t += 0.1

def main():
    rclpy.init()
    controller = ObstacleController()
    rclpy.spin(controller)
    rclpy.shutdown()
```

---

### Exercise 1.3: Multi-Robot Simulation

**Difficulty**: Advanced  
**Time**: 2 hours  
**Concepts**: Namespaces, multiple robots, coordination

**Task**: Spawn 3 robots in the same world and implement basic coordination.

**Requirements**:
1. Spawn 3 identical robots with different namespaces
2. Each robot has its own LiDAR sensor
3. Implement collision avoidance between robots
4. Make robots patrol different areas without interfering

**Launch File Example**:
```python
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Start Gazebo
        ExecuteProcess(
            cmd=['gazebo', '--verbose', 'multi_robot.world'],
            output='screen'
        ),
        
        # Spawn robot 1
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'robot1',
                '-file', 'robot.urdf',
                '-robot_namespace', '/robot1',
                '-x', '0', '-y', '0', '-z', '0.5'
            ]
        ),
        
        # Spawn robot 2
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'robot2',
                '-file', 'robot.urdf',
                '-robot_namespace', '/robot2',
                '-x', '2', '-y', '0', '-z', '0.5'
            ]
        ),
        
        # Spawn robot 3
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'robot3',
                '-file', 'robot.urdf',
                '-robot_namespace', '/robot3',
                '-x', '4', '-y', '0', '-z', '0.5'
            ]
        ),
    ])
```

---

## Exercise Set 2: Sensor Integration

### Exercise 2.1: Sensor Calibration

**Difficulty**: Intermediate  
**Time**: 1 hour  
**Concepts**: Sensor data processing, calibration

**Task**: Calibrate simulated sensors to remove bias and noise.

**Requirements**:
1. Collect 1000 IMU samples while robot is stationary
2. Calculate mean and standard deviation
3. Implement bias correction
4. Test before/after calibration quality

**Implementation**:
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np

class IMUCalibration(Node):
    def __init__(self):
        super().__init__('imu_calibration')
        
        self.samples = []
        self.num_samples = 1000
        self.calibrating = True
        
        self.subscription = self.create_subscription(
            Imu, '/robot/imu/data', self.collect_samples, 10
        )
        
        self.calibrated_pub = self.create_publisher(
            Imu, '/robot/imu/calibrated', 10
        )
        
        self.bias_accel = np.zeros(3)
        self.bias_gyro = np.zeros(3)
        
    def collect_samples(self, msg):
        if self.calibrating:
            sample = {
                'accel': [msg.linear_acceleration.x,
                         msg.linear_acceleration.y,
                         msg.linear_acceleration.z],
                'gyro': [msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z]
            }
            self.samples.append(sample)
            
            if len(self.samples) >= self.num_samples:
                self.compute_bias()
                self.calibrating = False
        else:
            # Apply calibration
            calibrated_msg = Imu()
            calibrated_msg.header = msg.header
            
            calibrated_msg.linear_acceleration.x = \
                msg.linear_acceleration.x - self.bias_accel[0]
            calibrated_msg.linear_acceleration.y = \
                msg.linear_acceleration.y - self.bias_accel[1]
            calibrated_msg.linear_acceleration.z = \
                msg.linear_acceleration.z - self.bias_accel[2]
            
            calibrated_msg.angular_velocity.x = \
                msg.angular_velocity.x - self.bias_gyro[0]
            calibrated_msg.angular_velocity.y = \
                msg.angular_velocity.y - self.bias_gyro[1]
            calibrated_msg.angular_velocity.z = \
                msg.angular_velocity.z - self.bias_gyro[2]
            
            self.calibrated_pub.publish(calibrated_msg)
    
    def compute_bias(self):
        accel_samples = np.array([s['accel'] for s in self.samples])
        gyro_samples = np.array([s['gyro'] for s in self.samples])
        
        self.bias_accel = np.mean(accel_samples, axis=0)
        self.bias_gyro = np.mean(gyro_samples, axis=0)
        
        # Account for gravity on Z-axis
        self.bias_accel[2] -= 9.81
        
        self.get_logger().info(f'Calibration complete!')
        self.get_logger().info(f'Accel bias: {self.bias_accel}')
        self.get_logger().info(f'Gyro bias: {self.bias_gyro}')

def main():
    rclpy.init()
    calibrator = IMUCalibration()
    rclpy.spin(calibrator)
    rclpy.shutdown()
```

---

### Exercise 2.2: Sensor Fusion for Localization

**Difficulty**: Advanced  
**Time**: 2 hours  
**Concepts**: Kalman filtering, multi-sensor fusion

**Task**: Fuse wheel odometry, IMU, and LiDAR for accurate localization.

**Requirements**:
1. Subscribe to:
   - Wheel odometry (`/robot/odom`)
   - IMU data (`/robot/imu/data`)
   - LiDAR scan (`/robot/scan`)

2. Implement Extended Kalman Filter (EKF)
3. Publish fused pose estimate
4. Compare accuracy vs. single-sensor estimates

**Simplified EKF Structure**:
```python
import numpy as np

class SimpleEKF:
    def __init__(self):
        # State: [x, y, theta, vx, vy, omega]
        self.state = np.zeros(6)
        self.covariance = np.eye(6) * 0.1
        
        # Process noise
        self.Q = np.eye(6) * 0.01
        
        # Measurement noise
        self.R_odom = np.eye(3) * 0.1  # x, y, theta
        self.R_imu = np.eye(3) * 0.05  # theta, omega, accel
        
    def predict(self, dt):
        """Predict step using motion model."""
        # Simple motion model: state propagates based on velocity
        self.state[0] += self.state[3] * dt  # x += vx * dt
        self.state[1] += self.state[4] * dt  # y += vy * dt
        self.state[2] += self.state[5] * dt  # theta += omega * dt
        
        # Jacobian of motion model
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        
        # Update covariance
        self.covariance = F @ self.covariance @ F.T + self.Q
        
    def update_odometry(self, x, y, theta):
        """Update with odometry measurement."""
        z = np.array([x, y, theta])
        H = np.zeros((3, 6))
        H[0, 0] = 1  # measure x
        H[1, 1] = 1  # measure y
        H[2, 2] = 1  # measure theta
        
        self._kalman_update(z, H, self.R_odom)
    
    def update_imu(self, theta, omega):
        """Update with IMU measurement."""
        z = np.array([theta, omega])
        H = np.zeros((2, 6))
        H[0, 2] = 1  # measure theta
        H[1, 5] = 1  # measure omega
        
        self._kalman_update(z, H, self.R_imu[:2, :2])
    
    def _kalman_update(self, z, H, R):
        """Generic Kalman update step."""
        # Innovation
        y = z - H @ self.state
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + R
        
        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        self.covariance = (np.eye(6) - K @ H) @ self.covariance
    
    def get_pose(self):
        return self.state[0], self.state[1], self.state[2]
```

---

## Exercise Set 3: Unity Integration

### Exercise 3.1: ROS-Unity Communication

**Difficulty**: Intermediate  
**Time**: 1.5 hours  
**Concepts**: Unity, ROS TCP Connector

**Task**: Create a Unity scene that visualizes robot position from ROS.

**Requirements**:
1. Set up Unity project with ROS TCP Connector
2. Subscribe to `/robot/pose` topic in Unity
3. Update Unity robot model position in real-time
4. Add visual effects (trail, speed indicator)

**Unity C# Script**:
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class RobotPositionSync : MonoBehaviour
{
    private ROSConnection ros;
    public string topicName = "/robot/pose";
    public GameObject robotModel;
    
    // Trail renderer for path visualization
    private TrailRenderer trail;
    
    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<PoseStampedMsg>(topicName, UpdatePosition);
        
        trail = robotModel.AddComponent<TrailRenderer>();
        trail.time = 10f;
        trail.startWidth = 0.1f;
        trail.endWidth = 0.05f;
        trail.material = new Material(Shader.Find("Sprites/Default"));
        trail.startColor = Color.blue;
        trail.endColor = Color.cyan;
    }
    
    void UpdatePosition(PoseStampedMsg msg)
    {
        // Convert ROS coordinates to Unity coordinates
        Vector3 position = new Vector3(
            (float)msg.pose.position.x,
            (float)msg.pose.position.z,  // ROS Z -> Unity Y
            (float)msg.pose.position.y   // ROS Y -> Unity Z
        );
        
        Quaternion rotation = new Quaternion(
            (float)msg.pose.orientation.x,
            (float)msg.pose.orientation.z,
            (float)msg.pose.orientation.y,
            (float)msg.pose.orientation.w
        );
        
        robotModel.transform.position = position;
        robotModel.transform.rotation = rotation;
    }
}
```

---

### Exercise 3.2: Human-Robot Interaction Scene

**Difficulty**: Advanced  
**Time**: 3 hours  
**Concepts**: Unity animations, interaction logic

**Task**: Create a scene where a human and robot collaborate to move boxes.

**Requirements**:
1. Import human character with animations
2. Place 5 boxes in scene
3. Robot and human take turns picking up boxes
4. When box is picked up, it parents to hand
5. Boxes are stacked at target location

**Interaction Manager Script**:
```csharp
using UnityEngine;
using System.Collections;

public class CollaborationManager : MonoBehaviour
{
    public GameObject human;
    public GameObject robot;
    public GameObject[] boxes;
    public Transform targetLocation;
    
    private int currentBox = 0;
    private bool humanTurn = true;
    
    void Start()
    {
        StartCoroutine(CollaborationRoutine());
    }
    
    IEnumerator CollaborationRoutine()
    {
        while (currentBox < boxes.Length)
        {
            GameObject actor = humanTurn ? human : robot;
            GameObject box = boxes[currentBox];
            
            // Move to box
            yield return MoveToTarget(actor, box.transform.position);
            
            // Pick up box
            yield return PickUpBox(actor, box);
            
            // Move to target
            yield return MoveToTarget(actor, targetLocation.position);
            
            // Place box
            yield return PlaceBox(actor, box, currentBox);
            
            // Switch turns
            humanTurn = !humanTurn;
            currentBox++;
            
            yield return new WaitForSeconds(1f);
        }
        
        Debug.Log("Collaboration complete!");
    }
    
    IEnumerator MoveToTarget(GameObject actor, Vector3 target)
    {
        float speed = 2f;
        while (Vector3.Distance(actor.transform.position, target) > 0.1f)
        {
            actor.transform.position = Vector3.MoveTowards(
                actor.transform.position, target, speed * Time.deltaTime
            );
            yield return null;
        }
    }
    
    IEnumerator PickUpBox(GameObject actor, GameObject box)
    {
        // Trigger pick up animation
        Animator anim = actor.GetComponent<Animator>();
        anim.SetTrigger("PickUp");
        
        yield return new WaitForSeconds(1f);
        
        // Parent box to hand
        Transform hand = actor.transform.Find("Hand");
        box.transform.SetParent(hand);
        box.transform.localPosition = Vector3.zero;
    }
    
    IEnumerator PlaceBox(GameObject actor, GameObject box, int stackIndex)
    {
        Animator anim = actor.GetComponent<Animator>();
        anim.SetTrigger("Place");
        
        yield return new WaitForSeconds(1f);
        
        // Unparent and stack
        box.transform.SetParent(null);
        box.transform.position = targetLocation.position + 
                                 Vector3.up * (stackIndex * 0.2f);
    }
}
```

---

## Mini-Project: Simulated Warehouse Robot

**Difficulty**: Advanced  
**Time**: 5-6 hours  
**Concepts**: Full integration of simulation, sensors, and navigation

### Project Description

Build a complete warehouse robot simulation that:
1. Navigates autonomously in Gazebo environment
2. Detects and avoids obstacles using LiDAR
3. Picks up packages using depth camera
4. Visualizes operation in Unity

### Requirements

**Gazebo World**:
- Warehouse with shelves and boxes
- Multiple packages to pick up
- Dynamic obstacles (forklifts, people)

**Robot**:
- Differential drive base
- LiDAR for navigation
- RGB-D camera for object detection
- Gripper for picking

**Nodes**:
1. **Navigation Node**: Path planning, obstacle avoidance
2. **Detection Node**: Find packages using camera
3. **Manipulation Node**: Control gripper
4. **State Machine**: Coordinate overall behavior

**Unity Visualization**:
- Real-time position tracking
- Camera feed display
- Status indicators

### Deliverables

1. Complete ROS 2 package
2. Gazebo world file
3. Unity scene
4. Launch files
5. Demo video (3-5 minutes)
6. Documentation

---

## Assessment Questions

### Conceptual Questions

1. What are the advantages of using Gazebo vs. Unity for robotics simulation?

2. Explain why sensor noise is important in simulation.

3. What is the "sim-to-real gap" and how can it be minimized?

4. When would you use a 2D LiDAR vs. a 3D LiDAR?

5. Why do we need sensor fusion? Give an example.

### Practical Questions

6. Write a command to spawn a robot at position (2, 3, 0.5):
   ```bash
   # Your answer
   ```

7. How do you visualize a LaserScan topic in RViz2?

8. What Gazebo physics engine is best for humanoid robots?

9. Calculate the field of view (in degrees) of a LiDAR with:
   - min_angle: -π/2
   - max_angle: π/2

10. How do you add Gaussian noise with stddev=0.05 to a sensor in Gazebo?

### Code Review

**Question 11**: Find the bug in this sensor processing code:

```python
def process_lidar(self, msg):
    ranges = msg.ranges
    min_dist = min(ranges)
    
    if min_dist < 0.5:
        self.get_logger().warn('Obstacle detected!')
```

**Hint**: What if there are invalid readings?

---

## Performance Optimization Challenge

**Task**: Optimize a simulation to run at 2x real-time.

**Starting Scenario**:
- 10 robots in Gazebo
- Each has camera (30 Hz), LiDAR (10 Hz), IMU (100 Hz)
- Running at 0.5x real-time (too slow!)

**Your Goal**: Make it run at 2x real-time without losing functionality.

**Techniques to Try**:
1. Reduce sensor update rates
2. Simplify collision geometry
3. Use simpler physics engine
4. Run headless (no GUI)
5. Reduce image resolution
6. Disable unnecessary sensors
7. Use GPU ray sensor for LiDAR

**Deliverable**: Document which changes you made and their impact on performance.

---

## Key Takeaways

✅ **World building** creates realistic test environments  
✅ **Sensor integration** enables perception pipelines  
✅ **Unity visualization** provides photorealistic feedback  
✅ **Performance optimization** is crucial for complex simulations  
✅ **Complete projects** integrate all simulation concepts

## What's Next?

Congratulations on completing Module 2! You now have comprehensive simulation skills.

In **Module 3**, we'll explore NVIDIA Isaac - the cutting-edge platform for AI-powered robotics. You'll learn Isaac Sim, Isaac ROS, and advanced navigation with Nav2.

Continue to: [Module 3: Isaac Sim Overview →](/docs/module3/isaac-sim)

---

**Navigation**: [← Previous: Sensor Simulation](/docs/module2/sensor-simulation) | [Next: Module 3 →](/docs/module3/isaac-sim)