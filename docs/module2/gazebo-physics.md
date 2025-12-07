# Gazebo Physics Simulation

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand the role of simulation in robotics development
- Set up and configure Gazebo for humanoid robot simulation
- Create custom worlds and environments
- Spawn robots dynamically in simulation
- Configure physics engines and their parameters
- Implement sensor plugins for realistic data generation
- Handle contact and collision detection

## Why Simulation?

Before we can deploy robots to the real world, we need to test them safely and efficiently. **Simulation** provides:

### Advantages of Simulation

1. **Safety**: Test dangerous scenarios without risk to hardware or people
2. **Cost**: No need to purchase expensive robots initially
3. **Speed**: Test faster than real-time, run multiple scenarios in parallel
4. **Repeatability**: Test exact same conditions repeatedly
5. **Accessibility**: Everyone can practice without physical hardware
6. **Debugging**: Pause, inspect, and modify during execution

### The Sim-to-Real Challenge

The challenge: **simulations are never perfect**. Differences between simulation and reality include:

- **Physics approximations**: Real-world physics is infinitely complex
- **Sensor noise**: Simulated sensors are often too perfect
- **Actuation delays**: Real motors have lag and backlash
- **Material properties**: Friction, elasticity, damping differ
- **Environmental factors**: Wind, temperature, lighting variations

Our goal: Make simulation **realistic enough** that controllers work on real hardware with minimal tuning.

## Introduction to Gazebo

**Gazebo** is the most popular robotics simulator, integrated tightly with ROS 2. It provides:

- High-fidelity physics simulation
- Realistic sensor simulation
- 3D visualization
- Plugin system for extensibility
- Support for complex robots and environments

### Gazebo Architecture

Gazebo consists of several components:

1. **Gazebo Server (gzserver)**: Runs physics simulation (headless)
2. **Gazebo Client (gzclient)**: Provides 3D visualization GUI
3. **Plugins**: Extend functionality (sensors, controllers, etc.)
4. **Models**: Robot and object descriptions (SDF format)
5. **Worlds**: Environment descriptions

You can run server and client separately - useful for running simulations on powerful servers while viewing on a laptop.

## Installing Gazebo

Gazebo Classic (Gazebo 11) works with ROS 2 Humble:

```bash
# Install Gazebo Classic
sudo apt install ros-humble-gazebo-ros-pkgs

# Install additional tools
sudo apt install ros-humble-gazebo-ros2-control
```

Test the installation:

```bash
# Launch Gazebo
gazebo
```

You should see an empty world with a ground plane.

### Gazebo vs. Ignition Gazebo

**Note**: There are two versions of Gazebo:

- **Gazebo Classic** (Gazebo 11): Mature, stable, widely used
- **Ignition Gazebo** (now called "Gazebo"): Modern rewrite, better performance

For this course, we use **Gazebo Classic** as it has better ROS 2 integration and more resources. Ignition is the future but still maturing.

## Understanding World Files

**World files** (.world) define simulation environments using SDF (Simulation Description Format).

### Basic World Structure

```xml
<?xml version="1.0"?>
<sdf version="1.6">
  <world name="simple_world">
    
    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Physics settings -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    
    <!-- Gravity -->
    <gravity>0 0 -9.81</gravity>
    
  </world>
</sdf>
```

### Adding Objects to a World

```xml
<world name="obstacle_course">
  <include>
    <uri>model://sun</uri>
  </include>
  
  <include>
    <uri>model://ground_plane</uri>
  </include>
  
  <!-- Add a box obstacle -->
  <model name="box1">
    <pose>2 0 0.5 0 0 0</pose>
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box>
            <size>1 1 1</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>1 1 1</size>
          </box>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
  
  <!-- Add a cylinder -->
  <model name="cylinder1">
    <pose>-2 0 0.5 0 0 0</pose>
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.5</radius>
            <length>1.0</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.5</radius>
            <length>1.0</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0 0 1 1</ambient>
        </material>
      </visual>
    </link>
  </model>
  
</world>
```

Save as `obstacle_course.world` and launch:

```bash
gazebo obstacle_course.world
```

## Physics Engines

Gazebo supports multiple physics engines. Each has tradeoffs:

### ODE (Open Dynamics Engine)

**Default engine** - good balance of speed and accuracy.

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_update_rate>1000</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>50</iters>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

**Pros**: Fast, stable, good for most robots  
**Cons**: Less accurate for complex contacts

### Bullet

Better for complex collision detection.

```xml
<physics type="bullet">
  <max_step_size>0.001</max_step_size>
  <real_time_update_rate>1000</real_time_update_rate>
</physics>
```

**Pros**: More accurate collisions, better for grasping  
**Cons**: Slower than ODE

### DART

Most accurate, best for complex kinematics.

```xml
<physics type="dart">
  <max_step_size>0.001</max_step_size>
  <real_time_update_rate>1000</real_time_update_rate>
</physics>
```

**Pros**: Most accurate, best for humanoids  
**Cons**: Slowest, may not run real-time

### Choosing a Physics Engine

- **Wheeled robots**: ODE
- **Arms and manipulators**: Bullet
- **Humanoids and complex dynamics**: DART
- **Speed critical**: ODE

## Spawning Robots in Gazebo

There are multiple ways to add robots to simulation:

### Method 1: Include in World File

```xml
<world name="robot_world">
  <!-- ... other world elements ... -->
  
  <include>
    <uri>model://my_robot</uri>
    <pose>0 0 0.5 0 0 0</pose>
  </include>
</world>
```

### Method 2: Spawn via ROS 2 Service

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
import os

class RobotSpawner(Node):
    def __init__(self):
        super().__init__('robot_spawner')
        
        # Create service client
        self.spawn_client = self.create_client(
            SpawnEntity,
            '/spawn_entity'
        )
        
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for spawn service...')
        
        self.spawn_robot()
    
    def spawn_robot(self):
        # Read URDF file
        urdf_path = '/path/to/robot.urdf'
        with open(urdf_path, 'r') as f:
            robot_desc = f.read()
        
        # Create spawn request
        request = SpawnEntity.Request()
        request.name = 'my_robot'
        request.xml = robot_desc
        request.robot_namespace = '/my_robot'
        request.initial_pose.position.x = 0.0
        request.initial_pose.position.y = 0.0
        request.initial_pose.position.z = 0.5
        
        # Call service
        future = self.spawn_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            self.get_logger().info('Robot spawned successfully!')
        else:
            self.get_logger().error('Failed to spawn robot')

def main():
    rclpy.init()
    spawner = RobotSpawner()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Method 3: Launch File

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        # Start Gazebo
        ExecuteProcess(
            cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
            output='screen'
        ),
        
        # Spawn robot
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'my_robot',
                '-file', '/path/to/robot.urdf',
                '-x', '0', '-y', '0', '-z', '0.5'
            ]
        ),
    ])
```

## Sensor Simulation

Accurate sensor simulation is crucial for developing perception algorithms.

### Camera Sensor Plugin

Add to your URDF/SDF:

```xml
<gazebo reference="camera_link">
  <sensor type="camera" name="camera1">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.02</near>
        <far>300</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/image_raw:=camera/image_raw</remapping>
        <remapping>~/camera_info:=camera/camera_info</remapping>
      </ros>
      <camera_name>camera</camera_name>
      <frame_name>camera_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

This publishes:
- `/my_robot/camera/image_raw`: Raw camera images
- `/my_robot/camera/camera_info`: Camera calibration data

### LiDAR Sensor Plugin

```xml
<gazebo reference="lidar_link">
  <sensor type="ray" name="lidar_sensor">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.10</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.01</stddev>
      </noise>
    </ray>
    <plugin name="gazebo_ros_head_hokuyo_controller" 
            filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>lidar_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

Publishes `/my_robot/scan` with laser scan data.

### IMU Sensor Plugin

```xml
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <plugin filename="libgazebo_ros_imu_sensor.so" name="imu_plugin">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/out:=imu</remapping>
      </ros>
      <initial_orientation_as_reference>false</initial_orientation_as_reference>
    </plugin>
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <!-- Similar for y, z -->
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <!-- Similar for y, z -->
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

## Contact and Collision Detection

### Contact Sensor Plugin

Detect when robot touches objects:

```xml
<gazebo reference="foot_link">
  <sensor name="foot_contact" type="contact">
    <plugin name="foot_contact_plugin" 
            filename="libgazebo_ros_bumper.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/out:=foot_contact</remapping>
      </ros>
      <frame_name>foot_link</frame_name>
    </plugin>
    <contact>
      <collision>foot_collision</collision>
    </contact>
    <update_rate>50.0</update_rate>
  </sensor>
</gazebo>
```

### Handling Collisions

Configure collision properties:

```xml
<collision name="foot_collision">
  <geometry>
    <box>
      <size>0.15 0.08 0.05</size>
    </box>
  </geometry>
  <surface>
    <friction>
      <ode>
        <mu>1.0</mu>
        <mu2>1.0</mu2>
      </ode>
    </friction>
    <contact>
      <ode>
        <kp>1000000.0</kp>
        <kd>100.0</kd>
        <max_vel>1.0</max_vel>
        <min_depth>0.001</min_depth>
      </ode>
    </contact>
  </surface>
</collision>
```

Parameters:
- **mu, mu2**: Friction coefficients
- **kp**: Contact stiffness (spring constant)
- **kd**: Contact damping
- **max_vel**: Maximum contact correction velocity
- **min_depth**: Minimum penetration before contact

## Performance Optimization

### Tips for Faster Simulation

1. **Simplify collision geometry**: Use boxes/cylinders instead of meshes
2. **Reduce update rates**: Sensors don't need 1000 Hz
3. **Run headless**: `gzserver` only (no GUI)
4. **Disable shadows**: In GUI, uncheck shadows
5. **Use simpler physics**: ODE instead of DART
6. **Increase time step**: Carefully (may reduce stability)

### Running Headless

```bash
# Server only (no GUI)
gzserver my_world.world
```

Or in a launch file:

```python
ExecuteProcess(
    cmd=['gzserver', '--verbose', 'my_world.world'],
    output='screen'
)
```

## Key Takeaways

✅ **Simulation** enables safe, fast, and cost-effective robot testing  
✅ **Gazebo** provides high-fidelity physics and sensor simulation  
✅ **World files** define simulation environments  
✅ **Physics engines** offer tradeoffs between speed and accuracy  
✅ **Sensor plugins** generate realistic data for perception algorithms  
✅ **Contact detection** enables interaction simulation

## What's Next?

In the next chapter, we'll explore **Unity** for photorealistic visualization and human-robot interaction scenarios.

Continue to: [Unity Visualization →](/docs/module2/unity-visualization)

## Exercises

### Exercise 1: Custom World
Create a world with a humanoid-sized room including walls, furniture, and obstacles.

### Exercise 2: Sensor Fusion
Add both camera and LiDAR to a robot. Compare their data in RViz.

### Exercise 3: Contact Forces
Implement a node that monitors foot contact forces and logs when they exceed safe thresholds.

### Exercise 4: Physics Comparison
Run the same robot with ODE, Bullet, and DART. Compare stability and performance.

### Exercise 5: Multi-Robot
Spawn 3 robots in the same world. Implement collision avoidance between them.

---

**Navigation**: [← Module 1](/docs/module1/ros2-exercises) | [Next: Unity Visualization →](/docs/module2/unity-visualization)