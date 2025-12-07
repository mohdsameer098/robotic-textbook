# Sensor Simulation

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand different types of robot sensors and their applications
- Simulate LiDAR sensors for range measurement
- Implement depth camera (RGB-D) simulation
- Configure IMU sensors for orientation and motion sensing
- Add realistic sensor noise models
- Implement sensor fusion techniques
- Process and filter sensor data in ROS 2
- Visualize sensor data in RViz2

## The Role of Sensors in Robotics

Sensors are the "eyes, ears, and touch" of robots. They enable robots to:

- **Perceive** the environment
- **Localize** themselves in space
- **Detect** obstacles and hazards
- **Measure** their own motion and orientation
- **Interact** safely with humans and objects

For humanoid robots, accurate sensor simulation is critical because:
- Balance requires precise IMU data
- Navigation needs reliable range sensors
- Manipulation depends on force/tactile feedback
- Human interaction requires vision systems

## Sensor Categories

### Exteroceptive Sensors
Measure external environment:
- **Cameras**: Visual perception
- **LiDAR**: Distance measurement
- **Ultrasonic**: Short-range proximity
- **Microphones**: Audio input

### Proprioceptive Sensors
Measure robot's own state:
- **IMU**: Orientation, acceleration
- **Encoders**: Joint positions
- **Force/Torque**: Contact forces
- **Current sensors**: Motor loads

## LiDAR Simulation

**LiDAR (Light Detection and Ranging)** uses laser pulses to measure distances.

### Types of LiDAR

1. **2D LiDAR**: Single plane scan (e.g., Hokuyo, SICK)
   - Common for mobile robots
   - 180-360° field of view
   - 10-30m range

2. **3D LiDAR**: Multi-plane or spinning (e.g., Velodyne)
   - Point cloud output
   - Full 360° × vertical FOV
   - Used in autonomous vehicles

### Gazebo 2D LiDAR Plugin

Add to your robot's URDF:

```xml
<gazebo reference="laser_link">
  <sensor name="laser_scanner" type="ray">
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
    
    <plugin name="laser_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/robot</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>laser_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

**Key Parameters**:
- `samples`: Number of rays per scan
- `min_angle/max_angle`: Field of view
- `min/max` range: Detection limits
- `stddev`: Noise level

### Processing LiDAR Data

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')
        
        self.subscription = self.create_subscription(
            LaserScan,
            '/robot/scan',
            self.scan_callback,
            10
        )
        
    def scan_callback(self, msg):
        # Extract ranges
        ranges = np.array(msg.ranges)
        
        # Replace inf with max range
        ranges[np.isinf(ranges)] = msg.range_max
        
        # Find minimum distance (closest obstacle)
        min_distance = np.min(ranges)
        min_index = np.argmin(ranges)
        
        # Calculate angle of closest obstacle
        angle = msg.angle_min + min_index * msg.angle_increment
        
        self.get_logger().info(
            f'Closest obstacle: {min_distance:.2f}m at {np.degrees(angle):.1f}°'
        )
        
        # Obstacle detection zones
        front_ranges = ranges[len(ranges)//3:2*len(ranges)//3]
        front_clear = np.min(front_ranges) > 0.5  # 0.5m threshold
        
        if not front_clear:
            self.get_logger().warn('OBSTACLE AHEAD!')

def main():
    rclpy.init()
    processor = LidarProcessor()
    rclpy.spin(processor)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3D LiDAR Simulation

For 3D LiDAR (point clouds):

```xml
<gazebo reference="velodyne_link">
  <sensor name="velodyne" type="gpu_ray">
    <pose>0 0 0 0 0 0</pose>
    <visualize>false</visualize>
    <update_rate>10</update_rate>
    
    <ray>
      <scan>
        <horizontal>
          <samples>1024</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
        <vertical>
          <samples>32</samples>
          <resolution>1</resolution>
          <min_angle>-0.2618</min_angle>
          <max_angle>0.2618</max_angle>
        </vertical>
      </scan>
      
      <range>
        <min>0.9</min>
        <max>100.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    
    <plugin name="velodyne_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <remapping>~/out:=velodyne_points</remapping>
      </ros>
      <output_type>sensor_msgs/PointCloud2</output_type>
      <frame_name>velodyne_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

## Depth Camera (RGB-D) Simulation

**RGB-D cameras** provide both color images and depth information (e.g., Intel RealSense, Kinect).

### Gazebo Depth Camera Plugin

```xml
<gazebo reference="camera_link">
  <sensor name="depth_camera" type="depth">
    <update_rate>30</update_rate>
    <camera>
      <horizontal_fov>1.047198</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.05</near>
        <far>10.0</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    
    <plugin name="depth_camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>/robot</namespace>
        <remapping>~/image_raw:=camera/rgb/image_raw</remapping>
        <remapping>~/depth/image_raw:=camera/depth/image_raw</remapping>
        <remapping>~/camera_info:=camera/rgb/camera_info</remapping>
        <remapping>~/depth/camera_info:=camera/depth/camera_info</remapping>
        <remapping>~/points:=camera/depth/points</remapping>
      </ros>
      
      <camera_name>depth_camera</camera_name>
      <frame_name>camera_link</frame_name>
      <hack_baseline>0.07</hack_baseline>
      <min_depth>0.05</min_depth>
      <max_depth>10.0</max_depth>
    </plugin>
  </sensor>
</gazebo>
```

**Topics Published**:
- `/robot/camera/rgb/image_raw`: Color image
- `/robot/camera/depth/image_raw`: Depth image
- `/robot/camera/depth/points`: Point cloud

### Processing Depth Images

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthImageProcessor(Node):
    def __init__(self):
        super().__init__('depth_processor')
        
        self.bridge = CvBridge()
        
        self.subscription = self.create_subscription(
            Image,
            '/robot/camera/depth/image_raw',
            self.depth_callback,
            10
        )
        
    def depth_callback(self, msg):
        # Convert ROS Image to OpenCV format
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        
        # Replace NaN/Inf with max depth
        depth_image = np.nan_to_num(depth_image, nan=10.0, posinf=10.0)
        
        # Find closest point
        min_depth = np.min(depth_image)
        min_location = np.unravel_index(np.argmin(depth_image), depth_image.shape)
        
        self.get_logger().info(
            f'Closest point: {min_depth:.2f}m at pixel {min_location}'
        )
        
        # Create visualization (normalize for display)
        depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_display = depth_display.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        
        cv2.imshow('Depth Image', depth_colored)
        cv2.waitKey(1)

def main():
    rclpy.init()
    processor = DepthImageProcessor()
    rclpy.spin(processor)
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Converting Depth to Point Cloud

```python
import numpy as np

def depth_to_pointcloud(depth_image, camera_info):
    """
    Convert depth image to 3D point cloud.
    
    Args:
        depth_image: HxW depth array in meters
        camera_info: Camera intrinsic parameters
    
    Returns:
        Nx3 array of 3D points
    """
    height, width = depth_image.shape
    
    # Camera intrinsics
    fx = camera_info.k[0]  # Focal length X
    fy = camera_info.k[4]  # Focal length Y
    cx = camera_info.k[2]  # Principal point X
    cy = camera_info.k[5]  # Principal point Y
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Back-project to 3D
    z = depth_image
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack into Nx3 array
    points = np.stack([x, y, z], axis=-1)
    points = points.reshape(-1, 3)
    
    # Remove invalid points
    valid = ~np.isnan(points).any(axis=1)
    points = points[valid]
    
    return points
```

## IMU (Inertial Measurement Unit) Simulation

**IMU** sensors measure:
- **Acceleration** (3-axis accelerometer)
- **Angular velocity** (3-axis gyroscope)
- **Orientation** (often computed from above via sensor fusion)

### Gazebo IMU Plugin

```xml
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <plugin filename="libgazebo_ros_imu_sensor.so" name="imu_plugin">
      <ros>
        <namespace>/robot</namespace>
        <remapping>~/out:=imu/data</remapping>
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
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

### Processing IMU Data

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np

class IMUProcessor(Node):
    def __init__(self):
        super().__init__('imu_processor')
        
        self.subscription = self.create_subscription(
            Imu,
            '/robot/imu/data',
            self.imu_callback,
            10
        )
        
        # For integration
        self.last_time = None
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.position = np.array([0.0, 0.0, 0.0])
        
    def imu_callback(self, msg):
        # Extract orientation (quaternion)
        orientation = msg.orientation
        
        # Extract angular velocity
        angular_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        
        # Extract linear acceleration
        linear_acc = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        
        # Convert quaternion to Euler angles
        roll, pitch, yaw = self.quaternion_to_euler(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        
        self.get_logger().info(
            f'Orientation: roll={np.degrees(roll):.1f}° '
            f'pitch={np.degrees(pitch):.1f}° '
            f'yaw={np.degrees(yaw):.1f}°'
        )
        
        # Dead reckoning (simple integration - accumulates error!)
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.last_time is not None:
            dt = current_time - self.last_time
            self.velocity += linear_acc * dt
            self.position += self.velocity * dt
        self.last_time = current_time
        
    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

def main():
    rclpy.init()
    processor = IMUProcessor()
    rclpy.spin(processor)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Noise Models

Realistic sensor simulation requires noise models.

### Types of Noise

1. **Gaussian (White) Noise**: Random fluctuations
   ```xml
   <noise type="gaussian">
     <mean>0.0</mean>
     <stddev>0.01</stddev>
   </noise>
   ```

2. **Bias**: Constant offset
   ```xml
   <bias_mean>0.05</bias_mean>
   <bias_stddev>0.001</bias_stddev>
   ```

3. **Drift**: Slow change over time (for IMU)

4. **Quantization**: Discrete measurement steps

### Simulating Realistic Noise

```python
import numpy as np

class SensorNoiseModel:
    def __init__(self, mean=0.0, stddev=0.01, bias=0.0, drift_rate=0.0):
        self.mean = mean
        self.stddev = stddev
        self.bias = bias
        self.drift_rate = drift_rate
        self.current_drift = 0.0
        
    def add_noise(self, true_value, dt=0.01):
        """Add realistic noise to sensor reading."""
        # Gaussian noise
        noise = np.random.normal(self.mean, self.stddev)
        
        # Drift (increases over time)
        self.current_drift += self.drift_rate * dt
        
        # Combined noisy measurement
        measured_value = true_value + noise + self.bias + self.current_drift
        
        return measured_value
    
    def reset_drift(self):
        """Reset drift (e.g., after sensor calibration)."""
        self.current_drift = 0.0

# Example usage
lidar_noise = SensorNoiseModel(mean=0.0, stddev=0.02, bias=0.01)
true_distance = 5.0
measured_distance = lidar_noise.add_noise(true_distance)
```

## Sensor Fusion

**Sensor fusion** combines data from multiple sensors for better estimates.

### Complementary Filter (Simple Fusion)

Combine accelerometer (reliable long-term) with gyroscope (reliable short-term):

```python
class ComplementaryFilter:
    def __init__(self, alpha=0.98):
        self.alpha = alpha  # Trust gyro more (0.98)
        self.angle = 0.0
        
    def update(self, accel_angle, gyro_rate, dt):
        """
        Fuse accelerometer angle with gyroscope rate.
        
        Args:
            accel_angle: Angle from accelerometer (rad)
            gyro_rate: Angular velocity from gyro (rad/s)
            dt: Time step (s)
        """
        # Integrate gyro
        gyro_angle = self.angle + gyro_rate * dt
        
        # Complementary filter
        self.angle = self.alpha * gyro_angle + (1 - self.alpha) * accel_angle
        
        return self.angle

# Usage
filter = ComplementaryFilter(alpha=0.98)

while True:
    accel_angle = get_angle_from_accelerometer()
    gyro_rate = get_gyro_rate()
    dt = 0.01
    
    fused_angle = filter.update(accel_angle, gyro_rate, dt)
```

### Kalman Filter (Optimal Fusion)

For more sophisticated fusion:

```python
import numpy as np

class KalmanFilter1D:
    def __init__(self, process_variance, measurement_variance):
        self.process_var = process_variance
        self.measurement_var = measurement_variance
        self.estimate = 0.0
        self.estimate_error = 1.0
        
    def update(self, measurement):
        """Update estimate with new measurement."""
        # Prediction step
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_var
        
        # Update step
        kalman_gain = prediction_error / (prediction_error + self.measurement_var)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error
        
        return self.estimate

# Example: Fusing noisy position measurements
kf = KalmanFilter1D(process_variance=0.01, measurement_variance=0.1)

for noisy_measurement in sensor_readings:
    filtered_value = kf.update(noisy_measurement)
```

## Visualizing Sensor Data in RViz2

### Viewing LiDAR Scans

```bash
# Launch RViz
rviz2

# Add displays:
# 1. LaserScan - Topic: /robot/scan
# 2. Set Fixed Frame: base_link or laser_link
```

### Viewing Point Clouds

```bash
# In RViz:
# Add -> PointCloud2
# Topic: /robot/camera/depth/points
# Style: Points, Flat Squares, or Spheres
# Size: 0.01
# Color Transformer: AxisColor (Z-axis)
```

### Viewing IMU Data

Create a custom visualization:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class IMUVisualizer(Node):
    def __init__(self):
        super().__init__('imu_visualizer')
        
        self.imu_sub = self.create_subscription(
            Imu, '/robot/imu/data', self.imu_callback, 10
        )
        
        self.marker_pub = self.create_publisher(
            Marker, '/imu_arrow', 10
        )
        
    def imu_callback(self, msg):
        # Create arrow showing acceleration direction
        marker = Marker()
        marker.header = msg.header
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Start point
        marker.points.append(Point(x=0.0, y=0.0, z=0.0))
        
        # End point (scaled by acceleration)
        scale = 0.1
        marker.points.append(Point(
            x=msg.linear_acceleration.x * scale,
            y=msg.linear_acceleration.y * scale,
            z=msg.linear_acceleration.z * scale
        ))
        
        marker.scale.x = 0.05  # Shaft diameter
        marker.scale.y = 0.1   # Head diameter
        marker.color.r = 1.0
        marker.color.a = 1.0
        
        self.marker_pub.publish(marker)

def main():
    rclpy.init()
    visualizer = IMUVisualizer()
    rclpy.spin(visualizer)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Key Takeaways

✅ **LiDAR** provides accurate range measurements for navigation  
✅ **Depth cameras** combine vision with distance sensing  
✅ **IMU** enables orientation tracking and balance control  
✅ **Sensor noise** must be modeled for realistic simulation  
✅ **Sensor fusion** combines multiple sensors for robust estimates  
✅ **RViz2** visualizes sensor data for debugging

## What's Next?

In the final chapter of Module 2, we'll work through hands-on simulation exercises integrating everything we've learned.

Continue to: [Simulation Exercises →](/docs/module2/simulation-exercises)

## Exercises

### Exercise 1: Obstacle Detection
Use LiDAR data to detect obstacles in 3 zones (left, front, right). Publish warnings when obstacles are too close.

### Exercise 2: Depth-Based Grasping
Use depth camera to find the closest object and calculate its 3D position for grasping.

### Exercise 3: IMU Calibration
Record IMU data while robot is stationary. Calculate and remove bias from measurements.

### Exercise 4: Multi-Sensor Localization
Fuse wheel odometry with IMU data using a complementary filter for better position estimates.

### Exercise 5: Point Cloud Processing
Convert depth image to point cloud. Segment the ground plane and identify objects above it.

---

**Navigation**: [← Previous: Unity Visualization](/docs/module2/unity-visualization) | [Next: Simulation Exercises →](/docs/module2/simulation-exercises)