# Isaac ROS - Hardware-Accelerated Perception

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand Isaac ROS GEMs and their benefits
- Install and configure Isaac ROS packages
- Implement Visual SLAM (VSLAM) for localization
- Use hardware-accelerated object detection
- Perform pose estimation for manipulation
- Deploy Isaac ROS on NVIDIA Jetson devices
- Integrate Isaac ROS with your robot stack

## What is Isaac ROS?

**Isaac ROS** is NVIDIA's collection of hardware-accelerated ROS 2 packages. These packages leverage GPU acceleration to provide real-time performance for computationally intensive robotics tasks.

### Key Advantages

Traditional ROS packages run on CPU, which limits performance for:
- Visual SLAM (CPU: 5-10 FPS → GPU: 30+ FPS)
- Object detection (CPU: 1-2 FPS → GPU: 30+ FPS)
- Image processing pipelines

Isaac ROS packages use:
- **CUDA**: Parallel GPU computation
- **TensorRT**: Optimized deep learning inference
- **VPI (Vision Programming Interface)**: Accelerated computer vision
- **Triton**: AI model serving

### Isaac ROS GEMs

**GEM (Graph-Enabled Modules)** are reusable, hardware-accelerated components:

| GEM | Function | Acceleration |
|-----|----------|--------------|
| **Visual SLAM** | Localization & mapping | GPU + VPI |
| **DNN Inference** | Object detection, segmentation | TensorRT |
| **Stereo Disparity** | Depth from stereo | VPI |
| **Image Processing** | Rectification, encoding | CUDA/VPI |
| **AprilTag** | Fiducial marker detection | GPU |
| **Pose Estimation** | 6D object pose | TensorRT |

## System Requirements

### Hardware

- **GPU**: NVIDIA GPU with Compute Capability 7.0+
  - Desktop: RTX 2060 or higher
  - Jetson: Orin, Xavier (AGX/NX), or Nano
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 20GB for Isaac ROS packages

### Software

- **OS**: Ubuntu 20.04 or 22.04
- **ROS**: ROS 2 Humble Hawksbill
- **CUDA**: 11.4 or later
- **TensorRT**: 8.5 or later (included with JetPack on Jetson)

## Installing Isaac ROS

### Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade

# Install Git LFS (for large model files)
sudo apt install git-lfs
git lfs install

# Install dependencies
sudo apt install python3-pip
pip3 install -U jetson-stats  # For Jetson devices only
```

### Docker-Based Installation (Recommended)

Isaac ROS uses Docker for easy deployment:

```bash
# Clone Isaac ROS Common
cd ~/workspaces
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git

# Build Docker image
cd isaac_ros_common
./scripts/run_dev.sh
```

This creates a Docker container with:
- ROS 2 Humble
- CUDA and TensorRT
- All Isaac ROS dependencies

### Native Installation (Advanced)

For native installation without Docker:

```bash
# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Clone Isaac ROS packages (example: Visual SLAM)
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git

# Install dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build --symlink-install

# Source
source install/setup.bash
```

## Visual SLAM with Isaac ROS

**Visual SLAM (VSLAM)** uses camera images to localize the robot and build a map simultaneously.

### Why Hardware-Accelerated VSLAM?

Traditional VSLAM (like ORB-SLAM) runs on CPU:
- 5-10 FPS on desktop
- Less than 5 FPS on Jetson

Isaac ROS VSLAM runs on GPU:
- 30+ FPS on desktop
- 15-30 FPS on Jetson Orin

### Installing Isaac ROS Visual SLAM

```bash
# Inside Docker container or native workspace
cd ~/ros2_ws/src
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git

cd ~/ros2_ws
rosdep install -i -r --from-paths src --rosdistro humble -y

colcon build --packages-select isaac_ros_visual_slam
source install/setup.bash
```

### Running Visual SLAM

**With RealSense Camera:**

```bash
# Terminal 1: Start camera
ros2 launch realsense2_camera rs_launch.py

# Terminal 2: Start Visual SLAM
ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_realsense.launch.py
```

**With Recorded Data (ROS bag):**

```bash
ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam.launch.py
```

### VSLAM Output Topics

Isaac ROS Visual SLAM publishes:

- `/visual_slam/tracking/odometry`: Robot pose estimate
- `/visual_slam/tracking/vo_pose`: Visual odometry pose
- `/visual_slam/vis/landmarks_cloud`: 3D map points
- `/visual_slam/vis/loop_closure_cloud`: Loop closure points

### Visualizing in RViz

```bash
rviz2 -d $(ros2 pkg prefix isaac_ros_visual_slam)/share/isaac_ros_visual_slam/rviz/default.rviz
```

Add displays:
- **Odometry**: `/visual_slam/tracking/odometry`
- **PointCloud2**: `/visual_slam/vis/landmarks_cloud`
- **TF**: Shows coordinate frames

### VSLAM Configuration

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='isaac_ros_visual_slam',
            executable='isaac_ros_visual_slam',
            name='visual_slam',
            parameters=[{
                'denoise_input_images': False,
                'rectified_images': True,
                'enable_debug_mode': False,
                'debug_dump_path': '/tmp/cuvslam',
                'enable_slam_visualization': True,
                'enable_landmarks_view': True,
                'enable_observations_view': True,
                'map_frame': 'map',
                'odom_frame': 'odom',
                'base_frame': 'base_link',
                'input_imu_frame': 'imu_link',
                'enable_imu_fusion': True,
            }],
            remappings=[
                ('stereo_camera/left/image', '/camera/infra1/image_rect_raw'),
                ('stereo_camera/left/camera_info', '/camera/infra1/camera_info'),
                ('stereo_camera/right/image', '/camera/infra2/image_rect_raw'),
                ('stereo_camera/right/camera_info', '/camera/infra2/camera_info'),
            ]
        )
    ])
```

## Object Detection with Isaac ROS DNN Inference

Isaac ROS provides GPU-accelerated object detection.

### Supported Models

- **YOLOv5**: Fast, accurate object detection
- **DOPE**: 6D pose estimation
- **PeopleSemSegNet**: Person segmentation
- **Custom TensorRT models**

### Installing Isaac ROS DNN Inference

```bash
cd ~/ros2_ws/src
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline.git

cd ~/ros2_ws
rosdep install -i -r --from-paths src --rosdistro humble -y
colcon build
source install/setup.bash
```

### Running Object Detection

```bash
# Download pre-trained YOLOv5 model (TensorRT engine)
# Models available at: https://catalog.ngc.nvidia.com/

# Launch object detection
ros2 launch isaac_ros_yolov5 isaac_ros_yolov5.launch.py model_file_path:=/path/to/yolov5.engine
```

### Detection Node Example

```python
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray

class ObjectDetectionSubscriber(Node):
    def __init__(self):
        super().__init__('detection_subscriber')
        
        self.subscription = self.create_subscription(
            Detection2DArray,
            '/detections_output',
            self.detection_callback,
            10
        )
        
    def detection_callback(self, msg):
        self.get_logger().info(f'Detected {len(msg.detections)} objects')
        
        for detection in msg.detections:
            # Get bounding box
            bbox = detection.bbox
            center_x = bbox.center.position.x
            center_y = bbox.center.position.y
            size_x = bbox.size_x
            size_y = bbox.size_y
            
            # Get class and confidence
            if detection.results:
                result = detection.results[0]
                class_id = result.hypothesis.class_id
                score = result.hypothesis.score
                
                self.get_logger().info(
                    f'Class: {class_id}, Confidence: {score:.2f}, '
                    f'Position: ({center_x:.0f}, {center_y:.0f})'
                )

def main():
    rclpy.init()
    subscriber = ObjectDetectionSubscriber()
    rclpy.spin(subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Stereo Depth with Isaac ROS

Isaac ROS provides hardware-accelerated stereo depth estimation.

### Installing Stereo Depth

```bash
cd ~/ros2_ws/src
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline.git

cd ~/ros2_ws
colcon build --packages-select isaac_ros_stereo_image_proc
source install/setup.bash
```

### Running Stereo Depth

```bash
ros2 launch isaac_ros_stereo_image_proc isaac_ros_stereo_image_pipeline_realsense.launch.py
```

**Output Topics:**
- `/disparity`: Disparity image
- `/depth`: Depth image
- `/points2`: Point cloud

### Using Depth Data

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class DepthProcessor(Node):
    def __init__(self):
        super().__init__('depth_processor')
        self.bridge = CvBridge()
        
        self.subscription = self.create_subscription(
            Image,
            '/depth',
            self.depth_callback,
            10
        )
        
    def depth_callback(self, msg):
        # Convert to numpy array
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        
        # Find closest point
        valid_depths = depth_image[depth_image > 0]
        if len(valid_depths) > 0:
            min_depth = np.min(valid_depths)
            self.get_logger().info(f'Closest obstacle: {min_depth:.2f}m')
            
            if min_depth < 0.5:
                self.get_logger().warn('OBSTACLE TOO CLOSE!')

def main():
    rclpy.init()
    processor = DepthProcessor()
    rclpy.spin(processor)
    rclpy.shutdown()
```

## AprilTag Detection

AprilTags are fiducial markers used for localization and object tracking.

### Installing Isaac ROS AprilTag

```bash
cd ~/ros2_ws/src
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag.git

cd ~/ros2_ws
colcon build --packages-select isaac_ros_apriltag
source install/setup.bash
```

### Running AprilTag Detection

```bash
ros2 launch isaac_ros_apriltag isaac_ros_apriltag.launch.py
```

### AprilTag Use Cases

1. **Robot Localization**: Tags at known positions
2. **Object Tracking**: Tags on objects
3. **Calibration**: Camera-robot calibration
4. **Landing Pads**: Drone landing

```python
import rclpy
from rclpy.node import Node
from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray

class AprilTagTracker(Node):
    def __init__(self):
        super().__init__('apriltag_tracker')
        
        self.subscription = self.create_subscription(
            AprilTagDetectionArray,
            '/tag_detections',
            self.detection_callback,
            10
        )
        
    def detection_callback(self, msg):
        for detection in msg.detections:
            tag_id = detection.id
            pose = detection.pose.pose.pose
            
            position = pose.position
            orientation = pose.orientation
            
            self.get_logger().info(
                f'Tag {tag_id} at ({position.x:.2f}, {position.y:.2f}, {position.z:.2f})'
            )

def main():
    rclpy.init()
    tracker = AprilTagTracker()
    rclpy.spin(tracker)
    rclpy.shutdown()
```

## Deploying on NVIDIA Jetson

Isaac ROS is optimized for NVIDIA Jetson edge devices.

### Jetson Setup

1. **Flash JetPack 5.1+** (includes CUDA, TensorRT)
2. **Install ROS 2 Humble**
3. **Clone Isaac ROS packages**
4. **Build and run**

### Performance on Jetson

| Task | Jetson Orin | Jetson Xavier | Jetson Nano |
|------|-------------|---------------|-------------|
| **Visual SLAM** | 30 FPS | 20 FPS | 10 FPS |
| **Object Detection** | 30 FPS | 20 FPS | 5 FPS |
| **Stereo Depth** | 30 FPS | 15 FPS | Less than 5 FPS |

### Power Management

```bash
# Set to max performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor stats
sudo tegrastats
```

## Integration Example: Complete Perception Pipeline

Combine multiple Isaac ROS GEMs:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Visual SLAM for localization
        Node(
            package='isaac_ros_visual_slam',
            executable='isaac_ros_visual_slam',
            name='visual_slam'
        ),
        
        # Object detection
        Node(
            package='isaac_ros_yolov5',
            executable='isaac_ros_yolov5',
            name='object_detector',
            parameters=[{
                'model_file_path': '/models/yolov5.engine',
                'confidence_threshold': 0.5
            }]
        ),
        
        # Stereo depth
        Node(
            package='isaac_ros_stereo_image_proc',
            executable='disparity_node',
            name='stereo_depth'
        ),
        
        # AprilTag detection
        Node(
            package='isaac_ros_apriltag',
            executable='isaac_ros_apriltag',
            name='apriltag_detector'
        ),
    ])
```

## Performance Benchmarking

### Measuring FPS

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import time

class FPSCounter(Node):
    def __init__(self):
        super().__init__('fps_counter')
        
        self.frame_count = 0
        self.start_time = time.time()
        
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.timer = self.create_timer(1.0, self.print_fps)
        
    def image_callback(self, msg):
        self.frame_count += 1
        
    def print_fps(self):
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed
        
        self.get_logger().info(f'FPS: {fps:.2f}')
        
        # Reset counter every 10 seconds
        if elapsed > 10.0:
            self.frame_count = 0
            self.start_time = time.time()

def main():
    rclpy.init()
    counter = FPSCounter()
    rclpy.spin(counter)
    rclpy.shutdown()
```

## Troubleshooting

### Common Issues

**Issue 1: GPU not detected**
```bash
# Check CUDA
nvcc --version

# Check GPU
nvidia-smi
```

**Issue 2: TensorRT errors**
```bash
# Rebuild TensorRT engine
trtexec --onnx=model.onnx --saveEngine=model.engine
```

**Issue 3: Low FPS**
- Reduce image resolution
- Lower detection confidence threshold
- Disable visualization
- Check CPU/GPU usage with `htop` and `nvidia-smi`

## Key Takeaways

✅ **Isaac ROS** provides GPU-accelerated perception packages  
✅ **Visual SLAM** achieves 30+ FPS with GPU acceleration  
✅ **Object detection** runs real-time with TensorRT  
✅ **Stereo depth** and **AprilTag** detection are hardware-accelerated  
✅ **Optimized for Jetson** edge devices  
✅ **Plug-and-play** with ROS 2 ecosystem

## What's Next?

In the next chapter, we'll explore **Nav2** - the navigation framework for autonomous mobile robots, including path planning and obstacle avoidance.

Continue to: [Nav2 Navigation →](/docs/module3/nav2-navigation)

## Exercises

### Exercise 1: Run Visual SLAM
Set up Isaac ROS Visual SLAM with a RealSense camera. Map your environment.

### Exercise 2: Object Detection Pipeline
Implement YOLOv5 object detection. Detect and track objects in real-time.

### Exercise 3: AprilTag Localization
Place AprilTags in your environment. Use them to localize your robot.

### Exercise 4: FPS Benchmark
Compare FPS of CPU-based vs Isaac ROS-based object detection.

### Exercise 5: Jetson Deployment
Deploy an Isaac ROS package to a Jetson device. Measure performance.

---

**Navigation**: [← Previous: Isaac Sim](/docs/module3/isaac-sim) | [Next: Nav2 Navigation →](/docs/module3/nav2-navigation)