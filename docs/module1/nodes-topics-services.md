# Nodes, Topics, and Services

## Learning Objectives

By the end of this chapter, you will be able to:

- Create custom ROS 2 messages for robot-specific data
- Implement multi-publisher and multi-subscriber systems
- Build service servers and clients for request-response patterns
- Understand and configure Quality of Service (QoS) settings
- Use ROS 2 parameters to configure nodes
- Debug ROS 2 systems using command-line tools

## Deep Dive into ROS 2 Nodes

In the previous chapter, we created simple publisher and subscriber nodes. Now let's explore nodes in greater depth and understand how they form the building blocks of complex robotic systems.

### Node Lifecycle

Every ROS 2 node goes through a lifecycle:

1. **Creation**: Node is instantiated with `rclpy.init()`
2. **Configuration**: Publishers, subscribers, timers are set up
3. **Activation**: Node begins processing (with `rclpy.spin()`)
4. **Execution**: Callbacks are invoked as events occur
5. **Shutdown**: Node cleans up resources with `destroy_node()` and `rclpy.shutdown()`

### Node Naming and Namespaces

Node names must be **unique** within a ROS 2 system. You can organize nodes using **namespaces**:

```python
# Create a node with a namespace
node = Node('camera', namespace='front')
# Full name: /front/camera

node = Node('camera', namespace='back')
# Full name: /back/camera
```

This is crucial for multi-robot systems where you might have multiple identical robots running the same nodes.

### Node Parameters

**Parameters** allow you to configure nodes without changing code. Think of them as runtime configuration values.

```python
class ConfigurableNode(Node):
    def __init__(self):
        super().__init__('configurable_node')
        
        # Declare parameters with default values
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('message_prefix', 'Hello')
        
        # Get parameter values
        rate = self.get_parameter('publish_rate').value
        prefix = self.get_parameter('message_prefix').value
        
        self.get_logger().info(f'Rate: {rate}, Prefix: {prefix}')
```

Set parameters from command line:

```bash
ros2 run my_package configurable_node --ros-args -p publish_rate:=5.0 -p message_prefix:="Hi"
```

Or view/set parameters while running:

```bash
# List parameters
ros2 param list

# Get a parameter value
ros2 param get /configurable_node publish_rate

# Set a parameter value
ros2 param set /configurable_node publish_rate 20.0
```

## Advanced Publisher-Subscriber Patterns

Let's explore more sophisticated pub/sub scenarios common in robotics.

### Multiple Publishers, One Subscriber

Multiple nodes can publish to the same topic. The subscriber receives messages from all publishers.

**Use case**: Multiple cameras publishing to `/camera/images`, one perception node subscribing.

```python
# Publisher 1
class CameraLeft(Node):
    def __init__(self):
        super().__init__('camera_left')
        self.pub = self.create_publisher(Image, '/camera/images', 10)

# Publisher 2
class CameraRight(Node):
    def __init__(self):
        super().__init__('camera_right')
        self.pub = self.create_publisher(Image, '/camera/images', 10)

# Subscriber receives from both
class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception')
        self.sub = self.create_subscription(
            Image, '/camera/images', self.callback, 10
        )
```

### One Publisher, Multiple Subscribers

One publisher can serve data to many subscribers.

**Use case**: A LiDAR node publishing scan data, with multiple nodes (mapping, obstacle avoidance, localization) subscribing.

```python
# One publisher
class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar')
        self.pub = self.create_publisher(LaserScan, '/scan', 10)

# Multiple subscribers
class MappingNode(Node):
    def __init__(self):
        super().__init__('mapping')
        self.sub = self.create_subscription(LaserScan, '/scan', self.map_callback, 10)

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        self.sub = self.create_subscription(LaserScan, '/scan', self.detect_callback, 10)
```

### Topic Remapping

You can remap topic names at runtime without changing code:

```bash
# Run node but remap '/camera/image' to '/front_camera/image'
ros2 run my_package camera_node --ros-args -r camera/image:=front_camera/image
```

This is essential for reusing code across different robot configurations.

## Custom Messages

While ROS 2 provides standard messages (`std_msgs`, `sensor_msgs`, etc.), you'll often need **custom messages** for your specific robot.

### Creating a Custom Message

Let's create a message for a humanoid robot's joint state:

**Step 1**: Create a message definition file `HumanoidJoint.msg`:

```
# HumanoidJoint.msg - Custom message for a single joint

string joint_name          # Name of the joint (e.g., "left_knee")
float64 position           # Joint angle in radians
float64 velocity           # Joint angular velocity in rad/s
float64 effort             # Torque applied to joint in Nm
float64 temperature        # Motor temperature in Celsius
bool is_moving             # Whether joint is currently moving
```

**Step 2**: Update `CMakeLists.txt` to build the message:

```cmake
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/HumanoidJoint.msg"
)
```

**Step 3**: Update `package.xml`:

```xml
<build_depend>rosidl_default_generators</build_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```

**Step 4**: Build and use:

```bash
colcon build
source install/setup.bash
```

```python
from my_package.msg import HumanoidJoint

class JointPublisher(Node):
    def __init__(self):
        super().__init__('joint_publisher')
        self.pub = self.create_publisher(HumanoidJoint, 'joint_state', 10)
        
    def publish_joint_state(self):
        msg = HumanoidJoint()
        msg.joint_name = 'left_knee'
        msg.position = 1.57  # 90 degrees
        msg.velocity = 0.1
        msg.effort = 5.2
        msg.temperature = 42.5
        msg.is_moving = True
        
        self.pub.publish(msg)
```

### Complex Message Types

Messages can contain arrays, nested messages, and more:

```
# RobotStatus.msg - Complex message example

Header header                           # Standard header with timestamp
string robot_id                         # Unique robot identifier
float64 battery_percentage              # 0-100
geometry_msgs/Pose pose                 # Current robot pose (nested message)
HumanoidJoint[] joint_states            # Array of joint states
uint8 error_code                        # Error status (0 = no error)
string[] active_behaviors               # List of running behaviors
```

## Services: Request-Response Communication

While topics are great for streaming data, **services** provide synchronous request-response communication. Use services when:

- You need immediate feedback
- The operation is quick (< 1 second typically)
- You want to ensure the request was received

### Creating a Service

**Step 1**: Define the service in `AddTwoInts.srv`:

```
# Request
int64 a
int64 b
---
# Response
int64 sum
```

The `---` separates request from response.

**Step 2**: Build the service interface (same as messages).

### Service Server Example

```python
from example_interfaces.srv import AddTwoInts

class AddTwoIntsServer(Node):
    def __init__(self):
        super().__init__('add_two_ints_server')
        
        # Create service
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_callback
        )
        
        self.get_logger().info('Add Two Ints Service ready')
    
    def add_callback(self, request, response):
        """
        Called when a client makes a request.
        
        Args:
            request: AddTwoInts.Request with 'a' and 'b'
            response: AddTwoInts.Response to fill with 'sum'
        """
        response.sum = request.a + request.b
        
        self.get_logger().info(
            f'Request: {request.a} + {request.b} = {response.sum}'
        )
        
        return response
```

### Service Client Example

```python
from example_interfaces.srv import AddTwoInts

class AddTwoIntsClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')
        
        # Create client
        self.client = self.create_client(AddTwoInts, 'add_two_ints')
        
        # Wait for service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service...')
        
        self.get_logger().info('Service available!')
    
    def send_request(self, a, b):
        """
        Send a request to add two integers.
        """
        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        
        # Call service asynchronously
        future = self.client.call_async(request)
        
        return future

def main():
    rclpy.init()
    
    client_node = AddTwoIntsClient()
    
    # Send request
    future = client_node.send_request(5, 7)
    
    # Wait for response
    rclpy.spin_until_future_complete(client_node, future)
    
    if future.result() is not None:
        response = future.result()
        client_node.get_logger().info(f'Result: {response.sum}')
    else:
        client_node.get_logger().error('Service call failed')
    
    client_node.destroy_node()
    rclpy.shutdown()
```

### Robotics Service Examples

Common services in robotics:

- **GetRobotPose**: Request current robot position
- **SetMotorSpeed**: Command motor to specific speed
- **CalibrateCamera**: Trigger camera calibration
- **EmergencyStop**: Immediately halt all motion
- **LoadMap**: Load a new map for navigation

## Quality of Service (QoS)

**Quality of Service** lets you tune communication reliability, latency, and resource usage. This is critical for real-time robotics.

### QoS Profiles

ROS 2 provides several preset QoS profiles:

| Profile | Reliability | Durability | History | Use Case |
|---------|-------------|------------|---------|----------|
| **Default** | Reliable | Volatile | Keep last 10 | General purpose |
| **Sensor Data** | Best effort | Volatile | Keep last 5 | High-frequency sensor streams |
| **Parameters** | Reliable | Volatile | Keep last 1000 | Parameter updates |
| **System Default** | Reliable | Volatile | Keep last 10 | Most topics |
| **Services** | Reliable | Volatile | Keep last 10 | Service calls |

### QoS Settings Explained

**Reliability**:
- **Reliable**: Guarantees message delivery (TCP-like)
- **Best Effort**: May drop messages (UDP-like, faster)

**Durability**:
- **Volatile**: Late subscribers don't get old messages
- **Transient Local**: Late subscribers get recent messages

**History**:
- **Keep Last N**: Keep only the N most recent messages
- **Keep All**: Keep all messages (until resource limits)

### Using QoS in Code

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')
        
        # Custom QoS for high-frequency sensor data
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Don't wait for acks
            history=HistoryPolicy.KEEP_LAST,
            depth=5  # Keep only last 5 messages
        )
        
        self.pub = self.create_publisher(
            LaserScan,
            'scan',
            qos  # Use custom QoS
        )
```

### When to Use Which QoS

**Best Effort** (fast, may drop):
- Camera images (60 FPS - a few dropped frames OK)
- LiDAR scans (10 Hz - missing one scan tolerable)
- IMU data (200 Hz - some loss acceptable)

**Reliable** (guaranteed delivery):
- Robot commands (must not lose "STOP" command!)
- Goal positions
- State transitions
- Configuration updates

## Multi-Node Communication Example

Let's build a complete system with multiple nodes communicating:

**Scenario**: A simple humanoid robot monitoring system with three nodes:
1. **SensorNode**: Publishes temperature and battery data
2. **MonitorNode**: Subscribes to sensor data, checks for issues
3. **AlertService**: Provides a service to trigger alerts

### Node 1: SensorNode

```python
from std_msgs.msg import Float32
import random

class SensorNode(Node):
    def __init__(self):
        super().__init__('sensor_node')
        
        # Publishers
        self.temp_pub = self.create_publisher(Float32, 'robot/temperature', 10)
        self.battery_pub = self.create_publisher(Float32, 'robot/battery', 10)
        
        # Timer: publish every 2 seconds
        self.timer = self.create_timer(2.0, self.publish_sensors)
        
        self.get_logger().info('Sensor Node started')
    
    def publish_sensors(self):
        # Simulate sensor readings
        temp = Float32()
        temp.data = 35.0 + random.uniform(-5, 15)  # 30-50°C
        
        battery = Float32()
        battery.data = max(0, random.uniform(20, 100))  # 20-100%
        
        self.temp_pub.publish(temp)
        self.battery_pub.publish(battery)
        
        self.get_logger().info(f'Temp: {temp.data:.1f}°C, Battery: {battery.data:.1f}%')
```

### Node 2: MonitorNode

```python
from std_msgs.msg import Float32, String
from example_interfaces.srv import Trigger

class MonitorNode(Node):
    def __init__(self):
        super().__init__('monitor_node')
        
        # Subscribers
        self.temp_sub = self.create_subscription(
            Float32, 'robot/temperature', self.temp_callback, 10
        )
        self.battery_sub = self.create_subscription(
            Float32, 'robot/battery', self.battery_callback, 10
        )
        
        # Service client for alerts
        self.alert_client = self.create_client(Trigger, 'send_alert')
        
        # Thresholds
        self.temp_threshold = 45.0  # °C
        self.battery_threshold = 30.0  # %
        
        self.get_logger().info('Monitor Node started')
    
    def temp_callback(self, msg):
        if msg.data > self.temp_threshold:
            self.get_logger().warn(f'HIGH TEMPERATURE: {msg.data:.1f}°C')
            self.trigger_alert(f'Temperature critical: {msg.data:.1f}°C')
    
    def battery_callback(self, msg):
        if msg.data < self.battery_threshold:
            self.get_logger().warn(f'LOW BATTERY: {msg.data:.1f}%')
            self.trigger_alert(f'Battery low: {msg.data:.1f}%')
    
    def trigger_alert(self, message):
        if self.alert_client.wait_for_service(timeout_sec=1.0):
            request = Trigger.Request()
            future = self.alert_client.call_async(request)
            # Note: In production, you'd handle the response
```

### Node 3: AlertService

```python
from example_interfaces.srv import Trigger

class AlertService(Node):
    def __init__(self):
        super().__init__('alert_service')
        
        # Service server
        self.srv = self.create_service(
            Trigger,
            'send_alert',
            self.alert_callback
        )
        
        self.get_logger().info('Alert Service ready')
    
    def alert_callback(self, request, response):
        # In a real system, this might send email, SMS, or dashboard notification
        self.get_logger().error('🚨 ALERT TRIGGERED! 🚨')
        
        response.success = True
        response.message = 'Alert sent successfully'
        
        return response
```

### Running the System

```bash
# Terminal 1
ros2 run robot_monitor sensor_node

# Terminal 2
ros2 run robot_monitor monitor_node

# Terminal 3
ros2 run robot_monitor alert_service
```

Watch as the monitor detects issues and triggers alerts via the service!

## ROS 2 Debugging Tools

Master these tools for efficient debugging:

### 1. Topic Introspection

```bash
# List all topics
ros2 topic list

# Show topic info (publishers, subscribers)
ros2 topic info /robot/temperature

# Display messages in real-time
ros2 topic echo /robot/temperature

# Measure publishing rate
ros2 topic hz /robot/temperature

# Publish a test message manually
ros2 topic pub /robot/temperature std_msgs/Float32 "data: 42.0"
```

### 2. Node Introspection

```bash
# List all nodes
ros2 node list

# Show node details
ros2 node info /sensor_node

# Kill a node
ros2 lifecycle set /sensor_node shutdown
```

### 3. Service Tools

```bash
# List all services
ros2 service list

# Show service type
ros2 service type /send_alert

# Call a service from command line
ros2 service call /send_alert example_interfaces/srv/Trigger
```

### 4. Parameter Tools

```bash
# List parameters for a node
ros2 param list /sensor_node

# Get parameter value
ros2 param get /sensor_node publish_rate

# Set parameter value
ros2 param set /sensor_node publish_rate 5.0

# Save parameters to file
ros2 param dump /sensor_node > params.yaml

# Load parameters from file
ros2 param load /sensor_node params.yaml
```

### 5. Recording and Playback

**rosbag2** records all topic data for later analysis:

```bash
# Record specific topics
ros2 bag record /robot/temperature /robot/battery

# Record all topics
ros2 bag record -a

# Play back recorded data
ros2 bag play my_recording

# Show bag info
ros2 bag info my_recording
```

This is invaluable for:
- Debugging issues that occurred in the field
- Testing perception algorithms with real sensor data
- Creating test datasets

## Best Practices

### 1. Node Design
- **Single Responsibility**: Each node should do one thing well
- **Small Nodes**: Easier to debug and reuse
- **Clear Names**: Use descriptive node and topic names

### 2. Topic Naming
- Use namespaces: `/robot1/camera/image` not `/camera_image_1`
- Be consistent: Always `snake_case` for topics
- Group related topics: `/sensors/lidar`, `/sensors/camera`

### 3. Error Handling
- Always check service availability
- Handle QoS mismatches gracefully
- Log errors with appropriate severity

### 4. Performance
- Use appropriate QoS for each topic
- Don't publish faster than subscribers can consume
- Monitor CPU and network usage

## Key Takeaways

✅ **Custom messages** allow robot-specific data structures  
✅ **Services** provide request-response communication  
✅ **QoS profiles** tune reliability vs. performance  
✅ **Parameters** enable runtime configuration  
✅ **ROS 2 CLI tools** are essential for debugging  
✅ **Multiple nodes** work together via topics and services

## What's Next?

In the next chapter, we'll learn how to model humanoid robots using **URDF (Unified Robot Description Format)** and visualize them in RViz2.

Continue to: [URDF for Humanoids →](/docs/module1/urdf-humanoids)

## Exercises

### Exercise 1: Custom Message
Create a custom message `RobotPose` with fields for x, y, z position and roll, pitch, yaw orientation.

### Exercise 2: Temperature Monitor
Build a service that returns "OK" if temperature is below 50°C, "WARNING" if 50-60°C, and "CRITICAL" above 60°C.

### Exercise 3: QoS Experiment
Create two subscribers to the same topic - one with RELIABLE and one with BEST_EFFORT QoS. Publish 1000 messages rapidly. How many does each receive?

### Exercise 4: Multi-Robot System
Modify the sensor example to support two robots using namespaces: `/robot1/` and `/robot2/`.

### Exercise 5: Data Logger
Create a node that subscribes to multiple topics and logs all data to a CSV file.

---

**Navigation**: [← Previous: Introduction to ROS 2](/docs/module1/intro-ros2) | [Next: URDF for Humanoids →](/docs/module1/urdf-humanoids)