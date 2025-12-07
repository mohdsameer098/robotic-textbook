# ROS 2 Exercises

## Overview

Congratulations on completing Module 1! You've learned the fundamentals of ROS 2, including nodes, topics, services, and URDF modeling. Now it's time to solidify your understanding through hands-on exercises.

This chapter contains:
- **Hands-on coding exercises** (beginner to advanced)
- **Mini-projects** that integrate multiple concepts
- **Troubleshooting scenarios**
- **Assessment questions** to test your knowledge

## Exercise Set 1: Basic Node Communication

### Exercise 1.1: Temperature Monitor

**Difficulty**: Beginner  
**Time**: 30 minutes  
**Concepts**: Publisher, Subscriber, Basic Python

**Task**: Create a system that monitors temperature and triggers warnings.

**Requirements**:
1. Create a `temperature_sensor` node that:
   - Publishes random temperature values (20-80°C) to `/temperature` topic
   - Publishes once per second
   - Uses `std_msgs/Float32` message type

2. Create a `temperature_monitor` node that:
   - Subscribes to `/temperature` topic
   - Prints "NORMAL" if temperature < 50°C
   - Prints "WARNING" if temperature between 50-70°C
   - Prints "CRITICAL" if temperature > 70°C

**Starter Code**:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import random

class TemperatureSensor(Node):
    def __init__(self):
        super().__init__('temperature_sensor')
        # TODO: Create publisher
        # TODO: Create timer
        
    def publish_temperature(self):
        # TODO: Generate random temperature
        # TODO: Publish message
        pass

# TODO: Create TemperatureMonitor class
```

**Expected Output**:
```
[temperature_sensor]: Publishing: 45.3°C
[temperature_monitor]: NORMAL

[temperature_sensor]: Publishing: 67.8°C
[temperature_monitor]: WARNING

[temperature_sensor]: Publishing: 75.2°C
[temperature_monitor]: CRITICAL
```

---

### Exercise 1.2: Multi-Sensor Fusion

**Difficulty**: Intermediate  
**Time**: 45 minutes  
**Concepts**: Multiple publishers/subscribers, Data processing

**Task**: Combine data from multiple sensors to estimate robot position.

**Requirements**:
1. Create three sensor nodes:
   - `gps_sensor`: Publishes x, y coordinates (with noise)
   - `imu_sensor`: Publishes heading angle (with noise)
   - `wheel_odometry`: Publishes velocity

2. Create a `sensor_fusion` node that:
   - Subscribes to all three sensors
   - Combines data to estimate robot pose
   - Publishes fused estimate to `/robot/pose`

**Hints**:
- Use `geometry_msgs/Pose2D` for position
- Apply simple averaging to reduce noise
- Consider timestamp synchronization

---

### Exercise 1.3: Robot State Machine

**Difficulty**: Intermediate  
**Time**: 1 hour  
**Concepts**: State machines, Multiple topics

**Task**: Implement a robot behavior state machine.

**States**:
1. **IDLE**: Waiting for commands
2. **MOVING**: Executing motion
3. **CHARGING**: Battery low, seeking charger
4. **ERROR**: Something went wrong

**Requirements**:
- Subscribe to `/battery_level` (Float32, 0-100%)
- Subscribe to `/obstacle_detected` (Bool)
- Publish current state to `/robot/state` (String)
- Publish velocity commands to `/cmd_vel` based on state

**State Transitions**:
- IDLE → MOVING when command received
- Any state → CHARGING when battery < 20%
- MOVING → ERROR when obstacle detected
- CHARGING → IDLE when battery > 80%

---

## Exercise Set 2: Services and Parameters

### Exercise 2.1: Calculator Service

**Difficulty**: Beginner  
**Time**: 30 minutes  
**Concepts**: Service server and client

**Task**: Create a math service for robot computations.

**Requirements**:
1. Define a service `RobotMath.srv`:
```
string operation  # "add", "subtract", "multiply", "divide"
float64 a
float64 b
---
float64 result
bool success
string message
```

2. Create `math_server` node that:
   - Provides the service on `/robot/calculate`
   - Handles all four operations
   - Returns error if division by zero

3. Create `math_client` node that:
   - Calls the service with test values
   - Prints results

**Test Cases**:
- 10 + 5 = 15
- 10 - 5 = 5
- 10 * 5 = 50
- 10 / 5 = 2
- 10 / 0 = ERROR

---

### Exercise 2.2: Dynamic Configuration

**Difficulty**: Intermediate  
**Time**: 45 minutes  
**Concepts**: Parameters, Runtime configuration

**Task**: Create a configurable robot controller.

**Requirements**:
1. Create a `motor_controller` node with parameters:
   - `max_speed` (default: 1.0 m/s)
   - `acceleration` (default: 0.5 m/s²)
   - `safety_distance` (default: 0.3 m)
   - `control_mode` (default: "velocity")

2. Node behavior:
   - Read parameters on startup
   - Allow runtime parameter changes
   - Log when parameters change
   - Validate parameter ranges

3. Test changing parameters while node is running:
```bash
ros2 param set /motor_controller max_speed 2.0
```

---

### Exercise 2.3: Remote Robot Control

**Difficulty**: Advanced  
**Time**: 1.5 hours  
**Concepts**: Services, Topics, Error handling

**Task**: Build a complete remote control system.

**Requirements**:
1. **Command Server Node**:
   - Service: `/robot/execute_command`
   - Commands: "move_forward", "turn_left", "turn_right", "stop"
   - Returns success/failure

2. **Status Publisher Node**:
   - Publishes robot status every 0.5 seconds
   - Status includes: position, velocity, battery, state

3. **Remote Client Node**:
   - Interactive CLI for sending commands
   - Displays robot status in real-time
   - Handles connection failures gracefully

---

## Exercise Set 3: URDF Modeling

### Exercise 3.1: Simple Robot Arm

**Difficulty**: Beginner  
**Time**: 45 minutes  
**Concepts**: URDF, Links, Joints

**Task**: Model a 2-DOF robot arm.

**Specifications**:
- **Link 1**: 0.3m length, 0.05m radius cylinder
- **Joint 1**: Revolute, ±90° range
- **Link 2**: 0.25m length, 0.04m radius cylinder  
- **Joint 2**: Revolute, ±120° range

**Requirements**:
1. Create URDF file with correct geometry
2. Add visual and collision elements
3. Calculate and add inertial properties
4. Visualize in RViz2
5. Test joint movement with sliders

---

### Exercise 3.2: Xacro Macros

**Difficulty**: Intermediate  
**Time**: 1 hour  
**Concepts**: Xacro, Macros, Parameterization

**Task**: Create reusable Xacro macros for robot components.

**Requirements**:
1. Create a `wheel` macro with parameters:
   - radius
   - width
   - position (xyz)
   - name

2. Create a `leg` macro with parameters:
   - length
   - side ("left" or "right")
   - Uses reflection for symmetric placement

3. Build a 4-wheeled robot using the macros

**Challenge**: Make the wheel macro automatically calculate correct inertia.

---

### Exercise 3.3: Humanoid Hand

**Difficulty**: Advanced  
**Time**: 2 hours  
**Concepts**: Complex URDF, Multiple joints

**Task**: Model a simple humanoid hand.

**Specifications**:
- Palm: 0.08m × 0.1m × 0.02m box
- 5 fingers, each with:
  - 3 phalanges (segments)
  - 2 revolute joints
  - Decreasing size toward fingertip

**Requirements**:
1. Use Xacro for finger macro
2. Proper joint limits (fingers only bend one direction)
3. Realistic proportions
4. Test grasping poses in RViz2

---

## Mini-Project 1: Autonomous Patrol Robot

**Difficulty**: Intermediate  
**Time**: 3-4 hours  
**Concepts**: Integration of multiple topics, services, and state machine

### Project Description

Build a robot that autonomously patrols waypoints while avoiding obstacles.

### Requirements

**Nodes**:
1. **Patrol Planner**:
   - Stores list of waypoints
   - Publishes next goal to `/goal_pose`
   - Provides service `/set_waypoints`

2. **Obstacle Detector**:
   - Subscribes to `/scan` (simulated LiDAR)
   - Publishes Bool to `/obstacle_detected`
   - Triggers when obstacle < 0.5m away

3. **Motion Controller**:
   - Subscribes to `/goal_pose` and `/obstacle_detected`
   - Publishes `/cmd_vel` (velocity commands)
   - Stops when obstacle detected
   - Resumes when clear

4. **Status Logger**:
   - Subscribes to all topics
   - Logs events to file
   - Publishes summary statistics

### Deliverables

1. ROS 2 package with all nodes
2. Launch file to start entire system
3. README with instructions
4. Demo video (2-3 minutes)

### Testing

Simulate the robot in an environment with obstacles. It should:
- Visit all waypoints in sequence
- Stop before hitting obstacles
- Resume motion when obstacles clear
- Handle missing sensor data gracefully

---

## Mini-Project 2: Multi-Robot Coordination

**Difficulty**: Advanced  
**Time**: 4-5 hours  
**Concepts**: Namespaces, Multi-robot systems, Coordination

### Project Description

Coordinate two robots to transport an object together.

### Requirements

**Robot Setup**:
- Two identical robots with namespaces `/robot1` and `/robot2`
- Each has: position sensor, gripper, communication

**Coordination Node**:
- Assigns roles: "leader" and "follower"
- Synchronizes motion
- Handles communication failures

**Tasks**:
1. Both robots approach object
2. Both grip object
3. Leader plans path
4. Follower maintains formation
5. Both release at destination

### Challenges

- Synchronize gripper actions
- Maintain constant distance during transport
- Handle if one robot fails

---

## Troubleshooting Scenarios

Practice debugging common ROS 2 issues:

### Scenario 1: No Messages Received

**Symptom**: Subscriber node runs but never receives messages.

**Possible Causes**:
- [ ] Publisher and subscriber on different topics?
- [ ] Message types don't match?
- [ ] QoS compatibility issue?
- [ ] Publisher node not running?

**Debug Steps**:
```bash
# Check if topic exists
ros2 topic list

# Check topic type
ros2 topic info /topic_name

# Echo messages
ros2 topic echo /topic_name

# Check nodes
ros2 node list
```

---

### Scenario 2: High CPU Usage

**Symptom**: Node consumes 100% CPU.

**Possible Causes**:
- [ ] Infinite loop in callback?
- [ ] Publishing too fast?
- [ ] Memory leak?
- [ ] No sleep in main loop?

**Solution Example**:
```python
# Bad - no rate limiting
while True:
    publish_data()

# Good - rate limiting
rate = node.create_rate(10)  # 10 Hz
while rclpy.ok():
    publish_data()
    rate.sleep()
```

---

### Scenario 3: Transform (TF) Errors

**Symptom**: "Transform from X to Y does not exist"

**Possible Causes**:
- [ ] URDF not loaded?
- [ ] robot_state_publisher not running?
- [ ] Joint states not published?
- [ ] Frame names misspelled?

**Debug Steps**:
```bash
# View TF tree
ros2 run tf2_tools view_frames

# Echo specific transform
ros2 run tf2_ros tf2_echo base_link end_effector

# Check for TF publishers
ros2 topic list | grep /tf
```

---

## Assessment Questions

Test your knowledge before moving to Module 2.

### Conceptual Questions

1. What is the difference between a topic and a service in ROS 2?

2. Why would you use "best effort" QoS instead of "reliable"?

3. Explain the parent-child relationship in a URDF robot model.

4. What happens if two nodes publish to the same topic simultaneously?

5. When should you use Xacro instead of plain URDF?

### Practical Questions

6. Write a command to list all topics currently active:
   ```bash
   # Your answer here
   ```

7. How do you change a parameter value while a node is running?

8. Calculate the inertia (ixx) for a cylinder: mass=0.5kg, radius=0.03m, height=0.4m

9. What's wrong with this URDF snippet?
   ```xml
   <joint name="elbow" type="revolute">
     <parent link="upper_arm"/>
     <child link="upper_arm"/>
     <axis xyz="0 1 0"/>
   </joint>
   ```

10. How do you convert a .xacro file to .urdf?

### Code Review

**Question 11**: Find and fix the bugs in this code:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class BuggyNode(Node):
    def __init__(self):
        super().__init__('buggy_node')
        self.publisher = self.create_publisher(String, 'output', 10)
        self.subscription = self.create_subscription(
            String, 'input', self.callback, 10
        )
        
    def callback(self, msg):
        response = String()
        response.data = msg.data.upper()
        self.publisher.publish(response)
        self.get_logger().info('Processed: {msg.data}')

def main():
    rclpy.init()
    node = BuggyNode()
    rclpy.spin(node)
    # What's missing here?
```

---

## Project Submission Guidelines

When you complete exercises and projects:

### Required Documentation

1. **README.md**:
   - Project description
   - Dependencies
   - Build instructions
   - Usage examples

2. **Package Structure**:
   ```
   my_robot_pkg/
   ├── package.xml
   ├── setup.py
   ├── my_robot_pkg/
   │   ├── __init__.py
   │   ├── node1.py
   │   └── node2.py
   ├── launch/
   │   └── system.launch.py
   ├── urdf/
   │   └── robot.urdf.xacro
   └── README.md
   ```

3. **Code Quality**:
   - Docstrings for all functions
   - Comments for complex logic
   - Consistent naming conventions
   - Error handling

4. **Testing**:
   - List test cases
   - Expected outputs
   - Known limitations

---

## Additional Challenges

For those who want to go further:

### Challenge 1: Performance Optimization
Optimize the patrol robot to minimize CPU usage while maintaining 100Hz control loop.

### Challenge 2: Fault Tolerance
Add fault detection and recovery to any exercise. Handle node crashes gracefully.

### Challenge 3: Visualization
Create a custom RViz plugin to visualize your robot's internal state.

### Challenge 4: Real Hardware
If you have access to a robot kit (TurtleBot, etc.), port one exercise to real hardware.

---

## Resources for Further Learning

### Official Documentation
- [ROS 2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [ROS 2 Design](https://design.ros2.org/)
- [URDF Tutorials](http://wiki.ros.org/urdf/Tutorials)

### Community
- [ROS Discourse](https://discourse.ros.org/)
- [ROS Answers](https://answers.ros.org/)
- [r/ROS on Reddit](https://www.reddit.com/r/ROS/)

### Books
- "A Gentle Introduction to ROS" by Jason M. O'Kane
- "Programming Robots with ROS" by Morgan Quigley et al.
- "Mastering ROS for Robotics Programming" by Lentin Joseph

---

## What's Next?

Congratulations on completing Module 1! You now have a solid foundation in ROS 2.

In **Module 2**, we'll learn to simulate robots in realistic environments using Gazebo and Unity. You'll create digital twins of robots and test them in physics-accurate virtual worlds.

Continue to: [Module 2: Gazebo Physics Simulation →](/docs/module2/gazebo-physics)

---

## Answer Key (For Self-Assessment)

<details>
<summary>Click to reveal answers to assessment questions</summary>

**1.** Topics are for continuous data streaming (asynchronous, one-way), while services are for request-response (synchronous, two-way).

**2.** Best effort is faster and suitable for high-frequency data where occasional packet loss is acceptable (e.g., sensor streams).

**3.** A child link moves relative to its parent link through the connecting joint. The parent link defines the reference frame for the child.

**4.** Both messages are delivered to subscribers. ROS 2 handles multiple publishers automatically.

**5.** Use Xacro when you have repetitive structures, need variables, or want to split large URDF files into modules.

**6.** `ros2 topic list`

**7.** `ros2 param set /node_name param_name value`

**8.** ixx = (m/12) × (3r² + h²) = (0.5/12) × (3×0.03² + 0.4²) = 0.00685 kg⋅m²

**9.** Parent and child are the same link (should be different). This creates a self-loop.

**10.** `ros2 run xacro xacro robot.urdf.xacro > robot.urdf`

**11.** Bugs:
- Missing f-string: `f'Processed: {msg.data}'`
- Missing cleanup: `node.destroy_node()` and `rclpy.shutdown()`

</details>

---

**Navigation**: [← Previous: URDF for Humanoids](/docs/module1/urdf-humanoids) | [Next: Module 2 →](/docs/module2/gazebo-physics)