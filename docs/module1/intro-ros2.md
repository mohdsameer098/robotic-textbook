# Introduction to ROS 2

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain what ROS 2 is and why it's essential for robotics
- Understand the key differences between ROS 1 and ROS 2
- Identify the core components of ROS 2 architecture
- Install ROS 2 Humble on Ubuntu 22.04
- Create and run your first ROS 2 node
- Understand the publish-subscribe communication pattern

## What is ROS 2?

**ROS 2 (Robot Operating System 2)** is not actually an operating system in the traditional sense. Rather, it's a **middleware framework** - a software layer that sits between your robot's hardware and your application code, making it easier to build complex robotic systems.

Think of ROS 2 as the "nervous system" of your robot. Just as your nervous system allows different parts of your body to communicate and coordinate (your eyes see an obstacle, your brain processes it, your legs move to avoid it), ROS 2 allows different parts of your robot to work together seamlessly.

### Why Do We Need ROS 2?

Modern robots are incredibly complex. A humanoid robot might have:

- **50+ motors** controlling joints and actuators
- **Multiple cameras** providing visual input
- **LiDAR sensors** for depth perception
- **IMU sensors** for balance and orientation
- **Force sensors** in hands and feet
- **AI models** for perception and decision-making
- **Navigation systems** for path planning

Without a framework like ROS 2, you'd need to write custom code to:
- Handle communication between all these components
- Manage timing and synchronization
- Deal with hardware failures gracefully
- Visualize data for debugging
- Record and replay sensor data

ROS 2 provides **battle-tested solutions** to all these problems, so you can focus on making your robot smarter rather than reinventing the wheel.

### Key Features of ROS 2

1. **Modular Architecture**: Break your robot software into small, reusable components (nodes)
2. **Language Agnostic**: Write nodes in Python, C++, or other supported languages
3. **Communication Middleware**: Built-in pub/sub, request/response, and action patterns
4. **Hardware Abstraction**: Same code works across different robot platforms
5. **Rich Tooling**: Visualization, logging, testing, and debugging tools included
6. **Large Ecosystem**: Thousands of pre-built packages for common robotics tasks

## ROS 1 vs ROS 2: What Changed?

If you've heard of ROS before, you might be wondering: why ROS 2? The original ROS (now called ROS 1) was revolutionary but had limitations that became apparent as robotics matured.

### Major Improvements in ROS 2

| Feature | ROS 1 | ROS 2 |
|---------|-------|-------|
| **Real-time Support** | Limited | Full support with DDS middleware |
| **Security** | None | Built-in encryption and authentication |
| **Multi-robot Systems** | Difficult | Native support |
| **Operating Systems** | Linux only | Linux, Windows, macOS |
| **Communication** | Single master node (failure point) | Distributed (no single point of failure) |
| **Quality of Service** | Best-effort only | Configurable reliability levels |
| **Embedded Systems** | Poor support | Optimized for resource-constrained devices |

For humanoid robotics, the **real-time capabilities** and **quality of service** features are crucial. A humanoid robot needs deterministic timing - if a balance control command arrives too late, the robot could fall!

### Should You Learn ROS 1?

**Short answer: No, start with ROS 2.**

ROS 1 reached end-of-life in 2025. The industry has migrated to ROS 2. All new robotics projects use ROS 2, and that's what we'll focus on in this course.

## ROS 2 Architecture Overview

Let's understand the fundamental building blocks of ROS 2.

### 1. Nodes

A **node** is a process that performs a specific computation. Think of nodes as specialized workers, each with a specific job:

- A **camera node** captures images
- A **perception node** identifies objects in those images
- A **planning node** decides what actions to take
- A **control node** sends commands to motors

Nodes are **modular and independent**. You can:
- Start and stop them independently
- Replace one node without affecting others
- Run multiple instances of the same node
- Distribute nodes across multiple computers

### 2. Topics (Publisher-Subscriber Pattern)

**Topics** are named buses over which nodes exchange messages. They implement the **publish-subscribe pattern**:

- **Publishers** send messages to a topic
- **Subscribers** receive messages from a topic
- Multiple publishers and subscribers can use the same topic
- Communication is **asynchronous** and **one-way**

Example: A camera node **publishes** image messages to the `/camera/image` topic. A perception node **subscribes** to this topic to receive and process the images.

```
[Camera Node] --publishes--> [/camera/image topic] --subscribes--> [Perception Node]
```

### 3. Services (Request-Response Pattern)

**Services** implement synchronous **request-response** communication:

- A **client** sends a request
- A **server** processes it and sends a response
- Communication is **synchronous** (client waits for response)
- Use for tasks that need immediate feedback

Example: A navigation node requests the current robot position from a localization service.

```
[Navigation Node] --request--> [Localization Service] --response--> [Navigation Node]
```

### 4. Actions (Goal-Based Pattern)

**Actions** are for long-running tasks that provide feedback:

- A **client** sends a goal
- An **action server** works on it
- Server sends **feedback** during execution
- Server sends a **result** when complete
- Client can **cancel** the goal

Example: "Move robot to position (x, y)" - you want feedback on progress and ability to cancel if needed.

```
[Planner] --goal--> [Navigation Action] --feedback--> [Planner]
                                       --result--> [Planner]
```

### 5. Messages

**Messages** are the data structures exchanged between nodes. ROS 2 provides standard message types:

- `std_msgs/String`: Text messages
- `sensor_msgs/Image`: Camera images
- `geometry_msgs/Twist`: Velocity commands
- `nav_msgs/Odometry`: Robot position and velocity

You can also define **custom messages** for your specific needs.

### ROS 2 Communication Layer: DDS

Under the hood, ROS 2 uses **DDS (Data Distribution Service)**, an industry-standard middleware. DDS provides:

- **Automatic discovery**: Nodes find each other without configuration
- **Quality of Service**: Configure reliability, latency, bandwidth tradeoffs
- **Real-time performance**: Deterministic message delivery
- **Security**: Encryption and access control

You don't need to understand DDS deeply to use ROS 2, but it's good to know it's there providing robust communication.

## Installing ROS 2 Humble

Let's get ROS 2 installed! We'll use **ROS 2 Humble Hawksbill**, the recommended long-term support (LTS) release.

### Prerequisites

- **Operating System**: Ubuntu 22.04 LTS (Jammy Jellyfish)
- **Architecture**: 64-bit (x86_64 or ARM64)
- **Locale**: UTF-8 locale

### Installation Steps

#### Step 1: Set Up Sources

```bash
# Ensure Ubuntu Universe repository is enabled
sudo apt install software-properties-common
sudo add-apt-repository universe

# Add ROS 2 GPG key
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add ROS 2 repository to sources list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

#### Step 2: Install ROS 2 Packages

```bash
# Update package index
sudo apt update

# Upgrade existing packages
sudo apt upgrade

# Install ROS 2 Humble Desktop (includes RViz, demos, tutorials)
sudo apt install ros-humble-desktop
```

This installation includes:
- Core ROS 2 libraries
- RViz (3D visualization tool)
- Example packages and tutorials
- Common message definitions

#### Step 3: Install Development Tools

```bash
# Install colcon build tool
sudo apt install python3-colcon-common-extensions

# Install rosdep (dependency management)
sudo apt install python3-rosdep
sudo rosdep init
rosdep update
```

#### Step 4: Set Up Environment

Add ROS 2 to your shell environment so commands are available:

```bash
# Add to .bashrc for automatic sourcing
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

#### Step 5: Verify Installation

```bash
# Check ROS 2 version
ros2 --version

# List available ROS 2 commands
ros2 --help
```

You should see output showing ROS 2 Humble is installed!

### Troubleshooting

**Issue**: `ros2: command not found`  
**Solution**: Run `source /opt/ros/humble/setup.bash` or restart your terminal

**Issue**: Permission denied errors  
**Solution**: Make sure you're using `sudo` for installation commands

**Issue**: Package not found  
**Solution**: Run `sudo apt update` again

## Your First ROS 2 Node: Hello World

Let's write your first ROS 2 program! We'll create a simple node that publishes "Hello, ROS 2!" messages.

### Step 1: Create a Workspace

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

A **workspace** is a directory where you organize your ROS 2 packages.

### Step 2: Create a Python Package

```bash
# Create a new package
ros2 pkg create --build-type ament_python hello_world_py --dependencies rclpy

cd hello_world_py
```

This creates a package named `hello_world_py` with Python support and `rclpy` (ROS Client Library for Python) as a dependency.

### Step 3: Write the Publisher Node

Create a file `hello_world_py/hello_publisher.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class HelloPublisher(Node):
    """
    A simple ROS 2 node that publishes Hello messages.
    """
    
    def __init__(self):
        # Initialize the node with a name
        super().__init__('hello_publisher')
        
        # Create a publisher
        # - Message type: String
        # - Topic name: 'hello_topic'
        # - Queue size: 10
        self.publisher = self.create_publisher(String, 'hello_topic', 10)
        
        # Create a timer that calls the callback every 1 second
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.counter = 0
        self.get_logger().info('Hello Publisher node started!')
    
    def timer_callback(self):
        """
        Called periodically by the timer.
        Publishes a message each time.
        """
        # Create a message
        msg = String()
        msg.data = f'Hello, ROS 2! Message #{self.counter}'
        
        # Publish the message
        self.publisher.publish(msg)
        
        # Log it
        self.get_logger().info(f'Publishing: "{msg.data}"')
        
        self.counter += 1


def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)
    
    # Create the node
    node = HelloPublisher()
    
    # Keep the node running
    # This will call timer_callback repeatedly
    rclpy.spin(node)
    
    # Cleanup
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 4: Write the Subscriber Node

Create `hello_world_py/hello_subscriber.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class HelloSubscriber(Node):
    """
    A simple ROS 2 node that subscribes to Hello messages.
    """
    
    def __init__(self):
        # Initialize the node
        super().__init__('hello_subscriber')
        
        # Create a subscriber
        # - Message type: String
        # - Topic name: 'hello_topic'
        # - Callback function: message_callback
        # - Queue size: 10
        self.subscription = self.create_subscription(
            String,
            'hello_topic',
            self.message_callback,
            10
        )
        
        self.get_logger().info('Hello Subscriber node started!')
    
    def message_callback(self, msg):
        """
        Called whenever a message arrives on the subscribed topic.
        
        Args:
            msg: The received message (String type)
        """
        self.get_logger().info(f'Received: "{msg.data}"')


def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)
    
    # Create the node
    node = HelloSubscriber()
    
    # Keep the node running and processing callbacks
    rclpy.spin(node)
    
    # Cleanup
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 5: Update Package Configuration

Edit `setup.py` to register your nodes:

```python
from setuptools import setup

package_name = 'hello_world_py'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Hello World ROS 2 package',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'hello_publisher = hello_world_py.hello_publisher:main',
            'hello_subscriber = hello_world_py.hello_subscriber:main',
        ],
    },
)
```

### Step 6: Build the Package

```bash
# Go to workspace root
cd ~/ros2_ws

# Build all packages
colcon build

# Source the workspace
source install/setup.bash
```

### Step 7: Run Your Nodes!

Open **two terminals**.

**Terminal 1** (Publisher):
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run hello_world_py hello_publisher
```

**Terminal 2** (Subscriber):
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run hello_world_py hello_subscriber
```

You should see the publisher sending messages and the subscriber receiving them!

**Output in Terminal 1:**
```
[INFO] [hello_publisher]: Hello Publisher node started!
[INFO] [hello_publisher]: Publishing: "Hello, ROS 2! Message #0"
[INFO] [hello_publisher]: Publishing: "Hello, ROS 2! Message #1"
...
```

**Output in Terminal 2:**
```
[INFO] [hello_subscriber]: Hello Subscriber node started!
[INFO] [hello_subscriber]: Received: "Hello, ROS 2! Message #0"
[INFO] [hello_subscriber]: Received: "Hello, ROS 2! Message #1"
...
```

Congratulations! You've created your first ROS 2 application! 🎉

## Understanding the Code

Let's break down what's happening:

### Publisher Node

1. **Initialize**: `super().__init__('hello_publisher')` creates a node named "hello_publisher"
2. **Create Publisher**: `create_publisher(String, 'hello_topic', 10)` sets up publishing to the topic
3. **Create Timer**: `create_timer(1.0, self.timer_callback)` calls the callback every second
4. **Publish Message**: `self.publisher.publish(msg)` sends the message

### Subscriber Node

1. **Initialize**: Creates node named "hello_subscriber"
2. **Create Subscription**: `create_subscription(...)` listens to 'hello_topic'
3. **Callback**: `message_callback` is called automatically when messages arrive
4. **Process Message**: We simply log the received message

### Key Concepts

- **rclpy**: ROS Client Library for Python - provides Python bindings for ROS 2
- **Node**: Base class for all ROS 2 nodes
- **Publisher/Subscriber**: Objects that handle communication
- **Topic**: Named channel ('hello_topic') where messages flow
- **Message**: Data structure (String) being transmitted
- **spin()**: Keeps node running and processing callbacks

## ROS 2 Command Line Tools

ROS 2 provides powerful command-line tools for introspection and debugging.

### List All Nodes

```bash
ros2 node list
```

Output:
```
/hello_publisher
/hello_subscriber
```

### Get Node Information

```bash
ros2 node info /hello_publisher
```

Shows publishers, subscribers, services, and actions for that node.

### List Topics

```bash
ros2 topic list
```

Output:
```
/hello_topic
/parameter_events
/rosout
```

### Echo Topic Messages

```bash
ros2 topic echo /hello_topic
```

Displays messages being published to the topic in real-time.

### Show Topic Info

```bash
ros2 topic info /hello_topic
```

Shows the number of publishers and subscribers.

### Check Message Type

```bash
ros2 interface show std_msgs/msg/String
```

Displays the structure of the String message type.

## Key Takeaways

✅ **ROS 2 is middleware** that simplifies building complex robot systems  
✅ **Nodes** are modular processes that perform specific tasks  
✅ **Topics** enable asynchronous pub/sub communication  
✅ **Services** provide synchronous request/response patterns  
✅ **Actions** support long-running tasks with feedback  
✅ **ROS 2 Humble** is the current LTS release (recommended)  
✅ **rclpy** is the Python library for writing ROS 2 nodes

## What's Next?

Now that you understand ROS 2 basics, we'll dive deeper into:

- **Custom messages** and complex data types
- **Services** for request-response patterns
- **Actions** for long-running tasks
- **Parameters** for configuring nodes
- **Launch files** for starting multiple nodes

Continue to the next chapter: [Nodes, Topics, and Services →](/docs/module1/nodes-topics-services)

## Exercises

### Exercise 1: Modify the Message
Change the publisher to send your name instead of "Hello, ROS 2!"

### Exercise 2: Change Frequency
Modify the timer to publish 5 times per second instead of once per second.

### Exercise 3: Multiple Subscribers
Run two subscriber nodes simultaneously. What happens?

### Exercise 4: Explore with CLI
Use `ros2 topic hz /hello_topic` to measure the publishing rate.

### Exercise 5: Custom Counter
Add a subscriber that counts how many messages it has received and prints the count.

---

**Navigation**: [← Back to Introduction](/docs/intro) | [Next: Nodes, Topics, and Services →](/docs/module1/nodes-topics-services)