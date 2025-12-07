# URDF for Humanoids

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand the URDF (Unified Robot Description Format) specification
- Create robot models with links and joints
- Define coordinate frames and transformations
- Build a simple humanoid robot model
- Use Xacro for modular and maintainable robot descriptions
- Visualize robots in RViz2
- Apply best practices for humanoid modeling

## What is URDF?

**URDF (Unified Robot Description Format)** is an XML-based format for describing the physical structure of a robot. Think of it as a blueprint that tells ROS 2:

- What parts the robot has (links)
- How those parts connect (joints)
- Where things are located (coordinate frames)
- What the robot looks like (visual geometry)
- How to simulate physics (collision geometry and inertia)

URDF is the standard way to describe robots in ROS 2, used by simulation tools (Gazebo), visualization tools (RViz), motion planning systems, and more.

### Why URDF Matters for Humanoids

Humanoid robots are complex:
- **30-60 joints** (compared to 6-7 for typical robot arms)
- **Complex kinematic chains** (legs, arms, spine, head)
- **Multiple end-effectors** (two hands, two feet)
- **Balance constraints** (must maintain center of mass)

A well-designed URDF is essential for:
- Accurate simulation
- Motion planning
- Collision detection
- Force/torque calculations

## URDF Fundamentals

### The Building Blocks

Every URDF consists of two main elements:

1. **Links**: Rigid bodies (like bones)
2. **Joints**: Connections between links (like articulations)

```xml
<robot name="simple_robot">
  <link name="base_link"/>
  
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
  </joint>
  
  <link name="link1"/>
</robot>
```

### Links

A **link** represents a rigid body. It can have:
- **Visual**: What you see (for rendering)
- **Collision**: Simplified geometry (for collision detection)
- **Inertial**: Mass and inertia properties (for physics simulation)

```xml
<link name="forearm">
  <!-- Visual representation (what you see) -->
  <visual>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.03" length="0.3"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 1 1"/>
    </material>
  </visual>
  
  <!-- Collision geometry (simplified for physics) -->
  <collision>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.03" length="0.3"/>
    </geometry>
  </collision>
  
  <!-- Physical properties -->
  <inertial>
    <mass value="0.5"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <inertia ixx="0.004" ixy="0" ixz="0"
             iyy="0.004" iyz="0"
             izz="0.0001"/>
  </inertial>
</link>
```

### Joints

A **joint** connects two links and defines how they can move relative to each other.

**Joint Types**:

| Type | Description | Degrees of Freedom |
|------|-------------|--------------------|
| **fixed** | No motion (welded together) | 0 |
| **revolute** | Rotation around axis (with limits) | 1 |
| **continuous** | Rotation around axis (no limits) | 1 |
| **prismatic** | Linear motion along axis | 1 |
| **planar** | Motion in a plane | 2 |
| **floating** | Full 6-DOF motion | 6 |

```xml
<joint name="elbow_joint" type="revolute">
  <!-- Parent link (upper arm) -->
  <parent link="upper_arm"/>
  
  <!-- Child link (forearm) -->
  <child link="forearm"/>
  
  <!-- Origin: where child attaches to parent -->
  <origin xyz="0 0 0.3" rpy="0 0 0"/>
  
  <!-- Axis of rotation -->
  <axis xyz="0 1 0"/>
  
  <!-- Joint limits (for revolute joints) -->
  <limit lower="-2.5" upper="0" effort="100" velocity="2.0"/>
</joint>
```

### Coordinate Frames

Understanding coordinate frames is crucial:

- **Origin**: The reference point of a link (usually its center or joint location)
- **xyz**: Translation (x, y, z) in meters
- **rpy**: Rotation (roll, pitch, yaw) in radians

```
Roll:  Rotation around X-axis
Pitch: Rotation around Y-axis  
Yaw:   Rotation around Z-axis
```

ROS 2 uses the **right-hand rule**:
- X: Forward (red)
- Y: Left (green)
- Z: Up (blue)

## Building a Simple Humanoid Arm

Let's build a simple 3-joint arm to understand the concepts.

### The Arm Structure

Our arm will have:
1. **Shoulder**: Connects to torso, rotates around Z (yaw)
2. **Upper Arm**: Rotates at shoulder around Y (pitch)
3. **Elbow**: Connects upper arm to forearm, rotates around Y
4. **Forearm**: Connected to upper arm
5. **Wrist**: Simple fixed connection to hand
6. **Hand**: End effector

```xml
<?xml version="1.0"?>
<robot name="humanoid_arm">
  
  <!-- Base link (torso connection point) -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
  </link>
  
  <!-- Shoulder joint (Z-axis rotation) -->
  <joint name="shoulder_yaw" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1.0"/>
  </joint>
  
  <link name="shoulder_link">
    <visual>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.0001" ixy="0" ixz="0"
               iyy="0.0001" iyz="0"
               izz="0.0001"/>
    </inertial>
  </link>
  
  <!-- Shoulder pitch joint -->
  <joint name="shoulder_pitch" type="revolute">
    <parent link="shoulder_link"/>
    <child link="upper_arm"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="1.0"/>
  </joint>
  
  <link name="upper_arm">
    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.03" length="0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.03" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <inertia ixx="0.004" ixy="0" ixz="0"
               iyy="0.004" iyz="0"
               izz="0.0001"/>
    </inertial>
  </link>
  
  <!-- Elbow joint -->
  <joint name="elbow_joint" type="revolute">
    <parent link="upper_arm"/>
    <child link="forearm"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="0" effort="30" velocity="1.5"/>
  </joint>
  
  <link name="forearm">
    <visual>
      <origin xyz="0 0 0.125" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.025" length="0.25"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.125" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.025" length="0.25"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <origin xyz="0 0 0.125" rpy="0 0 0"/>
      <inertia ixx="0.002" ixy="0" ixz="0"
               iyy="0.002" iyz="0"
               izz="0.0001"/>
    </inertial>
  </link>
  
  <!-- Wrist (fixed for now) -->
  <joint name="wrist_joint" type="fixed">
    <parent link="forearm"/>
    <child link="hand"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
  </joint>
  
  <link name="hand">
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.08 0.04 0.1"/>
      </geometry>
      <material name="yellow">
        <color rgba="1 1 0 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.0001" ixy="0" ixz="0"
               iyy="0.0001" iyz="0"
               izz="0.0001"/>
    </inertial>
  </link>
  
</robot>
```

### Testing the Arm

Save this as `humanoid_arm.urdf` and visualize it:

```bash
# Install URDF tools if needed
sudo apt install ros-humble-urdf-tutorial

# Check URDF validity
check_urdf humanoid_arm.urdf

# Visualize in RViz with joint control
ros2 launch urdf_tutorial display.launch.py model:=humanoid_arm.urdf
```

You should see the arm in RViz with sliders to control each joint!

## Xacro: Programmable URDF

Writing URDF by hand becomes tedious for complex robots. **Xacro (XML Macros)** adds programming features to URDF:

- **Variables**: Define values once, reuse everywhere
- **Macros**: Parameterized templates for repetitive structures
- **Math**: Compute values dynamically
- **Include**: Split large files into modules

### Basic Xacro Example

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="robot_with_xacro">
  
  <!-- Define properties (variables) -->
  <xacro:property name="arm_length" value="0.3"/>
  <xacro:property name="arm_radius" value="0.03"/>
  <xacro:property name="arm_mass" value="0.5"/>
  
  <!-- Define a macro for creating arm segments -->
  <xacro:macro name="arm_segment" params="name length radius mass parent_link">
    
    <joint name="${name}_joint" type="revolute">
      <parent link="${parent_link}"/>
      <child link="${name}_link"/>
      <origin xyz="0 0 ${length}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-2.0" upper="2.0" effort="50" velocity="1.0"/>
    </joint>
    
    <link name="${name}_link">
      <visual>
        <origin xyz="0 0 ${length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${radius}" length="${length}"/>
        </geometry>
        <material name="blue">
          <color rgba="0 0 1 1"/>
        </material>
      </visual>
      <inertial>
        <mass value="${mass}"/>
        <inertia ixx="${(mass*(3*radius*radius + length*length))/12}" 
                 ixy="0" ixz="0"
                 iyy="${(mass*(3*radius*radius + length*length))/12}" 
                 iyz="0"
                 izz="${(mass*radius*radius)/2}"/>
      </inertial>
    </link>
    
  </xacro:macro>
  
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
  </link>
  
  <!-- Use the macro to create arm segments -->
  <xacro:arm_segment name="upper_arm" 
                     length="${arm_length}" 
                     radius="${arm_radius}"
                     mass="${arm_mass}"
                     parent_link="base_link"/>
  
  <xacro:arm_segment name="forearm" 
                     length="${arm_length * 0.8}" 
                     radius="${arm_radius * 0.8}"
                     mass="${arm_mass * 0.6}"
                     parent_link="upper_arm_link"/>
  
</robot>
```

### Converting Xacro to URDF

```bash
# Convert xacro to URDF
ros2 run xacro xacro robot.urdf.xacro > robot.urdf

# Or use it directly in launch files (no conversion needed)
```

### Xacro Best Practices

1. **Use properties for repeated values**: Wheel radius, link lengths, etc.
2. **Create macros for symmetric parts**: Left and right arms/legs
3. **Math expressions**: `${arm_length * 0.8}` for proportional sizing
4. **Separate files**: `robot.urdf.xacro` includes `arm.xacro`, `leg.xacro`, etc.

## Complete Humanoid Model Structure

A full humanoid URDF typically has this hierarchy:

```
base_link (pelvis)
├── torso
│   ├── chest
│   │   ├── neck
│   │   │   └── head
│   │   ├── left_shoulder
│   │   │   ├── left_upper_arm
│   │   │   │   ├── left_forearm
│   │   │   │   │   └── left_hand
│   │   └── right_shoulder
│   │       ├── right_upper_arm
│   │       │   ├── right_forearm
│   │       │   │   └── right_hand
├── left_hip
│   ├── left_thigh
│   │   ├── left_shin
│   │   │   └── left_foot
└── right_hip
    ├── right_thigh
        ├── right_shin
            └── right_foot
```

### Simplified Humanoid Xacro

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="simple_humanoid">
  
  <!-- Properties -->
  <xacro:property name="torso_height" value="0.5"/>
  <xacro:property name="leg_length" value="0.4"/>
  <xacro:property name="arm_length" value="0.3"/>
  
  <!-- Macro for a leg -->
  <xacro:macro name="leg" params="side reflect">
    
    <joint name="${side}_hip_joint" type="revolute">
      <parent link="pelvis"/>
      <child link="${side}_thigh"/>
      <origin xyz="0 ${reflect * 0.1} -0.1" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
    </joint>
    
    <link name="${side}_thigh">
      <visual>
        <origin xyz="0 0 -${leg_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.05" length="${leg_length}"/>
        </geometry>
        <material name="blue"><color rgba="0 0 1 1"/></material>
      </visual>
      <inertial>
        <mass value="1.0"/>
        <inertia ixx="0.01" ixy="0" ixz="0" 
                 iyy="0.01" iyz="0" izz="0.001"/>
      </inertial>
    </link>
    
    <joint name="${side}_knee_joint" type="revolute">
      <parent link="${side}_thigh"/>
      <child link="${side}_shin"/>
      <origin xyz="0 0 -${leg_length}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-2.5" upper="0" effort="80" velocity="1.0"/>
    </joint>
    
    <link name="${side}_shin">
      <visual>
        <origin xyz="0 0 -${leg_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.04" length="${leg_length}"/>
        </geometry>
        <material name="green"><color rgba="0 1 0 1"/></material>
      </visual>
      <inertial>
        <mass value="0.7"/>
        <inertia ixx="0.008" ixy="0" ixz="0" 
                 iyy="0.008" iyz="0" izz="0.0008"/>
      </inertial>
    </link>
    
    <joint name="${side}_ankle_joint" type="fixed">
      <parent link="${side}_shin"/>
      <child link="${side}_foot"/>
      <origin xyz="0 0 -${leg_length}" rpy="0 0 0"/>
    </joint>
    
    <link name="${side}_foot">
      <visual>
        <origin xyz="0.05 0 -0.025" rpy="0 0 0"/>
        <geometry>
          <box size="0.15 0.08 0.05"/>
        </geometry>
        <material name="red"><color rgba="1 0 0 1"/></material>
      </visual>
      <inertial>
        <mass value="0.3"/>
        <inertia ixx="0.0005" ixy="0" ixz="0" 
                 iyy="0.001" iyz="0" izz="0.001"/>
      </inertial>
    </link>
    
  </xacro:macro>
  
  <!-- Pelvis (base) -->
  <link name="pelvis">
    <visual>
      <geometry>
        <box size="0.2 0.3 0.15"/>
      </geometry>
      <material name="gray"><color rgba="0.5 0.5 0.5 1"/></material>
    </visual>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" 
               iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>
  
  <!-- Instantiate legs -->
  <xacro:leg side="left" reflect="1"/>
  <xacro:leg side="right" reflect="-1"/>
  
  <!-- Add torso, arms, head using similar macros... -->
  
</robot>
```

## Visualizing in RViz2

RViz2 is the standard 3D visualization tool for ROS 2.

### Launching RViz with a URDF

Create a launch file `display.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model',
            default_value='robot.urdf.xacro',
            description='Path to robot URDF/Xacro file'
        ),
        
        # Robot State Publisher - publishes robot transforms
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': ParameterValue(
                    Command(['xacro ', LaunchConfiguration('model')]),
                    value_type=str
                )
            }]
        ),
        
        # Joint State Publisher GUI - control joints with sliders
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui'
        ),
        
        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', '$(find urdf_tutorial)/rviz/urdf.rviz']
        ),
    ])
```

Launch it:

```bash
ros2 launch my_robot display.launch.py model:=path/to/robot.urdf.xacro
```

### RViz Configuration

In RViz:
1. **Add** → **RobotModel** - displays the URDF
2. **Add** → **TF** - shows coordinate frames
3. Set **Fixed Frame** to `base_link` or `pelvis`
4. Use **Joint State Publisher** sliders to move joints

## Calculating Inertia

Accurate inertia values are crucial for realistic simulation. Here are formulas for common shapes:

### Box
```
mass = m
dimensions = (x, y, z)

ixx = (m/12) * (y² + z²)
iyy = (m/12) * (x² + z²)
izz = (m/12) * (x² + y²)
```

### Cylinder
```
mass = m
radius = r
height = h

ixx = iyy = (m/12) * (3*r² + h²)
izz = (m/2) * r²
```

### Sphere
```
mass = m
radius = r

ixx = iyy = izz = (2/5) * m * r²
```

## Best Practices for Humanoid URDF

### 1. Start Simple, Add Detail Later
- Begin with boxes and cylinders
- Add complex meshes only when needed
- Test at each stage

### 2. Use Consistent Naming
```
{side}_{body_part}_{type}
Examples:
- left_upper_arm_link
- right_hip_joint
- head_camera_frame
```

### 3. Collision vs Visual Geometry
- **Visual**: Can be detailed meshes (for looks)
- **Collision**: Should be simple shapes (for performance)

```xml
<visual>
  <geometry>
    <mesh filename="package://my_robot/meshes/hand_detailed.stl"/>
  </geometry>
</visual>

<collision>
  <geometry>
    <box size="0.08 0.04 0.1"/>  <!-- Simple box for collision -->
  </geometry>
</collision>
```

### 4. Center of Mass Considerations
For humanoid balance, the center of mass (COM) is critical:

```xml
<inertial>
  <!-- Place inertial origin at the link's center of mass -->
  <origin xyz="0 0 0.15" rpy="0 0 0"/>
  <mass value="0.5"/>
  <inertia .../>
</inertial>
```

### 5. Joint Limits
Set realistic limits based on human anatomy:

```xml
<!-- Knee can only bend backward (0 to -150 degrees) -->
<limit lower="-2.618" upper="0" effort="100" velocity="2.0"/>

<!-- Hip has larger range of motion -->
<limit lower="-1.57" upper="1.57" effort="150" velocity="2.0"/>
```

## Common URDF Errors and Fixes

### Error: "Link has no inertia"
```xml
<!-- Missing inertial - add this -->
<inertial>
  <mass value="0.1"/>
  <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
</inertial>
```

### Error: "Joint axis is zero"
```xml
<!-- Incorrect -->
<axis xyz="0 0 0"/>

<!-- Correct - specify a valid axis -->
<axis xyz="0 1 0"/>
```

### Error: "Tree does not contain root link"
Every robot must have a root link that isn't a child of any joint. Make sure `base_link` or `pelvis` is at the top of the hierarchy.

## Key Takeaways

✅ **URDF** describes robot structure in XML format  
✅ **Links** are rigid bodies, **joints** connect them  
✅ **Xacro** adds programming features to URDF  
✅ **RViz2** visualizes robots and coordinate frames  
✅ **Inertia** must be accurate for realistic simulation  
✅ **Humanoids** require careful kinematic design

## What's Next?

Now that you can model robots, let's put your knowledge into practice with hands-on exercises!

Continue to: [ROS 2 Exercises →](/docs/module1/ros2-exercises)

## Exercises

### Exercise 1: Modify the Arm
Add a 4th joint to the arm - a wrist rotation around the Z-axis.

### Exercise 2: Create a Leg
Build a complete leg with hip (3 DOF), knee (1 DOF), and ankle (2 DOF) joints.

### Exercise 3: Calculate Inertia
Calculate the inertia tensor for a cylindrical link with mass 0.8 kg, radius 0.04 m, length 0.35 m.

### Exercise 4: Xacro Macro
Create a Xacro macro for a finger with 3 phalanges (segments) and 2 joints.

### Exercise 5: Full Humanoid
Assemble a complete humanoid with torso, head, two arms, and two legs. Test in RViz2.

---

**Navigation**: [← Previous: Nodes, Topics, and Services](/docs/module1/nodes-topics-services) | [Next: ROS 2 Exercises →](/docs/module1/ros2-exercises)