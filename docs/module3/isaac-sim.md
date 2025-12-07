# NVIDIA Isaac Sim Overview

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand NVIDIA Isaac platform and its ecosystem
- Install and configure Isaac Sim
- Navigate the Isaac Sim interface
- Import and simulate robots in Isaac Sim
- Generate synthetic training data
- Understand the advantages of Isaac Sim over Gazebo
- Create photorealistic simulation environments
- Export data for machine learning workflows

## What is NVIDIA Isaac?

**NVIDIA Isaac** is a comprehensive robotics platform that accelerates robot development through simulation, AI, and high-performance computing. It consists of several components:

### Isaac Platform Components

1. **Isaac Sim**: Photorealistic robotics simulator built on Omniverse
2. **Isaac ROS**: Hardware-accelerated ROS 2 packages
3. **Isaac Manipulator**: Tools for robotic manipulation
4. **Isaac AMR**: Autonomous mobile robot reference designs
5. **Isaac Cortex**: Coordination framework for robot behaviors

In this module, we focus on **Isaac Sim** and **Isaac ROS**.

## Why Isaac Sim?

### Advantages Over Traditional Simulators

| Feature | Gazebo | Isaac Sim |
|---------|--------|-----------|
| **Graphics** | Basic 3D | Photorealistic (RTX ray tracing) |
| **Physics** | ODE/Bullet/DART | PhysX (GPU-accelerated) |
| **Scale** | Single robot | Thousands of robots |
| **AI Training** | Limited | Built-in synthetic data generation |
| **Performance** | CPU-bound | GPU-accelerated |
| **Sensor Simulation** | Good | Excellent (physically accurate) |
| **Learning Curve** | Easier | Steeper |

### When to Use Isaac Sim

**Use Isaac Sim when**:
- Training vision-based AI models (need photorealistic data)
- Simulating large fleets of robots
- Requiring physically accurate sensor simulation
- Working with NVIDIA hardware (Jetson, GPU)
- Need real-time performance with complex scenes

**Use Gazebo when**:
- Learning ROS basics
- Quick prototyping
- Limited GPU resources
- Simpler robots and environments

## Isaac Sim Architecture

Isaac Sim is built on **NVIDIA Omniverse**, a platform for 3D collaboration and simulation.

### Key Technologies

1. **USD (Universal Scene Description)**: Scene representation format
2. **PhysX**: GPU-accelerated physics engine
3. **RTX Ray Tracing**: Photorealistic rendering
4. **Synthetic Data Generation**: Training data pipeline
5. **ROS/ROS 2 Bridge**: Communication with robot stack

### Architecture Diagram

```
Isaac Sim (Omniverse)
├── USD Scene
│   ├── Robots (URDF/USD)
│   ├── Environment (3D assets)
│   └── Sensors (Cameras, LiDAR)
├── PhysX Physics
├── RTX Rendering
└── Extensions
    ├── ROS 2 Bridge
    ├── Synthetic Data
    └── Robot Control
```

## System Requirements

Isaac Sim is computationally demanding. Here are the requirements:

### Minimum Requirements

- **OS**: Ubuntu 20.04 or 22.04 (Windows also supported)
- **GPU**: NVIDIA RTX 2070 or higher (8GB VRAM minimum)
- **CPU**: Intel Core i7 or AMD Ryzen 7
- **RAM**: 32GB
- **Storage**: 50GB free space (SSD recommended)
- **Driver**: NVIDIA Driver 525.60 or newer

### Recommended Requirements

- **GPU**: RTX 3090 or RTX 4080 (24GB VRAM)
- **CPU**: Intel Core i9 or AMD Ryzen 9
- **RAM**: 64GB
- **Storage**: 100GB NVMe SSD

**Note**: Without RTX GPU, Isaac Sim will not run properly.

## Installing Isaac Sim

### Method 1: Omniverse Launcher (Recommended)

**Step 1**: Download Omniverse Launcher

```bash
# Visit: https://www.nvidia.com/en-us/omniverse/download/
# Download the AppImage for Linux

# Make it executable
chmod +x omniverse-launcher-linux.AppImage

# Run it
./omniverse-launcher-linux.AppImage
```

**Step 2**: Install Isaac Sim from Launcher

1. Open Omniverse Launcher
2. Go to **Exchange** tab
3. Search for "Isaac Sim"
4. Click **Install** (current version: 2023.1.1 or newer)
5. Wait for download (~10-15 GB)

**Step 3**: Verify Installation

1. In Launcher, go to **Library**
2. Find "Isaac Sim"
3. Click **Launch**
4. First launch takes 2-3 minutes (shader compilation)

### Method 2: Docker Container

For headless or cloud deployments:

```bash
# Pull Isaac Sim container
docker pull nvcr.io/nvidia/isaac-sim:2023.1.1

# Run container
docker run --name isaac-sim --entrypoint bash \
  --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -it nvcr.io/nvidia/isaac-sim:2023.1.1
```

## Isaac Sim Interface Tour

### Main Components

1. **Viewport**: 3D scene visualization
2. **Stage**: Scene hierarchy (USD tree)
3. **Property Panel**: Object properties
4. **Content Browser**: Asset library
5. **Console**: Python scripting and logs

### Navigation Controls

- **Rotate**: Middle mouse button + drag
- **Pan**: Shift + middle mouse + drag
- **Zoom**: Scroll wheel
- **Select**: Left click
- **Focus**: F key (focus on selected object)

### Coordinate System

Isaac Sim uses **Y-up** coordinate system:
- **X**: Right
- **Y**: Up
- **Z**: Forward (toward camera in default view)

**Note**: ROS uses Z-up, so coordinate conversion is needed!

## Creating Your First Scene

### Step 1: Start Isaac Sim

Launch Isaac Sim from Omniverse Launcher.

### Step 2: Load Environment

1. Go to **Isaac Examples** menu
2. Select **Environments**
3. Choose "Hospital" or "Office"
4. Environment loads with physics enabled

### Step 3: Add a Robot

**Option A: From Isaac Examples**
1. **Isaac Examples** → **Simple Objects**
2. Select "Carter" (NVIDIA reference robot)
3. Robot spawns in scene

**Option B: Import URDF**
1. **Isaac Utils** → **Workflows** → **URDF Importer**
2. Browse to your URDF file
3. Configure import settings:
   - Fix Base Link: Check if robot should be fixed
   - Self Collisions: Enable
4. Click **Import**

### Step 4: Run Simulation

1. Click **Play** button (or press Space)
2. Physics starts running
3. Click **Stop** to pause

## Importing and Simulating Robots

### URDF Import Process

```python
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.extensions import enable_extension

# Enable URDF importer extension
enable_extension("omni.isaac.urdf")

# Import URDF
import omni.kit.commands
omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path="/path/to/robot.urdf",
    import_config={
        "merge_fixed_joints": False,
        "fix_base": False,
        "self_collision": True,
        "default_drive_type": "none"
    }
)
```

### Configuring Robot Articulation

After import, configure joints:

1. Select robot in **Stage**
2. In **Property Panel**, find **Articulation Root**
3. Set solver parameters:
   - Position Iteration Count: 32
   - Velocity Iteration Count: 16
   - Stabilization Threshold: 0.001

### Adding Controllers

```python
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction

# Get robot articulation
robot = Articulation("/World/Robot")
robot.initialize()

# Set joint positions
joint_positions = [0.0, 0.5, -0.5, 0.0, 0.0, 0.0]
robot.set_joint_positions(joint_positions)

# Apply joint efforts (torques)
efforts = [10.0, 5.0, 5.0, 2.0, 2.0, 1.0]
robot.set_joint_efforts(efforts)

# Or use high-level action
action = ArticulationAction(joint_positions=joint_positions)
robot.apply_action(action)
```

## Sensor Simulation

Isaac Sim provides physically accurate sensor simulation.

### RGB Camera

```python
from omni.isaac.sensor import Camera
import numpy as np

# Create camera
camera = Camera(
    prim_path="/World/Camera",
    position=np.array([2.0, 2.0, 1.5]),
    frequency=30,  # Hz
    resolution=(1280, 720)
)

# Initialize
camera.initialize()

# Get image data
rgb_data = camera.get_rgb()  # Returns numpy array (H, W, 3)
```

### Depth Camera

```python
# Add depth output
camera.add_depth_data_to_frame()

# Get depth
depth_data = camera.get_depth()  # Returns numpy array (H, W)
```

### LiDAR

```python
from omni.isaac.sensor import RotatingLidarPhysX

lidar = RotatingLidarPhysX(
    prim_path="/World/Lidar",
    name="lidar",
    rotation_frequency=20,  # Hz
    resolution_horizontal=0.4,  # degrees
    resolution_vertical=4.0,
    min_range=0.4,  # meters
    max_range=100.0
)

lidar.initialize()

# Get point cloud
point_cloud = lidar.get_current_frame()  # Returns dict with 'data' key
```

## Generating Synthetic Training Data

Isaac Sim excels at generating training data for AI models.

### Domain Randomization

Randomize scene elements to improve model generalization:

```python
from omni.replicator.core import randomizer

# Randomize lighting
with randomizer.randomize():
    light = randomizer.get_prim("/World/Light")
    randomizer.attribute(light, "intensity", 
                        distribution="uniform", 
                        min_val=500, max_val=2000)
    randomizer.attribute(light, "color", 
                        distribution="uniform",
                        min_val=(0.8, 0.8, 0.8), 
                        max_val=(1.0, 1.0, 1.0))

# Randomize object positions
objects = randomizer.get_prims("/World/Objects/*")
randomizer.scatter_2d(objects, 
                     surface="/World/Floor",
                     check_for_collisions=True)

# Randomize textures
textures = randomizer.get_assets("textures")
randomizer.texture(objects, textures)
```

### Synthetic Data Generation Pipeline

```python
import omni.replicator.core as rep

# Set up camera
camera = rep.create.camera(position=(2, 2, 2))

# Define render products
render_product = rep.create.render_product(camera, (1024, 1024))

# Annotators (what to capture)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
depth = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
bbox = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
semantic = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")

rgb.attach([render_product])
depth.attach([render_product])
bbox.attach([render_product])
semantic.attach([render_product])

# Define randomization
with rep.trigger.on_frame(num_frames=1000):
    with rep.create.group(objects):
        rep.modify.pose(
            position=rep.distribution.uniform((-2, 0, -2), (2, 2, 2))
        )
    
    with rep.create.group([light]):
        rep.modify.attribute("intensity", 
                           rep.distribution.uniform(500, 2000))

# Write data
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir="/path/to/output", rgb=True, 
                 bounding_box_2d=True, semantic_segmentation=True)

# Run
rep.orchestrator.run()
```

This generates:
- 1000 RGB images
- Corresponding depth maps
- 2D bounding boxes (JSON)
- Semantic segmentation masks

## ROS 2 Integration

Isaac Sim has native ROS 2 support.

### Enabling ROS 2 Bridge

```python
from omni.isaac.core.utils.extensions import enable_extension

# Enable ROS 2 bridge
enable_extension("omni.isaac.ros2_bridge")
```

**Or via GUI**:
1. **Window** → **Extensions**
2. Search "ROS2 Bridge"
3. Enable extension

### Publishing Robot State

```python
# Add ROS 2 component to robot
import omni.graph.core as og

# Create ROS 2 graph
keys = og.Controller.Keys
(graph, nodes, _, _) = og.Controller.edit(
    {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("ROS2PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
        ],
        keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "ROS2PublishJointState.inputs:execIn"),
        ],
        keys.SET_VALUES: [
            ("ROS2PublishJointState.inputs:targetPrim", "/World/Robot"),
            ("ROS2PublishJointState.inputs:topicName", "/joint_states"),
        ],
    },
)
```

## Isaac Sim vs Gazebo: A Comparison

### When to Use Each

**Use Isaac Sim for**:
- Computer vision model training
- Large-scale multi-robot simulation
- Photorealistic visualization
- GPU-accelerated physics (>100 objects)
- Accurate sensor simulation (cameras, LiDAR)

**Use Gazebo for**:
- Learning and prototyping
- Simple environments
- Standard robot platforms (TurtleBot, etc.)
- Limited GPU resources
- Quick iteration

**Use Both**:
- Develop logic in Gazebo (faster iteration)
- Validate with Isaac Sim (realistic testing)
- Train AI models with Isaac Sim data
- Deploy on hardware

## Performance Tips

### Optimization Strategies

1. **Reduce Visual Quality** (for faster physics):
   - **Edit** → **Preferences** → **Rendering**
   - Set Anti-Aliasing to "FXAA"
   - Disable ray tracing during development

2. **Simplify Collision Meshes**:
   - Use primitive shapes (boxes, spheres) for collision
   - Detailed meshes only for visuals

3. **Adjust Physics Settings**:
   - Reduce solver iterations for faster simulation
   - Increase timestep (e.g., 1/60 → 1/30) if stability allows

4. **LOD (Level of Detail)**:
   - Use lower poly models for distant objects

5. **Headless Mode** (no GUI):
   ```bash
   ./isaac-sim.sh --headless
   ```

## Key Takeaways

✅ **Isaac Sim** provides photorealistic robotics simulation  
✅ **Built on Omniverse** with USD scene format  
✅ **GPU-accelerated** physics and rendering  
✅ **Synthetic data generation** for AI training  
✅ **ROS 2 integration** for robot development  
✅ **Requires NVIDIA RTX GPU** for optimal performance

## What's Next?

In the next chapter, we'll explore **Isaac ROS**, NVIDIA's hardware-accelerated ROS 2 packages for perception, localization, and navigation.

Continue to: [Isaac ROS →](/docs/module3/isaac-ros)

## Exercises

### Exercise 1: First Isaac Sim Scene
Create a simple scene with a robot, obstacles, and lighting. Run the simulation.

### Exercise 2: URDF Import
Import your humanoid robot URDF from Module 1 into Isaac Sim. Configure articulations.

### Exercise 3: Camera Data Capture
Add an RGB camera to your robot. Capture 50 images as it moves through an environment.

### Exercise 4: Domain Randomization
Implement basic domain randomization: randomize lighting and object positions.

### Exercise 5: Performance Benchmark
Compare simulation speed of Isaac Sim vs Gazebo for the same robot and environment.

---

**Navigation**: [← Module 2](/docs/module2/simulation-exercises) | [Next: Isaac ROS →](/docs/module3/isaac-ros)