# Module 4, Chapter 4 - Capstone Project: The Autonomous Humanoid

## Project Overview

This capstone project integrates everything you've learned throughout the course to build a fully autonomous humanoid robot system. Your robot will:

1. **Listen** to natural language voice commands (Whisper)
2. **Plan** multi-step actions using LLM cognitive reasoning (GPT-4/Claude)
3. **Navigate** through complex environments avoiding obstacles (Nav2)
4. **Perceive** objects using computer vision (Isaac ROS, GroundingDINO)
5. **Manipulate** objects with precision grasping (Isaac Sim)
6. **Execute** the complete task autonomously

### Project Goal

**Create a simulated humanoid robot that can successfully complete this task:**

> **User Command**: "Go to the kitchen, find the red cup on the table, and bring it to me."

**Expected Behavior**:
1. Robot hears and transcribes the command
2. LLM breaks down the task into subtasks
3. Robot navigates to the kitchen using Nav2
4. Computer vision detects "red cup" on table
5. Robot approaches table and grasps cup
6. Robot navigates back to user
7. Robot hands cup to user

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CAPSTONE SYSTEM                          │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐  ┌────────────────┐  ┌──────────────┐
│  Voice Input  │  │  LLM Planning  │  │  Execution   │
│   (Whisper)   │  │   (GPT-4)      │  │   Engine     │
└───────┬───────┘  └────────┬───────┘  └──────┬───────┘
        │                   │                  │
        └───────────────────┼──────────────────┘
                            ▼
                ┌───────────────────────┐
                │   ROS 2 Middleware    │
                └───────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐  ┌────────────────┐  ┌──────────────┐
│  Navigation   │  │    Vision      │  │ Manipulation │
│    (Nav2)     │  │  (Isaac ROS)   │  │(Isaac Gym/Sim│
└───────────────┘  └────────────────┘  └──────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Isaac Sim (Physics)  │
                │  Simulated Humanoid   │
                └───────────────────────┘
```

## Prerequisites

### Hardware Requirements

**Minimum (for development)**:
- CPU: Intel i7 or AMD Ryzen 7
- RAM: 32 GB
- GPU: NVIDIA RTX 3060 (12GB VRAM)
- Storage: 100 GB SSD

**Recommended (for smooth simulation)**:
- CPU: Intel i9 or AMD Ryzen 9
- RAM: 64 GB
- GPU: NVIDIA RTX 4080 or RTX 3090 (16GB+ VRAM)
- Storage: 256 GB NVMe SSD

### Software Requirements

```bash
# Ubuntu 22.04 LTS
# ROS 2 Humble
# NVIDIA Isaac Sim 2023.1.1+
# Python 3.10+
# CUDA 11.8+

# Install dependencies
sudo apt update
sudo apt install -y \
    ros-humble-desktop \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-slam-toolbox \
    python3-pip \
    git

pip3 install \
    openai-whisper \
    openai \
    transformers \
    torch \
    opencv-python \
    pyaudio
```

## Phase 1: Environment Setup

### Step 1: Create Isaac Sim Environment

Create a realistic kitchen environment in Isaac Sim:

```python
# kitchen_environment.py
from omni.isaac.kit import SimulationApp

# Initialize Isaac Sim
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.core.prims import XFormPrim
import numpy as np

class KitchenEnvironment:
    def __init__(self):
        self.world = World()
        self.setup_kitchen()
        
    def setup_kitchen(self):
        """Create kitchen with furniture and objects"""
        
        # Add ground plane
        self.world.scene.add_default_ground_plane()
        
        # Create kitchen table
        table = VisualCuboid(
            prim_path="/World/Kitchen/Table",
            name="kitchen_table",
            position=np.array([2.0, 0.0, 0.4]),
            size=np.array([1.2, 0.8, 0.05]),
            color=np.array([0.6, 0.4, 0.2])
        )
        
        # Create red cup on table
        red_cup = DynamicCuboid(
            prim_path="/World/Kitchen/RedCup",
            name="red_cup",
            position=np.array([2.0, 0.2, 0.5]),
            size=np.array([0.08, 0.08, 0.12]),
            color=np.array([0.9, 0.1, 0.1]),  # Red color
            mass=0.2
        )
        
        # Create blue cup on table
        blue_cup = DynamicCuboid(
            prim_path="/World/Kitchen/BlueCup",
            name="blue_cup",
            position=np.array([2.0, -0.2, 0.5]),
            size=np.array([0.08, 0.08, 0.12]),
            color=np.array([0.1, 0.1, 0.9]),  # Blue color
            mass=0.2
        )
        
        # Add to scene
        self.world.scene.add(table)
        self.world.scene.add(red_cup)
        self.world.scene.add(blue_cup)
        
        # Create walls
        self.create_walls()
        
        # Add lighting
        self.setup_lighting()
        
    def create_walls(self):
        """Create kitchen walls"""
        # Back wall
        back_wall = VisualCuboid(
            prim_path="/World/Kitchen/BackWall",
            name="back_wall",
            position=np.array([3.0, 0.0, 1.5]),
            size=np.array([0.1, 4.0, 3.0]),
            color=np.array([0.9, 0.9, 0.8])
        )
        self.world.scene.add(back_wall)
        
        # Side walls...
        
    def setup_lighting(self):
        """Add realistic lighting"""
        from omni.isaac.core.utils.prims import create_prim
        
        # Create dome light
        create_prim(
            "/World/DomeLight",
            "DomeLight",
            attributes={"intensity": 1000}
        )

# Initialize environment
env = KitchenEnvironment()
env.world.reset()

# Keep simulation running
while simulation_app.is_running():
    env.world.step(render=True)

simulation_app.close()
```

### Step 2: Load Humanoid Robot

```python
# humanoid_robot.py
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage

class HumanoidRobot:
    def __init__(self, world):
        self.world = world
        self.robot = None
        self.load_robot()
        
    def load_robot(self):
        """Load humanoid robot model"""
        
        # Load robot USD file (e.g., Unitree H1 or custom URDF)
        robot_usd_path = "/Isaac/Robots/Humanoid/humanoid.usd"
        
        add_reference_to_stage(
            usd_path=robot_usd_path,
            prim_path="/World/Humanoid"
        )
        
        # Create robot instance
        self.robot = self.world.scene.add(
            Robot(
                prim_path="/World/Humanoid",
                name="humanoid_robot",
                position=np.array([0.0, 0.0, 1.0])
            )
        )
        
        return self.robot
    
    def get_joint_positions(self):
        """Get current joint positions"""
        return self.robot.get_joint_positions()
    
    def set_joint_positions(self, positions):
        """Set target joint positions"""
        self.robot.set_joint_positions(positions)
    
    def get_end_effector_pose(self):
        """Get end effector (hand) pose"""
        # Get position and orientation of right hand
        # Implementation depends on robot model
        pass

# Usage
env = KitchenEnvironment()
robot = HumanoidRobot(env.world)
```

## Phase 2: Perception System

### Step 3: Object Detection with Computer Vision

```python
# vision_system.py
import cv2
import numpy as np
from groundingdino.util.inference import load_model, predict
import torch

class VisionSystem:
    def __init__(self, camera_interface):
        self.camera = camera_interface
        
        # Load GroundingDINO for open-vocabulary detection
        self.model = load_model(
            config_path="GroundingDINO/config/GroundingDINO_SwinT_OGC.py",
            checkpoint_path="weights/groundingdino_swint_ogc.pth"
        )
        
        # Camera intrinsics (from Isaac Sim camera)
        self.fx = 525.0
        self.fy = 525.0
        self.cx = 320.0
        self.cy = 240.0
        
    def detect_object(self, text_prompt, confidence_threshold=0.35):
        """
        Detect object matching text description
        
        Args:
            text_prompt: Natural language description (e.g., "red cup")
            confidence_threshold: Minimum detection confidence
            
        Returns:
            dict with object_id, bbox, 3d_position, confidence
        """
        # Get RGB and depth images from camera
        rgb_image = self.camera.get_rgb()
        depth_image = self.camera.get_depth()
        
        # Detect objects
        boxes, confidences, labels = predict(
            model=self.model,
            image=rgb_image,
            caption=text_prompt,
            box_threshold=confidence_threshold,
            text_threshold=0.25
        )
        
        if len(boxes) == 0:
            return None
        
        # Get highest confidence detection
        best_idx = np.argmax(confidences)
        bbox = boxes[best_idx]
        confidence = confidences[best_idx]
        
        # Convert 2D bbox to 3D position
        position_3d = self.bbox_to_3d(bbox, depth_image)
        
        return {
            'object_id': f"{text_prompt}_{best_idx}",
            'bbox': bbox,
            'position_3d': position_3d,
            'confidence': confidence
        }
    
    def bbox_to_3d(self, bbox, depth_image):
        """Convert 2D bounding box to 3D position using depth"""
        # Get center of bounding box
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Get depth at center (with some averaging for robustness)
        depth_region = depth_image[
            max(0, center_y-5):min(depth_image.shape[0], center_y+5),
            max(0, center_x-5):min(depth_image.shape[1], center_x+5)
        ]
        depth = np.median(depth_region)
        
        # Unproject to 3D using camera intrinsics
        x = (center_x - self.cx) * depth / self.fx
        y = (center_y - self.cy) * depth / self.fy
        z = depth
        
        return np.array([x, y, z])
    
    def visualize_detection(self, rgb_image, detection):
        """Draw detection on image for debugging"""
        if detection is None:
            return rgb_image
        
        img = rgb_image.copy()
        x1, y1, x2, y2 = detection['bbox']
        
        # Draw bounding box
        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )
        
        # Draw label
        label = f"{detection['object_id']} ({detection['confidence']:.2f})"
        cv2.putText(
            img,
            label,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
        
        return img

# Usage
camera = IsaacSimCamera()  # Your camera interface
vision = VisionSystem(camera)

detection = vision.detect_object("red cup")
if detection:
    print(f"Found object at 3D position: {detection['position_3d']}")
```

## Phase 3: Integration - Complete System

### Step 4: Main Control System

```python
# autonomous_humanoid.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import json

class AutonomousHumanoid(Node):
    def __init__(self):
        super().__init__('autonomous_humanoid')
        
        # Initialize components
        self.whisper = WhisperNode()
        self.llm_planner = GPT4Planner()
        self.vision = VisionSystem(camera)
        self.navigation = NavigationController()
        self.manipulation = ManipulationController()
        
        # State machine
        self.current_state = "IDLE"
        self.current_plan = None
        self.current_action_index = 0
        
        # Publishers/Subscribers
        self.command_sub = self.create_subscription(
            String,
            'voice_commands',
            self.command_callback,
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            'robot_status',
            10
        )
        
        self.get_logger().info("Autonomous Humanoid System Ready")
        
    def command_callback(self, msg):
        """Process voice command"""
        command = msg.data
        self.get_logger().info(f"Received command: {command}")
        
        # Generate plan
        plan = self.llm_planner.plan_task(command)
        self.current_plan = plan['plan']
        self.current_action_index = 0
        
        self.get_logger().info(f"Generated plan with {len(self.current_plan)} actions")
        
        # Start execution
        self.current_state = "EXECUTING"
        self.execute_next_action()
        
    def execute_next_action(self):
        """Execute next action in plan"""
        if self.current_action_index >= len(self.current_plan):
            self.get_logger().info("Plan completed successfully!")
            self.current_state = "IDLE"
            self.publish_status("Task completed")
            return
        
        action = self.current_plan[self.current_action_index]
        action_type = action['action']
        parameters = action['parameters']
        
        self.get_logger().info(f"Executing: {action_type}({parameters})")
        
        # Route to appropriate controller
        if action_type == 'navigate':
            self.execute_navigation(parameters)
        elif action_type == 'search':
            self.execute_search(parameters)
        elif action_type == 'pick':
            self.execute_pick(parameters)
        elif action_type == 'place':
            self.execute_place(parameters)
        else:
            self.get_logger().warn(f"Unknown action: {action_type}")
            self.advance_to_next_action()
    
    def execute_navigation(self, params):
        """Navigate to location"""
        location = params['location']
        self.get_logger().info(f"Navigating to {location}")
        
        # Get goal pose for location
        goal_pose = self.get_location_pose(location)
        
        # Send navigation goal
        success = self.navigation.navigate_to_pose(goal_pose)
        
        if success:
            self.get_logger().info(f"Reached {location}")
            self.advance_to_next_action()
        else:
            self.get_logger().error(f"Navigation failed")
            self.handle_failure()
    
    def execute_search(self, params):
        """Search for object using vision"""
        object_type = params['object_type']
        self.get_logger().info(f"Searching for {object_type}")
        
        # Use vision system to detect object
        detection = self.vision.detect_object(object_type)
        
        if detection:
            self.get_logger().info(
                f"Found {object_type} at {detection['position_3d']}"
            )
            # Store detection for next action
            self.current_detection = detection
            self.advance_to_next_action()
        else:
            self.get_logger().error(f"Could not find {object_type}")
            self.handle_failure()
    
    def execute_pick(self, params):
        """Pick up object"""
        object_id = params.get('object_id')
        
        if not hasattr(self, 'current_detection'):
            self.get_logger().error("No object detected to pick")
            self.handle_failure()
            return
        
        target_pose = self.current_detection['position_3d']
        self.get_logger().info(f"Picking object at {target_pose}")
        
        # Execute pick action
        success = self.manipulation.pick_object(target_pose)
        
        if success:
            self.get_logger().info("Successfully picked object")
            self.advance_to_next_action()
        else:
            self.get_logger().error("Pick failed")
            self.handle_failure()
    
    def execute_place(self, params):
        """Place held object"""
        location = params['location']
        self.get_logger().info(f"Placing object at {location}")
        
        # Get placement pose
        place_pose = self.get_location_pose(location)
        
        # Execute place action
        success = self.manipulation.place_object(place_pose)
        
        if success:
            self.get_logger().info("Successfully placed object")
            self.advance_to_next_action()
        else:
            self.get_logger().error("Place failed")
            self.handle_failure()
    
    def advance_to_next_action(self):
        """Move to next action in plan"""
        self.current_action_index += 1
        self.execute_next_action()
    
    def handle_failure(self):
        """Handle action failure"""
        self.get_logger().error("Action failed, attempting recovery...")
        
        # Could implement recovery strategies here
        # For now, just abort
        self.current_state = "IDLE"
        self.publish_status("Task failed")
    
    def get_location_pose(self, location_name):
        """Get pose for named location"""
        # Define known locations in environment
        locations = {
            'kitchen': PoseStamped(
                pose=Pose(position=Point(x=2.0, y=0.0, z=0.0))
            ),
            'living_room': PoseStamped(
                pose=Pose(position=Point(x=-2.0, y=0.0, z=0.0))
            ),
            'user': PoseStamped(
                pose=Pose(position=Point(x=0.0, y=2.0, z=0.0))
            )
        }
        
        return locations.get(location_name)
    
    def publish_status(self, status):
        """Publish status message"""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousHumanoid()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Phase 4: Testing and Evaluation

### Step 5: Create Test Scenarios

```python
# test_scenarios.py
class CapstoneEvaluator:
    def __init__(self, robot_system):
        self.robot = robot_system
        self.test_results = []
        
    def run_all_tests(self):
        """Run complete test suite"""
        
        test_scenarios = [
            {
                'name': 'Basic Fetch',
                'command': 'Go to the kitchen and bring me the red cup',
                'expected_steps': ['navigate', 'search', 'pick', 'navigate', 'place'],
                'success_criteria': {
                    'object_retrieved': 'red_cup',
                    'final_location': 'user'
                }
            },
            {
                'name': 'Object Disambiguation',
                'command': 'Bring me the blue cup from the kitchen',
                'expected_steps': ['navigate', 'search', 'pick', 'navigate', 'place'],
                'success_criteria': {
                    'object_retrieved': 'blue_cup',
                    'correct_object': True
                }
            },
            {
                'name': 'Multi-Step Task',
                'command': 'Put all cups in the sink',
                'expected_steps': ['navigate', 'search', 'pick', 'place', 'search', 'pick', 'place'],
                'success_criteria': {
                    'objects_moved': 2,
                    'final_location': 'sink'
                }
            },
            {
                'name': 'Navigation with Obstacles',
                'command': 'Navigate around the chair to reach the table',
                'expected_steps': ['navigate'],
                'success_criteria': {
                    'path_clear': True,
                    'collision_free': True
                }
            }
        ]
        
        for scenario in test_scenarios:
            result = self.run_scenario(scenario)
            self.test_results.append(result)
            
        return self.generate_report()
    
    def run_scenario(self, scenario):
        """Run single test scenario"""
        print(f"\n{'='*60}")
        print(f"Testing: {scenario['name']}")
        print(f"Command: {scenario['command']}")
        print(f"{'='*60}")
        
        # Reset environment
        self.robot.reset()
        
        # Execute command
        start_time = time.time()
        success = self.robot.execute_command(scenario['command'])
        execution_time = time.time() - start_time
        
        # Evaluate results
        criteria_met = self.evaluate_success_criteria(
            scenario['success_criteria']
        )
        
        result = {
            'scenario': scenario['name'],
            'success': success and criteria_met,
            'execution_time': execution_time,
            'criteria_met': criteria_met
        }
        
        print(f"Result: {'✓ PASS' if result['success'] else '✗ FAIL'}")
        print(f"Time: {execution_time:.2f}s")
        
        return result
    
    def evaluate_success_criteria(self, criteria):
        """Check if success criteria are met"""
        # Implementation depends on your environment
        # Check object positions, robot state, etc.
        pass
    
    def generate_report(self):
        """Generate evaluation report"""
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['success'])
        
        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT")
        print(f"{'='*60}")
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {passed/total*100:.1f}%")
        print(f"{'='*60}")
        
        return {
            'total': total,
            'passed': passed,
            'success_rate': passed / total
        }

# Run evaluation
evaluator = CapstoneEvaluator(autonomous_humanoid)
report = evaluator.run_all_tests()
```

## Phase 5: Documentation and Presentation

### Project Documentation Structure

Create a comprehensive project report:

```markdown
# Capstone Project: Autonomous Humanoid Robot

## Executive Summary
- Project overview
- Key achievements
- Challenges faced
- Lessons learned

## System Architecture
- Component diagram
- Data flow
- Integration points

## Implementation Details

### 1. Voice Interface (Whisper)
- Model selection rationale
- Performance metrics
- Accuracy on test set

### 2. Cognitive Planning (LLM)
- Prompt engineering approach
- Planning accuracy
- Execution success rate

### 3. Navigation (Nav2)
- Map building process
- Path planning algorithm
- Obstacle avoidance performance

### 4. Vision System
- Object detection accuracy
- 3D localization precision
- Robustness to lighting conditions

### 5. Manipulation
- Grasping success rate
- Placement accuracy
- Force control implementation

## Evaluation Results
- Test scenarios
- Success rates
- Performance metrics
- Comparison to baseline

## Demo Video Script
1. Introduction (0:00-0:15)
   - Project overview
   - System components

2. Voice Command (0:15-0:30)
   - Show voice input
   - Display transcription

3. Planning Phase (0:30-0:45)
   - Show LLM generating plan
   - Display action sequence

4. Execution (0:45-1:20)
   - Navigation to kitchen
   - Object detection
   - Grasping
   - Return to user

5. Conclusion (1:20-1:30)
   - Results summary
   - Future improvements

## Future Enhancements
- Multi-robot coordination
- Learning from demonstration
- Improved error recovery
- Real hardware deployment
```

### Creating the Demo Video

```python
# demo_video_recorder.py
import cv2
import numpy as np

class DemoRecorder:
    def __init__(self, video_path="capstone_demo.mp4"):
        self.video_path = video_path
        self.writer = None
        self.fps = 30
        self.frame_size = (1920, 1080)
        
    def start_recording(self):
        """Initialize video writer"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.video_path,
            fourcc,
            self.fps,
            self.frame_size
        )
        
    def add_frame(self, frame, overlay_text=None):
        """Add frame to video with optional text overlay"""
        # Resize frame to target size
        frame_resized = cv2.resize(frame, self.frame_size)
        
        # Add text overlay if provided
        if overlay_text:
            cv2.putText(
                frame_resized,
                overlay_text,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )
        
        self.writer.write(frame_resized)
        
    def finish_recording(self):
        """Finalize video"""
        self.writer.release()
        print(f"Demo video saved to {self.video_path}")

# Usage
recorder = DemoRecorder()
recorder.start_recording()

# Record demo execution
# ... your demo code ...

recorder.finish_recording()
```

## Grading Rubric

Your capstone project will be evaluated on:

### Technical Implementation (60 points)
- **Voice Recognition (10 pts)**: Accurate transcription, low latency
- **LLM Planning (15 pts)**: Correct task decomposition, robust planning
- **Navigation (10 pts)**: Collision-free, efficient paths
- **Vision (10 pts)**: Accurate object detection and 3D localization
- **Manipulation (10 pts)**: Successful grasping and placement
- **Integration (5 pts)**: Smooth component interaction

### Performance (20 points)
- **Success Rate (10 pts)**: Percentage of successful task completions
- **Robustness (10 pts)**: Handling of edge cases and failures

### Documentation (10 points)
- **Code Quality (5 pts)**: Clean, well-commented code
- **Report (5 pts)**: Comprehensive documentation

### Presentation (10 points)
- **Demo Video (5 pts)**: Clear demonstration under 90 seconds
- **Technical Explanation (5 pts)**: Clear articulation of approach

## Common Challenges and Solutions

### Challenge 1: Isaac Sim Crashes
**Solution**: Reduce simulation complexity, use lower-quality rendering

### Challenge 2: LLM API Costs
**Solution**: Cache plans for common commands, use local models for development

### Challenge 3: Object Detection Failures
**Solution**: Improve lighting, add data augmentation, use multiple views

### Challenge 4: Navigation Gets Stuck
**Solution**: Tune Nav2 parameters, add recovery behaviors, improve costmap

### Challenge 5: Grasping Misses Object
**Solution**: Fine-tune grasp pose estimation, add force feedback, use multiple attempts

## Conclusion

This capstone project brings together all the skills you've developed:
- Physical AI fundamentals
- ROS 2 programming
- Simulation with Isaac and Gazebo
- Deep learning for perception
- LLMs for cognitive reasoning
- Navigation and manipulation

Successfully completing this project demonstrates you can build real-world autonomous robotics systems—a critical skill as humanoid robots enter homes, hospitals, and workplaces.

**Your autonomous humanoid is ready to enter the real world!**

## Next Steps

After completing this capstone:
1. Deploy to real hardware (Jetson + humanoid platform)
2. Contribute to open-source robotics projects
3. Build your own AI robotics startup
4. Join the Panaversity core team as a founding member

Congratulations on completing the Physical AI & Humanoid Robotics course!