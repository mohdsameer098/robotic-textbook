# Module 3, Chapter 4 - Reinforcement Learning for Robot Control

## Introduction to Reinforcement Learning in Robotics

Reinforcement Learning (RL) represents a paradigm shift in how we program robots. Instead of explicitly coding every behavior, we teach robots to learn optimal actions through trial and error, much like how humans learn new skills.

### Why RL for Humanoid Robots?

Traditional robotics relies on hand-crafted control policies that require:
- Extensive domain knowledge
- Manual tuning for each scenario
- Difficulty adapting to new situations

Humanoid robots present unique challenges:
- **High-dimensional control**: 30+ degrees of freedom
- **Dynamic balance**: Constant adjustment needed
- **Complex interactions**: Contact forces with environment
- **Generalization**: Must work in varied environments

RL addresses these challenges by:
- Learning from experience rather than explicit programming
- Discovering optimal behaviors through exploration
- Adapting to new situations through continuous learning
- Handling complex, high-dimensional state spaces

## Core Concepts of Reinforcement Learning

### The RL Framework

At its core, RL involves an **agent** (the robot) interacting with an **environment**:

1. **Agent** observes the current **state** (s)
2. **Agent** takes an **action** (a)
3. **Environment** transitions to a new **state** (s')
4. **Agent** receives a **reward** (r)
5. Goal: Learn a **policy** (π) that maximizes cumulative reward

### Key Components

#### 1. State Space (S)
For humanoid robots, the state includes:
- Joint positions and velocities (30-50 dimensions)
- Base orientation and linear/angular velocity
- Contact forces with ground
- Goal location and orientation
- Obstacle positions

```python
# Example state representation
state = {
    'joint_positions': [θ1, θ2, ..., θn],
    'joint_velocities': [ω1, ω2, ..., ωn],
    'base_position': [x, y, z],
    'base_orientation': [roll, pitch, yaw],
    'base_linear_velocity': [vx, vy, vz],
    'base_angular_velocity': [ωx, ωy, ωz],
    'contact_forces': [f_left_foot, f_right_foot],
    'goal_relative_position': [dx, dy, dz]
}
```

#### 2. Action Space (A)
Actions represent what the robot can do:

**Discrete Actions** (simpler, less control):
- Walk forward
- Turn left
- Turn right
- Stop

**Continuous Actions** (more natural, complex):
- Target joint positions or torques
- Desired velocities for each joint
- Force commands for actuators

```python
# Continuous action space for humanoid
action = {
    'joint_torques': [τ1, τ2, ..., τn],  # Torque for each joint
    # or
    'joint_target_positions': [θ1_target, θ2_target, ..., θn_target]
}
```

#### 3. Reward Function (R)
The reward function shapes the learned behavior. For humanoid locomotion:

```python
def compute_reward(state, action, next_state):
    reward = 0.0
    
    # Forward progress reward
    reward += 1.0 * next_state['base_linear_velocity'][0]  # vx
    
    # Penalty for falling
    if next_state['base_position'][2] < 0.3:  # height threshold
        reward -= 10.0
    
    # Penalty for excessive energy use
    reward -= 0.001 * sum(action['joint_torques']**2)
    
    # Penalty for deviation from upright posture
    reward -= 0.5 * (abs(next_state['base_orientation'][0]) + 
                     abs(next_state['base_orientation'][1]))
    
    # Bonus for reaching goal
    distance_to_goal = np.linalg.norm(next_state['goal_relative_position'])
    if distance_to_goal < 0.5:
        reward += 100.0
    
    return reward
```

#### 4. Policy (π)
The policy maps states to actions: π(s) → a

Types of policies:
- **Deterministic**: Always outputs same action for same state
- **Stochastic**: Outputs probability distribution over actions

## Popular RL Algorithms for Robotics

### 1. Proximal Policy Optimization (PPO)

**Why PPO?**
- Most popular for robotics
- Stable training
- Sample efficient
- Works well with continuous control

**How PPO Works:**
- On-policy algorithm (learns from recent experience)
- Clips policy updates to prevent large, destabilizing changes
- Uses advantage function to determine good/bad actions

**Key Hyperparameters:**
```python
ppo_config = {
    'learning_rate': 3e-4,
    'gamma': 0.99,              # Discount factor
    'clip_range': 0.2,          # Clipping parameter
    'n_epochs': 10,             # Training epochs per update
    'batch_size': 64,
    'n_steps': 2048,            # Steps before update
}
```

### 2. Soft Actor-Critic (SAC)

**Why SAC?**
- Off-policy (learns from any experience)
- Sample efficient
- Encourages exploration through entropy maximization
- Excellent for continuous control

**Advantages for Humanoids:**
- Can reuse old experience (replay buffer)
- More sample efficient than PPO
- Better exploration in complex environments

### 3. Deep Deterministic Policy Gradient (DDPG)

**Characteristics:**
- Off-policy actor-critic
- Deterministic policy
- Uses experience replay
- Good for precise control tasks

### 4. Trust Region Policy Optimization (TRPO)

**Characteristics:**
- Predecessor to PPO
- Theoretically sound policy updates
- More complex to implement
- Guarantees monotonic improvement

## NVIDIA Isaac Gym for RL Training

### What is Isaac Gym?

Isaac Gym is NVIDIA's physics simulation environment specifically designed for RL:
- **Massively parallel**: Thousands of environments simultaneously
- **GPU-accelerated**: Physics runs entirely on GPU
- **Fast**: 100x-1000x faster than CPU-based simulators
- **Integrated**: Direct tensor interface with PyTorch

### Key Features

1. **Parallel Environments**
   - Train with 4096+ environments at once
   - Each environment runs independently
   - Massive speedup in data collection

2. **GPU-Accelerated Physics**
   - PhysX 5.0 engine
   - Runs entirely on GPU
   - No CPU-GPU data transfer bottleneck

3. **Tensor API**
   - Direct access to simulation tensors
   - Zero-copy operations with PyTorch
   - Seamless integration with RL frameworks

### Installing Isaac Gym

Isaac Gym is part of the Isaac Sim ecosystem:

```bash
# Download from NVIDIA (requires NVIDIA account)
# https://developer.nvidia.com/isaac-gym

# Extract the package
tar -xvf IsaacGym_Preview_4_Package.tar.gz

# Navigate to directory
cd isaacgym/python

# Install dependencies
pip install -e .
```

**System Requirements:**
- NVIDIA GPU (RTX series recommended)
- Ubuntu 20.04 or 22.04
- CUDA 11.3+
- Python 3.7+

### Setting Up Isaac Gym Environment

```python
from isaacgym import gymapi
from isaacgym import gymutil
import torch

# Initialize gym
gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments(
    description="Humanoid RL Training"
)

# Create simulation parameters
sim_params = gymapi.SimParams()
sim_params.dt = 0.01  # 100 Hz simulation
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

# Configure physics engine
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.02
sim_params.physx.rest_offset = 0.001

# Create simulation
sim = gym.create_sim(
    args.compute_device_id,
    args.graphics_device_id,
    gymapi.SIM_PHYSX,
    sim_params
)
```

## Building a Humanoid RL Training Environment

### Step 1: Define the Environment Class

```python
import torch
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

class HumanoidEnv:
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        self.sim_device = sim_device
        self.graphics_device_id = graphics_device_id
        self.headless = headless
        
        # Environment configuration
        self.num_envs = cfg["env"]["numEnvs"]
        self.num_obs = cfg["env"]["numObservations"]
        self.num_actions = cfg["env"]["numActions"]
        
        # Initialize Isaac Gym
        self.gym = gymapi.acquire_gym()
        self.sim = self._create_sim()
        
        # Create environments
        self.envs = []
        self.humanoid_handles = []
        self._create_envs()
        
        # Prepare simulation
        self.gym.prepare_sim(self.sim)
        
        # Get initial state tensors
        self._init_tensors()
        
    def _create_sim(self):
        """Create simulation with physics parameters"""
        sim_params = gymapi.SimParams()
        sim_params.dt = 0.01
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        
        return self.gym.create_sim(
            self.sim_device,
            self.graphics_device_id,
            gymapi.SIM_PHYSX,
            sim_params
        )
    
    def _create_envs(self):
        """Create parallel environments with humanoid robots"""
        # Load humanoid asset
        asset_root = "assets"
        asset_file = "humanoid.urdf"
        
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.angular_damping = 0.01
        
        humanoid_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        
        # Configure environment
        spacing = 2.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(np.sqrt(self.num_envs))
        
        # Create ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        
        # Create environments
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)
            
            # Add humanoid to environment
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0, 1.0)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            
            humanoid_handle = self.gym.create_actor(
                env, humanoid_asset, pose, "humanoid", i, 1
            )
            self.humanoid_handles.append(humanoid_handle)
    
    def _init_tensors(self):
        """Initialize state tensors from simulation"""
        # Get actor root state tensor (position, orientation, velocities)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        
        # Get DOF state tensor (joint positions and velocities)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        
        # Get rigid body state tensor
        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(rb_state_tensor)
        
        # Refresh tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
    
    def reset(self):
        """Reset all environments"""
        # Reset root states
        self.root_states[:, :3] = torch.tensor([0, 0, 1.0])  # Position
        self.root_states[:, 3:7] = torch.tensor([0, 0, 0, 1])  # Orientation (quat)
        self.root_states[:, 7:] = 0  # Velocities
        
        # Reset DOF states
        self.dof_states[:, :] = 0
        
        # Apply resets
        env_ids = torch.arange(self.num_envs, device=self.sim_device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids.to(torch.int32)),
            len(env_ids)
        )
        
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(env_ids.to(torch.int32)),
            len(env_ids)
        )
        
        return self.get_observations()
    
    def get_observations(self):
        """Compute observations from current state"""
        # Extract relevant states
        root_pos = self.root_states[:, :3]
        root_orient = self.root_states[:, 3:7]
        root_lin_vel = self.root_states[:, 7:10]
        root_ang_vel = self.root_states[:, 10:13]
        
        dof_pos = self.dof_states[:, 0].view(self.num_envs, -1)
        dof_vel = self.dof_states[:, 1].view(self.num_envs, -1)
        
        # Concatenate observations
        obs = torch.cat([
            root_pos,
            root_orient,
            root_lin_vel,
            root_ang_vel,
            dof_pos,
            dof_vel
        ], dim=-1)
        
        return obs
    
    def step(self, actions):
        """Execute actions and step simulation"""
        # Apply actions as joint torques
        torques = actions * self.cfg["env"]["maxEffort"]
        self.gym.set_dof_actuation_force_tensor(
            self.sim,
            gymtorch.unwrap_tensor(torques)
        )
        
        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # Refresh tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # Compute observations and rewards
        obs = self.get_observations()
        rewards = self.compute_rewards()
        dones = self.compute_dones()
        
        return obs, rewards, dones, {}
    
    def compute_rewards(self):
        """Compute rewards for current state"""
        # Forward velocity reward
        forward_vel = self.root_states[:, 7]  # vx
        rewards = forward_vel
        
        # Penalty for falling
        height = self.root_states[:, 2]
        rewards -= torch.where(height < 0.3, torch.ones_like(height) * 10.0, torch.zeros_like(height))
        
        # Penalty for excessive energy
        dof_vel = self.dof_states[:, 1].view(self.num_envs, -1)
        energy_penalty = 0.001 * torch.sum(dof_vel ** 2, dim=-1)
        rewards -= energy_penalty
        
        return rewards
    
    def compute_dones(self):
        """Determine which environments are done"""
        # Terminate if robot falls
        height = self.root_states[:, 2]
        dones = height < 0.3
        
        return dones
```

### Step 2: Integrate with RL Framework (PPO)