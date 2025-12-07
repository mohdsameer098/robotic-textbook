# Module 3, Chapter 5 - Sim-to-Real Transfer

## Introduction to the Reality Gap

The **reality gap** is one of the most challenging problems in robotics: policies trained in simulation often fail catastrophically when deployed to real robots. Despite perfect performance in virtual environments, real-world deployment reveals fundamental differences between simulation and reality.

### Why Does the Reality Gap Exist?

**Physical Discrepancies:**
- Simulation uses simplified physics models
- Real-world friction, damping, and contact dynamics are complex
- Actuator delays and mechanical compliance aren't perfectly modeled
- Sensor noise and latency differ from simulation

**Environmental Differences:**
- Lighting conditions affect vision systems
- Ground surfaces vary in texture and compliance
- Air resistance and temperature affect performance
- Unexpected obstacles and dynamic environments

**Model Inaccuracies:**
- Robot parameters (mass, inertia, dimensions) have manufacturing tolerances
- Wear and tear changes robot properties over time
- Simplified kinematic and dynamic models
- Unmodeled degrees of freedom (cable flexibility, joint backlash)

### The Cost of the Reality Gap

For humanoid robots, the reality gap is particularly expensive:
- **Safety risks**: Falling humanoids can damage themselves or surroundings
- **Hardware costs**: Real humanoid robots cost $15,000-$150,000+
- **Training time**: Real-world learning is 100-1000x slower than simulation
- **Human supervision**: Requires constant monitoring for safety

**Goal of Sim-to-Real Transfer**: Train policies in fast, safe, cheap simulation that work reliably on expensive, fragile, slow real hardware.

## Fundamental Approaches to Sim-to-Real Transfer

### 1. System Identification (The Classical Approach)

**Concept**: Measure real robot parameters precisely, then update simulation to match reality.

**Process:**
1. Measure physical properties (masses, lengths, friction coefficients)
2. Calibrate sensors and actuators
3. Update simulation parameters
4. Validate with real-world tests
5. Iterate until simulation matches reality

**Example: Measuring Joint Friction**

```python
import numpy as np
import matplotlib.pyplot as plt

def measure_joint_friction(robot, joint_id, velocities):
    """
    Apply constant velocities and measure required torques
    to estimate friction parameters
    """
    measured_torques = []
    
    for vel in velocities:
        # Command constant velocity
        robot.set_joint_velocity(joint_id, vel)
        
        # Wait for steady state
        time.sleep(2.0)
        
        # Measure required torque
        torque = robot.get_joint_torque(joint_id)
        measured_torques.append(torque)
    
    # Fit friction model: τ = b*v + c*sign(v)
    # b = viscous friction, c = coulomb friction
    
    # Separate positive and negative velocities
    pos_mask = np.array(velocities) > 0
    
    # Linear regression for viscous component
    b = np.polyfit(velocities, measured_torques, 1)[0]
    
    # Coulomb friction from offset
    c = np.mean(np.abs(measured_torques) - b * np.abs(velocities))
    
    return {'viscous': b, 'coulomb': c}

# Example usage
velocities = np.linspace(-2.0, 2.0, 20)
friction_params = measure_joint_friction(robot, joint_id=0, velocities=velocities)

print(f"Viscous friction: {friction_params['viscous']:.4f} Nm/(rad/s)")
print(f"Coulomb friction: {friction_params['coulomb']:.4f} Nm")
```

**Updating Simulation Parameters:**

```python
# In Isaac Gym or Gazebo configuration
joint_properties = {
    'damping': friction_params['viscous'],
    'friction_loss': friction_params['coulomb'],
    'armature': 0.001,  # Motor inertia
}
```

**Limitations:**
- Time-consuming (days to weeks)
- Requires specialized equipment
- Parameters change over time (wear and tear)
- Some properties difficult to measure (contact dynamics)
- Doesn't capture all complexity

### 2. Domain Randomization (The Modern Approach)

**Concept**: Instead of making simulation match one reality, make the policy robust to many possible realities.

**Philosophy**: If the policy works across a wide range of simulated parameters, it should work in the real world (which is just one point in that range).

**What to Randomize:**

#### Physical Parameters
- Mass: ±20% variation
- Dimensions: ±5% variation  
- Center of mass location: ±2cm
- Joint friction: 0.5x to 2x nominal
- Joint damping: 0.5x to 2x nominal
- Ground friction: 0.3 to 1.5
- Motor strength: ±15%

#### Sensor Parameters
- Camera noise: Gaussian, salt-and-pepper
- Depth sensor accuracy: ±5mm
- IMU drift and bias
- Latency: 0-50ms random delays
- Dropped measurements: 5% packet loss

#### Environmental Parameters  
- Lighting: brightness, shadows, highlights
- Ground slope: ±10 degrees
- Ground compliance: rigid to soft
- Wind forces: random perturbations
- Object textures and colors

**Implementation in Isaac Gym:**

```python
import torch
from isaacgym import gymapi

class DomainRandomization:
    def __init__(self, gym, sim, env_ids):
        self.gym = gym
        self.sim = sim
        self.env_ids = env_ids
        
    def randomize_physics(self, actor_handles):
        """Randomize physical properties of actors"""
        num_actors = len(actor_handles)
        
        # Randomize mass
        mass_scale = torch.rand(num_actors) * 0.4 + 0.8  # 0.8 to 1.2
        
        # Randomize friction
        friction_scale = torch.rand(num_actors) * 1.0 + 0.5  # 0.5 to 1.5
        
        # Randomize damping
        damping_scale = torch.rand(num_actors) * 1.0 + 0.5  # 0.5 to 1.5
        
        for i, actor_handle in enumerate(actor_handles):
            # Apply mass scaling
            props = self.gym.get_actor_rigid_body_properties(
                self.env_ids[i], actor_handle
            )
            for prop in props:
                prop.mass *= mass_scale[i].item()
            self.gym.set_actor_rigid_body_properties(
                self.env_ids[i], actor_handle, props
            )
            
            # Apply friction scaling
            shape_props = self.gym.get_actor_rigid_shape_properties(
                self.env_ids[i], actor_handle
            )
            for prop in shape_props:
                prop.friction *= friction_scale[i].item()
            self.gym.set_actor_rigid_shape_properties(
                self.env_ids[i], actor_handle, shape_props
            )
            
            # Apply damping scaling
            dof_props = self.gym.get_actor_dof_properties(
                self.env_ids[i], actor_handle
            )
            for prop in dof_props:
                prop.damping *= damping_scale[i].item()
            self.gym.set_actor_dof_properties(
                self.env_ids[i], actor_handle, dof_props
            )
    
    def randomize_visual(self, camera_handles):
        """Randomize visual sensor properties"""
        for cam_handle in camera_handles:
            # Randomize lighting
            brightness = torch.rand(1).item() * 0.4 + 0.8  # 0.8 to 1.2
            
            # Apply color gain (simulates lighting changes)
            color_gain = torch.rand(3) * 0.4 + 0.8  # Per-channel variation
            
            # This would be applied in the image processing pipeline
            # (Isaac Gym doesn't have direct API for this, so we'd post-process)
    
    def add_noise_to_observations(self, observations):
        """Add sensor noise to observations"""
        # Gaussian noise on joint positions
        joint_pos_noise = torch.randn_like(observations['joint_pos']) * 0.01
        
        # Gaussian noise on joint velocities  
        joint_vel_noise = torch.randn_like(observations['joint_vel']) * 0.1
        
        # IMU noise (orientation)
        imu_noise = torch.randn_like(observations['orientation']) * 0.02
        
        # Add random latency (simulated by using slightly old data)
        # This requires maintaining a history buffer
        
        noisy_obs = observations.copy()
        noisy_obs['joint_pos'] += joint_pos_noise
        noisy_obs['joint_vel'] += joint_vel_noise  
        noisy_obs['orientation'] += imu_noise
        
        return noisy_obs
    
    def randomize_external_forces(self, actor_handles):
        """Apply random external forces (wind, pushes)"""
        num_actors = len(actor_handles)
        
        # Random force magnitudes
        force_mag = torch.rand(num_actors) * 20.0  # 0-20N
        
        # Random directions
        force_dir = torch.randn(num_actors, 3)
        force_dir = force_dir / torch.norm(force_dir, dim=1, keepdim=True)
        
        forces = force_mag.unsqueeze(1) * force_dir
        
        for i, actor_handle in enumerate(actor_handles):
            force = gymapi.Vec3(
                forces[i, 0].item(),
                forces[i, 1].item(),
                forces[i, 2].item()
            )
            self.gym.apply_rigid_body_force_at_pos_tensors(
                self.sim,
                gymtorch.unwrap_tensor(forces),
                gymtorch.unwrap_tensor(torch.zeros_like(forces)),  # positions
                gymapi.ENV_SPACE
            )
```

**Training Loop with Domain Randomization:**

```python
def train_with_domain_randomization(env, policy, num_iterations):
    randomizer = DomainRandomization(env.gym, env.sim, env.env_ids)
    
    for iteration in range(num_iterations):
        # Randomize at the start of each episode
        if iteration % env.max_episode_length == 0:
            randomizer.randomize_physics(env.actor_handles)
            
        # Every few steps, apply random disturbances
        if iteration % 10 == 0:
            randomizer.randomize_external_forces(env.actor_handles)
        
        # Get observations with noise
        obs = env.get_observations()
        obs = randomizer.add_noise_to_observations(obs)
        
        # Policy acts on noisy observations
        actions = policy(obs)
        
        # Step environment
        next_obs, rewards, dones, info = env.step(actions)
        
        # Train policy
        policy.update(obs, actions, rewards, next_obs, dones)
```

**Advantages:**
- No need for accurate system identification
- Naturally handles manufacturing variations
- Policy becomes robust to uncertainties
- Works even with imperfect simulation

**Disadvantages:**
- May be overly conservative (too cautious)
- Requires massive compute (many randomized simulations)
- Can slow down learning
- Harder to debug failures

### 3. Adversarial Domain Randomization (ADR)

**Concept**: Automatically adjust randomization ranges to make training challenging but not impossible.

**Problem with Fixed Randomization:** 
- Too little → policy doesn't generalize
- Too much → policy never learns

**ADR Solution**: Start with narrow ranges, gradually increase difficulty based on policy performance.

```python
class AdversarialDomainRandomization:
    def __init__(self, initial_ranges):
        self.param_ranges = initial_ranges  # e.g., {'mass_scale': [0.9, 1.1]}
        self.performance_buffer = []
        self.adjustment_rate = 0.1
        
    def update_ranges(self, success_rate):
        """Adjust randomization ranges based on policy performance"""
        self.performance_buffer.append(success_rate)
        
        # If policy succeeds > 80%, increase difficulty
        if len(self.performance_buffer) > 100:
            avg_success = np.mean(self.performance_buffer[-100:])
            
            if avg_success > 0.8:
                # Expand randomization ranges
                for param, (low, high) in self.param_ranges.items():
                    range_width = high - low
                    expansion = range_width * self.adjustment_rate
                    self.param_ranges[param] = [
                        low - expansion/2,
                        high + expansion/2
                    ]
                print(f"Increased difficulty: {self.param_ranges}")
                
            elif avg_success < 0.3:
                # Reduce randomization ranges  
                for param, (low, high) in self.param_ranges.items():
                    range_width = high - low
                    reduction = range_width * self.adjustment_rate
                    self.param_ranges[param] = [
                        low + reduction/2,
                        high - reduction/2
                    ]
                print(f"Decreased difficulty: {self.param_ranges}")
    
    def sample_parameters(self):
        """Sample random parameters within current ranges"""
        sampled = {}
        for param, (low, high) in self.param_ranges.items():
            sampled[param] = np.random.uniform(low, high)
        return sampled
```

### 4. Sim-to-Real via Privileged Learning

**Concept**: Train with access to privileged information (ground truth) in simulation, then deploy without it.

**Use Case**: Real sensors are noisy, but simulation knows ground truth. Use this during training.

```python
class PrivilegedPolicy:
    def __init__(self, obs_dim, privileged_dim, action_dim):
        # Student network (deployed to robot)
        self.student = StudentNetwork(obs_dim, action_dim)
        
        # Teacher network (only in simulation)
        self.teacher = TeacherNetwork(obs_dim + privileged_dim, action_dim)
        
    def train_step(self, obs, privileged_info, true_action):
        """Train both networks"""
        # Teacher has access to privileged info
        teacher_action = self.teacher(
            torch.cat([obs, privileged_info], dim=-1)
        )
        
        # Student only sees observations
        student_action = self.student(obs)
        
        # Teacher learns from environment reward
        teacher_loss = compute_rl_loss(teacher_action, true_action)
        
        # Student learns to imitate teacher (distillation)
        student_loss = F.mse_loss(student_action, teacher_action.detach())
        
        return teacher_loss, student_loss
    
    def deploy(self, obs):
        """Only use student network for deployment"""
        return self.student(obs)
```

**Privileged Information Examples:**
- Ground truth object positions (vs noisy vision)
- Perfect contact forces (vs noisy force sensors)
- True terrain elevation (vs uncertain depth maps)
- Exact external forces (vs unknown disturbances)

## Progressive Training Strategies

### Curriculum Learning

**Concept**: Start with easy tasks, gradually increase difficulty.

**For Humanoid Locomotion:**

```python
class LocomotionCurriculum:
    def __init__(self):
        self.current_stage = 0
        self.stages = [
            {
                'name': 'standing',
                'target_velocity': 0.0,
                'terrain': 'flat',
                'duration': 50000,
            },
            {
                'name': 'slow_walk',
                'target_velocity': 0.5,
                'terrain': 'flat',
                'duration': 100000,
            },
            {
                'name': 'normal_walk',
                'target_velocity': 1.0,
                'terrain': 'flat',  
                'duration': 150000,
            },
            {
                'name': 'rough_terrain',
                'target_velocity': 1.0,
                'terrain': 'rough',
                'duration': 200000,
            },
            {
                'name': 'stairs',
                'target_velocity': 0.8,
                'terrain': 'stairs',
                'duration': 200000,
            },
        ]
        self.total_steps = 0
    
    def get_current_stage(self):
        """Get current training stage configuration"""
        cumulative_steps = 0
        for stage in self.stages:
            cumulative_steps += stage['duration']
            if self.total_steps < cumulative_steps:
                return stage
        return self.stages[-1]  # Final stage
    
    def step(self):
        """Increment training steps"""
        self.total_steps += 1
        
        # Check if we should advance to next stage
        stage = self.get_current_stage()
        if self.total_steps % 10000 == 0:
            print(f"Stage: {stage['name']}, Step: {self.total_steps}")
```

### Reward Shaping Through Stages

Early stages provide more guidance:

```python
def compute_staged_reward(state, action, stage_name):
    reward = 0.0
    
    if stage_name == 'standing':
        # Dense rewards for maintaining balance
        reward += 1.0  # Just for staying upright
        reward -= 0.1 * abs(state['roll']) + 0.1 * abs(state['pitch'])
        reward -= 0.001 * torch.sum(action ** 2)  # Energy penalty
        
    elif stage_name == 'slow_walk':
        # Reward forward progress, but gently
        reward += 2.0 * state['forward_velocity']
        reward -= 0.5 * abs(state['roll']) + 0.5 * abs(state['pitch'])
        reward -= 0.01 * torch.sum(action ** 2)
        
    elif stage_name == 'normal_walk':
        # Stronger emphasis on speed
        reward += 5.0 * state['forward_velocity']
        reward -= 1.0 * abs(state['roll']) + 1.0 * abs(state['pitch'])
        reward -= 0.01 * torch.sum(action ** 2)
        
    elif stage_name == 'rough_terrain':
        # Prioritize stability over speed
        reward += 3.0 * state['forward_velocity']
        reward -= 2.0 * abs(state['roll']) + 2.0 * abs(state['pitch'])
        reward += 0.5 if state['foot_contact_stable'] else -1.0
        
    return reward
```

## Validation and Testing

### Sim-to-Sim Transfer (Sanity Check)

Before deploying to hardware, test across different simulators:

```python
# Train in Isaac Gym
policy = train_in_isaac_gym(env_config)

# Test in Gazebo  
gazebo_env = create_gazebo_env(env_config)
gazebo_success_rate = evaluate_policy(policy, gazebo_env, num_episodes=100)

# Test in PyBullet
pybullet_env = create_pybullet_env(env_config)
pybullet_success_rate = evaluate_policy(policy, pybullet_env, num_episodes=100)

print(f"Isaac Gym → Gazebo: {gazebo_success_rate:.2%}")
print(f"Isaac Gym → PyBullet: {pybullet_success_rate:.2%}")

# If policy fails in other simulators, it will likely fail in reality
```

### Gradual Real-World Deployment

**Phase 1: Constrained Testing**
- Robot suspended in harness
- Limited joint ranges
- Emergency stop always ready
- Test individual behaviors (stand, shift weight)

**Phase 2: Supported Testing**  
- Robot on soft mat
- Helper holds robot initially
- Short tests (10-30 seconds)
- Test transitions (stand → walk)