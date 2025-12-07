# Module 4, Chapter 3 - Cognitive Planning with LLMs

## From Voice to Action: The Planning Challenge

In the previous chapter, we learned how to convert speech to text using Whisper. But transcription alone isn't enough. Consider this command:

**User says**: "Clean up the living room"

**Whisper outputs**: "clean up the living room"

**What the robot needs**:
```python
[
  navigate_to("living_room"),
  detect_objects(category="clutter"),
  for each object:
    pick_up(object),
    classify_object(object),
    navigate_to(destination_for(object)),
    place_object(object)
]
```

This gap between natural language and executable robot actions is where **Large Language Models (LLMs)** excel. They provide the "cognitive" layer that understands intent, breaks down tasks, and generates action sequences.

## Why LLMs for Robot Planning?

### Traditional Approach (Rigid Rules)

```python
def parse_command(text):
    if "pick up" in text and "cup" in text:
        return PickAction(object="cup")
    elif "go to" in text and "kitchen" in text:
        return NavigateAction(location="kitchen")
    # ... hundreds of hardcoded rules
```

**Problems**:
- Doesn't generalize to new phrasings
- Can't handle complex, multi-step commands
- Breaks with slight variations ("grab the cup" vs "pick up the cup")
- No reasoning about context or constraints

### LLM Approach (Flexible Reasoning)

```python
def parse_command(text):
    response = llm.query(f"""
    Break down this robot command into specific actions:
    "{text}"
    
    Available actions: navigate, pick, place, search, wait
    Output as JSON list.
    """)
    return parse_json(response)
```

**Advantages**:
- Understands intent regardless of phrasing
- Handles complex, multi-step tasks
- Can reason about preconditions and constraints
- Adapts to context and environment

## LLM Options for Robotics

### Cloud-Based LLMs

**OpenAI GPT-4**
- **Pros**: Most capable, excellent reasoning, large context window (128K tokens)
- **Cons**: API costs (~$0.01 per 1K tokens), latency (~1-3 seconds), requires internet
- **Best for**: Complex planning, research, prototyping

**Anthropic Claude**
- **Pros**: Strong reasoning, 200K context window, good at following instructions
- **Cons**: API costs, latency, requires internet
- **Best for**: Long-form planning, multi-step reasoning

**Google Gemini**
- **Pros**: Multimodal (text + images), fast, competitive pricing
- **Cons**: Requires Google Cloud, some regions restricted
- **Best for**: Vision-language tasks

### Local/Open-Source LLMs

**Llama 3 (8B/70B)**
- **Pros**: Open-source, can run locally, no API costs
- **Cons**: Requires powerful GPU (24GB+ for 70B), slower than cloud
- **Best for**: Edge deployment, offline operation, privacy

**Mistral (7B)**
- **Pros**: Efficient, good performance for size, fast inference
- **Cons**: Smaller than GPT-4, less capable for complex reasoning
- **Best for**: Resource-constrained environments

**Phi-3 (3.8B)**
- **Pros**: Tiny model, runs on CPU, very fast
- **Cons**: Limited capabilities compared to larger models
- **Best for**: Simple command parsing, edge devices

### Recommendation for Humanoid Robots

**Development**: Use GPT-4 or Claude for rapid prototyping and testing

**Production (Cloud)**: Use GPT-4 Turbo or Claude Sonnet for best performance

**Production (Edge)**: Use Llama 3 8B on Jetson Orin or server-side inference

## Setting Up LLM Integration

### OpenAI GPT-4 Setup

```bash
pip install openai
```

```python
from openai import OpenAI
import os

class GPT4Planner:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4-turbo-preview"
        
    def plan_task(self, command, context=None):
        """
        Convert natural language command to action plan
        
        Args:
            command: Natural language instruction
            context: Optional context about environment, robot state
            
        Returns:
            List of actions with parameters
        """
        system_prompt = """You are a task planner for a humanoid robot.

Available actions:
- navigate(location: str): Move to a location
- search(object_type: str, location: str): Look for objects
- pick(object_id: str): Pick up an object
- place(object_id: str, location: str): Place object at location
- open(object_id: str): Open a container/door
- close(object_id: str): Close a container/door
- wait(duration: float): Wait for specified seconds

Output ONLY valid JSON in this format:
{
  "plan": [
    {"action": "navigate", "parameters": {"location": "kitchen"}},
    {"action": "pick", "parameters": {"object_id": "cup_1"}}
  ],
  "reasoning": "Brief explanation of the plan"
}

Do not include any text before or after the JSON."""

        user_prompt = f"Command: {command}"
        if context:
            user_prompt += f"\n\nContext: {context}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent planning
            max_tokens=2000
        )
        
        return self.parse_response(response.choices[0].message.content)
    
    def parse_response(self, text):
        """Parse LLM response into structured plan"""
        import json
        
        # Remove markdown code blocks if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            plan_data = json.loads(text)
            return plan_data
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Response was: {text}")
            return {"plan": [], "reasoning": "Parse error"}

# Example usage
planner = GPT4Planner()

command = "Go to the kitchen and bring me a glass of water"
plan = planner.plan_task(command)

print("Plan:")
for step in plan["plan"]:
    print(f"  {step['action']}({step['parameters']})")

print(f"\nReasoning: {plan['reasoning']}")
```

**Output**:
```
Plan:
  navigate({'location': 'kitchen'})
  search({'object_type': 'glass', 'location': 'kitchen'})
  pick({'object_id': 'glass_1'})
  navigate({'location': 'sink'})
  fill_glass({'object_id': 'glass_1'})
  navigate({'location': 'user'})
  place({'object_id': 'glass_1', 'location': 'user_hand'})

Reasoning: First navigate to kitchen, find a glass, fill it with water, then return to user
```

### Local LLM Setup (Llama 3)

```bash
pip install transformers torch accelerate
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

class LlamaPlanner:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        print("Loading Llama model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True  # Quantization for lower memory
        )
        print("Model loaded")
        
    def plan_task(self, command, context=None):
        """Generate action plan using Llama"""
        
        system_prompt = """You are a robot task planner. Convert commands to JSON action sequences.

Actions: navigate, search, pick, place, open, close, wait

Format:
{"plan": [{"action": "...", "parameters": {...}}], "reasoning": "..."}"""

        user_prompt = f"Command: {command}"
        if context:
            user_prompt += f"\nContext: {context}"
        
        # Format for Llama 3 chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.1,
            do_sample=True,
            top_p=0.9
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        return self.parse_response(response)
    
    def parse_response(self, text):
        """Extract JSON from model output"""
        # Find JSON in response
        start = text.find('{')
        end = text.rfind('}') + 1
        
        if start != -1 and end > start:
            json_str = text[start:end]
            try:
                return json.loads(json_str)
            except:
                pass
        
        return {"plan": [], "reasoning": "Failed to parse"}

# Usage
planner = LlamaPlanner()
plan = planner.plan_task("Clean the table")
print(plan)
```

## Advanced Planning Techniques

### 1. Contextual Planning with Environment State

Provide the LLM with information about the current environment:

```python
class ContextualPlanner:
    def __init__(self, llm_client):
        self.llm = llm_client
        
    def plan_with_context(self, command, robot_state, environment_state):
        """Plan with full context"""
        
        context = f"""
Robot State:
- Current location: {robot_state['location']}
- Holding: {robot_state.get('holding', 'nothing')}
- Battery: {robot_state['battery']}%
- Capabilities: {robot_state['capabilities']}

Environment:
- Detected objects: {environment_state['objects']}
- Accessible locations: {environment_state['locations']}
- Obstacles: {environment_state['obstacles']}
"""
        
        plan = self.llm.plan_task(command, context=context)
        return plan

# Example
robot_state = {
    'location': 'living_room',
    'holding': None,
    'battery': 85,
    'capabilities': ['navigate', 'pick', 'place', 'search']
}

environment_state = {
    'objects': [
        {'id': 'cup_1', 'type': 'cup', 'location': 'table', 'color': 'red'},
        {'id': 'book_1', 'type': 'book', 'location': 'floor', 'title': 'AI Textbook'}
    ],
    'locations': ['living_room', 'kitchen', 'bedroom'],
    'obstacles': ['chair_1', 'plant_1']
}

planner = ContextualPlanner(GPT4Planner())
plan = planner.plan_with_context(
    "Put the book on the shelf",
    robot_state,
    environment_state
)
```

### 2. Hierarchical Planning (Multi-Level)

Break complex tasks into hierarchical subtasks:

```python
class HierarchicalPlanner:
    def __init__(self, llm_client):
        self.llm = llm_client
        
    def plan_hierarchical(self, command):
        """Generate hierarchical plan: goals → tasks → actions"""
        
        # Level 1: High-level goals
        goal_prompt = f"""
Break this command into high-level goals:
"{command}"

Output as JSON list of goals.
Example: {{"goals": ["goal 1", "goal 2"]}}
"""
        goals_response = self.llm.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": goal_prompt}],
            temperature=0.1
        )
        
        goals = json.loads(goals_response.choices[0].message.content)['goals']
        
        # Level 2: Tasks for each goal
        full_plan = []
        for goal in goals:
            task_plan = self.llm.plan_task(goal)
            full_plan.append({
                'goal': goal,
                'actions': task_plan['plan']
            })
        
        return full_plan

# Example
planner = HierarchicalPlanner(GPT4Planner())
plan = planner.plan_hierarchical("Clean and organize the living room")

print("Hierarchical Plan:")
for i, level in enumerate(plan):
    print(f"\nGoal {i+1}: {level['goal']}")
    for action in level['actions']:
        print(f"  - {action['action']}({action['parameters']})")
```

**Output**:
```
Hierarchical Plan:

Goal 1: Pick up clutter from the floor
  - navigate({'location': 'living_room'})
  - search({'object_type': 'clutter', 'location': 'floor'})
  - pick({'object_id': 'toy_1'})
  - navigate({'location': 'toy_box'})
  - place({'object_id': 'toy_1', 'location': 'toy_box'})

Goal 2: Arrange items on surfaces
  - search({'object_type': 'books', 'location': 'table'})
  - pick({'object_id': 'book_1'})
  - navigate({'location': 'bookshelf'})
  - place({'object_id': 'book_1', 'location': 'shelf'})
```

### 3. Interactive Planning with Clarification

Ask for clarification when commands are ambiguous:

```python
class InteractivePlanner:
    def __init__(self, llm_client):
        self.llm = llm_client
        
    def plan_with_clarification(self, command, environment_state):
        """Generate plan, asking for clarification if needed"""
        
        # First, check if command is ambiguous
        ambiguity_check = f"""
Analyze this command for ambiguity:
"{command}"

Environment: {environment_state}

Is there any ambiguity? What information is missing?
Output JSON: {{"is_ambiguous": true/false, "questions": ["question 1", ...]}}
"""
        
        response = self.llm.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": ambiguity_check}],
            temperature=0.1
        )
        
        analysis = json.loads(response.choices[0].message.content)
        
        if analysis['is_ambiguous']:
            # Return questions for user
            return {
                'status': 'needs_clarification',
                'questions': analysis['questions']
            }
        else:
            # Generate plan
            plan = self.llm.plan_task(command, context=str(environment_state))
            return {
                'status': 'ready',
                'plan': plan
            }

# Example
planner = InteractivePlanner(GPT4Planner())

environment = {
    'objects': [
        {'id': 'cup_1', 'type': 'cup', 'color': 'red'},
        {'id': 'cup_2', 'type': 'cup', 'color': 'blue'}
    ]
}

result = planner.plan_with_clarification("Bring me the cup", environment)

if result['status'] == 'needs_clarification':
    print("Questions:")
    for q in result['questions']:
        print(f"  - {q}")
    # Output: "Which cup? The red one or the blue one?"
```

### 4. Error Recovery and Replanning

Handle execution failures by replanning:

```python
class AdaptivePlanner:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.execution_history = []
        
    def execute_with_recovery(self, command, robot, max_retries=3):
        """Execute plan with automatic replanning on failure"""
        
        plan = self.llm.plan_task(command)
        
        for action_step in plan['plan']:
            action = action_step['action']
            params = action_step['parameters']
            
            # Try to execute
            success = robot.execute_action(action, params)
            
            self.execution_history.append({
                'action': action,
                'parameters': params,
                'success': success
            })
            
            if not success:
                # Get failure reason
                failure_reason = robot.get_last_error()
                
                # Replan from current state
                recovery_plan = self.replan_after_failure(
                    original_command=command,
                    failure_point=action_step,
                    failure_reason=failure_reason,
                    history=self.execution_history
                )
                
                if recovery_plan:
                    print(f"Replanning after failure: {failure_reason}")
                    # Continue with recovery plan
                    plan['plan'] = recovery_plan['plan']
                else:
                    print("Unable to recover from failure")
                    return False
        
        return True
    
    def replan_after_failure(self, original_command, failure_point, 
                            failure_reason, history):
        """Generate recovery plan after failure"""
        
        recovery_prompt = f"""
Original command: {original_command}
Failed action: {failure_point}
Failure reason: {failure_reason}
Execution history: {history}

Generate a recovery plan to complete the original command.
Consider alternative approaches.
Output as JSON with new action sequence.
"""
        
        response = self.llm.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": recovery_prompt}],
            temperature=0.3  # Slightly higher for creative solutions
        )
        
        return self.llm.parse_response(response.choices[0].message.content)

# Example usage
planner = AdaptivePlanner(GPT4Planner())
success = planner.execute_with_recovery(
    "Move the box to the shelf",
    robot=my_robot
)
```

## Integrating with ROS 2

### LLM Planning Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from custom_msgs.msg import ActionPlan, Action
import json

class LLMPlannerNode(Node):
    def __init__(self):
        super().__init__('llm_planner_node')
        
        # Subscribe to voice commands
        self.subscription = self.create_subscription(
            String,
            'voice_commands',
            self.command_callback,
            10
        )
        
        # Publish action plans
        self.plan_publisher = self.create_publisher(
            ActionPlan,
            'action_plan',
            10
        )
        
        # Initialize LLM planner
        self.planner = GPT4Planner()
        
        self.get_logger().info("LLM Planner Node ready")
        
    def command_callback(self, msg):
        """Receive voice command and generate plan"""
        command = msg.data
        self.get_logger().info(f"Planning for command: {command}")
        
        try:
            # Generate plan
            plan_data = self.planner.plan_task(command)
            
            # Convert to ROS message
            plan_msg = ActionPlan()
            plan_msg.command = command
            plan_msg.reasoning = plan_data.get('reasoning', '')
            
            for action_dict in plan_data['plan']:
                action_msg = Action()
                action_msg.action_type = action_dict['action']
                action_msg.parameters = json.dumps(action_dict['parameters'])
                plan_msg.actions.append(action_msg)
            
            # Publish plan
            self.plan_publisher.publish(plan_msg)
            self.get_logger().info(f"Published plan with {len(plan_msg.actions)} actions")
            
        except Exception as e:
            self.get_logger().error(f"Planning failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = LLMPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

### Action Executor Node

```python
class ActionExecutorNode(Node):
    def __init__(self):
        super().__init__('action_executor_node')
        
        # Subscribe to action plans
        self.subscription = self.create_subscription(
            ActionPlan,
            'action_plan',
            self.plan_callback,
            10
        )
        
        # Action clients for robot control
        self.nav_client = self.create_client(Navigate, 'navigate')
        self.pick_client = self.create_client(Pick, 'pick')
        self.place_client = self.create_client(Place, 'place')
        
        self.get_logger().info("Action Executor ready")
        
    def plan_callback(self, plan_msg):
        """Execute action plan"""
        self.get_logger().info(f"Executing plan: {plan_msg.reasoning}")
        
        for action_msg in plan_msg.actions:
            action_type = action_msg.action_type
            parameters = json.loads(action_msg.parameters)
            
            self.get_logger().info(f"Executing: {action_type}({parameters})")
            
            # Route to appropriate action
            if action_type == 'navigate':
                self.execute_navigate(parameters)
            elif action_type == 'pick':
                self.execute_pick(parameters)
            elif action_type == 'place':
                self.execute_place(parameters)
            # ... other actions
            
    def execute_navigate(self, params):
        """Execute navigation action"""
        request = Navigate.Request()
        request.location = params['location']
        
        future = self.nav_client.call_async(request)
        # Wait for completion or handle asynchronously
```

## Prompt Engineering for Better Plans

### 1. Few-Shot Examples

Provide examples of good plans:

```python
few_shot_prompt = """You are a robot task planner.

Example 1:
Command: "Put the dishes in the dishwasher"
Plan:
{"plan": [
  {"action": "navigate", "parameters": {"location": "kitchen"}},
  {"action": "search", "parameters": {"object_type": "dish", "location": "sink"}},
  {"action": "pick", "parameters": {"object_id": "dish_1"}},
  {"action": "open", "parameters": {"object_id": "dishwasher"}},
  {"action": "place", "parameters": {"object_id": "dish_1", "location": "dishwasher"}}
]}

Example 2:
Command: "Water the plants"
Plan:
{"plan": [
  {"action": "navigate", "parameters": {"location": "kitchen"}},
  {"action": "pick", "parameters": {"object_id": "watering_can"}},
  {"action": "fill", "parameters": {"object_id": "watering_can", "location": "sink"}},
  {"action": "navigate", "parameters": {"location": "living_room"}},
  {"action": "search", "parameters": {"object_type": "plant"}},
  {"action": "pour", "parameters": {"source": "watering_can", "target": "plant_1"}}
]}

Now plan for: {user_command}
"""
```

### 2. Chain-of-Thought Prompting

Ask LLM to reason step-by-step:

```python
cot_prompt = """
Command: "{command}"

Think step-by-step:
1. What is the goal?
2. What objects are involved?
3. What locations are needed?
4. What is the sequence of actions?
5. Are there any preconditions?

Then output the plan as JSON.
"""
```

### 3. Constrained Generation

Enforce output format:

```python
constrained_prompt = """
You MUST output valid JSON with this exact structure:
{
  "plan": [
    {"action": "action_name", "parameters": {"param1": "value1"}}
  ],
  "reasoning": "explanation"
}

Valid actions ONLY: navigate, search, pick, place, open, close, wait

Do NOT include any text outside the JSON.
"""
```

## Evaluation and Testing

### Test Suite for Planning

```python
class PlannerEvaluator:
    def __init__(self, planner):
        self.planner = planner
        
    def test_commands(self):
        """Test planner on various commands"""
        
        test_cases = [
            {
                'command': "Bring me a water bottle",
                'expected_actions': ['navigate', 'search', 'pick', 'navigate', 'place']
            },
            {
                'command': "Clean the table",
                'expected_actions': ['navigate', 'search', 'pick', 'place']
            },
            {
                'command': "Open the door",
                'expected_actions': ['navigate', 'open']
            }
        ]
        
        results = []
        for test in test_cases:
            plan = self.planner.plan_task(test['command'])
            actual_actions = [a['action'] for a in plan['plan']]
            
            passed = all(exp in actual_actions for exp in test['expected_actions'])
            results.append({
                'command': test['command'],
                'passed': passed,
                'expected': test['expected_actions'],
                'actual': actual_actions
            })
        
        return results

# Run evaluation
evaluator = PlannerEvaluator(GPT4Planner())
results = evaluator.test_commands()

for result in results:
    status = "✓" if result['passed'] else "✗"
    print(f"{status} {result['command']}")
    if not result['passed']:
        print(f"  Expected: {result['expected']}")
        print(f"  Got: {result['actual']}")
```

## Summary

In this chapter, we covered:
- Why LLMs are essential for cognitive planning in robotics
- Setting up cloud (GPT-4, Claude) and local (Llama 3) LLMs
- Converting natural language commands to structured action plans
- Advanced techniques: contextual, hierarchical, and interactive planning
- ROS 2 integration for real-world deployment
- Prompt engineering and evaluation

**Next Chapter**: We'll bring everything together in the Capstone Project—building a complete autonomous humanoid that listens, plans, and executes complex tasks.