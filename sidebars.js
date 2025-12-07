module.exports = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: ROS 2 Fundamentals',
      items: [
        'module1/intro-ros2',
        'module1/nodes-topics-services',
        'module1/urdf-humanoids',
        'module1/ros2-exercises',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Robot Simulation',
      items: [
        'module2/gazebo-physics',
        'module2/unity-visualization',
        'module2/sensor-simulation',
        'module2/simulation-exercises',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: NVIDIA Isaac Platform',
      items: [
        'module3/isaac-sim',
        'module3/isaac-ros',
        'module3/nav2-navigation',
        'module3/reinforcement-learning-robot-control',
        'module3/sim-to-real-transfer',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action',
      items: [
        'module4/introduction-vla',
        'module4/voice-to-action-whisper',
        'module4/cognitive-planning-llms',
        'module4/capstone-autonomous-humanoid',
      ],
    },
  ],
};