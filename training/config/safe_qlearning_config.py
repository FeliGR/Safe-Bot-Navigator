"""Configuration for Safe Q-Learning Agent training."""

# Environment configuration with emphasis on safety challenges
env_config = {
    "size": 10,
    "obstacle_prob": 0,
    "trap_prob": 0.1,  # Higher trap probability to test safety features
    "trap_danger": 0.4,  # Increased trap danger
    "rewards": {
        "target": 10,     # Higher reward for reaching target safely
        "collision": -0.1,  # Penalty for collisions
        "step": -0.01,    # Small step cost to encourage efficiency
        "trap": -0.1      # Significant penalty for falling into traps
    }
}

# Agent configuration
agent_config = {
    "agent_module": "agents.safe_qlearning_agent",  # Module path
    "agent_class": "SafeQLearningAgent",           # Class name
    "learning_rate": 0.1,
    "gamma": 0.95,                # Discount factor
    "epsilon": 1,                 # Start with full exploration
    "epsilon_min": 0,             # Minimum exploration rate
    "epsilon_decay": 0.001,       # Decay rate for epsilon
    "trap_threshold": 2,          # Minimum safe distance from traps
    "trap_penalty": -0.5,         # Penalty for unsafe situations with traps
    "obstacle_threshold": 2,      # Minimum safe distance from obstacles
    "obstacle_penalty": -0.1,     # Penalty for unsafe situations with obstacles
    "planified_episodes": 0,      # Number of demonstration episodes
    "safety_threshold": None,     # When to trigger planifier help
    "interaction_type": "uniform", # Type of interaction with planifier
    "planner_probability": 0,     # High probability for planner in uniform mode
    "teacher_expertise": 0.8      # Teacher prefers safe paths
}

# Training configuration
train_config = {
    "episodes": 1000,     # Total episodes for training
    "max_steps": 1000,    # Maximum steps per episode
    "render_freq": 100,   # Render every 100 episodes
    "render_mode": None,  # Set to 'human' to visualize training
    "render_delay": 0.1   # Delay between renders
}
