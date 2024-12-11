"""Configuration for Marie 2005 Manual Agent training."""

# Environment configuration
env_config = {
    "size": 8,
    "obstacle_prob": 0.2,
    "trap_prob": 0.1,
    "trap_danger": 0.3,
    "rewards": {"target": 1, "collision": -0.1, "step": -0.01, "trap": -0.2},
}

# Agent configuration
agent_config = {
    "agent_module": "agents.q_learning_marie_2005_manual",
    "agent_class": "QLearningAgent",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "manual_episodes": 15,
}

# Training configuration
train_config = {
    "episodes": 1000,
    "max_steps": 100,
    "render_freq": 1,
    "render_mode": "human",
    "render_delay": 0.2,
}
