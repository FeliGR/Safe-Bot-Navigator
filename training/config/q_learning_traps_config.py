"""Configuration for Teacher Agent training."""

# Environment configuration
env_config = {
    "size": 15,
    "obstacle_prob": 0.2,
    "trap_prob": 0.2,
    "trap_danger": 0.3,
    "rewards": {"target": 1, "collision": 0, "step": 0, "trap": -0.5},
}

# Agent configuration
agent_config = {
    "agent_module": "agents.basic_qlearning",
    "agent_class": "BasicQLearningAgent",
    "learning_rate": 0.1,
    "gamma": 0.5,
    "epsilon": 1,
    "epsilon_min": 0,
    }

# Training configuration
train_config = {
    "episodes": 10000,
    "max_steps": 1000,
    "render_freq": 1,
    "render_mode": None,
    "render_delay": 0.2,
}

agent_config["epsilon_decay"] = (
    agent_config["epsilon"] - agent_config["epsilon_min"]
) / train_config["episodes"]
