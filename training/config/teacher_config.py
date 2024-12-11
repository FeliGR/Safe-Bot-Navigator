"""Configuration for Teacher Agent training."""

# Environment configuration
env_config = {
    "size": 8,
    "obstacle_prob": 0.2,
    "trap_prob": 0.1,
    "trap_danger": 0.3,
    "rewards": {
        "target": 1,
        "collision": 0,
        "step": 0,
        "trap": 0
    }
}

# Agent configuration
agent_config = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0,
    "planified_episodes": 0,
    "q_threshold": 0.2,
    "interaction_type": "stochastic",
    "planner_probability": 0.2,
    "teacher_expertise": 1
}

# Training configuration
train_config = {
    "episodes": 1000,
    "max_steps": 100,
    "render_freq": 1,
    "render_mode": None,
    "render_delay": 0.2
}
