from train_functions import create_scenario

env_config = {
    "size": 8,
    "obstacle_prob": 0.2,
    "trap_prob": 0.1,
    "trap_danger": 0.3,
    "rewards": {"target": 1, "collision": 0, "step": 0, "trap": -0.0},
}

train_config = {
    "episodes": 1000,
    "max_steps": 100,
    "render_freq": 1,
    "render_mode": None,
    "render_delay": 0.2,
}

agent_config = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 1,
    "epsilon_min": 0,
    "planified_episodes": 0,
    "q_threshold": 0.2,  # Threshold for deciding between planifier and RL actions
    "interaction_type": "stochastic",  # "uniform" or "stochastic"
    "planner_probability": 0.2,
    "teacher_expertise": 0.8,  # Probability of teacher trying trap-free path first
}

agent_config["epsilon_decay"] = (
    agent_config["epsilon"] - agent_config["epsilon_min"]
) / train_config["episodes"]

create_scenario(env_config, agent_config, train_config)
