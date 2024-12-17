"""Configuration for Teacher Agent training - scenario_2_variation_2"""

# Base del entorno
env_config = {
    "size": 15,
    "obstacle_prob": 0.2,
    "trap_prob": 0.2,
    "trap_danger": 0.3,
    "rewards": {"target": 1, "collision": 0, "step": 0, "trap": 0},
}

# Base de entrenamiento
train_config = {
    "episodes": 1000,
    "max_steps": 100,
    "render_freq": 1,
    "render_mode": None,
    "render_delay": 0.2,
}

# Configuración completa del agente
agent_config = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0,
    "planified_episodes": 0,
    "q_threshold": 1e-06,
    "interaction_type": "uniform",
    "planner_probability": 0.5,
    "teacher_expertise": 0.7
}
