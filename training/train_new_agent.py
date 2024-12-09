import json
import os
import time
from train_functions import train_and_compare

env_config = {
    "size": 10,
    "obstacle_prob": 0.2,
    "trap_prob": 0.2,
    "trap_danger": 0.3,
    "rewards": {"target": 1, "collision": 0, "step": 0, "trap": 0},
}

train_config = {
    "episodes": 10000,
    "max_steps": 1000,
    "render_freq": 1000,
    "render_mode": None,
    "render_delay": 0.1,
}

agent_config_no_risk = {
    "agent_module": "agents.basic_qlearning",
    "agent_class": "BasicQLearningAgent",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.000099,  # Ajusta según corresponda
    "xi": 0.0,  # Sin sensibilidad al riesgo
    "state_size": env_config["size"] ** 2,
    "n_actions": 4,  # Asumiendo 4 acciones (arriba, abajo, izquierda, derecha)
}

agent_config_with_risk = {
    "agent_module": "agents.basic_qlearning",
    "agent_class": "BasicQLearningAgent",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.000099,  # Ajusta según corresponda
    "xi": 1.0,  # Con sensibilidad al riesgo
    "state_size": env_config["size"] ** 2,
    "n_actions": 4,  # Asumiendo 4 acciones
}

train_and_compare(env_config, agent_config_no_risk, agent_config_with_risk, train_config)
