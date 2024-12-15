"""Configuration for Teacher Agent training - Escenarios con variaciones de parámetros
Todos iguales a la base, modificando únicamente los parámetros especificados.
Se mantendrá el mismo formato y estructura, cambiando sólo los valores requeridos.
Los comentarios posteriores se añadirán más tarde.
"""

# Base del entorno (común a todos)
env_config = {
    "size": 15,
    "obstacle_prob": 0.2,
    "trap_prob": 0.2,
    "trap_danger": 0.3,
    "rewards": {"target": 1, "collision": 0, "step": 0, "trap": 0},
}

# Base de entrenamiento (común a todos)
train_config = {
    "episodes": 1000,
    "max_steps": 100,
    "render_freq": 1,
    "render_mode": None,
    "render_delay": 0.2,
}

#============================================================
# Escenario 1: Diferencia de Valores Q (Sin Demostraciones)
# Variaciones en q_threshold: 0.1, 0.2, 0.3
# interaction_type = "stochastic"
# planified_episodes = 0
# planner_probability = 0.2
#============================================================

agent_config_esc1_q01 = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0,
    "planified_episodes": 0,
    "q_threshold": 0.1,
    "interaction_type": "stochastic",
    "planner_probability": 0.2,
    "teacher_expertise": 1,
}

agent_config_esc1_q02 = {
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
    "teacher_expertise": 1,
}

agent_config_esc1_q03 = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0,
    "planified_episodes": 0,
    "q_threshold": 0.3,
    "interaction_type": "stochastic",
    "planner_probability": 0.2,
    "teacher_expertise": 1,
}

#============================================================
# Escenario 2: Distribución Uniforme (Sin Demostraciones)
# Variaciones en planner_probability: 0.1, 0.3, 0.5
# interaction_type = "uniform"
# planified_episodes = 0
#============================================================

agent_config_esc2_p01 = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0,
    "planified_episodes": 0,
    "q_threshold": 0.000001,
    "interaction_type": "uniform",
    "planner_probability": 0.1,
    "teacher_expertise": 1,
}

agent_config_esc2_p03 = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0,
    "planified_episodes": 0,
    "q_threshold": 0.000001,
    "interaction_type": "uniform",
    "planner_probability": 0.3,
    "teacher_expertise": 1,
}

agent_config_esc2_p05 = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0,
    "planified_episodes": 0,
    "q_threshold": 0.000001,
    "interaction_type": "uniform",
    "planner_probability": 0.5,
    "teacher_expertise": 1,
}

#============================================================
# Escenario 3: Demostraciones (Puras)
# Variaciones en planified_episodes: 20, 50, 100
# interaction_type = "uniform"
# planner_probability = 0
#============================================================

agent_config_esc3_pe20 = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0,
    "planified_episodes": 20,
    "q_threshold": 0.000001,
    "interaction_type": "uniform",
    "planner_probability": 0,
    "teacher_expertise": 1,
}

agent_config_esc3_pe50 = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0,
    "planified_episodes": 50,
    "q_threshold": 0.000001,
    "interaction_type": "uniform",
    "planner_probability": 0,
    "teacher_expertise": 1,
}

agent_config_esc3_pe100 = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0,
    "planified_episodes": 100,
    "q_threshold": 0.000001,
    "interaction_type": "uniform",
    "planner_probability": 0,
    "teacher_expertise": 1,
}

#============================================================
# Escenario 4: Demostraciones con Distribución Uniforme + Q Threshold
# Variaciones en q_threshold: 0.1, 0.2
# Variaciones en planified_episodes: 50, 100
# interaction_type = "stochastic"
# planner_probability = 0.2
#============================================================

agent_config_esc4_pe50_q01 = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0,
    "planified_episodes": 50,
    "q_threshold": 0.1,
    "interaction_type": "stochastic",
    "planner_probability": 0.2,
    "teacher_expertise": 1,
}

agent_config_esc4_pe100_q01 = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0,
    "planified_episodes": 100,
    "q_threshold": 0.1,
    "interaction_type": "stochastic",
    "planner_probability": 0.2,
    "teacher_expertise": 1,
}

agent_config_esc4_pe50_q02 = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0,
    "planified_episodes": 50,
    "q_threshold": 0.2,
    "interaction_type": "stochastic",
    "planner_probability": 0.2,
    "teacher_expertise": 1,
}

agent_config_esc4_pe100_q02 = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0,
    "planified_episodes": 100,
    "q_threshold": 0.2,
    "interaction_type": "stochastic",
    "planner_probability": 0.2,
    "teacher_expertise": 1,
}

#============================================================
# Escenario 5: Demostraciones con Diferencia de Valores Q + Uniform
# q_threshold = 0.2
# Variaciones en planner_probability: 0.1, 0.3
# Variaciones en planified_episodes: 50, 100
# interaction_type = "uniform"
#============================================================

agent_config_esc5_pe50_p01 = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0,
    "planified_episodes": 50,
    "q_threshold": 0.2,
    "interaction_type": "uniform",
    "planner_probability": 0.1,
    "teacher_expertise": 1,
}

agent_config_esc5_pe100_p01 = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0,
    "planified_episodes": 100,
    "q_threshold": 0.2,
    "interaction_type": "uniform",
    "planner_probability": 0.1,
    "teacher_expertise": 1,
}

agent_config_esc5_pe50_p03 = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0,
    "planified_episodes": 50,
    "q_threshold": 0.2,
    "interaction_type": "uniform",
    "planner_probability": 0.3,
    "teacher_expertise": 1,
}

agent_config_esc5_pe100_p03 = {
    "agent_module": "agents.q_learning_Clouse_AFH",
    "agent_class": "QLearningAgentTeacher",
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 0,
    "epsilon_min": 0,
    "planified_episodes": 100,
    "q_threshold": 0.2,
    "interaction_type": "uniform",
    "planner_probability": 0.3,
    "teacher_expertise": 1,
}
