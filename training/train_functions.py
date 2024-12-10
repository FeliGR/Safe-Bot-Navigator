import pickle
import importlib
import sys
import os
import time

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.environment import GridEnvironment
from agents.basic_qlearning import BasicQLearningAgent, plot_comparison
import json


def create_scenario(env_config, agent_config, train_config):
    """Create a new scenario with agent specified in agent_config

    Args:
        env_config: Environment configuration
        agent_config: Agent configuration including 'agent_module' and 'agent_class'
        train_config: Training configuration
    """

    try:
        agent_module = importlib.import_module(agent_config["agent_module"])
        AgentClass = getattr(agent_module, agent_config["agent_class"])
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Failed to load agent: {e}. Make sure agent_module and agent_class are correctly specified in agent_config"
        )

    agent_params = agent_config.copy()
    agent_params.pop("agent_module", None)
    agent_params.pop("agent_class", None)

    agent = AgentClass(**agent_params)

    env = GridEnvironment(**env_config)

    agent.train(env, **train_config)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    agent_dir = f"{agent_config['agent_class']}_size_{env_config['size']}_obstacle_prob_{env_config['obstacle_prob']}_trap_prob_{env_config['trap_prob']}_trap_danger_{env_config['trap_danger']}_{timestamp}"
    full_path = os.path.join("trained_agents", agent_dir)
    os.makedirs(full_path, exist_ok=True)

    with open(os.path.join(full_path, "agent.pkl"), "wb") as f:
        pickle.dump(agent, f)

    with open(os.path.join(full_path, "env.pkl"), "wb") as f:
        pickle.dump(env, f)

    with open(os.path.join(full_path, "train_config.json"), "w") as f:
        json.dump(train_config, f)

    with open(os.path.join(full_path, "agent_config.json"), "w") as f:
        json.dump(agent_config, f)

    with open(os.path.join(full_path, "env_config.json"), "w") as f:
        json.dump(env_config, f)

    agent.plot_history()

    agent.plot_risk_map(env, save_path=os.path.join(full_path, "risk_map.png"))

    print(f"Escenario creado y guardado en {full_path}")


def retrain_agent(agent, env, train_config):
    agent.train(env, **train_config)

    save_path = "trained_agents"
    folder_name = f"{agent.__class__.__name__}_size_{env.size}_obstacle_prob_{env.obstacle_prob}_trap_prob_{env.trap_prob}_trap_danger_{env.trap_danger}_{time.strftime('%Y%m%d_%H%M%S')}"
    full_path = os.path.join(save_path, folder_name)
    os.makedirs(full_path, exist_ok=True)

    with open(os.path.join(full_path, "agent.pkl"), "wb") as f:
        pickle.dump(agent, f)

    with open(os.path.join(full_path, "env.pkl"), "wb") as f:
        pickle.dump(env, f)

    with open(os.path.join(full_path, "train_config.json"), "w") as f:
        json.dump(train_config, f)

    agent.plot_history()


def train_and_compare(env_config, agent_config_no_risk, agent_config_with_risk, train_config):
    """Entrena dos agentes y los compara."""

    # Entrenar agente sin riesgo
    agent_params_no_risk = agent_config_no_risk.copy()
    agent_params_no_risk.pop("agent_module", None)
    agent_params_no_risk.pop("agent_class", None)

    agent_no_risk = BasicQLearningAgent(**agent_params_no_risk)
    env_no_risk = GridEnvironment(**env_config)
    agent_no_risk.train(env_no_risk, **train_config)

    # Guardar agente sin riesgo
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    agent_dir_no_risk = f"{agent_config_no_risk['agent_class']}_no_risk_{timestamp}"
    full_path_no_risk = os.path.join("trained_agents", agent_dir_no_risk)
    os.makedirs(full_path_no_risk, exist_ok=True)

    with open(os.path.join(full_path_no_risk, "agent.pkl"), "wb") as f:
        pickle.dump(agent_no_risk, f)

    with open(os.path.join(full_path_no_risk, "env.pkl"), "wb") as f:
        pickle.dump(env_no_risk, f)

    with open(os.path.join(full_path_no_risk, "train_config.json"), "w") as f:
        json.dump(train_config, f)

    with open(os.path.join(full_path_no_risk, "agent_config.json"), "w") as f:
        json.dump(agent_config_no_risk, f)

    with open(os.path.join(full_path_no_risk, "env_config.json"), "w") as f:
        json.dump(env_config, f)

    agent_no_risk.plot_history()

    # Entrenar agente con riesgo
    agent_params_with_risk = agent_config_with_risk.copy()
    agent_params_with_risk.pop("agent_module", None)
    agent_params_with_risk.pop("agent_class", None)

    agent_with_risk = BasicQLearningAgent(**agent_params_with_risk)
    env_with_risk = GridEnvironment(**env_config)
    agent_with_risk.train(env_with_risk, **train_config)

    # Guardar agente con riesgo
    agent_dir_with_risk = f"{agent_config_with_risk['agent_class']}_with_risk_{timestamp}"
    full_path_with_risk = os.path.join("trained_agents", agent_dir_with_risk)
    os.makedirs(full_path_with_risk, exist_ok=True)

    with open(os.path.join(full_path_with_risk, "agent.pkl"), "wb") as f:
        pickle.dump(agent_with_risk, f)

    with open(os.path.join(full_path_with_risk, "env.pkl"), "wb") as f:
        pickle.dump(env_with_risk, f)

    with open(os.path.join(full_path_with_risk, "train_config.json"), "w") as f:
        json.dump(train_config, f)

    with open(os.path.join(full_path_with_risk, "agent_config.json"), "w") as f:
        json.dump(agent_config_with_risk, f)

    with open(os.path.join(full_path_with_risk, "env_config.json"), "w") as f:
        json.dump(env_config, f)

    agent_with_risk.plot_history()
    agent_with_risk.plot_risk_map(env_with_risk, save_path=os.path.join(full_path_with_risk, "risk_map_with_risk.png"))

    # Comparar ambos agentes utilizando la función externa
    plot_comparison(agent_no_risk, agent_with_risk)