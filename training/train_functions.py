import pickle
import importlib
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.environment import GridEnvironment
import json
import time


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

    env = GridEnvironment(**env_config)
    agent = AgentClass(env.size**2, len(env.ACTIONS), **agent_params)
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

    with open(os.path.join(full_path, "agent_config.json"), "w") as f:
        json.dump(agent_config, f)

    with open(os.path.join(full_path, "env_config.json"), "w") as f:
        json.dump(env_config, f)

    agent.plot_history()


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
