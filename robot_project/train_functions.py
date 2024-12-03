import pickle
from agents.basic_qlearning import BasicQLearningAgent
from environment import GridEnvironment
import json
import os
import time


def create_scenario(env_config, agent_config, train_config):

    env = GridEnvironment(**env_config)
    agent = BasicQLearningAgent(env.size**2, len(env.ACTIONS), **agent_config)    
    agent.train(env, **train_config)

    save_path = "trained_agents"
    folder_name = f"{agent.__class__.__name__}_{env.size}_{env.obstacle_prob}_{time.strftime('%Y%m%d_%H%M%S')}"
    full_path = os.path.join(save_path, folder_name)
    os.makedirs(full_path, exist_ok=True)

    with open(os.path.join(full_path, "agent.pkl"), 'wb') as f:
        pickle.dump(agent, f)

    with open(os.path.join(full_path, "env.pkl"), 'wb') as f:
        pickle.dump(env, f)

    with open(os.path.join(full_path, "train_config.json"), 'w') as f:
        json.dump(train_config, f)

    with open(os.path.join(full_path, "agent_config.json"), 'w') as f:
        json.dump(agent_config, f)

    with open(os.path.join(full_path, "env_config.json"), 'w') as f:
        json.dump(env_config, f)

    #plot history
    agent.plot_history()

def retrain_agent(agent, env, train_config):
    agent.train(env, **train_config)

    save_path = "trained_agents"
    folder_name = f"{agent.__class__.__name__}_{env.size}_{env.obstacle_prob}_{time.strftime('%Y%m%d_%H%M%S')}"
    full_path = os.path.join(save_path, folder_name)
    os.makedirs(full_path, exist_ok=True)

    with open(os.path.join(full_path, "agent.pkl"), 'wb') as f:
        pickle.dump(agent, f)

    with open(os.path.join(full_path, "env.pkl"), 'wb') as f:
        pickle.dump(env, f)

    with open(os.path.join(full_path, "train_config.json"), 'w') as f:
        json.dump(train_config, f)

    #plot history
    agent.plot_history()
    