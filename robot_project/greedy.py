import os
import json
import pickle
import time
import numpy as np
from environment import GridEnvironment
from agents.basic_qlearning import BasicQLearningAgent

def list_saved_agents(base_dir='trained_agents'):
    """List all saved agents and their configurations"""
    if not os.path.exists(base_dir):
        print(f"No trained agents found in {base_dir}")
        return []
    
    agents = []
    for agent_dir in os.listdir(base_dir):
        agent_path = os.path.join(base_dir, agent_dir)
        if os.path.isdir(agent_path):
            # Check for agent and configuration files
            agent_file = os.path.join(agent_path, "agent.pkl")
            env_config_file = os.path.join(agent_path, "env_config.json")
            agent_config_file = os.path.join(agent_path, "agent_config.json")
            train_config_file = os.path.join(agent_path, "train_config.json")
            env_file = os.path.join(agent_path, "env.pkl")
            
            if all(os.path.exists(f) for f in [agent_file, env_config_file, agent_config_file, train_config_file, env_file]):
                # Load configurations
                with open(env_config_file, 'r') as f:
                    env_config = json.load(f)
                with open(agent_config_file, 'r') as f:
                    agent_config = json.load(f)
                with open(train_config_file, 'r') as f:
                    train_config = json.load(f)
                
                # Create agent summary
                summary = (
                    f"Agent Directory: {agent_dir}\n"
                    f"Grid Size: {env_config['size']}\n"
                    f"Obstacle Probability: {env_config['obstacle_prob']}\n"
                    f"Training Episodes: {train_config['episodes']}\n"
                    f"Learning Rate: {agent_config['learning_rate']}\n"
                    f"Gamma: {agent_config['gamma']}\n"
                    f"Initial Epsilon: {agent_config['epsilon']}"
                )
                
                agents.append({
                    'name': agent_dir,
                    'path': agent_path,
                    'env_config': env_config,
                    'agent_config': agent_config,
                    'train_config': train_config,
                    'summary': summary
                })
    
    return agents

def select_agent():
    """Interactive agent selection"""
    agents = list_saved_agents()
    if not agents:
        return None
    
    print("\nAvailable trained agents:")
    print("=" * 50)
    for i, agent in enumerate(agents):
        print(f"\n{i + 1}. {agent['name']}")
        print("-" * 50)
        print(agent['summary'])
    
    while True:
        try:
            choice = int(input("\nSelect an agent (number) or 0 to exit: "))
            if choice == 0:
                return None
            if 1 <= choice <= len(agents):
                return agents[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def run_greedy_evaluation(agent_info, episodes=5, render_delay=0.5):
    """Run greedy evaluation of the selected agent"""
    # Load agent and environment
    with open(os.path.join(agent_info['path'], 'agent.pkl'), 'rb') as f:
        agent = pickle.load(f)
    
    with open(os.path.join(agent_info['path'], 'env.pkl'), 'rb') as f:
        env = pickle.load(f)
    
    print(f"\nRunning {episodes} greedy episodes...")
    print("=" * 50)
    
    # Run episodes
    agent.run_greedy(env, episodes=episodes, max_steps=1000, render_mode='human', render_delay=render_delay)
    
    # Print summary from agent's history
    print("\nEvaluation Summary:")
    print("-" * 30)
    history = agent.greedy_history
    print(f"Average Reward: {np.mean(history['rewards']):.2f}")
    print(f"Average Steps: {np.mean(history['steps']):.2f}")
    print(f"Success Rate: {np.mean(history['success']) * 100:.1f}%")
    
    agent.plot_history(history_type=agent.GREEDY_HISTORY)

    env.close()

if __name__ == "__main__":
    # Select and run agent
    selected_agent = select_agent()
    if selected_agent:
        run_greedy_evaluation(selected_agent, episodes=50, render_delay=0.2)
    else:
        print("No agent selected. Exiting.")