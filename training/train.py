"""Main training script for Safe-Bot-Navigator agents."""

import os
import sys
import argparse
import importlib
import pickle
import json
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from environment.environment import GridEnvironment
from agents.safe_qlearning_agent import SafeQLearningAgent
from agents.q_learning_marie_2005_manual import QLearningAgent as ManualAgent
from agents.q_learning_marie_2005_planifier import QLearningAgentMarie2005 as PlanifierAgent
from agents.q_learning_Clouse_AFH import QLearningAgentTeacher

def get_available_configs():
    """Get all available configuration files from the config directory."""
    config_dir = os.path.join(os.path.dirname(__file__), 'config')
    configs = []
    for file in os.listdir(config_dir):
        if file.endswith('_config.py') and file != '__init__.py':
            configs.append(file[:-10])  # Remove '_config.py'
    return configs

def load_config(config_name):
    """Load configuration from the config directory."""
    try:
        config_module = importlib.import_module(f'training.config.{config_name}_config')
        return (
            getattr(config_module, 'env_config', {}),
            getattr(config_module, 'agent_config', {}),
            getattr(config_module, 'train_config', {})
        )
    except ImportError as e:
        print(f"Error loading configuration '{config_name}': {str(e)}")
        return None

def create_agent_directory(agent_type):
    """Create a directory for saving agent results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_dir = os.path.join(PROJECT_ROOT, '..', 'trained_agents', f'{agent_type}_{timestamp}')
    os.makedirs(agent_dir, exist_ok=True)
    return agent_dir

def get_agent_class(agent_type):
    """Get the appropriate agent class based on agent type."""
    agent_classes = {
        'safe_qlearning': SafeQLearningAgent,
        'marie_2005_manual': ManualAgent,
        'marie_2025_planifier': PlanifierAgent,
        'teacher': QLearningAgentTeacher
    }
    return agent_classes.get(agent_type)

def train_agent(agent_type):
    """Train an agent using the specified configuration."""
    # Load configuration
    config_result = load_config(agent_type)
    if config_result is None:
        return None
        
    env_config, agent_config, train_config = config_result
    
    # Create and configure environment
    env = GridEnvironment(**env_config)
    
    # Create agent directory
    agent_dir = create_agent_directory(agent_type)
    
    # Get agent class and initialize
    AgentClass = get_agent_class(agent_type)
    if AgentClass is None:
        print(f"Unknown agent type: {agent_type}")
        return None
        
    # Remove module and class info from agent_config before passing to constructor
    agent_params = agent_config.copy()
    agent_params.pop('agent_module', None)
    agent_params.pop('agent_class', None)
    
    agent = AgentClass(
        state_size=env.size**2,
        n_actions=len(env.ACTIONS),
        **agent_params
    )
    
    # Train the agent
    print(f"\nStarting training for {agent_type} agent...")
    agent.train(env, **train_config)
    
    # Save environment, agent and all configurations
    env_path = os.path.join(agent_dir, 'env.pkl')
    agent_path = os.path.join(agent_dir, 'agent.pkl')
    train_config_path = os.path.join(agent_dir, 'train_config.json')
    agent_config_path = os.path.join(agent_dir, 'agent_config.json')
    env_config_path = os.path.join(agent_dir, 'env_config.json')
    
    # Save pickled objects
    with open(env_path, 'wb') as f:
        pickle.dump(env, f)
    
    with open(agent_path, 'wb') as f:
        pickle.dump(agent, f)
    
    # Save configurations
    with open(train_config_path, 'w') as f:
        json.dump(train_config, f, indent=4)
        
    with open(agent_config_path, 'w') as f:
        json.dump(agent_config, f, indent=4)
        
    with open(env_config_path, 'w') as f:
        json.dump(env_config, f, indent=4)
    
    print(f"\nTraining completed!")
    print(f"Agent saved to: {agent_path}")
    print(f"Environment saved to: {env_path}")
    print(f"Configurations saved to:")
    print(f"  - Training config: {train_config_path}")
    print(f"  - Agent config: {agent_config_path}")
    print(f"  - Environment config: {env_config_path}")
    
    return agent_dir

def main():
    print("Training all available agents...")
    configs = get_available_configs()
    
    print(f"\nFound {len(configs)} configurations: {', '.join(configs)}")
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Training agent: {config}")
        print(f"{'='*50}")
        train_agent(config)
        print(f"\nCompleted training for {config}")

if __name__ == "__main__":
    main()
