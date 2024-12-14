"""Train a specific agent configuration."""

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


def list_available_configs():
    """List all available configuration files in the config directory."""
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    config_files = []

    for file in os.listdir(config_dir):
        if file.endswith("_config.py"):
            config_name = file[:-3]  # Remove .py extension
            config_files.append(config_name)

    return sorted(config_files)


def load_config(config_name):
    """
    Load a configuration module by name.

    Args:
        config_name (str): Name of the configuration file (without .py extension)

    Returns:
        module: Loaded configuration module
    """
    config_path = os.path.join(os.path.dirname(__file__), "config", f"{config_name}.py")

    if not os.path.exists(config_path):
        raise ValueError(f"Configuration file not found: {config_path}")

    spec = importlib.util.spec_from_file_location(config_name, config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    return config


def train_agent(config):
    """
    Train an agent using the specified configuration.

    Args:
        config (module): Configuration module containing training parameters
    """
    # Create environment
    env = GridEnvironment(**config.env_config)

    # Import and create agent
    module = importlib.import_module(config.agent_config["agent_module"])
    agent_class = getattr(module, config.agent_config["agent_class"])

    # Remove module and class info before passing to constructor
    agent_params = config.agent_config.copy()
    agent_params.pop("agent_module")
    agent_params.pop("agent_class")

    # Create agent with environment parameters
    agent = agent_class(
        state_size=env.size**2,  # Grid size squared for flattened state
        n_actions=len(env.ACTIONS),
        **agent_params,
    )

    # Train the agent
    print(f"\nStarting training with configuration: {config.__name__}")
    print(f"Training parameters: {config.train_config}")

    history = agent.train(env, **config.train_config)

    # Save the trained agent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        PROJECT_ROOT,
        "trained_agents",
        f"{agent.__class__.__name__}_{timestamp}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Save agent
    with open(os.path.join(save_dir, "agent.pkl"), "wb") as f:
        pickle.dump(agent, f)

    # Save environment
    with open(os.path.join(save_dir, "env.pkl"), "wb") as f:
        pickle.dump(env, f)

    # Save configurations
    with open(os.path.join(save_dir, "agent_config.json"), "w") as f:
        json.dump(config.agent_config, f, indent=4)

    with open(os.path.join(save_dir, "env_config.json"), "w") as f:
        json.dump(config.env_config, f, indent=4)

    with open(os.path.join(save_dir, "train_config.json"), "w") as f:
        json.dump(config.train_config, f, indent=4)

    print(f"\nTraining completed. Agent saved to: {save_dir}")

    # Plot training history if available
    if hasattr(agent, "plot_history"):
        agent.plot_history()


def select_config():
    """
    Interactively select a configuration from available options.

    Returns:
        str: Selected configuration name
    """
    configs = list_available_configs()

    if not configs:
        print("No configurations found in config directory")
        sys.exit(1)

    print("\nAvailable configurations:")
    for i, config in enumerate(configs, 1):
        print(f"{i}. {config}")

    while True:
        try:
            choice = input("\nSelect a configuration (enter number): ")
            idx = int(choice) - 1
            if 0 <= idx < len(configs):
                return configs[idx]
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def main():
    parser = argparse.ArgumentParser(
        description="Train an agent using a specific configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Name of the configuration to use (without .py extension)",
    )

    args = parser.parse_args()

    try:
        config_name = args.config if args.config else select_config()
        print(f"\nLoading configuration: {config_name}")
        config = load_config(config_name)
        train_agent(config)
    except KeyboardInterrupt:
        print("\nTraining cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
