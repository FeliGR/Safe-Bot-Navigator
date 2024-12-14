"""Generate combined visualizations for selected agents."""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINED_AGENTS_DIR = os.path.join(PROJECT_ROOT, "trained_agents")
SAFE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "safe_results")
sys.path.append(PROJECT_ROOT)

from visualization.visualize_all import (
    find_trained_agents,
    find_related_files,
    load_pickle,
    load_json,
)


def list_available_agents(base_dir: str = TRAINED_AGENTS_DIR) -> List[str]:
    """List all available trained agents."""
    agents = find_trained_agents(base_dir)
    print("\nAvailable agents:")
    for idx, agent in enumerate(agents, 1):
        print(f"{idx}. {agent}")
    return agents


def select_agents() -> List[str]:
    """Allow user to select multiple agents for comparison."""
    agents = list_available_agents()

    print(
        "\nSelect agents to compare (enter numbers separated by spaces, e.g., '1 3 4'):"
    )
    while True:
        try:
            selections = input("> ").strip().split()
            indices = [int(s) - 1 for s in selections]
            selected_agents = [agents[i] for i in indices]
            return selected_agents
        except (ValueError, IndexError):
            print("Invalid selection. Please enter valid numbers separated by spaces.")


def load_agent_histories(agent_name: str, base_dir: str = TRAINED_AGENTS_DIR):
    """Load both training and greedy histories from agent's pickle file."""
    agent_dir = os.path.join(base_dir, agent_name)
    agent_file = os.path.join(agent_dir, "agent.pkl")

    if os.path.exists(agent_file):
        agent = load_pickle(agent_file)
        train_history = None
        greedy_history = None
        
        # Get training history
        if hasattr(agent, "history"):
            train_history = agent.history
        elif hasattr(agent, "train_history"):
            train_history = agent.train_history
            
        # Get greedy history
        if hasattr(agent, "greedy_history"):
            greedy_history = agent.greedy_history
            
        return train_history, greedy_history
    return None, None


def create_combined_graphs(agent_histories: Dict[str, dict], output_dir: str, history_type: str):
    """Create combined graphs for all metrics, both episode and cumulative."""
    if not agent_histories:
        print(f"No agent {history_type} histories to visualize.")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all available metrics from all agents
    all_metrics = set()
    for history in agent_histories.values():
        if isinstance(history, list):
            if history and isinstance(history[0], dict):
                all_metrics.update(history[0].keys())
        elif isinstance(history, dict):
            all_metrics.update(history.keys())

    print(f"Available metrics for {history_type}: {', '.join(all_metrics)}")

    # Create plots for each metric
    for metric in all_metrics:
        # Episode-by-episode plot
        plt.figure(figsize=(12, 6))

        for agent_name, history in agent_histories.items():
            values = []

            # Handle list of dictionaries format
            if isinstance(history, list):
                values = [episode.get(metric, 0) for episode in history]
            # Handle dictionary format with metric keys
            elif isinstance(history, dict) and metric in history:
                if isinstance(history[metric], list):
                    values = history[metric]
                else:
                    values = [history[metric]]

            if values:
                episodes = range(1, len(values) + 1)
                plt.plot(episodes, values, label=agent_name, linewidth=2)

        plt.title(f'{history_type} - Episode {metric.replace("_", " ").title()}')
        plt.xlabel("Episode")
        plt.ylabel(metric.replace("_", " ").title())
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save the episode plot
        plt.savefig(os.path.join(output_dir, f"episode_{metric}.png"))
        plt.close()

        # Cumulative plot
        plt.figure(figsize=(12, 6))

        for agent_name, history in agent_histories.items():
            values = []

            # Handle list of dictionaries format
            if isinstance(history, list):
                values = [episode.get(metric, 0) for episode in history]
            # Handle dictionary format with metric keys
            elif isinstance(history, dict) and metric in history:
                if isinstance(history[metric], list):
                    values = history[metric]
                else:
                    values = [history[metric]]

            if values:
                # Calculate cumulative values
                cumulative_values = np.cumsum(values)
                episodes = range(1, len(cumulative_values) + 1)
                plt.plot(episodes, cumulative_values, label=agent_name, linewidth=2)

        plt.title(f'{history_type} - Cumulative {metric.replace("_", " ").title()}')
        plt.xlabel("Episode")
        plt.ylabel(f'Cumulative {metric.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save the cumulative plot
        plt.savefig(os.path.join(output_dir, f"cumulative_{metric}.png"))
        plt.close()


def main():
    """Main function to run the combined visualization."""
    print("Welcome to Combined Agent Visualization!")

    # Let user select agents
    selected_agents = select_agents()
    print(f"\nSelected agents: {', '.join(selected_agents)}")

    # Load histories for all selected agents
    train_histories = {}
    greedy_histories = {}
    
    for agent_name in selected_agents:
        train_history, greedy_history = load_agent_histories(agent_name)
        if train_history:
            train_histories[agent_name] = train_history
        if greedy_history:
            greedy_histories[agent_name] = greedy_history
        if not train_history and not greedy_history:
            print(f"Warning: Could not load histories for {agent_name}")

    if not train_histories and not greedy_histories:
        print("No agent histories could be loaded. Exiting.")
        return

    # Create output directories for combined visualizations
    train_output_dir = os.path.join(SAFE_RESULTS_DIR, "combined_visualization", "training")
    greedy_output_dir = os.path.join(SAFE_RESULTS_DIR, "combined_visualization", "greedy")

    # Generate combined visualizations
    print("\nGenerating combined visualizations...")
    
    if train_histories:
        print("\nGenerating training visualizations...")
        create_combined_graphs(train_histories, train_output_dir, "Training")
        print(f"Training visualizations saved in: {train_output_dir}")
        
    if greedy_histories:
        print("\nGenerating greedy visualizations...")
        create_combined_graphs(greedy_histories, greedy_output_dir, "Greedy")
        print(f"Greedy visualizations saved in: {greedy_output_dir}")

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
