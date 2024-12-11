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

from visualization.visualize_all import find_trained_agents, find_related_files, load_pickle, load_json

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
    
    print("\nSelect agents to compare (enter numbers separated by spaces, e.g., '1 3 4'):")
    while True:
        try:
            selections = input("> ").strip().split()
            indices = [int(s) - 1 for s in selections]
            selected_agents = [agents[i] for i in indices]
            return selected_agents
        except (ValueError, IndexError):
            print("Invalid selection. Please enter valid numbers separated by spaces.")

def load_agent_history(agent_name: str, base_dir: str = TRAINED_AGENTS_DIR):
    """Load history data from agent's pickle file."""
    agent_dir = os.path.join(base_dir, agent_name)
    agent_file = os.path.join(agent_dir, "agent.pkl")
    
    if os.path.exists(agent_file):
        agent = load_pickle(agent_file)
        if hasattr(agent, 'history'):
            return agent.history
        elif hasattr(agent, 'train_history'):
            return agent.train_history
    return None

def create_combined_graphs(agent_histories: Dict[str, dict], output_dir: str):
    """Create combined graphs for all metrics, both episode and cumulative."""
    if not agent_histories:
        print("No agent histories to visualize.")
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

    print(f"Available metrics: {', '.join(all_metrics)}")

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

        plt.title(f'Episode {metric.replace("_", " ").title()}')
        plt.xlabel('Episode')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the episode plot
        plt.savefig(os.path.join(output_dir, f'episode_{metric}.png'))
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

        plt.title(f'Cumulative {metric.replace("_", " ").title()}')
        plt.xlabel('Episode')
        plt.ylabel(f'Cumulative {metric.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the cumulative plot
        plt.savefig(os.path.join(output_dir, f'cumulative_{metric}.png'))
        plt.close()

def main():
    """Main function to run the combined visualization."""
    print("Welcome to Combined Agent Visualization!")
    
    # Let user select agents
    selected_agents = select_agents()
    print(f"\nSelected agents: {', '.join(selected_agents)}")

    # Load histories for all selected agents
    agent_histories = {}
    for agent_name in selected_agents:
        history = load_agent_history(agent_name)
        if history:
            agent_histories[agent_name] = history
        else:
            print(f"Warning: Could not load history for {agent_name}")

    if not agent_histories:
        print("No agent histories could be loaded. Exiting.")
        return

    # Create output directory for combined visualizations
    output_dir = os.path.join(SAFE_RESULTS_DIR, "combined_visualization")
    
    # Generate combined visualizations
    print("\nGenerating combined visualizations...")
    create_combined_graphs(agent_histories, output_dir)
    print(f"\nVisualization complete! Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
