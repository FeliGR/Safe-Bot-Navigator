"""Visualize all trained agents' results."""

import os
import sys
import argparse
from datetime import datetime
import glob

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINED_AGENTS_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'trained_agents')
SAFE_RESULTS_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'safe_results')
sys.path.append(PROJECT_ROOT)

from environment.environment import GridEnvironment
from visualization.visualize_episodes import create_episode_graphs
from visualization.visualize_cumulative import create_cumulative_graphs
from visualization.visualize_politic import create_policy_visualization, load_pickle, load_json

def find_trained_agents(base_dir):
    """Find all trained agent directories."""
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

def find_related_files(agent_dir):
    """Find all related files for a given agent directory."""
    files = {
        'agent': None,
        'env': None
    }
    
    # Search patterns
    patterns = {
        'agent': 'agent.pkl',
        'env': 'env.pkl'
    }
    
    # Search in agent directory
    for file_type, pattern in patterns.items():
        file_path = os.path.join(agent_dir, pattern)
        if os.path.exists(file_path):
            files[file_type] = file_path
    
    return files

def create_visualization_directory(agent_name):
    """Create a directory for storing visualizations."""
    # Use exact agent name for the visualization directory
    vis_dir = os.path.join(SAFE_RESULTS_DIR, agent_name)
    os.makedirs(vis_dir, exist_ok=True)
    return vis_dir

def process_history(history, prefix, vis_dir):
    """Process and visualize a specific history (training or greedy)."""
    if not history:
        return False
        
    if isinstance(history, dict):
        print(f"Converting {prefix} dictionary history format...")
        # Convert dictionary format to list of dictionaries if needed
        episodes = len(next((v for v in history.values() if isinstance(v, list)), []))
        list_history = []
        
        # Get all available keys from the history dictionary
        available_keys = list(history.keys())
        print(f"Available metrics in history: {available_keys}")
        
        for i in range(episodes):
            episode_dict = {}
            for key in available_keys:
                try:
                    episode_dict[key] = history[key][i] if key in history else None
                except (IndexError, KeyError) as e:
                    print(f"Warning: Could not process {key} at episode {i}: {str(e)}")
                    episode_dict[key] = None
            list_history.append(episode_dict)
        history = list_history

    # Create specific directory for this history type
    history_dir = os.path.join(vis_dir, f'{prefix}_history')
    os.makedirs(history_dir, exist_ok=True)

    # Generate episode-by-episode graphs
    print(f"Generating {prefix} episode graphs...")
    try:
        create_episode_graphs(history, history_dir)
        print(f"{prefix} episode graphs generated successfully")
    except Exception as e:
        print(f"Error generating {prefix} episode graphs: {str(e)}")
        
    # Generate cumulative graphs
    print(f"Generating {prefix} cumulative graphs...")
    try:
        create_cumulative_graphs(history, history_dir)
        print(f"{prefix} cumulative graphs generated successfully")
    except Exception as e:
        print(f"Error generating {prefix} cumulative graphs: {str(e)}")
    
    return True

def visualize_agent_results(agent_dir, base_dir=None):
    """Generate all visualizations for an agent's results."""
    if base_dir is None:
        base_dir = os.path.dirname(agent_dir)
        
    agent_name = os.path.basename(agent_dir)
    
    # Find all related files
    files = find_related_files(agent_dir)
    
    if not files['agent']:
        print(f"Warning: No agent file found for: {agent_name}")
        return None
    
    # Create visualization directory
    vis_dir = create_visualization_directory(agent_name)
    
    # Load agent and extract history
    print(f"Loading agent from: {files['agent']}")
    try:
        agent = load_pickle(files['agent'])
    except Exception as e:
        print(f"Error loading agent: {str(e)}")
        return None
    
    # Process training history
    train_history = agent.train_history if hasattr(agent, 'train_history') else None
    if train_history:
        print("Found training history")
        process_history(train_history, 'training', vis_dir)
    else:
        print("No training history found")
        
    # Process greedy history
    greedy_history = agent.greedy_history if hasattr(agent, 'greedy_history') else None
    if greedy_history:
        print("Found greedy evaluation history")
        process_history(greedy_history, 'greedy', vis_dir)
    else:
        print("No greedy evaluation history found")
        
    # Process generic history if no specific histories found
    if not train_history and not greedy_history:
        generic_history = agent.history if hasattr(agent, 'history') else None
        if generic_history:
            print("Using generic history")
            process_history(generic_history, 'generic', vis_dir)
        else:
            print("Warning: No history found in agent")
    
    # Generate policy visualization if environment is available
    if files['env']:
        print("Generating policy visualization...")
        try:
            env = load_pickle(files['env'])
            policy_path = os.path.join(vis_dir, 'policy')
            create_policy_visualization(agent, env, agent_name, policy_path)
            print("Policy visualization generated successfully")
        except Exception as e:
            print(f"Error generating policy visualization: {str(e)}")
    else:
        print("Warning: Missing environment file, skipping policy visualization")
    
    print(f"\nAll visualizations have been saved to: {vis_dir}")
    print("\nFiles used:")
    for file_type, filepath in files.items():
        status = "✓ Found" if filepath else "✗ Not found"
        print(f"{file_type.capitalize()}: {status}")
        if filepath:
            print(f"  Path: {filepath}")
    
    print("\nHistory types processed:")
    print(f"Training history: {'✓ Found' if train_history else '✗ Not found'}")
    print(f"Greedy history: {'✓ Found' if greedy_history else '✗ Not found'}")
    if not train_history and not greedy_history:
        print(f"Generic history: {'✓ Found' if generic_history else '✗ Not found'}")
    
    return vis_dir

def process_all_agents(base_dir=None):
    """Process all trained agents in the given directory."""
    if base_dir is None:
        base_dir = TRAINED_AGENTS_DIR
    
    print(f"Searching for trained agents in: {base_dir}")
    print(f"Results will be saved in: {SAFE_RESULTS_DIR}\n")
    
    # Find all agent directories
    agent_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if 'agent.pkl' in files:
            agent_dirs.append(root)
    
    if not agent_dirs:
        print("No trained agents found!")
        return
    
    print(f"Found {len(agent_dirs)} trained agents\n")
    
    # Process each agent
    successful = []
    failed = []
    
    for agent_dir in agent_dirs:
        print("=" * 50)
        agent_name = os.path.basename(agent_dir)
        print(f"Processing agent: {agent_name}")
        print("=" * 50)
        
        try:
            vis_dir = visualize_agent_results(agent_dir)
            if vis_dir:
                successful.append((agent_name, vis_dir))
            else:
                failed.append((agent_name, "Failed to generate visualizations"))
        except Exception as e:
            print(f"Error processing agent: {str(e)}")
            failed.append((agent_name, str(e)))
        print("\n")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Processing Summary")
    print("=" * 50)
    
    if successful:
        print("\nSuccessfully processed agents:")
        for agent_name, vis_dir in successful:
            print(f"✓ {agent_name}")
            print(f"  Visualizations saved to: {vis_dir}")
    
    if failed:
        print("\nFailed to process agents:")
        for agent_name, error in failed:
            print(f"✗ {agent_name}")
            print(f"  Error: {error}")

def main():
    parser = argparse.ArgumentParser(description='Generate all visualizations for agent results')
    parser.add_argument('--agents_dir', type=str, help='Directory containing trained agents', default=TRAINED_AGENTS_DIR)
    
    args = parser.parse_args()
    process_all_agents(args.agents_dir)

if __name__ == "__main__":
    main()
