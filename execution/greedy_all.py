"""Run greedy evaluation on all trained agents."""

import os
import json
import pickle
import numpy as np
import importlib
import sys
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from greedy import list_saved_agents, load_agent, run_greedy_evaluation

def run_all_agents(episodes=50, render_mode=None, render_delay=0.1):
    """
    Run greedy evaluation on all trained agents.
    
    Args:
        episodes (int): Number of episodes to run for each agent
        render_mode (str): Rendering mode ('human' or None)
        render_delay (float): Delay between renders in seconds
    """
    agents = list_saved_agents()
    if not agents:
        print("No trained agents found.")
        return

    print(f"\nFound {len(agents)} trained agents. Running greedy evaluation on each...")
    
    for i, agent_info in enumerate(agents, 1):
        print(f"\n{'='*50}")
        print(f"Agent {i}/{len(agents)}: {agent_info['name']}")
        print(f"{'='*50}")
        print(agent_info['summary'])
        print("\nStarting greedy evaluation...")
        
        try:
            run_greedy_evaluation(
                agent_info,
                episodes=episodes,
                render_mode=render_mode,
                render_delay=render_delay,
                plot=False
            )
            print(f"\nCompleted evaluation of {agent_info['name']}")
        except Exception as e:
            print(f"Error evaluating {agent_info['name']}: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser(description='Run greedy evaluation on all trained agents')
    parser.add_argument('--episodes', type=int, default=50,
                      help='Number of episodes to run for each agent (default: 50)')
    parser.add_argument('--render', action='store_true',
                      help='Enable rendering (human mode)')
    parser.add_argument('--delay', type=float, default=0.1,
                      help='Delay between renders in seconds (default: 0.1)')
    
    args = parser.parse_args()
    render_mode = None
    
    run_all_agents(
        episodes=args.episodes,
        render_mode=render_mode,
        render_delay=args.delay
    )

if __name__ == "__main__":
    main()
