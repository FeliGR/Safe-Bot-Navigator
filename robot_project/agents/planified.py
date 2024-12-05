import numpy as np
import json
import time
import matplotlib.pyplot as plt

class PlanifiedAgent:
    def __init__(self):
        """Initialize the planified agent"""
        self.current_plan = None
        self.steps_taken = 0
        self.total_reward = 0
        
    def get_action(self, env):
        """Get the next action from the planned path
        
        Args:
            env: The environment instance
            
        Returns:
            int: The next action to take
        """
        # If we don't have a plan or finished the current one, get a new plan
        if self.current_plan is None or not self.current_plan:
            # First try to find a path without using traps
            self.current_plan = env.find_shortest_path(allow_traps=False)
            
            # If no path found, try again allowing traps
            if self.current_plan is None:
                self.current_plan = env.find_shortest_path(allow_traps=True)
                
            if self.current_plan is None:
                # This should never happen since allowing traps should always find a path
                print("Warning: No path found even with traps allowed!")
                return env.ACTIONS[0]
        
        # Return and remove the first action from the plan
        return self.current_plan.pop(0)
    
    def run_episode(self, env, render=False, render_delay=0.1):
        """Run a single episode
        
        Args:
            env: The environment to run on
            render: Whether to render the environment
            render_delay: Delay between renders in seconds
            
        Returns:
            tuple: (total_reward, steps_taken, success)
        """
        state = env.reset()
        self.current_plan = None
        self.steps_taken = 0
        self.total_reward = 0
        done = False
        
        while not done:
            if render:
                env.render(mode='human')
                time.sleep(render_delay)
            
            action = self.get_action(env)
            _, reward, done = env.step(action)
            
            self.total_reward += reward
            self.steps_taken += 1
        
        if render:
            env.render(mode='human')
            time.sleep(render_delay)
        
        return self.total_reward, self.steps_taken, self.total_reward > 0
    
    def run_episodes(self, env, num_episodes=100, render_freq=10):
        """Run multiple episodes and track performance
        
        Args:
            env: The environment to run on
            num_episodes: Number of episodes to run
            render_freq: How often to render the environment (episodes)
            
        Returns:
            dict: Statistics about the episodes
        """
        history = {
            'rewards': [],
            'steps': [],
            'success_rate': [],
            'episodes': list(range(num_episodes))
        }
        
        successes = 0
        
        for episode in range(num_episodes):
            render = episode % render_freq == 0
            reward, steps, success = self.run_episode(env, render=render)
            
            if success:
                successes += 1
            
            history['rewards'].append(reward)
            history['steps'].append(steps)
            history['success_rate'].append(successes / (episode + 1))
            
            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes}, Steps: {steps}, "
                      f"Reward: {reward:.2f}, Success Rate: {(successes / (episode + 1)):.2%}")
        
        return history
    
    def plot_history(self, history, save_path=None):
        """Plot agent histories.
        
        Args:
            history: The history to plot
            save_path: Optional path to save the plot
        """
        metrics = [k for k in history.keys() if k != 'episodes']
        episodes = history['episodes']
        
        # Create figure with enough height for all plots
        plt.figure(figsize=(12, 4 * len(metrics)))
        
        # Track the current subplot index
        plot_idx = 1
        
        for metric in metrics:
            plt.subplot(len(metrics), 1, plot_idx)
            plt.plot(episodes, history[metric], label=f'{metric}')
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.xlabel('Episodes')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True)
            plt.legend()
            
            # Add more space between subplots
            plt.subplots_adjust(hspace=0.4)
            
            plot_idx += 1
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()