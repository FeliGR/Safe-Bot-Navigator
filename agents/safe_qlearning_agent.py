import numpy as np
from .basic_qlearning import BasicQLearningAgent
from .planified import PlanifiedAgent
import time
import random


class SafeQLearningAgent(BasicQLearningAgent):
    """A Q-learning agent that implements safety-aware exploration and learning.
    
    This agent extends the basic Q-learning agent by incorporating safety constraints
    and maintaining a minimum safe distance from dangerous elements (traps and obstacles).
    It can also request help from a planifier when safety is compromised.
    """
    
    UNIFORM = "uniform"
    STOCHASTIC = "stochastic"
    
    def __init__(
        self,
        state_size,
        n_actions,
        learning_rate=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.001,
        trap_threshold=2,
        trap_penalty=-0.1,
        obstacle_threshold=2,
        obstacle_penalty=-0.1,
        planified_episodes=20,
        safety_threshold=0.5,  # Threshold for safety factor to trigger planifier
        interaction_type=UNIFORM,
        planner_probability=0.5,
        teacher_expertise=0.8,
        teacher_safety=2
    ):
        """Initialize the safe Q-learning agent.
        
        Args:
            state_size (int): Number of possible states
            n_actions (int): Number of possible actions
            learning_rate (float): Learning rate for Q-value updates
            gamma (float): Discount factor
            epsilon (float): Initial exploration rate
            epsilon_min (float): Minimum exploration rate
            epsilon_decay (float): Exploration rate decay
            trap_threshold (int): Minimum safe distance from traps
            trap_penalty (float): Penalty for unsafe situations with traps
            obstacle_threshold (int): Minimum safe distance from obstacles
            obstacle_penalty (float): Penalty for unsafe situations with obstacles
            planified_episodes (int): Number of episodes for teacher demonstrations
            safety_threshold (float): Threshold for safety factor to trigger planifier
            interaction_type (str): Type of interaction ('uniform' or 'stochastic')
            planner_probability (float): Probability of using planifier in uniform mode
            teacher_expertise (float): Probability that teacher prefers safe paths
        """
        super().__init__(
            state_size,
            n_actions,
            learning_rate,
            gamma,
            epsilon,
            epsilon_min,
            epsilon_decay,
        )
        self.trap_threshold = trap_threshold
        self.trap_penalty = trap_penalty
        self.obstacle_threshold = obstacle_threshold
        self.obstacle_penalty = obstacle_penalty
        self.safety_values = {}  # Store safety values for each state
        
        # Planifier-related attributes
        self.planified_episodes = planified_episodes
        self.planifier = PlanifiedAgent()
        self.safety_threshold = safety_threshold
        self.interaction_type = interaction_type
        self.planner_probability = planner_probability
        self.teacher_expertise = teacher_expertise
        self.teacher_safety = teacher_safety
        
    def get_manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_nearest_trap_distance(self, state, env):
        """Calculate distance to nearest trap from current state."""
        grid_size = int(np.sqrt(self.state_size))  # Assuming square grid
        pos = [state // grid_size, state % grid_size]
        min_distance = float('inf')
        
        for i in range(grid_size):
            for j in range(grid_size):
                if env.grid[i, j] == env.TRAP:
                    distance = self.get_manhattan_distance(pos, [i, j])
                    min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else grid_size * 2
    
    def get_nearest_obstacle_distance(self, state, env):
        """Calculate distance to nearest obstacle from current state."""
        grid_size = int(np.sqrt(self.state_size))  # Assuming square grid
        pos = [state // grid_size, state % grid_size]
        min_distance = float('inf')
        
        for i in range(grid_size):
            for j in range(grid_size):
                if env.grid[i, j] == env.OBSTACLE:
                    distance = self.get_manhattan_distance(pos, [i, j])
                    min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else grid_size * 2

    def is_safe_state(self, state, env):
        """Determine if a state is safe based on distance to traps and obstacles."""
        trap_distance = self.get_nearest_trap_distance(state, env)
        obstacle_distance = self.get_nearest_obstacle_distance(state, env)
        return trap_distance >= self.trap_threshold and obstacle_distance >= self.obstacle_threshold
    
    def get_safe_action(self, state, env):
        """Select an action that maintains safety constraints while optimizing for the goal."""
        grid_size = int(np.sqrt(self.state_size))
        
        if np.random.random() < self.epsilon:
            # Safe exploration
            safe_actions = []
            pos = [state // grid_size, state % grid_size]
            
            for action in range(self.n_actions):
                new_pos = pos.copy()
                
                if action == env.MOVE_RIGHT:
                    new_pos[1] += 1
                elif action == env.MOVE_DOWN:
                    new_pos[0] += 1
                elif action == env.MOVE_LEFT:
                    new_pos[1] -= 1
                elif action == env.MOVE_UP:
                    new_pos[0] -= 1
                
                # Check if new position is within bounds
                if (0 <= new_pos[0] < grid_size and 
                    0 <= new_pos[1] < grid_size):
                    new_state = new_pos[0] * grid_size + new_pos[1]
                    if self.is_safe_state(new_state, env):
                        safe_actions.append(action)
            
            if safe_actions:
                return np.random.choice(safe_actions)
            
        # If no safe exploratory actions or not exploring, use Q-values
        options = np.argwhere(
            self.q_table[state] == np.max(self.q_table[state])
        ).flatten()
        return np.random.choice(options)
            
    def get_safety_factor(self, state, env):
        """Calculate the combined safety factor based on trap and obstacle distances."""
        trap_distance = self.get_nearest_trap_distance(state, env)
        obstacle_distance = self.get_nearest_obstacle_distance(state, env)
        
        trap_safety = min(1.0, trap_distance / self.trap_threshold)
        obstacle_safety = min(1.0, obstacle_distance / self.obstacle_threshold)
        
        return min(trap_safety, obstacle_safety)

    def get_planified_action(self, env):
        """Get an action from the planner based on teacher expertise.
        
        The planner will first attempt to find a path avoiding both traps and obstacles
        with probability equal to teacher_expertise. If this fails or if the teacher
        decides to allow risky paths, it will find any available path.
        
        Args:
            env: The environment providing path planning capabilities.
        
        Returns:
            int: The planned action to take.
        """
        if np.random.random() < self.teacher_expertise:
            planned_actions = env.find_shortest_path(allow_traps=False, safety_distance=self.teacher_safety)
            if planned_actions is not None:
                return planned_actions[0]
        
        planned_actions = env.find_shortest_path(allow_traps=True, safety_distance=random.randint(0, self.teacher_safety))
        if planned_actions is None:
            return np.random.randint(0, self.n_actions)
        return planned_actions[0]

    def get_action(self, state, env=None):
        """Select an action based on safety considerations and interaction type.
        
        Returns:
            tuple: (action, used_planifier) where used_planifier is True if the action came from planifier
        """
        if env is None:
            return super().get_action(state), False
            
        safety_factor = self.get_safety_factor(state, env)
        
        # Skip planifier if safety_threshold is None
        if self.safety_threshold is not None and safety_factor < self.safety_threshold:
            if self.interaction_type == self.UNIFORM:
                if np.random.random() < self.planner_probability:
                    print("Using planifier due to safety concerns (uniform)...")
                    return self.get_planified_action(env), True
            else:  # STOCHASTIC
                state_q_values = self.q_table[state]
                q_diff = np.max(state_q_values) - np.min(state_q_values)
                if q_diff < safety_factor:  # Use Q-value uncertainty as additional criterion
                    print("Using planifier due to safety concerns (stochastic)...")
                    return self.get_planified_action(env), True
        
        return self.get_safe_action(state, env), False

    def update(self, state, action, reward, next_state, env=None):
        """Update Q-values with safety considerations for both traps and obstacles."""
        if env is not None:
            # Apply safety penalties if too close to traps or obstacles
            trap_distance = self.get_nearest_trap_distance(state, env)
            obstacle_distance = self.get_nearest_obstacle_distance(state, env)
            
            if trap_distance < self.trap_threshold:
                reward += self.trap_penalty
            if obstacle_distance < self.obstacle_threshold:
                reward += self.obstacle_penalty
            
            # Weight the update based on combined safety
            trap_safety = min(1.0, trap_distance / self.trap_threshold)
            obstacle_safety = min(1.0, obstacle_distance / self.obstacle_threshold)
            # safety_factor = min(trap_safety, obstacle_safety)  # Original, left it there, just in case
            safety_factor = trap_safety * obstacle_safety
            
            # Update Q-value with safety weighting
            current_q = self.q_table[state][action]
            future_q = np.max(self.q_table[next_state])
            self.q_table[state][action] = current_q + self.learning_rate * safety_factor * (
                reward + self.gamma * future_q - current_q
            )
        else:
            super().update(state, action, reward, next_state)

    def train(
        self,
        env,
        episodes=1000,
        max_steps=1000,
        render_freq=100,
        render_mode=None,
        render_delay=0.1,
    ):
        """Train the agent with safety considerations and planifier demonstrations."""
        print(f"\nStarting {self.planified_episodes} planified demonstration episodes...")
        
        history = {
            "rewards": [],
            "epsilon": [],
            "max_q": [],
            "steps": [],
            "collisions": [],
            "trap_steps": [],  # Will now be a counter
            "trap_activations": [],
            "episodes": list(range(episodes)),
        }
        
        # Start with planified episodes
        for episode in range(self.planified_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            episode_collisions = 0
            episode_trap_steps = 0
            episode_trap_activations = 0
            done = False
            
            print(f"\nPlanified Episode {episode + 1}/{self.planified_episodes}")
            
            planned_actions = env.find_shortest_path(allow_traps=False)
            if planned_actions is None:
                planned_actions = env.find_shortest_path(allow_traps=True)
            
            if planned_actions is None:
                print("Warning: No path found! Skipping episode.")
                continue
            
            while not done and steps < max_steps and steps < len(planned_actions):
                if render_mode:
                    env.render(mode=render_mode)
                    time.sleep(render_delay)
                
                action = planned_actions[steps]
                next_state, reward, done, info = env.step(action)
                
                episode_collisions += int(info.get("collision", False))
                episode_trap_steps += info.get("trap_step", 0)  # Add trap steps
                episode_trap_activations += int(info.get("trap_activation", False))
                
                self.update(state, action, reward, next_state, env)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            history["rewards"].append(total_reward)
            history["epsilon"].append(self.epsilon)
            history["max_q"].append(np.max(self.q_table))
            history["steps"].append(steps)
            history["collisions"].append(episode_collisions)
            history["trap_steps"].append(episode_trap_steps)
            history["trap_activations"].append(episode_trap_activations)
            
            print(f"Episode completed - Steps: {steps}, Reward: {total_reward:.2f}")
        
        # Continue with regular training
        print("\nStarting autonomous learning phase...")
        for episode in range(self.planified_episodes, episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            episode_collisions = 0
            episode_trap_steps = 0
            episode_trap_activations = 0
            done = False
            
            while not done and steps < max_steps:
                if render_mode and episode % render_freq == 0:
                    env.render(mode=render_mode)
                    time.sleep(render_delay)
                
                action, used_planifier = self.get_action(state, env)
                if used_planifier:
                    episode_trap_steps += 1
                
                next_state, reward, done, info = env.step(action)
                
                episode_collisions += int(info.get("collision", False))
                episode_trap_steps += info.get("trap_step", 0)  # Add trap steps
                episode_trap_activations += int(info.get("trap_activation", False))
                
                self.update(state, action, reward, next_state, env)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            history["rewards"].append(total_reward)
            history["epsilon"].append(self.epsilon)
            history["max_q"].append(np.max(self.q_table))
            history["steps"].append(steps)
            history["collisions"].append(episode_collisions)
            history["trap_steps"].append(episode_trap_steps)
            history["trap_activations"].append(episode_trap_activations)
            
            if episode % 100 == 0:
                print(
                    f"Episode {episode}/{episodes}, Steps: {steps}, "
                    f"Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}, "
                    f"Trap Violations: {episode_trap_steps}, "
                    f"Obstacle Violations: {episode_collisions}, "
                    f"Planifier Uses: {episode_trap_steps}"
                )
            
            self.decay_epsilon()
        
        if render_mode:
            env.close()
        
        self.train_history = history

    def _update_history(self, history, total_reward, steps, trap_violations, 
                       obstacle_violations, planifier_uses, info):
        """Helper method to update training history."""
        history["rewards"].append(total_reward)
        history["epsilon"].append(self.epsilon)
        history["max_q"].append(np.max(self.q_table))
        history["steps"].append(steps)
        history["trap_violations"].append(trap_violations)
        history["obstacle_violations"].append(obstacle_violations)
        history["planifier_uses"].append(planifier_uses)
        history["collisions"].append(info.get("collision", False))
        history["trap_steps"].append(info.get("trap_step", False))
        history["trap_activations"].append(info.get("trap_activation", False))

    def plot_history(self, history_type=BasicQLearningAgent.BOTH_HISTORIES, save_path=None):
        """Plot training history including safety metrics for both traps and obstacles."""
        import matplotlib.pyplot as plt
        
        if history_type not in self.VALID_HISTORY_TYPES:
            raise ValueError(f"Invalid history type. Must be one of {self.VALID_HISTORY_TYPES}")
        
        histories = []
        if history_type in [self.TRAIN_HISTORY, self.BOTH_HISTORIES]:
            histories.append((self.train_history, "Training"))
        if history_type in [self.GREEDY_HISTORY, self.BOTH_HISTORIES]:
            histories.append((self.greedy_history, "Greedy"))
            
        if not histories:
            print("No history available to plot.")
            return
            
        fig = plt.figure(figsize=(15, 12))
        
        for i, (history, label) in enumerate(histories):
            if not history:
                continue
                
            # Plot rewards
            plt.subplot(3, 2, 1)
            plt.plot(history["episodes"], history["rewards"], label=f"{label} Rewards")
            plt.title("Rewards per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.legend()
            
            # Plot steps
            plt.subplot(3, 2, 2)
            plt.plot(history["episodes"], history["steps"], label=f"{label} Steps")
            plt.title("Steps per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Steps")
            plt.legend()
            
            if "epsilon" in history:
                plt.subplot(3, 2, 3)
                plt.plot(history["episodes"], history["epsilon"], label="Epsilon")
                plt.title("Exploration Rate")
                plt.xlabel("Episode")
                plt.ylabel("Epsilon")
                plt.legend()
            
            if "trap_violations" in history:
                plt.subplot(3, 2, 4)
                plt.plot(history["episodes"], history["trap_violations"], label="Trap Violations")
                plt.title("Trap Safety Violations per Episode")
                plt.xlabel("Episode")
                plt.ylabel("Number of Violations")
                plt.legend()
                
            if "obstacle_violations" in history:
                plt.subplot(3, 2, 5)
                plt.plot(history["episodes"], history["obstacle_violations"], label="Obstacle Violations")
                plt.title("Obstacle Safety Violations per Episode")
                plt.xlabel("Episode")
                plt.ylabel("Number of Violations")
                plt.legend()
            
            if "planifier_uses" in history:
                plt.subplot(3, 2, 6)
                plt.plot(history["episodes"], history["planifier_uses"], label="Planifier Uses")
                plt.title("Planifier Uses per Episode")
                plt.xlabel("Episode")
                plt.ylabel("Number of Uses")
                plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
