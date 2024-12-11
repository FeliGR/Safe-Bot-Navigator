import numpy as np
import time
import matplotlib.pyplot as plt


class BasicQLearningAgent:
    """A basic Q-learning agent for reinforcement learning tasks.

    Attributes:
        state_size (int): The number of possible states in the environment.
        n_actions (int): The number of possible actions the agent can take.
        learning_rate (float): The learning rate for updating Q-values.
        gamma (float): The discount factor for future rewards.
        epsilon (float): The exploration rate for the epsilon-greedy policy.
        epsilon_min (float): The minimum exploration rate.
        epsilon_decay (float): The rate at which the exploration rate decays.
        q_table (np.ndarray): The Q-value table storing state-action values.
        train_history (dict): A history of training metrics.
        greedy_history (dict): A history of evaluation metrics using a greedy policy.
    """

    TRAIN_HISTORY = "train"
    GREEDY_HISTORY = "greedy"
    BOTH_HISTORIES = "both"
    VALID_HISTORY_TYPES = {TRAIN_HISTORY, GREEDY_HISTORY, BOTH_HISTORIES}

    def __init__(
        self,
        state_size,
        n_actions,
        learning_rate=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.001,
    ):
        """Initialize the Q-learning agent with given parameters.

        Args:
            state_size (int): Number of possible states.
            n_actions (int): Number of possible actions.
            learning_rate (float, optional): Learning rate for Q-value updates.
            gamma (float, optional): Discount factor for future rewards.
            epsilon (float, optional): Initial exploration rate.
            epsilon_min (float, optional): Minimum exploration rate after decay.
            epsilon_decay (float, optional): Decay rate for the exploration rate.
        """
        self.state_size = state_size
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((state_size, n_actions))
        self.train_history = {}
        self.greedy_history = {}

    def get_action(self, state):
        """Select an action using the epsilon-greedy policy.

        Args:
            state (int): The current state of the environment.

        Returns:
            int: The action selected.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        options = np.argwhere(
            self.q_table[state] == np.max(self.q_table[state])
        ).flatten()
        return np.random.choice(options)

    def update(self, state, action, reward, next_state):
        """Update the Q-value for a state-action pair.

        Args:
            state (int): The current state.
            action (int): The action taken.
            reward (float): The reward received after taking the action.
            next_state (int): The next state after taking the action.
        """
        current_q = self.q_table[state][action]
        future_q = np.max(self.q_table[next_state])
        self.q_table[state][action] = current_q + self.learning_rate * (
            reward + self.gamma * future_q - current_q
        )

    def decay_epsilon(self):
        """Decay epsilon after an episode by subtracting epsilon_decay"""
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def train(
        self,
        env,
        episodes=1000,
        max_steps=1000,
        render_mode=None,
        render_freq=100,
        render_delay=0.1,
    ):
        """Train the agent using Q-learning"""
        history = {
            "rewards": [],
            "epsilon": [],
            "max_q": [],
            "steps": [],
            "episodes": list(range(episodes)),
        }
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < max_steps:
                if render_mode and episode % render_freq == 0:
                    env.render(mode=render_mode)
                    time.sleep(render_delay)
                
                action = self.get_action(state)
                next_state, reward, done, info = env.step(action)  # Update to unpack 4 values
                
                self.update(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            history["rewards"].append(total_reward)
            history["epsilon"].append(self.epsilon)
            history["max_q"].append(np.max(self.q_table))
            history["steps"].append(steps)
            
            if episode % 100 == 0:
                print(
                    f"Episode {episode}/{episodes}, "
                    f"Reward: {total_reward:.2f}, "
                    f"Epsilon: {self.epsilon:.2f}"
                )
            
            self.decay_epsilon()
        
        return history

    def run_greedy(
        self, env, episodes=1, max_steps=1000, render_mode="human", render_delay=0.1
    ):
        """Evaluate the agent using a greedy policy over multiple episodes.

        Args:
            env: The environment for evaluation.
            episodes (int, optional): Number of evaluation episodes.
            max_steps (int, optional): Maximum steps per episode.
            render_mode (str, optional): Mode to render ('human' or None).
            render_delay (float, optional): Delay between renders in seconds.
        """
        history = {
            "rewards": [],
            "steps": [],
            "success": [],
            "episodes": list(range(episodes)),
        }

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done and steps < max_steps:
                if render_mode:
                    env.render(mode=render_mode)
                    time.sleep(render_delay)

                # action = np.argmax(self.q_table[state])
                options = np.argwhere(
                    self.q_table[state] == np.max(self.q_table[state])
                ).flatten()
                action = np.random.choice(options)
                next_state, reward, done, info = env.step(action)

                state = next_state
                total_reward += reward
                steps += 1

            if render_mode:
                env.render(mode=render_mode)
                time.sleep(render_delay)

            history["rewards"].append(total_reward)
            history["steps"].append(steps)
            history["success"].append(done)

            if episode % 10 == 0:
                success_rate = sum(history["success"][-10:]) / min(10, episode + 1)
                print(
                    f"Episode {episode}/{episodes}, Steps: {steps}, "
                    f"Reward: {total_reward:.2f}, Success Rate: {success_rate:.2f}, "
                    f"Epsilon: {self.epsilon:.2f}"
                )

        if render_mode:
            env.close()

        self.greedy_history = history

    def plot_history(self, history_type=BOTH_HISTORIES, save_path=None):
        """Plot the training and evaluation history of the agent.

        Args:
            history_type (str, optional): The type of history to plot ('train', 'greedy', or 'both').
            save_path (str, optional): Path to save the plot image.
        """
        if history_type not in self.VALID_HISTORY_TYPES:
            raise ValueError(
                f"Invalid history type. Must be one of {self.VALID_HISTORY_TYPES}"
            )

        histories = []
        if (
            history_type in [self.TRAIN_HISTORY, self.BOTH_HISTORIES]
            and self.train_history
        ):
            histories.append(("Training", self.train_history))
        if (
            history_type in [self.GREEDY_HISTORY, self.BOTH_HISTORIES]
            and self.greedy_history
        ):
            histories.append(("Greedy", self.greedy_history))

        if not histories:
            print("No history available to plot")
            return

        n_metrics = sum(
            len([k for k in h[1].keys() if k != "episodes"]) for h in histories
        )

        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        plt.figure(figsize=(8 * n_cols, 5 * n_rows))

        plot_idx = 1

        for title, history in histories:
            metrics = [k for k in history.keys() if k != "episodes"]
            episodes = history["episodes"]

            for metric in metrics:
                plt.subplot(n_rows, n_cols, plot_idx)
                plt.plot(episodes, history[metric], label=f"{title} {metric}")
                plt.title(f'{title} {metric.replace("_", " ").title()}')
                plt.xlabel("Episodes")
                plt.ylabel(metric.replace("_", " ").title())
                plt.grid(True)
                plt.legend()

                plot_idx += 1

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        plt.close()
