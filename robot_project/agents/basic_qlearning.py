import numpy as np
import json
import time
import matplotlib.pyplot as plt


class BasicQLearningAgent:

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
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        options = np.argwhere(
            self.q_table[state] == np.max(self.q_table[state])
        ).flatten()
        return np.random.choice(options)

    def update(self, state, action, reward, next_state):
        """Update Q-value for state-action pair"""
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
        episodes=10000,
        max_steps=1000,
        render_freq=100,
        render_mode=None,
        render_delay=0.1,
    ):
        """Train the agent using Q-learning

        Args:
            env: The environment to train on
            episodes: Number of episodes to train for
            max_steps: Maximum steps per episode
            render_freq: How often to render the environment (episodes)
            render_mode: Rendering mode ('human' or None)
            render_delay: Delay between renders in seconds

        Returns:
            dict: Training history containing rewards, epsilon values and max Q-values
        """
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
                next_state, reward, done = env.step(action)

                self.update(state, action, reward, next_state)

                state = next_state
                total_reward += reward
                steps += 1

            if render_mode and episode % render_freq == 0:
                env.render(mode=render_mode)
                time.sleep(render_delay)

            history["rewards"].append(total_reward)
            history["epsilon"].append(self.epsilon)
            history["max_q"].append(np.max(self.q_table))
            history["steps"].append(steps)

            if episode % 100 == 0:
                print(
                    f"Episode {episode}/{episodes}, Steps: {steps}, "
                    f"Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}, "
                    f"Max Q-value: {np.max(self.q_table):.2f}"
                )

            self.decay_epsilon()

        if render_mode:
            env.close()

        self.train_history = history

    def run_greedy(
        self, env, episodes=1, max_steps=1000, render_mode="human", render_delay=0.1
    ):
        """Run the agent using a greedy policy (no exploration) for multiple episodes.

        Args:
            env: The environment to run in
            episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            render_mode: How to render the environment (None or 'human')
            render_delay: Delay between renders in seconds

        Returns:
            dict: History containing rewards, steps, and success rate
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

                action = np.argmax(self.q_table[state])
                next_state, reward, done = env.step(action)

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
        """Plot agent histories.

        Args:
            history_type: Which history to plot (TRAIN_HISTORY, GREEDY_HISTORY, or BOTH_HISTORIES)
            save_path: Optional path to save the plot
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

        plt.figure(figsize=(12, 4 * n_metrics))

        plot_idx = 1

        for title, history in histories:
            metrics = [k for k in history.keys() if k != "episodes"]
            episodes = history["episodes"]

            for metric in metrics:
                plt.subplot(n_metrics, 1, plot_idx)
                plt.plot(episodes, history[metric], label=f"{title} {metric}")
                plt.title(f'{title} {metric.replace("_", " ").title()}')
                plt.xlabel("Episodes")
                plt.ylabel(metric.replace("_", " ").title())
                plt.grid(True)
                plt.legend()

                plt.subplots_adjust(hspace=0.4)

                plot_idx += 1

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        plt.close()
