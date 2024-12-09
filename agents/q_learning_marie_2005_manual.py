import numpy as np
import pygame
import time
from agents.basic_qlearning import BasicQLearningAgent


class QLearningAgent(BasicQLearningAgent):
    """
    Q-Learning Agent with Manual Demonstrations based on Marie's 2005 methodology.

    This agent allows for an initial phase of manual demonstrations where a human operator
    controls the agent using keyboard inputs. After the demonstration phase, the agent continues
    to learn autonomously using the Q-learning algorithm.

    Attributes:
        manual_episodes (int): Number of episodes to run with manual control.
        action_keys (dict): Mapping of keyboard keys to action indices.
    """

    def __init__(
        self,
        state_size,
        n_actions,
        learning_rate=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.001,
        manual_episodes=20,
    ):
        """
        Initialize the QLearningAgent with manual demonstration parameters.

        Args:
            state_size (int): Size of the state space.
            n_actions (int): Number of possible actions.
            learning_rate (float, optional): Learning rate for Q-value updates.
            gamma (float, optional): Discount factor for future rewards.
            epsilon (float, optional): Initial exploration rate.
            epsilon_min (float, optional): Minimum exploration rate.
            epsilon_decay (float, optional): Decay rate for the exploration rate.
            manual_episodes (int, optional): Number of episodes for manual demonstrations.
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
        self.manual_episodes = manual_episodes
        self.action_keys = {
            pygame.K_RIGHT: 0,
            pygame.K_DOWN: 1,
            pygame.K_LEFT: 2,
            pygame.K_UP: 3,
        }

    def _get_manual_action(self, env):
        """
        Get an action from the user via keyboard input.

        Args:
            env: The environment instance (used for handling events).

        Returns:
            int or None: The action selected by the user, or None if the window is closed.
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key in self.action_keys:
                        return self.action_keys[event.key]
            time.sleep(0.1)

    def train(
        self,
        env,
        episodes=10000,
        max_steps=1000,
        render_freq=100,
        render_mode="human",
        render_delay=0.1,
    ):
        """
        Train the agent using manual demonstrations followed by autonomous learning.

        Args:
            env: The environment to train the agent in.
            episodes (int, optional): Total number of training episodes.
            max_steps (int, optional): Maximum steps per episode.
            render_freq (int, optional): Frequency to render the environment.
            render_mode (str, optional): Mode to render ('human' or None).
            render_delay (float, optional): Delay between renders in seconds.
        """
        print(f"\nStarting {self.manual_episodes} manual demonstration episodes...")
        print("Use arrow keys to control the agent")

        history = {
            "rewards": [],
            "epsilon": [],
            "max_q": [],
            "steps": [],
            "episodes": list(range(episodes)),
        }

        for episode in range(self.manual_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False

            print(f"\nManual Episode {episode + 1}/{self.manual_episodes}")
            while not done and steps < max_steps:
                env.render(mode=render_mode)
                time.sleep(render_delay)

                action = self._get_manual_action(env)
                if action is None:
                    env.close()
                    return

                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state)

                state = next_state
                total_reward += reward
                steps += 1

            history["rewards"].append(total_reward)
            history["epsilon"].append(self.epsilon)
            history["max_q"].append(np.max(self.q_table))
            history["steps"].append(steps)

            print(f"Episode completed - Steps: {steps}, Reward: {total_reward:.2f}")

        print("\nStarting autonomous learning phase...")
        for episode in range(self.manual_episodes, episodes):
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
