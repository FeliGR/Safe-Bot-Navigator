import numpy as np
import pygame
import time
from agents.basic_qlearning import BasicQLearningAgent


class QLearningAgent(BasicQLearningAgent):
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
            pygame.K_RIGHT: 0,  # Move right
            pygame.K_DOWN: 1,  # Move down
            pygame.K_LEFT: 2,  # Move left
            pygame.K_UP: 3,  # Move up
        }

    def _get_manual_action(self, env):
        """Get action from keyboard input"""
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
        """Train with manual demonstrations followed by autonomous learning"""
        print(f"\nStarting {self.manual_episodes} manual demonstration episodes...")
        print("Use arrow keys to control the agent")

        history = {
            "rewards": [],
            "epsilon": [],
            "max_q": [],
            "steps": [],
            "episodes": list(range(episodes)),
        }

        # Manual demonstration phase
        for episode in range(self.manual_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False

            print(f"\nManual Episode {episode + 1}/{self.manual_episodes}")
            while not done and steps < max_steps:
                env.render(mode=render_mode)
                time.sleep(render_delay)

                # Get action from keyboard
                action = self._get_manual_action(env)
                if action is None:  # Quit requested
                    env.close()
                    return

                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state)

                state = next_state
                total_reward += reward
                steps += 1

            # Track metrics
            history["rewards"].append(total_reward)
            history["epsilon"].append(self.epsilon)
            history["max_q"].append(np.max(self.q_table))
            history["steps"].append(steps)

            print(f"Episode completed - Steps: {steps}, Reward: {total_reward:.2f}")

        # Rest of training code remains the same...
        # Autonomous learning phase
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
