import numpy as np
import time
from agents.basic_qlearning import BasicQLearningAgent
from agents.planified import PlanifiedAgent

class QLearningAgentMarie2005(BasicQLearningAgent):
    def __init__(
        self,
        state_size,
        n_actions,
        learning_rate=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.001,
        planified_episodes=20,
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
        self.planified_episodes = planified_episodes
        self.planifier = PlanifiedAgent()

    def train(
        self,
        env,
        episodes=10000,
        max_steps=1000,
        render_freq=100,
        render_mode=None,
        render_delay=0.1,
    ):
        """Train with planified demonstrations followed by autonomous learning"""
        print(f"\nStarting {self.planified_episodes} planified demonstration episodes...")

        history = {
            "rewards": [],
            "epsilon": [],
            "max_q": [],
            "steps": [],
            "episodes": list(range(episodes)),
        }

        # Planified demonstration phase
        for episode in range(self.planified_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False

            print(f"\nPlanified Episode {episode + 1}/{self.planified_episodes}")
            
            # Get the planned path for this episode
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

                # Get action from the planned path
                action = planned_actions[steps]
                next_state, reward, done = env.step(action)
                
                # Update Q-table with the planified action
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

        # Autonomous learning phase
        print("\nStarting autonomous learning phase...")
        for episode in range(self.planified_episodes, episodes):
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
