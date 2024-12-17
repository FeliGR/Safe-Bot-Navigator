import numpy as np
import time
from agents.basic_qlearning import BasicQLearningAgent
from agents.planified import PlanifiedAgent
import random


class QLearningAgentTeacher(BasicQLearningAgent):
    """
    Clouse Ask for Help Q-Learning Agent with configurable interaction types and teacher expertise.

    This agent implements a Q-learning algorithm that can learn from both autonomous exploration
    and a teaching agent (planner). The interaction between the agent and the teacher can be
    configured in different ways.

    Attributes:
        state_size (int): Size of the state space.
        n_actions (int): Number of possible actions.
        learning_rate (float): Learning rate for Q-value updates (α in Q-learning).
        gamma (float): Discount factor for future rewards (γ in Q-learning).
        epsilon (float): Exploration rate for epsilon-greedy policy.
        epsilon_min (float): Minimum exploration rate.
        epsilon_decay (float): Decay rate for epsilon after each episode.
        planified_episodes (int): Number of episodes for teacher demonstrations.
        q_threshold (float): Threshold for Q-value difference in stochastic interaction.
        interaction_type (str): Type of interaction between agent and teacher ('uniform' or 'stochastic').
        planner_probability (float): Probability of using teacher's action in uniform interaction mode.
        teacher_expertise (float): Probability that the teacher will attempt to find a path without traps first.
        planifier (PlanifiedAgent): The planning agent used for demonstrations.
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
        planified_episodes=20,
        q_threshold=0.2,
        interaction_type=UNIFORM,
        planner_probability=0.5,
        teacher_expertise=0.8,
        teacher_safety=0,
    ):
        """
        Initialize the QLearningAgentTeacher with given parameters.

        Args:
            state_size (int): Size of the state space.
            n_actions (int): Number of possible actions.
            learning_rate (float, optional): Learning rate for Q-value updates.
            gamma (float, optional): Discount factor for future rewards.
            epsilon (float, optional): Initial exploration rate.
            epsilon_min (float, optional): Minimum exploration rate.
            epsilon_decay (float, optional): Decay rate for the exploration rate.
            planified_episodes (int, optional): Number of episodes for teacher demonstrations.
            q_threshold (float, optional): Threshold for Q-value difference to ask for help.
            interaction_type (str, optional): Type of interaction ('uniform' or 'stochastic').
            planner_probability (float, optional): Probability of using teacher's action in uniform interaction.
            teacher_expertise (float, optional): Probability that the teacher prefers paths without traps.
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
        self.planified_episodes = planified_episodes
        self.planifier = PlanifiedAgent()
        self.q_threshold = q_threshold
        self.interaction_type = interaction_type
        self.planner_probability = planner_probability
        self.teacher_expertise = teacher_expertise
        self.teacher_safety = teacher_safety

    def train(
        self,
        env,
        episodes=10000,
        max_steps=1000,
        render_freq=100,
        render_mode=None,
        render_delay=0.1,
    ):
        """
        Train the agent using a combination of teacher demonstrations and autonomous learning.

        Args:
            env: The environment to train the agent in.
            episodes (int, optional): Total number of training episodes.
            max_steps (int, optional): Maximum steps per episode.
            render_freq (int, optional): Frequency to render the environment.
            render_mode (str, optional): Mode to render ('human' or None).
            render_delay (float, optional): Delay between renders in seconds.
        """
        print(
            f"\nStarting {self.planified_episodes} planified demonstration episodes..."
        )

        history = {
            "rewards": [],
            "epsilon": [],
            "max_q": [],
            "steps": [],
            "planifier_uses": [],
            "collisions": [],
            "trap_steps": [],
            "trap_activations": [],
            "episodes": list(range(episodes)),
        }

        # Demonstration phase
        for episode in range(self.planified_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            planifier_uses = 0
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
                planifier_uses += 1
                next_state, reward, done, info = env.step(action)

                episode_collisions += int(info.get("collision", False))
                episode_trap_steps += info.get("trap_step", 0)
                episode_trap_activations += int(info.get("trap_activation", False))

                self.update(state, action, reward, next_state)

                state = next_state
                total_reward += reward
                steps += 1

            history["rewards"].append(total_reward)
            history["epsilon"].append(self.epsilon)
            history["max_q"].append(np.max(self.q_table))
            history["steps"].append(steps)
            history["planifier_uses"].append(planifier_uses)
            history["collisions"].append(episode_collisions)
            history["trap_steps"].append(episode_trap_steps)
            history["trap_activations"].append(episode_trap_activations)

            print(f"Episode completed - Steps: {steps}, Reward: {total_reward:.2f}")

        # Autonomous learning phase
        print("\nStarting autonomous learning phase...")
        for episode in range(self.planified_episodes, episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            planifier_uses = 0
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
                    planifier_uses += 1

                next_state, reward, done, info = env.step(action)

                episode_collisions += int(info.get("collision", False))
                episode_trap_steps += info.get("trap_step", 0)
                episode_trap_activations += int(info.get("trap_activation", False))

                self.update(state, action, reward, next_state)

                state = next_state
                total_reward += reward
                steps += 1

            history["rewards"].append(total_reward)
            history["epsilon"].append(self.epsilon)
            history["max_q"].append(np.max(self.q_table))
            history["steps"].append(steps)
            history["planifier_uses"].append(planifier_uses)
            history["collisions"].append(episode_collisions)
            history["trap_steps"].append(episode_trap_steps)
            history["trap_activations"].append(episode_trap_activations)

            if episode % 100 == 0:
                print(
                    f"Episode {episode}/{episodes}, Steps: {steps}, "
                    f"Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}, "
                    f"Max Q-value: {np.max(self.q_table):.2f}, "
                    f"Planifier Uses: {planifier_uses}"
                )

            self.decay_epsilon()

        if render_mode:
            env.close()

        self.train_history = history

    def get_random_action(self):
        """
        Get a random action.

        Returns:
            int: A randomly selected action.
        """
        return np.random.randint(0, self.n_actions)

    def get_planified_action(self, env):
        """
        Get an action from the planner based on teacher expertise.

        The teacher will first attempt to find a path without traps with probability
        equal to teacher_expertise. If this fails or if the teacher decides to allow
        traps (based on expertise), it will find a path allowing traps.

        Args:
            env: The environment providing path planning capabilities.

        Returns:
            int: The planned action to take.
        """
        if np.random.random() < self.teacher_expertise:
            planned_actions = env.find_shortest_path(
                allow_traps=False, safety_distance=self.teacher_safety
            )
            if planned_actions is not None:
                return planned_actions[0]
        else:
            planned_actions = env.find_shortest_path(
                allow_traps=True, safety_distance=random.randint(0, self.teacher_safety)
            )
            if planned_actions is None:
                return self.get_random_action()
            return planned_actions[0]

    def get_action(self, state, env=None):
        """
        Select an action based on the current state and interaction type.

        Args:
            state (int): The current state.
            env: The environment instance (required for stochastic interaction).

        Returns:
            tuple: (action, used_planifier) where used_planifier is True if the action came from planifier
        """
        if env is None:
            raise ValueError("Environment must be provided for stochastic interaction")

        if self.interaction_type == self.STOCHASTIC:
            state_q_values = self.q_table[state]
            q_diff = np.max(state_q_values) - np.min(state_q_values)

            if q_diff < self.q_threshold:
                print("Using planifier (stochastic)...")
                return self.get_planified_action(env), True

        elif self.interaction_type == self.UNIFORM:
            if np.random.random() < self.planner_probability:
                print("Using planifier (uniform)...")
                return self.get_planified_action(env), True

        if np.random.random() < self.epsilon:
            return self.get_random_action(), False

        options = np.argwhere(
            self.q_table[state] == np.max(self.q_table[state])
        ).flatten()
        return np.random.choice(options), False
