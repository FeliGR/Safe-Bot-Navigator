import numpy as np
import time
from agents.basic_qlearning import BasicQLearningAgent
from agents.planified import PlanifiedAgent


class QLearningAgentTeacher(BasicQLearningAgent):
    """
    Clouse Ask for Help Q-Learning Agent with configurable interaction types and teacher expertise.

    This agent implements a Q-learning algorithm that can learn from both autonomous exploration
    and a teaching agent (planner). The interaction between the agent and the teacher can be
    configured in different ways.

    Parameters
    ----------
    state_size : int
        Size of the state space
    n_actions : int
        Number of possible actions
    learning_rate : float, optional (default=0.1)
        Learning rate for Q-value updates (α in Q-learning equation)
    gamma : float, optional (default=0.95)
        Discount factor for future rewards (γ in Q-learning equation)
    epsilon : float, optional (default=1.0)
        Initial exploration rate for epsilon-greedy policy
    epsilon_min : float, optional (default=0.01)
        Minimum exploration rate
    epsilon_decay : float, optional (default=0.001)
        Decay rate for epsilon after each episode
    planified_episodes : int, optional (default=20)
        Number of episodes where the agent learns from teacher demonstrations
    q_threshold : float, optional (default=0.2)
        Threshold for Q-value difference to decide when to ask for help in stochastic mode
    interaction_type : str, optional (default='uniform')
        Type of interaction between agent and teacher. Options:
        - 'uniform': Uses teacher's action with fixed probability
        - 'stochastic': Uses teacher based on Q-value uncertainty
    planner_probability : float, optional (default=0.5)
        Probability of using teacher's action in uniform interaction mode
    teacher_expertise : float, optional (default=0.8)
        Probability that the teacher will attempt to find a path without traps first
    """

    # Interaction type constants
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
        self.q_threshold = q_threshold
        self.interaction_type = interaction_type
        self.planner_probability = planner_probability
        self.teacher_expertise = teacher_expertise

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
        print(
            f"\nStarting {self.planified_episodes} planified demonstration episodes..."
        )

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

                action = self.get_action(state, env)
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

    def get_random_action(self):
        return np.random.randint(0, self.n_actions)

    def get_planified_action(self, env):
        """
        Get action from the planifier based on teacher expertise.

        The teacher will first attempt to find a path without traps with probability
        equal to teacher_expertise. If this fails or if the teacher decides to allow
        traps (based on expertise), it will find a path allowing traps.

        Parameters
        ----------
        env : Environment
            The environment instance that provides path planning capabilities

        Returns
        -------
        int
            The planned action to take
        """
        # First attempt: try path without traps based on teacher expertise
        if np.random.random() < self.teacher_expertise:
            planned_actions = env.find_shortest_path(allow_traps=False)
            if planned_actions is not None:
                return planned_actions[0]

        # Second attempt: allow traps
        planned_actions = env.find_shortest_path(allow_traps=True)
        if planned_actions is None:
            return (
                self.get_random_action()
            )  # Fallback to random action if no path found
        return planned_actions[0]  # Return first action from the planned path

    def get_action(self, state, env=None):
        """
        Get action based on current state and interaction type.
        - For stochastic: Uses Q-values difference threshold
        - For uniform: Uses fixed probability
        """
        if env is None:
            raise ValueError("Environment must be provided for stochastic interaction")

        if self.interaction_type == self.STOCHASTIC:
            state_q_values = self.q_table[state]
            q_diff = np.max(state_q_values) - np.min(state_q_values)

            if q_diff < self.q_threshold:
                print("Using planifier (stochastic)...")
                return self.get_planified_action(env)

        elif self.interaction_type == self.UNIFORM:
            if np.random.random() < self.planner_probability:
                print("Using planifier (uniform)...")
                return self.get_planified_action(env)

        if np.random.random() < self.epsilon:
            return self.get_random_action()

        options = np.argwhere(
            self.q_table[state] == np.max(self.q_table[state])
        ).flatten()
        return np.random.choice(options)
