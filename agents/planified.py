import time
import matplotlib.pyplot as plt


class PlanifiedAgent:
    """Agent that follows a planned path in the environment.

    Attributes:
        current_plan (list): The current sequence of actions to reach the target.
        steps_taken (int): The number of steps taken in the current episode.
        total_reward (float): The total reward accumulated in the current episode.
    """

    def __init__(self):
        """Initialize the PlanifiedAgent."""
        self.current_plan = None
        self.steps_taken = 0
        self.total_reward = 0

    def get_action(self, env):
        """Get the next action from the planned path.

        Args:
            env: The environment instance.

        Returns:
            int: The next action to take.
        """
        if self.current_plan is None or not self.current_plan:

            self.current_plan = env.find_shortest_path(allow_traps=False, safety_distance=0)

            if self.current_plan is None:
                self.current_plan = env.find_shortest_path(allow_traps=True)

            if self.current_plan is None:

                print("Warning: No path found even with traps allowed!")
                return env.ACTIONS[0]

        return self.current_plan.pop(0)

    def run_episode(self, env, render=False, render_delay=0.1):
        """Run a single episode in the environment.

        Args:
            env: The environment to run on.
            render (bool, optional): Whether to render the environment. Defaults to False.
            render_delay (float, optional): Delay between renders in seconds. Defaults to 0.1.

        Returns:
            tuple: Total reward, steps taken, and success flag.
        """
        state = env.reset()
        self.current_plan = None
        self.steps_taken = 0
        self.total_reward = 0
        done = False

        while not done:
            if render:
                env.render(mode="human")
                time.sleep(render_delay)

            action = self.get_action(env)
            _, reward, done, info = env.step(action)

            self.total_reward += reward
            self.steps_taken += 1

        if render:
            env.render(mode="human")
            time.sleep(render_delay)

        return self.total_reward, self.steps_taken, self.total_reward > 0

    def run_episodes(self, env, num_episodes=100, render_freq=10):
        """Run multiple episodes and track performance.

        Args:
            env: The environment to run on.
            num_episodes (int, optional): Number of episodes to run. Defaults to 100.
            render_freq (int, optional): Frequency of rendering the environment. Defaults to 10.

        Returns:
            dict: A history of rewards, steps, and success rates.
        """
        history = {
            "rewards": [],
            "steps": [],
            "success_rate": [],
            "episodes": list(range(num_episodes)),
        }

        successes = 0

        for episode in range(num_episodes):
            render = episode % render_freq == 0
            reward, steps, success = self.run_episode(env, render=render)

            if success:
                successes += 1

            history["rewards"].append(reward)
            history["steps"].append(steps)
            history["success_rate"].append(successes / (episode + 1))

            if episode % 10 == 0:
                print(
                    f"Episode {episode}/{num_episodes}, Steps: {steps}, "
                    f"Reward: {reward:.2f}, Success Rate: {(successes / (episode + 1)):.2%}"
                )

        return history

    def plot_history(self, history, save_path=None):
        """Plot the agent's performance history.

        Args:
            history (dict): The history data to plot.
            save_path (str, optional): Path to save the plot image. Defaults to None.
        """
        metrics = [k for k in history.keys() if k != "episodes"]
        episodes = history["episodes"]

        plt.figure(figsize=(12, 4 * len(metrics)))

        plot_idx = 1

        for metric in metrics:
            plt.subplot(len(metrics), 1, plot_idx)
            plt.plot(episodes, history[metric], label=f"{metric}")
            plt.title(f'{metric.replace("_", " ").title()}')
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
