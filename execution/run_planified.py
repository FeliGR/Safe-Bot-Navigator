
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.environment import GridEnvironment
from agents.planified import PlanifiedAgent
import time
def main():
    """
    Main function to run a continuous simulation of a planified agent in a grid environment.

    Creates a grid environment and a planified agent, then runs episodes continuously until 
    interrupted by the user (Ctrl+C). Each episode is visualized with delays between steps
    and displays summary statistics after completion.

    Environment Parameters:
        - Grid size: 8x8
        - Obstacle probability: 0.2
        - Trap probability: 0.1
        - Trap danger: 0.3

    The simulation includes:
        - Visual rendering of each episode
        - Step-by-step execution with delays
        - Episode results display (success/failure)
        - Performance metrics (steps taken, total reward)
        - 2-second pause between episodes
    """
    env = GridEnvironment(
        size=15,
        obstacle_prob=0,
        trap_prob=0.1,
        trap_danger=0.3,
    )

    agent = PlanifiedAgent()

    print("\nRunning episodes with visualization...")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            reward, steps, success, info = agent.run_episode(
                env,
                render=True,
                render_delay=0.5,
            )

            result = "SUCCESS" if success else "FAILED"
            print(f"Episode finished - {result}")
            print(f"Steps taken: {steps}")
            print(f"Total reward: {reward}\n")

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        env.close()


if __name__ == "__main__":
    main()
