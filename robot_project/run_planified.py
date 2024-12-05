from environment import GridEnvironment
from agents.planified import PlanifiedAgent
import time

def main():

    env = GridEnvironment(
        size=8,
        obstacle_prob=0.2,
        trap_prob=0.1,
        trap_danger=0.3,
    )

    agent = PlanifiedAgent()

    print("\nRunning episodes with visualization...")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            reward, steps, success = agent.run_episode(
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
