from environment import GridEnvironment
from agents.planified import PlanifiedAgent
import time

def main():
    # Create environment (you can adjust these parameters)
    env = GridEnvironment(
        size=8,  # 8x8 grid
        obstacle_prob=0.2,  # 20% chance of obstacles
        trap_prob=0.1,  # 10% chance of traps
        trap_danger=0.3  # 30% chance of trap ending episode
    )
    
    # Create agent
    agent = PlanifiedAgent()
    
    # Run episodes with visualization
    print("\nRunning episodes with visualization...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:  # Run indefinitely until Ctrl+C
            reward, steps, success = agent.run_episode(
                env,
                render=True,  # Enable visualization
                render_delay=0.5  # Half second delay between steps
            )
            
            # Print episode results
            result = "SUCCESS" if success else "FAILED"
            print(f"Episode finished - {result}")
            print(f"Steps taken: {steps}")
            print(f"Total reward: {reward}\n")
            
            # Small pause between episodes
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        env.close()

if __name__ == "__main__":
    main()
