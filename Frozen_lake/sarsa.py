import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
MIN_EPSILON = 0
EPISODES = 50000
EPSILON = 1.0
EPSILON_DECAY = (EPSILON - MIN_EPSILON) / EPISODES

# Initialize the environment
env = gym.make("FrozenLake-v1")

# Initialize Q-table with random values
Q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Lists to store metrics
rewards_per_episode = []
steps_per_episode = []
max_q_values = []  # Track Q* (maximum Q-value) over episodes

def print_q_table(q_table):
    actions = ['←', '↓', '→', '↑']  # LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3
    print("\nQ-table Grid (showing best action and its value for each state):")
    print("-" * 50)
    
    for row in range(4):  # FrozenLake is 4x4
        for col in range(4):
            state = row * 4 + col
            best_action = np.argmax(q_table[state])
            best_value = np.max(q_table[state])
            print(f"|{actions[best_action]}:{best_value:.2f}", end=" ")
        print("|")
        print("-" * 50)
        print("\nDetailed Q-values for each state:")
    print("State | Left(←) | Down(↓) | Right(→) | Up(↑)")
    print("-" * 50)
    for state in range(16):
        print(f"{state:5d} |", end=" ")
        for action in range(4):
            print(f"{q_table[state, action]:7.2f} |", end=" ")
        print()  # New line after each state
    print("-" * 50)

def run_greedy_policy(env, Q_table, num_episodes=100):
    total_rewards = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Always choose the best action (greedy policy)
            action = np.argmax(Q_table[state])
            print(f"State: {state}, Action: {action}, Q-Table: {Q_table[state]}, Action: {Q_table[state][action]}")

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            time.sleep(0.5)
        
        total_rewards += episode_reward
    
    return total_rewards / num_episodes

# Training loop
for episode in range(EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    # Select initial action using epsilon-greedy
    if np.random.random() < EPSILON:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q_table[state])
    
    while not done:
        # Take action and observe next state
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Select next action using epsilon-greedy (SARSA is on-policy)
        if np.random.random() < EPSILON:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(Q_table[next_state])
        
        # SARSA update rule
        old_q = Q_table[state, action]
        next_q = Q_table[next_state, next_action]  # Use actual next action instead of max
        new_q = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_q - old_q)
        Q_table[state, action] = new_q
        
        state = next_state
        action = next_action  # Update action for next iteration
        total_reward += reward
        steps += 1
    
    # Track metrics
    EPSILON = max(MIN_EPSILON, EPSILON - EPSILON_DECAY)
    rewards_per_episode.append(total_reward)
    steps_per_episode.append(steps)
    max_q_values.append(np.max(Q_table))  # Track Q*
    
    if episode % 100 == 0:
        avg_reward = np.mean(rewards_per_episode[-100:])
        print(f"Episode: {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {EPSILON:.2f}")

# Save the Q-table
with open('q_table.pkl', 'wb') as f:
    pickle.dump(Q_table, f)

# Plot training results
plt.figure(figsize=(15, 5))

# Plot rewards
plt.subplot(1, 3, 1)
plt.plot(rewards_per_episode)
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

# Plot steps
plt.subplot(1, 3, 2)
plt.plot(steps_per_episode)
plt.title('Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Steps')

# Plot Q*
plt.subplot(1, 3, 3)
plt.plot(max_q_values)
plt.title('Maximum Q-value (Q*) per Episode')
plt.xlabel('Episode')
plt.ylabel('Q*')

plt.tight_layout()
plt.show()




    
    

# Print final Q-table
print("\nFinal Q-table Analysis:")
print_q_table(Q_table)


# Evaluate the learned policy
print("\nEvaluating learned policy...")
env_test = gym.make("FrozenLake-v1", render_mode="human")
average_reward = run_greedy_policy(env_test, Q_table)
print(f"Average reward over 100 episodes with greedy policy: {average_reward:.2f}")

env.close()
env_test.close()