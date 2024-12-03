import gymnasium as gym
import numpy as np
from collections import defaultdict
import pickle

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=1.0):
        self.action_space = action_space
        self.lr = learning_rate          # α (alpha) - learning rate
        self.gamma = discount_factor     # γ (gamma) - discount factor
        self.epsilon = epsilon           # ε (epsilon) - exploration rate
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        
    def discretize_state(self, state):
        """Convert continuous state to discrete for Q-table"""
        # Discretize each component of the state
        pos_x = np.digitize(state[0], bins=np.linspace(-1.5, 1.5, 10))
        pos_y = np.digitize(state[1], bins=np.linspace(-1.5, 1.5, 10))
        vel_x = np.digitize(state[2], bins=np.linspace(-2, 2, 10))
        vel_y = np.digitize(state[3], bins=np.linspace(-2, 2, 10))
        angle = np.digitize(state[4], bins=np.linspace(-3.14, 3.14, 10))
        ang_vel = np.digitize(state[5], bins=np.linspace(-5, 5, 10))
        # Legs contact points are already discrete (boolean)
        return tuple([pos_x, pos_y, vel_x, vel_y, angle, ang_vel, state[6], state[7]])
    
    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return self.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
    
    def update(self, state, action, reward, next_state):
        """Q-learning update rule"""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error
    
    def decay_epsilon(self, episode, total_episodes):
        """Decay exploration rate"""
        self.epsilon = max(0.01, 1.0 - episode / (total_episodes * 0.8))

def train_agent(env, agent, episodes=1000):
    scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = agent.discretize_state(state)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.discretize_state(next_state)
            total_reward += reward
            
            # Update Q-table
            agent.update(state, action, reward, next_state)
            state = next_state
            done = terminated or truncated
        
        agent.decay_epsilon(episode, episodes)
        scores.append(total_reward)
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    return scores

# Initialize environment and agent
env = gym.make("LunarLander-v2", render_mode="human")
agent = QLearningAgent(env.action_space)

# Train the agent
scores = train_agent(env, agent)

# Save the trained Q-table
with open('q_table.pkl', 'wb') as f:
    pickle.dump(dict(agent.q_table), f)

env.close()