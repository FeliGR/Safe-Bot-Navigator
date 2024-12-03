import numpy as np
import gymnasium as gym

def policy_evaluation(policy, env, discount_factor=0.9, theta=1e-8):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env.
        discount_factor: gamma discount factor.
        theta: We stop evaluation once our value function change is less than theta for all states.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.observation_space.n):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states
                for prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state] * (not done))
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return V

def policy_improvement(env, V, discount_factor=0.9):
    """
    Given the value function from policy evaluation step, we update the policy by selecting the best action in each state.
    
    Args:
        env: The OpenAI environment.
        V: Value function array of length env.nS.
        discount_factor: gamma discount factor.
    
    Returns:
        A matrix of shape [S, A] where each state s contains a valid probability distribution over actions.
    """
    policy = np.zeros([env.observation_space.n, env.action_space.n])
    
    for s in range(env.observation_space.n):
        # For each state, find the Q values for all possible actions
        A = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[s][a]:
                A[a] += prob * (reward + discount_factor * V[next_state] * (not done))
        
        # Select the action with highest Q value
        best_action = np.argmax(A)
        # Update the policy to select the best action with probability 1
        policy[s] = np.eye(env.action_space.n)[best_action]
    
    return policy

def policy_iteration(env, discount_factor=0.9, theta=1e-8):
    """
    Policy Iteration Algorithm.
    
    Args:
        env: The OpenAI environment.
        discount_factor: gamma discount factor.
        theta: Stopping criteria for value function update.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    # Start with a random policy
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    
    while True:
        # Evaluate the current policy
        V = policy_evaluation(policy, env, discount_factor, theta)
        
        # Get the greedy policy for the current value function
        new_policy = policy_improvement(env, V, discount_factor)
        
        # If the policy is stable (no change), we've found the optimal policy
        if np.all(policy == new_policy):
            break
            
        policy = new_policy
    
    return policy, V

def run_policy_iteration():
    """
    Run the Policy Iteration algorithm on the FrozenLake environment.
    """
    # Create the environment
    env = gym.make('FrozenLake-v1')
    
    # Run policy iteration
    optimal_policy, optimal_value = policy_iteration(env)
    
    print("Optimal Policy:")
    print(optimal_policy)
    print("\nOptimal Value Function:")
    print(optimal_value.reshape(4, 4))
    
    return optimal_policy, optimal_value

if __name__ == "__main__":
    run_policy_iteration()