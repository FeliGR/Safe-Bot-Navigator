import gymnasium as gym
from pynput import keyboard
import time

# Initialize the environment
env = gym.make("FrozenLake-v1", render_mode="human")

# Reset the environment to generate the first state
observation, info = env.reset(seed=42)

# Global variables for key state
current_key = None
should_quit = False

# Dictionary to map keys to actions
# 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP
key_to_action = {
    'a': 0,  # left
    's': 1,  # down
    'd': 2,  # right
    'w': 3   # up
}

def on_press(key):
    global current_key, should_quit
    try:
        key_char = key.char.lower()
        if key_char in key_to_action:
            current_key = key_char
        elif key_char == 'q':
            should_quit = True
    except AttributeError:
        pass

def on_release(key):
    global current_key
    try:
        if key.char.lower() == current_key:
            current_key = None
    except AttributeError:
        pass

# Set up the keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

total_reward = 0
env.render()

try:
    while not should_quit:
        # Wait for a valid key press
        while current_key is None and not should_quit:
            time.sleep(0.1)
        
        if should_quit:
            break

        # Get action from current key
        action = key_to_action.get(current_key, None)
        if action is not None:
            # Perform the action and transition to the next state
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            current_key = None  # Reset the current key
            
            env.render()
            
            # If episode ends, print score and reset
            if terminated or truncated:
                print(f"Episode finished! Total reward: {total_reward}")
                observation, info = env.reset()
                total_reward = 0

finally:
    # Clean up
    listener.stop()
    env.close()