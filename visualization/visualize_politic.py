import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import json

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add the project root to the Python path
sys.path.append(os.path.join(PROJECT_ROOT, 'Safe-Bot-Navigator'))

from environment.environment import GridEnvironment

def load_pickle(filepath):
    """Load a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_json(filepath):
    """Load a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_policy_visualization(agent, env, run_name, save_path=None):
    """Create a visualization of the agent's policy on the grid."""
    grid = env.grid
    size = env.size
    
    # Create figure and axis with padding
    fig, ax = plt.subplots(figsize=(14, 14))  # Increased figure size for padding
    
    # Add padding around the grid
    padding = 1  # One cell padding
    
    def get_arrow_vertical_offset(num_arrows, arrow_index):
        """Get vertical offset for arrows arranged from top to bottom"""
        if num_arrows == 1:
            return 0  # Center
        
        # Same positions for 2-4 arrows
        positions = [-0.2, -0.066, 0.066, 0.2]  # Fixed positions for up to 4 arrows
        return positions[arrow_index]

    def get_arrow_horizontal_offset(num_arrows, arrow_index):
        """Get horizontal offset for arrows arranged from left to right"""
        if num_arrows == 1:
            return 0  # Center
        
        # Same positions for 2-4 arrows
        positions = [-0.2, -0.066, 0.066, 0.2]  # Fixed positions for up to 4 arrows
        return positions[arrow_index]
    
    # Draw grid cells
    for i in range(size):
        for j in range(size):
            cell_type = grid[i, j]
            cell_color = 'white'
            if cell_type == GridEnvironment.OBSTACLE:
                cell_color = 'gray'
            elif cell_type == GridEnvironment.TRAP:
                cell_color = 'red'
            elif cell_type == GridEnvironment.TARGET:
                cell_color = 'green'
            
            # Add cell with padding
            ax.add_patch(plt.Rectangle((j + padding, size-1-i + padding), 1, 1, 
                                     facecolor=cell_color, edgecolor='black'))
            
            # Get optimal action(s) for this state
            state = i * size + j
            q_values = agent.q_table[state]
            
            # Show arrows for all states except obstacles and target
            if cell_type != GridEnvironment.OBSTACLE and cell_type != GridEnvironment.TARGET:
                # Find all actions with maximum Q-value
                max_q = np.max(q_values)
                best_actions = np.where(q_values == max_q)[0]
                
                # Separate vertical and horizontal actions and sort them
                vertical_actions = sorted([a for a in best_actions if a in [GridEnvironment.MOVE_UP, GridEnvironment.MOVE_DOWN]])
                horizontal_actions = sorted([a for a in best_actions if a in [GridEnvironment.MOVE_LEFT, GridEnvironment.MOVE_RIGHT]])
                
                # Calculate cell center
                center_x = j + padding + 0.5
                center_y = size-1-i + padding + 0.5
                
                # Draw vertical arrows
                for idx, action in enumerate(vertical_actions):
                    # All arrows start from center
                    arrow_y = center_y
                    dx, dy = 0, 0
                    
                    if action == GridEnvironment.MOVE_UP:
                        dy = 0.25  # Arrow length
                    elif action == GridEnvironment.MOVE_DOWN:
                        dy = -0.25  # Arrow length
                    
                    arrow_color = 'white' if cell_type == GridEnvironment.TRAP else 'blue'
                    ax.arrow(center_x,  # Start from center x
                            arrow_y,    # Start from center y
                            dx, dy,
                            head_width=0.1, head_length=0.1, 
                            fc=arrow_color, ec=arrow_color,
                            length_includes_head=True)
                
                # Draw horizontal arrows
                for idx, action in enumerate(horizontal_actions):
                    # All arrows start from center
                    arrow_x = center_x
                    dx, dy = 0, 0
                    
                    if action == GridEnvironment.MOVE_LEFT:
                        dx = -0.25  # Arrow length
                    elif action == GridEnvironment.MOVE_RIGHT:
                        dx = 0.25   # Arrow length
                    
                    arrow_color = 'white' if cell_type == GridEnvironment.TRAP else 'blue'
                    ax.arrow(arrow_x,    # Start from center x
                            center_y,    # Start from center y
                            dx, dy,
                            head_width=0.1, head_length=0.1, 
                            fc=arrow_color, ec=arrow_color,
                            length_includes_head=True)
    
    if save_path:
        # Ensure the directory exists
        os.makedirs(save_path, exist_ok=True)

        # Save the first image with only arrows
        policy_image_path = os.path.join(save_path, f"policy.png")
        plt.savefig(policy_image_path, bbox_inches='tight', dpi=300, format='png')

        # Add text annotations for each action and Q value
        for i in range(size):
            for j in range(size):
                cell_type = grid[i, j]
                if cell_type != GridEnvironment.OBSTACLE and cell_type != GridEnvironment.TARGET:
                    state = i * size + j
                    q_values = agent.q_table[state]
                    for action, q_value in enumerate(q_values):
                        action_name = ''
                        if action == GridEnvironment.MOVE_UP:
                            action_name = 'Up'
                        elif action == GridEnvironment.MOVE_DOWN:
                            action_name = 'Down'
                        elif action == GridEnvironment.MOVE_LEFT:
                            action_name = 'Left'
                        elif action == GridEnvironment.MOVE_RIGHT:
                            action_name = 'Right'
                        
                        # Position the text slightly above the arrows
                        ax.text(j + padding + 0.5, size-1-i + padding + 0.3 + 0.1 * action, f'{action_name}: {q_value:.2f}',
                                fontsize=8, ha='center', va='center', color='black')

        # Save the second image with action names and Q values
        q_value_image_path = os.path.join(save_path, f"policy_with_q.png")
        plt.savefig(q_value_image_path, bbox_inches='tight', dpi=300, format='png')
        
    # Set grid properties with padding
    ax.set_xlim(-0.5 + padding, size + 0.5 + padding)
    ax.set_ylim(-0.5 + padding, size + 0.5 + padding)
    ax.set_xticks(np.arange(padding, size + padding, 1))
    ax.set_yticks(np.arange(padding, size + padding, 1))
    ax.grid(True)
    
    # Remove axis labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', label='Empty'),
        plt.Rectangle((0, 0), 1, 1, facecolor='gray', edgecolor='black', label='Obstacle'),
        plt.Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='black', label='Trap'),
        plt.Rectangle((0, 0), 1, 1, facecolor='green', edgecolor='black', label='Target'),
        plt.Line2D([0], [0], color='blue', marker='>', linestyle='-', label='Optimal Action')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add title with run configuration
    plt.title(f"Agent's Learned Policy\n{run_name}")
    plt.tight_layout()
    
    if save_path:
        plt.close(fig)
    else:
        plt.show()

def process_run_directory(run_dir):
    """Process a single run directory and create visualization."""
    # Get run name from directory
    run_name = os.path.basename(run_dir)
    
    # Load agent and environment
    agent = load_pickle(os.path.join(run_dir, 'agent.pkl'))
    env = load_pickle(os.path.join(run_dir, 'env.pkl'))
    
    # Load configurations
    agent_config = load_json(os.path.join(run_dir, 'agent_config.json'))
    env_config = load_json(os.path.join(run_dir, 'env_config.json'))
    train_config = load_json(os.path.join(run_dir, 'train_config.json'))
    
    # Create visualization directory if it doesn't exist
    vis_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create and save visualization
    save_path = os.path.join(vis_dir, run_name)
    create_policy_visualization(agent, env, run_name, save_path)
    
    return save_path

def main():
    # Get paths relative to the project root
    trained_agents_dir = os.path.join(PROJECT_ROOT, 'trained_agents')
    
    # Process each run directory
    for run_name in os.listdir(trained_agents_dir):
        run_dir = os.path.join(trained_agents_dir, run_name)
        if os.path.isdir(run_dir):
            try:
                print(f"Processing {run_name}...")
                save_path = process_run_directory(run_dir)
                print(f"Saved visualization to {save_path}")
            except Exception as e:
                print(f"Error processing {run_name}: {str(e)}")

if __name__ == "__main__":
    main()
