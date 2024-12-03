import numpy as np
import pygame
import time

class GridEnvironment:
    # Action constants
    MOVE_RIGHT = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_UP = 3
    ACTIONS = [MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT, MOVE_UP]
    
    def __init__(self, size=5, obstacle_prob=0.3, rewards=None):
        """Initialize environment
        
        Args:
            size: Size of the grid (size x size)
            obstacle_prob: Probability of obstacles
            rewards: Dictionary of rewards for different events
        """
        self.size = size
        self.obstacle_prob = obstacle_prob
        
        # Set rewards
        default_rewards = {
            'target': 1,
            'step': 0,
            'collision': 0
        }
        self.rewards = default_rewards if rewards is None else {**default_rewards, **rewards}
        
        # Track used robot starting positions
        self.used_starts = set()
        
        self._generate_environment()
        
        # Pygame setup
        # Fixed window size (800x800 pixels)
        WINDOW_SIZE = 800
        self.cell_size = WINDOW_SIZE // self.size
        self.window_size = (WINDOW_SIZE, WINDOW_SIZE)
        self.screen = None
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)

    def _has_adjacent_obstacle(self, pos):
        """Check if a position has any adjacent obstacles"""
        i, j = pos
        # Check all 8 surrounding positions
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:  # Skip current position
                    continue
                ni, nj = i + di, j + dj
                if (0 <= ni < self.size and 
                    0 <= nj < self.size and 
                    self.grid[ni, nj] == 1):  # 1 represents obstacle
                    return True
        return False

    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_max_distance_from_target(self):
        """Get maximum possible Manhattan distance from target to any border"""
        # Calculate distances to all corners
        corners = [(0, 0), (0, self.size-1), (self.size-1, 0), (self.size-1, self.size-1)]
        max_dist = max(self._manhattan_distance(self.target_pos, corner) for corner in corners)
        return max_dist

    def _generate_environment(self):
        """Generate a new environment ensuring no adjacent obstacles"""
        # Reset grid
        self.grid = np.zeros((self.size, self.size))
        
        # Try to place obstacles while respecting the no-adjacent rule
        for i in range(self.size):
            for j in range(self.size):
                if (i != 0 or j != 0):  # Don't place obstacle at start
                    if np.random.random() < self.obstacle_prob:
                        # Only place obstacle if there are no adjacent obstacles
                        if not self._has_adjacent_obstacle((i, j)):
                            self.grid[i, j] = 1
        
        # Find empty cells for target (excluding start position)
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) 
                      if self.grid[i, j] == 0 and (i != 0 or j != 0)]
        
        if empty_cells:
            # Place target in random empty cell
            target_i, target_j = empty_cells[np.random.randint(len(empty_cells))]
            self.grid[target_i, target_j] = 2
            self.target_pos = [target_i, target_j]
            self.robot_pos = [0, 0]
        else:
            # If no empty cells, create simple environment
            print("Warning: Could not generate environment with valid positions. Creating simple environment.")
            self.grid = np.zeros((self.size, self.size))
            self.grid[self.size-1, self.size-1] = 2
            self.target_pos = [self.size-1, self.size-1]
            self.robot_pos = [0, 0]
    
    def _pos_to_state(self, pos):
        """Convert a 2D position to a scalar state index."""
        return pos[0] * self.size + pos[1]
    
    def reset(self):
        """Reset the environment by moving the robot to a random valid position.
        
        Returns:
            int: Initial state index
        """
        # Clear current robot position
        self.grid[self.robot_pos[0], self.robot_pos[1]] = 0
        
        # Ensure target is in grid
        self.grid[self.target_pos[0], self.target_pos[1]] = 2
        
        # Find all empty positions (excluding target position)
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) 
                    if self.grid[i, j] == 0]
        
        # Get unused positions (excluding target position)
        unused_cells = [pos for pos in empty_cells if tuple(pos) not in self.used_starts]
        
        # If all positions have been used, reset tracking
        if not unused_cells:
            self.used_starts.clear()
            unused_cells = empty_cells
        
        if unused_cells:
            # Choose random position from unused cells
            i, j = unused_cells[np.random.randint(len(unused_cells))]
            self.robot_pos = [i, j]
            self.used_starts.add(tuple(self.robot_pos))
            return self._pos_to_state(tuple(self.robot_pos))
        
        # If no valid positions found, start at (0,0)
        self.robot_pos = [0, 0]
        self.used_starts.add((0, 0))
        return self._pos_to_state(tuple(self.robot_pos))
    
    def step(self, action):
        """Execute action and return new state, reward and done flag"""
        # Store original position for collision detection
        original_pos = self.robot_pos.copy()
        
        # Apply action
        if action == self.MOVE_RIGHT:
            self.robot_pos[1] += 1
        elif action == self.MOVE_DOWN:
            self.robot_pos[0] += 1
        elif action == self.MOVE_LEFT:
            self.robot_pos[1] -= 1
        elif action == self.MOVE_UP:
            self.robot_pos[0] -= 1
        
        # Check if new position is valid
        if (0 <= self.robot_pos[0] < self.size and 
            0 <= self.robot_pos[1] < self.size):
            if self.grid[self.robot_pos[0], self.robot_pos[1]] == 1:
                # Collision with obstacle
                self.robot_pos = original_pos  # Reset position
                return self._pos_to_state(tuple(self.robot_pos)), self.rewards['collision'], True
        else:
            # Collision with wall
            self.robot_pos = original_pos  # Reset position
            return self._pos_to_state(tuple(self.robot_pos)), self.rewards['collision'], True
        
        # Check if target reached
        done = tuple(self.robot_pos) == tuple(self.target_pos)
        reward = self.rewards['target'] if done else self.rewards['step']
        
        return self._pos_to_state(tuple(self.robot_pos)), reward, done
    
    def render(self, mode='human'):
        """Render the environment."""
        if self.screen is None and mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size))
            self.clock = pygame.time.Clock()
            
        if mode == 'human':
            self.screen.fill(self.WHITE)
            
            # Draw grid cells
            for i in range(self.size):
                for j in range(self.size):
                    pygame.draw.rect(self.screen, self.BLACK,
                                  (j * self.cell_size, i * self.cell_size,
                                   self.cell_size, self.cell_size), 1)
                    
                    if self.grid[i, j] == 1:  # Obstacle
                        pygame.draw.rect(self.screen, self.BLACK,
                                      (j * self.cell_size + 2, i * self.cell_size + 2,
                                       self.cell_size - 4, self.cell_size - 4))
                    elif self.grid[i, j] == 2:  # Target
                        pygame.draw.circle(self.screen, self.GREEN,
                                        (j * self.cell_size + self.cell_size // 2,
                                         i * self.cell_size + self.cell_size // 2),
                                        self.cell_size // 3)
            
            # Draw robot
            robot_center = (self.robot_pos[1] * self.cell_size + self.cell_size // 2,
                          self.robot_pos[0] * self.cell_size + self.cell_size // 2)
            
            # Draw robot as a blue circle
            pygame.draw.circle(self.screen, self.BLUE, robot_center, self.cell_size // 3)
            
            pygame.display.flip()
            self.clock.tick(30)
            
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
