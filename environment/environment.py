import numpy as np
import pygame
import time


class GridEnvironment:

    MOVE_RIGHT = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_UP = 3
    ACTIONS = [MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT, MOVE_UP]

    EMPTY = 0
    OBSTACLE = 1
    TARGET = 2
    TRAP = 3

    def __init__(
        self, size=5, obstacle_prob=0.2, trap_prob=0.1, trap_danger=0.3, rewards=None
    ):
        """Initialize environment

        Args:
            size: Size of the grid (size x size)
            obstacle_prob: Probability of obstacles
            trap_prob: Probability of placing a trap in empty cells
            trap_danger: Probability of trap ending the episode
            rewards: Dictionary of rewards for different events
        """
        self.size = size
        self.obstacle_prob = obstacle_prob
        self.trap_prob = trap_prob
        self.trap_danger = trap_danger

        default_rewards = {"target": 1, "step": 0, "collision": 0, "trap": 0}
        self.rewards = (
            default_rewards if rewards is None else {**default_rewards, **rewards}
        )

        self.used_starts = set()

        self._generate_environment()

        WINDOW_SIZE = 800
        self.cell_size = WINDOW_SIZE // self.size
        self.window_size = (WINDOW_SIZE, WINDOW_SIZE)
        self.screen = None

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.LIGHT_RED = (255, 200, 200)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)

        self.show_text = False
        self.text_message = ""
        self.text_start_time = 0
        self.text_color = self.BLACK

    def _has_adjacent_obstacle(self, pos):
        """Check if a position has any adjacent obstacles"""
        i, j = pos

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if (
                    0 <= ni < self.size
                    and 0 <= nj < self.size
                    and self.grid[ni, nj] == self.OBSTACLE
                ):
                    return True
        return False

    def _has_nearby_trap(self, pos, range_size=2):
        """Check if a position has any traps within the specified range"""
        i, j = pos
        for di in range(-range_size + 1, range_size):
            for dj in range(-range_size + 1, range_size):
                ni, nj = i + di, j + dj
                if (
                    0 <= ni < self.size
                    and 0 <= nj < self.size
                    and self.grid[ni, nj] == self.TRAP
                ):
                    return True
        return False

    def _generate_environment(self):
        """Generate a new environment ensuring no adjacent obstacles and properly spaced traps"""

        self.grid = np.zeros((self.size, self.size))

        for i in range(self.size):
            for j in range(self.size):
                if i != 0 or j != 0:
                    if np.random.random() < self.obstacle_prob:

                        if not self._has_adjacent_obstacle((i, j)):
                            self.grid[i, j] = self.OBSTACLE

        empty_cells = [
            (i, j)
            for i in range(self.size)
            for j in range(self.size)
            if self.grid[i, j] == self.EMPTY and (i != 0 or j != 0)
        ]

        if empty_cells:

            target_i, target_j = empty_cells[np.random.randint(len(empty_cells))]
            self.grid[target_i, target_j] = self.TARGET
            self.target_pos = [target_i, target_j]
            self.robot_pos = [0, 0]

            empty_cells = [
                (i, j)
                for i in range(self.size)
                for j in range(self.size)
                if self.grid[i, j] == self.EMPTY and (i != 0 or j != 0)
            ]

            for i, j in empty_cells:
                if np.random.random() < self.trap_prob:
                    if not self._has_nearby_trap((i, j)):
                        self.grid[i, j] = self.TRAP
        else:

            print(
                "Warning: Could not generate environment with valid positions. Creating simple environment."
            )
            self.grid = np.zeros((self.size, self.size))
            self.grid[self.size - 1, self.size - 1] = self.TARGET
            self.target_pos = [self.size - 1, self.size - 1]
            self.robot_pos = [0, 0]

    def _pos_to_state(self, pos):
        """Convert a 2D position to a scalar state index."""
        return pos[0] * self.size + pos[1]

    def reset(self):
        """Reset the environment by moving the robot to a random valid position.

        Returns:
            int: Initial state index
        """

        current_grid = self.grid.copy()

        self.grid[self.robot_pos[0], self.robot_pos[1]] = self.EMPTY

        self.grid[self.target_pos[0], self.target_pos[1]] = self.TARGET

        for i in range(self.size):
            for j in range(self.size):
                if current_grid[i, j] == self.TRAP:
                    self.grid[i, j] = self.TRAP

        empty_cells = [
            (i, j)
            for i in range(self.size)
            for j in range(self.size)
            if self.grid[i, j] == self.EMPTY
        ]

        unused_cells = [
            pos for pos in empty_cells if tuple(pos) not in self.used_starts
        ]

        if not unused_cells:
            self.used_starts.clear()
            unused_cells = empty_cells

        if unused_cells:

            i, j = unused_cells[np.random.randint(len(unused_cells))]
            self.robot_pos = [i, j]
            self.used_starts.add(tuple(self.robot_pos))
            return self._pos_to_state(tuple(self.robot_pos))

        self.robot_pos = [0, 0]
        self.used_starts.add((0, 0))
        return self._pos_to_state(tuple(self.robot_pos))

    def _wait_for_message(self):
        if self.screen is not None:
            start_time = time.time()
            while time.time() - start_time < 1.0:
                self.render()
                time.sleep(0.01)

    def step(self, action):
        """Execute action and return new state, reward and done flag"""

        original_pos = self.robot_pos.copy()

        if action == self.MOVE_RIGHT:
            self.robot_pos[1] += 1
        elif action == self.MOVE_DOWN:
            self.robot_pos[0] += 1
        elif action == self.MOVE_LEFT:
            self.robot_pos[1] -= 1
        elif action == self.MOVE_UP:
            self.robot_pos[0] -= 1

        info = {"collision": False, "trap_step": 0, "trap_activation": False}

        if 0 <= self.robot_pos[0] < self.size and 0 <= self.robot_pos[1] < self.size:
            cell_type = self.grid[self.robot_pos[0], self.robot_pos[1]]

            if cell_type == self.OBSTACLE:
                self.robot_pos = original_pos
                self.show_text = True
                self.text_message = "FAIL - Collision!"
                self.text_start_time = time.time()
                self.text_color = self.RED
                self._wait_for_message()
                info["collision"] = True
                return (
                    self._pos_to_state(tuple(self.robot_pos)),
                    self.rewards["collision"],
                    True,
                    info,
                )
            elif cell_type == self.TRAP:
                info["trap_step"] += 1
                if np.random.random() < self.trap_danger:
                    self.show_text = True
                    self.text_message = "FAIL - Trap!"
                    self.text_start_time = time.time()
                    self.text_color = self.RED
                    self._wait_for_message()
                    info["trap_activation"] = True
                    return (
                        self._pos_to_state(tuple(self.robot_pos)),
                        self.rewards["trap"],
                        True,
                        info,
                    )
        else:
            self.robot_pos = original_pos
            self.show_text = True
            self.text_message = "FAIL - Wall!"
            self.text_start_time = time.time()
            self.text_color = self.RED
            self._wait_for_message()
            info["collision"] = True
            return (
                self._pos_to_state(tuple(self.robot_pos)),
                self.rewards["collision"],
                True,
                info,
            )

        done = tuple(self.robot_pos) == tuple(self.target_pos)
        reward = self.rewards["target"] if done else self.rewards["step"]

        if done:
            self.show_text = True
            self.text_message = "SUCCESS!"
            self.text_start_time = time.time()
            self.text_color = self.GREEN
            self._wait_for_message()

        return self._pos_to_state(tuple(self.robot_pos)), reward, done, info

    def find_shortest_path(self, allow_traps=False, safety_distance=0):
        """Find the shortest path from current position to target.

        Args:
            allow_traps (bool): If True, allows passing through traps but tries to avoid them
                              If False, treats traps as obstacles
            safety_distance (int): Minimum desired distance from obstacles and traps.
                                 0 means no safety distance is enforced.

        Returns:
            list: Sequence of actions to reach target, or None if no path exists
        """

        def manhattan_distance(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        def get_min_distance_to_hazards(pos):
            min_dist = float("inf")
            hazard_count = 0
            for i in range(self.size):
                for j in range(self.size):
                    cell_type = self.grid[i, j]
                    if cell_type == self.OBSTACLE or (
                        not allow_traps and cell_type == self.TRAP
                    ):
                        dist = manhattan_distance(pos, (i, j))
                        if dist <= safety_distance:
                            hazard_count += 1
                        min_dist = min(min_dist, dist)
            return min_dist, hazard_count

        def get_neighbors(pos):
            neighbors = []
            for action in self.ACTIONS:
                new_pos = list(pos)
                if action == self.MOVE_RIGHT:
                    new_pos[1] += 1
                elif action == self.MOVE_DOWN:
                    new_pos[0] += 1
                elif action == self.MOVE_LEFT:
                    new_pos[1] -= 1
                elif action == self.MOVE_UP:
                    new_pos[0] -= 1

                if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
                    cell_type = self.grid[new_pos[0], new_pos[1]]

                    if cell_type == self.OBSTACLE or (
                        not allow_traps and cell_type == self.TRAP
                    ):
                        continue
                    neighbors.append((tuple(new_pos), action))
            return neighbors

        def get_cost(current, next_pos):
            base_cost = 1
            if allow_traps and self.grid[next_pos[0], next_pos[1]] == self.TRAP:
                base_cost = 0

            if safety_distance > 0:
                min_dist, hazard_count = get_min_distance_to_hazards(next_pos)
                if min_dist < safety_distance:
                    # Add penalty for being closer than safety_distance
                    safety_penalty = (safety_distance - min_dist) * 5
                    # Add additional penalty based on number of nearby hazards
                    hazard_penalty = hazard_count * 2
                    base_cost += safety_penalty + hazard_penalty

            return base_cost

        start = tuple(self.robot_pos)
        goal = tuple(self.target_pos)
        frontier = [(manhattan_distance(start, goal), 0, start, [])]
        visited = set()

        while frontier:
            _, g_score, current, path = frontier.pop(0)

            if current == goal:
                return path

            if current in visited:
                continue

            visited.add(current)

            for next_pos, action in get_neighbors(current):
                if next_pos not in visited:
                    new_g = g_score + get_cost(current, next_pos)
                    new_f = new_g + manhattan_distance(next_pos, goal)
                    new_path = path + [action]

                    insert_idx = 0
                    while (
                        insert_idx < len(frontier) and frontier[insert_idx][0] < new_f
                    ):
                        insert_idx += 1
                    frontier.insert(insert_idx, (new_f, new_g, next_pos, new_path))

        return None

    def render(self, mode="human"):
        """Render the environment."""
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 48, bold=True)
            self.screen = pygame.display.set_mode((self.window_size))
            self.clock = pygame.time.Clock()

        if mode == "human":
            self.screen.fill(self.WHITE)

            for i in range(self.size):
                for j in range(self.size):
                    pygame.draw.rect(
                        self.screen,
                        self.BLACK,
                        (
                            j * self.cell_size,
                            i * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        ),
                        1,
                    )

                    if self.grid[i, j] == self.OBSTACLE:
                        pygame.draw.rect(
                            self.screen,
                            self.BLACK,
                            (
                                j * self.cell_size + 2,
                                i * self.cell_size + 2,
                                self.cell_size - 4,
                                self.cell_size - 4,
                            ),
                        )
                    elif self.grid[i, j] == self.TARGET:
                        pygame.draw.circle(
                            self.screen,
                            self.GREEN,
                            (
                                j * self.cell_size + self.cell_size // 2,
                                i * self.cell_size + self.cell_size // 2,
                            ),
                            self.cell_size // 3,
                        )
                    elif self.grid[i, j] == self.TRAP:
                        pygame.draw.rect(
                            self.screen,
                            self.LIGHT_RED,
                            (
                                j * self.cell_size + 2,
                                i * self.cell_size + 2,
                                self.cell_size - 4,
                                self.cell_size - 4,
                            ),
                        )

            robot_center = (
                self.robot_pos[1] * self.cell_size + self.cell_size // 2,
                self.robot_pos[0] * self.cell_size + self.cell_size // 2,
            )

            pygame.draw.circle(
                self.screen, self.BLUE, robot_center, self.cell_size // 3
            )

            if self.show_text and time.time() - self.text_start_time < 1.0:
                text_surface = self.font.render(
                    self.text_message, True, self.text_color
                )
                text_rect = text_surface.get_rect(
                    center=(self.window_size[0] // 2, self.window_size[1] // 2)
                )

                padding = 20
                bg_rect = text_rect.inflate(padding, padding)
                pygame.draw.rect(self.screen, self.WHITE, bg_rect)
                pygame.draw.rect(self.screen, self.text_color, bg_rect, 2)
                self.screen.blit(text_surface, text_rect)
            else:
                self.show_text = False

            pygame.display.flip()
            self.clock.tick(30)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
