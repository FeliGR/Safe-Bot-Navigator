import numpy as np


class RiskAssessment:
    def __init__(
        self,
        grid,
        trap_penalty=0.05,
        obstacle_penalty=0.2,
        base_risk=0.5,
        risk_decay=0.05,
        risk_increase=0.5,
    ):
        """Initialize risk assessment with penalties for traps and obstacles."""
        self.grid = grid
        self.trap_penalty = trap_penalty
        self.obstacle_penalty = obstacle_penalty
        self.base_risk = base_risk
        self.risk_decay = risk_decay
        self.risk_increase = risk_increase
        self.risk_table = np.full(grid.shape, base_risk)

    def calculate_risk(self, pos, uncertainty_factor=0.1):
        """Calculate risk based on proximity to traps and obstacles, including uncertainty."""
        i, j = pos
        risk = self.risk_table[i, j]
        risk = float(risk) * (
            1 + uncertainty_factor
        )  # Convert to float to ensure it's a scalar
        return risk

    def adjust_reward(self, reward, pos):
        """Adjust reward by subtracting risk, ensuring non-negative reward."""
        i, j = pos
        risk = self.calculate_risk(pos)
        adjusted_reward = max(0, reward - risk)
        # Decay risk for visited position
        self.risk_table[i, j] = max(0, self.risk_table[i, j] - self.risk_decay)
        return adjusted_reward

    def increase_risk(self, pos):
        """Increase risk for a position when the agent 'dies' there."""
        i, j = pos
        self.risk_table[i, j] += self.risk_increase
