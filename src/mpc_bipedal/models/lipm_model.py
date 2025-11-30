"""Linear Inverted Pendulum Model (LIPM) for bipedal locomotion."""

import numpy as np
from typing import Tuple
from ..config import ModelConfig


class LIPMModel:
    """Linear Inverted Pendulum Model implementation."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.g = config.g
        self.h = config.h
        self.dt = config.dt
        self.A = np.array([
            [1.0, self.dt, self.dt**2 / 2.0],
            [0.0, 1.0, self.dt],
            [0.0, 0.0, 1.0]
        ])
        self.B = np.array([
            [self.dt**3 / 6.0],
            [self.dt**2 / 2.0],
            [self.dt]
        ])
        self.C = np.array([[1.0, 0.0, -self.h / self.g]])

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute next state given current state and control input."""
        return self.A @ x + self.B @ u

    def get_zmp(self, x: np.ndarray) -> float:
        """Compute Zero Moment Point (ZMP) from state."""
        return self.C @ x
    
    def get_state_dimension(self) -> int:
        """Return the dimension of the state vector."""
        return self.A.shape[1]

