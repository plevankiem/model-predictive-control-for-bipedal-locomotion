from config import ModelConfig
import numpy as np
from typing import Tuple

class LIPMModel:

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
        return self.A @ x + self.B @ u

    def get_zmp(self, x: np.ndarray) -> float:
        return self.C @ x
    
    def get_state_dimension(self) -> int:
        return self.A.shape[1]    