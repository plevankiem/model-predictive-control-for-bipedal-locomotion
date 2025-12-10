"""Speed and state trajectory generation utilities."""

import numpy as np
from typing import Tuple, List

from .cop_generator import CoPGenerator, State
from ..controllers.zmp_controller import ZMPController
from ..config import MPCConfig


class SpeedTrajectoryGenerator:
    """Generates reference CoM speed trajectories aligned with CoP/state timeline."""

    def __init__(self, config: MPCConfig):
        self.config = config
        self._cop_generator = CoPGenerator(config)
        self._zmp_controller = ZMPController(config)

    def generate_speed_and_state(
        self,
        save_footsteps: bool = True,
        output_dir: str = "results",
    ) -> Tuple[np.ndarray, np.ndarray, List[State]]:
        """
        Generate vx/vy reference speeds and state sequence.

        Two modes controlled by config.speed_generation:
          - "wieber": derive speeds from MPC (Wieber) COM trajectory.
          - "classic": fixed vx=0.3 m/s (except 0 when STANDING), vy=0.

        Args:
            save_footsteps: Whether to save the footsteps visualization produced by the CoP generator.
            output_dir: Directory for any generated plots.

        Returns:
            Tuple of (v_x, v_y, states) where:
                - v_x: np.ndarray of x-velocity references (shape: [n_steps])
                - v_y: np.ndarray of y-velocity references (shape: [n_steps])
                - states: list of State values for each step (length n_steps)
        """
        z_max, z_min, states = self._cop_generator.generate_cop_trajectory(
            save_footsteps=save_footsteps,
            output_dir=output_dir,
        )

        mode = (self.config.speed_generation or "wieber").lower()

        if mode == "classic":
            v_x = [0.0 if s == State.STANDING else 0.3 for s in states]
            v_y = [0.0 for _ in states]
            return np.array(v_x), np.array(v_y), states

        if mode == "wieber":
            # Initial CoM state: position, velocity, acceleration all zero
            x_init = np.zeros((3, 1))
            y_init = np.zeros((3, 1))

            x_hist, y_hist = self._zmp_controller.generate_state_trajectory_wieber(
                x_init=x_init,
                y_init=y_init,
                z_max=z_max,
                z_min=z_min,
            )

            v_x = x_hist[:, 1, 0]
            v_y = y_hist[:, 1, 0]
            return v_x, v_y, states

        raise ValueError(f"Unknown speed_generation mode: {self.config.speed_generation}")