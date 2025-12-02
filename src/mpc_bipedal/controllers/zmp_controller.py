"""Zero Moment Point (ZMP) controller using Model Predictive Control."""

import numpy as np
import cvxpy as cp
from typing import Tuple
from tqdm import tqdm
from ..config import MPCConfig


class ZMPController:
    """ZMP Controller using Model Predictive Control for bipedal locomotion."""
    
    def __init__(self, config: MPCConfig):
        T = config.dt
        self.config = config
        self.A = np.array([[1., T, T ** 2 / 2.], [0., 1., T], [0., 0., 1.]])
        self.B = np.array([T ** 3 / 6., T ** 2 / 2., T]).reshape((3, 1))
        self.C = np.array([1., 0., -self.config.h / self.config.g])
        self.external_force = True

    def generate_com_trajectory(self, x_init: np.ndarray, y_init: np.ndarray, 
                                z_max: np.ndarray, z_min: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the COM trajectory using linear predictive control.
        
        Args:
            x_init: The initial x state of the system.
            y_init: The initial y state of the system.
            z_max: Maximum ZMP bounds (shape: [n_steps, 2])
            z_min: Minimum ZMP bounds (shape: [n_steps, 2])
        
        Returns:
            Tuple of (com_trajectory, y_hist) where:
            - com_trajectory: COM positions (shape: [n_steps, 2])
            - y_hist: Full y state history (shape: [n_steps, 3, 1])
        """
        x_hist, y_hist = [], []
        x_hist.append(x_init)
        y_hist.append(y_init)

        n_steps = len(z_min)
        
        z_max_extended = np.vstack([
            z_max,
            np.tile(z_max[-1:, :], (self.config.horizon, 1))
        ])
        z_min_extended = np.vstack([
            z_min,
            np.tile(z_min[-1:, :], (self.config.horizon, 1))
        ])

        force_time = n_steps // 2
        print(f"Time of the external force: {(force_time*self.config.dt):.2f}s")
        
        for i in tqdm(range(n_steps-1), desc="Generating COM trajectory"):
            preview_n_steps = self.config.horizon
            x_hist.append(self.predict(
                x_hist[-1], preview_n_steps, 
                z_max_extended[i+1:i+1+preview_n_steps, 0:1], 
                z_min_extended[i+1:i+1+preview_n_steps, 0:1]
            ))
            y_hist.append(self.predict(
                y_hist[-1], preview_n_steps, 
                z_max_extended[i+1:i+1+preview_n_steps, 1:2], 
                z_min_extended[i+1:i+1+preview_n_steps, 1:2]
            ))
            if self.config.add_force and i == force_time:
                y_hist[-1] = y_hist[-1] - np.array([[0., self.config.dt * self.config.F_ext / self.config.m, 0.]]).T
                
        return np.array([[x[0, 0], y[0, 0]] for x, y in zip(x_hist, y_hist)]), np.array(y_hist)

    def predict(self, x_init: np.ndarray, nb_steps: int, z_max: np.ndarray, z_min: np.ndarray) -> np.ndarray:
        """
        Predict the next state of the system using linear predictive control.
        
        Args:
            x_init: The initial state of the system (shape: [3, 1]).
            nb_steps: Number of preview steps.
            z_max: Maximum ZMP bounds (shape: [n_steps, 1]).
            z_min: Minimum ZMP bounds (shape: [n_steps, 1]).
        
        Returns:
            The predicted state of the system (shape: [3, 1]).
        """
        Px = np.zeros((nb_steps, 3))
        Pu = np.zeros((nb_steps, nb_steps))
        T = self.config.dt

        for i in range(nb_steps):
            Px[i, 0] = 1
            Px[i, 1] = T * (i + 1)
            Px[i, 2] = (T ** 2) / 2 * (i + 1) ** 2 - self.config.h / self.config.g
            for j in range(i + 1):
                Pu[i, j] = (T ** 3) / 6 * (1 + 3*(i-j) + 3*(i-j)**2) - T * self.config.h / self.config.g
        
        if self.config.strict:
            H = self.config.R * np.eye(nb_steps)
            A_ineq = np.concatenate([Pu, -Pu], axis=0)
            # Ensure dimensions are consistent (all 1D)
            Px_x_init = (Px @ x_init).flatten()
            z_max_flat = z_max.flatten()
            z_min_flat = z_min.flatten()
            b_ineq = np.concatenate([z_max_flat - Px_x_init, -z_min_flat + Px_x_init], axis=0)
            J = cp.Variable(nb_steps)

            # Calculate z_ref and z_pred for tracking
            z_ref = (z_max_flat + z_min_flat) / 2
            z_pred = Px_x_init + Pu @ J  # Predicted ZMP
            
            # Objective function with tracking toward z_ref and regularization
            objective = 0.5 * self.config.Q * cp.sum_squares(z_pred - z_ref) + 0.5 * cp.quad_form(J, H)
            constraints = [A_ineq @ J <= b_ineq]
            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver=cp.OSQP, warm_start=True)

            if J.value is None:
                raise RuntimeError("QP solver did not find a solution (infeasible or other).")
            X_burst = np.expand_dims(np.array(J.value), axis=1)
        else:
            z_ref = (z_max + z_min) / 2
            X_burst = - np.linalg.inv(Pu.T @ Pu + self.config.R/self.config.Q * np.eye(nb_steps)) @ Pu.T @ (Px @ x_init - z_ref)
        result = self.A @ x_init + self.B @ X_burst[0:1, :]

        return result

