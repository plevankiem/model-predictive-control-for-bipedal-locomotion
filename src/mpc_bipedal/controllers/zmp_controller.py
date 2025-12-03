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
    
    def predict(
        self,
        x_init: np.ndarray,
        nb_steps: int,
        z_max: np.ndarray = None,
        z_min: np.ndarray = None,
        v_ref: float = None,
        axis: str = 'x'
    ) -> np.ndarray:
        """
        Unified predict method that routes to either wieber or herdt based on config.method.
        
        Args:
            x_init: The initial state of the system (shape: [3, 1]).
            nb_steps: Number of preview steps.
            z_max: Maximum ZMP bounds. For wieber: array (shape: [n_steps, 1]). For herdt: scalar (optional).
            z_min: Minimum ZMP bounds. For wieber: array (shape: [n_steps, 1]). For herdt: scalar (optional).
            v_ref: Reference velocity for herdt method (optional, uses config.vx_ref or config.vy_ref if None).
            axis: 'x' or 'y' axis, used for herdt method to select appropriate v_ref.
        
        Returns:
            The predicted state of the system (shape: [3, 1]).
        """
        method = self.config.method.lower()
        
        if method == "wieber":
            if z_max is None or z_min is None:
                raise ValueError("z_max and z_min are required for wieber method")
            return self.predict_wieber_axis(x_init, nb_steps, z_max, z_min)
        elif method == "herdt":
            # For herdt, use scalar bounds from config if not provided
            if z_max is None or z_min is None:
                if axis == 'x':
                    z_max = self.config.foot_length / 2.0
                    z_min = -self.config.foot_length / 2.0
                else:  # axis == 'y'
                    z_max = self.config.foot_width / 2.0
                    z_min = -self.config.foot_width / 2.0
            else:
                # If arrays are provided, extract scalar values (use first element)
                if isinstance(z_max, np.ndarray):
                    z_max = float(z_max.flat[0])
                if isinstance(z_min, np.ndarray):
                    z_min = float(z_min.flat[0])
            
            # Set v_ref from config if not provided
            if v_ref is None:
                if axis == 'x':
                    v_ref = self.config.vx_ref
                else:
                    v_ref = self.config.vy_ref
            
            return self.predict_herdt_axis(x_init, nb_steps, z_min, z_max, v_ref)
        else:
            raise ValueError(f"Unknown method: {method}. Must be 'wieber' or 'herdt'")
    
    def generate_com_trajectory(
        self,
        x_init: np.ndarray,
        y_init: np.ndarray,
        z_max: np.ndarray = None,
        z_min: np.ndarray = None,
        v_ref: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unified generate_com_trajectory method that routes to either wieber or herdt based on config.method.
        
        Args:
            x_init: The initial x state of the system.
            y_init: The initial y state of the system.
            z_max: Maximum ZMP bounds (shape: [n_steps, 2]). Required for wieber method.
            z_min: Minimum ZMP bounds (shape: [n_steps, 2]). Required for wieber method.
        
        Returns:
            Tuple of (com_trajectory, y_hist) where:
            - com_trajectory: COM positions (shape: [n_steps, 2])
            - y_hist: Full y state history (shape: [n_steps, 3, 1])
        """
        method = self.config.method.lower()
        
        if method == "wieber":
            if z_max is None or z_min is None:
                raise ValueError("z_max and z_min are required for wieber method")
            return self.generate_com_trajectory_wieber(x_init, y_init, z_max, z_min)
        elif method == "herdt":
            return self.generate_com_trajectory_herdt(x_init, y_init, v_ref)
        else:
            raise ValueError(f"Unknown method: {method}. Must be 'wieber' or 'herdt'")

    def generate_com_trajectory_wieber(self, x_init: np.ndarray, y_init: np.ndarray, 
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
            x_hist.append(self.predict_wieber_axis(
                x_hist[-1], preview_n_steps, 
                z_max_extended[i+1:i+1+preview_n_steps, 0:1], 
                z_min_extended[i+1:i+1+preview_n_steps, 0:1]
            ))
            y_hist.append(self.predict_wieber_axis(
                y_hist[-1], preview_n_steps, 
                z_max_extended[i+1:i+1+preview_n_steps, 1:2], 
                z_min_extended[i+1:i+1+preview_n_steps, 1:2]
            ))
            if self.config.add_force and i == force_time:
                y_hist[-1] = y_hist[-1] - np.array([[0., self.config.dt * self.config.F_ext / self.config.m, 0.]]).T
                
        return np.array([[x[0, 0], y[0, 0]] for x, y in zip(x_hist, y_hist)]), np.array(y_hist)

    def predict_wieber_axis(self, x_init: np.ndarray, nb_steps: int, z_max: np.ndarray, z_min: np.ndarray) -> np.ndarray:
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

    def generate_com_trajectory_herdt(
        self, x_init: np.ndarray, y_init: np.ndarray, v_ref: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the COM trajectory using herdt method.
        
        Args:
            x_init: The initial x state of the system.
            y_init: The initial y state of the system.
            v_ref: The reference velocity of the system.
        """
        x_hist, y_hist = [], []
        x_hist.append(x_init)
        y_hist.append(y_init)
        n_steps = len(v_ref)
        v_ref_extended = np.vstack([
            v_ref,
            np.tile(v_ref[-1:], (self.config.horizon, 1))
        ])
        

        return (np.empty, np.empty)
        

    def predict_herdt_axis(
        self,
        x_init: np.ndarray,
        v_ref: np.ndarray,
        x_fc: np.ndarray,
        nb_steps: int,
    ) -> np.ndarray:
        """
        Herdt-style prediction for ONE axis (x or y).

        Args:
            x_init: 3x1 state
            v_ref: 1xnb_steps speed
            x_fc: 1xnb_steps foot position
            nb_steps: preview horizon

        Returns:
            Next state (3x1) after applying the first jerk.
        """
        T = self.config.dt
        h, g = self.config.h, self.config.g

        # ----- Build ZMP prediction matrices Pzx, Pzu -----
        Pzx = np.zeros((nb_steps, 3))
        Pzu = np.zeros((nb_steps, nb_steps))
        for i in range(nb_steps):
            Pzx[i, 0] = 1.0
            Pzx[i, 1] = T * (i + 1)
            Pzx[i, 2] = (T**2)/2.0 * (i + 1)**2 - h/g
            for j in range(i + 1):
                Pzu[i, j] = (T**3)/6.0 * (1 + 3*(i-j) + 3*(i-j)**2) - T * h/g

        # ----- Build velocity prediction matrices Pvs, Pvu -----
        Pvs = np.zeros((nb_steps, 3))
        Pvu = np.zeros((nb_steps, nb_steps))

        for i in range(nb_steps):
            Pvs[i, 1] = 1.0
            Pvs[i, 2] = (i + 1) * T
            for j in range(i + 1):
                Pvu[i, j] = (T**2)/2.0 * (2*(i-j) + 1)
        
        # ----- Build the U matrices U_c, U -----
        dt_footstep = self.config.ssp_duration + self.config.dsp_duration
        nb_steps_foot = int(dt_footstep / self.config.dt)
        footstep_horizon = (nb_steps // nb_steps_foot)
        U = np.zeros((nb_steps, footstep_horizon))
        U_c = np.zeros((nb_steps, 1))

        U_c[0:nb_steps_foot, 0] = 1
        for m in range(1, footstep_horizon+1):
            U[m*nb_steps_foot:(m+1)*nb_steps_foot, m] = 1






        


        