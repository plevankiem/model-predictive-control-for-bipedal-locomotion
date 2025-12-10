"""Zero Moment Point (ZMP) controller using Model Predictive Control."""

import numpy as np
import cvxpy as cp
from typing import Tuple
from tqdm import tqdm
from scipy.spatial import ConvexHull
from ..config import MPCConfig
from ..generators.cop_generator import State


class ZMPController:
    """ZMP Controller using Model Predictive Control for bipedal locomotion."""
    
    def __init__(self, config: MPCConfig):
        T = config.dt
        self.config = config
        self.A = np.array([[1., T, T ** 2 / 2.], [0., 1., T], [0., 0., 1.]])
        self.B = np.array([T ** 3 / 6., T ** 2 / 2., T]).reshape((3, 1))
        self.C = np.array([1., 0., -self.config.h / self.config.g])
        self.external_force = True
    
    def generate_com_trajectory(
        self,
        x_init: np.ndarray,
        y_init: np.ndarray,
        z_max: np.ndarray = None,
        z_min: np.ndarray = None,
        v_ref: np.ndarray = None,
        state_ref: np.ndarray = None,
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
            if v_ref is None or state_ref is None:
                raise ValueError("v_ref and state_ref are required for herdt method")
            return self.generate_com_trajectory_herdt(x_init, y_init, v_ref, state_ref)
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

    def generate_state_trajectory_wieber(
        self, x_init: np.ndarray, y_init: np.ndarray, z_max: np.ndarray, z_min: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate full state trajectories (x and y) using the Wieber method.

        Returns:
            Tuple of (x_hist, y_hist) each with shape [n_steps, 3, 1]
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

        for i in tqdm(range(n_steps-1), desc="Generating COM state trajectory"):
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

        return np.array(x_hist), np.array(y_hist)

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

    def find_nb_steps(self, state_ref: np.ndarray) -> list:
        """
        Given a list/array of State (STANDING, DOUBLE_SUPPORT, SINGLE_SUPPORT), compute
        for each index i the number of time steps needed to reach the next "change of footstep"
        and the total number of timesteps of the current footstep phase.
        
        Rules for nb_steps_to_next_state:
            - If in STANDING: find the first DOUBLE_SUPPORT, then the next SINGLE_SUPPORT;
                              the first DOUBLE_SUPPORT and first SINGLE_SUPPORT belong to the same foot (wait through both)
            - If in DOUBLE_SUPPORT: count steps to the next DOUBLE_SUPPORT
            - If in SINGLE_SUPPORT: count steps to the next DOUBLE_SUPPORT
        If no future footstep change found, returns the distance to the end of state_ref.
        
        Rules for current_footstep_total_nb_steps:
            - If in STANDING: use the nb_steps_to_next_state of the previous DOUBLE_SUPPORT (if exists), else use index i
            - If in DOUBLE_SUPPORT: total timesteps of the DOUBLE_SUPPORT phase (from first DOUBLE_SUPPORT to end)
            - If in SINGLE_SUPPORT: total timesteps from first DOUBLE_SUPPORT of the same footstep to end of this phase
        
        Args:
            state_ref: Sequence of State for each step
        
        Returns:
            List of tuples where result[i] = (nb_steps_to_next_state, current_footstep_total_nb_steps)
        """
        n = len(state_ref)
        nb_steps_to_next = [0] * n  # First pass: compute all nb_steps_to_next_state values
        result = [(0, 0)] * n
        
        # First pass: compute all nb_steps_to_next_state values
        for i in range(n - 1, -1, -1):
            cur_state = state_ref[i]
            remaining = n - i
            
            if cur_state == State.STANDING:
                # Find first DOUBLE_SUPPORT
                idx_ds = None
                for j in range(i + 1, n):
                    if state_ref[j] == State.DOUBLE_SUPPORT:
                        idx_ds = j
                        break
                if idx_ds is None:
                    nb_steps_to_next[i] = remaining  # No further DS found
                else:
                    # Find first SINGLE_SUPPORT after that
                    idx_ss = None
                    for k in range(idx_ds + 1, n):
                        if state_ref[k] == State.SINGLE_SUPPORT:
                            idx_ss = k
                            break
                    if idx_ss is None:
                        nb_steps_to_next[i] = remaining  # No further SS found after DS
                    else:
                        nb_steps_to_next[i] = idx_ss - i - 1  # number of steps to the SS
                    
            elif cur_state == State.DOUBLE_SUPPORT:
                # Wait to next DOUBLE_SUPPORT (excluding current index)
                found = False
                for j in range(i + 1, n):
                    if state_ref[j] == State.DOUBLE_SUPPORT:
                        nb_steps_to_next[i] = j - i
                        found = True
                        break
                if not found:
                    nb_steps_to_next[i] = remaining  # No other DS left
                    
            elif cur_state == State.SINGLE_SUPPORT:
                # Wait to next DOUBLE_SUPPORT
                found = False
                for j in range(i + 1, n):
                    if state_ref[j] == State.DOUBLE_SUPPORT:
                        nb_steps_to_next[i] = j - i
                        found = True
                        break
                if not found:
                    nb_steps_to_next[i] = remaining
            else:
                # Unknown state? Be conservative
                nb_steps_to_next[i] = remaining
        
        # Second pass: compute current_footstep_total_nb_steps for all states
        # First, handle index 0 specially (needed for STANDING states that reference it)
        i = 0
        if i < n:
            cur_state = state_ref[i]
            remaining = n - i
            
            if cur_state == State.STANDING:
                # For index 0 STANDING: check if there's a previous DOUBLE_SUPPORT
                prev_ds_idx = None
                # There can't be one before index 0, so use nb_steps_to_next[0] as default
                current_footstep_total_nb_steps = nb_steps_to_next[0]
            elif cur_state == State.DOUBLE_SUPPORT:
                # Wait to next DOUBLE_SUPPORT (excluding current index)
                found = False
                next_ds_idx = None
                for j in range(i + 1, n):
                    if state_ref[j] == State.DOUBLE_SUPPORT:
                        next_ds_idx = j
                        found = True
                        break
                if not found:
                    next_ds_idx = n
                
                # For DOUBLE_SUPPORT: find the first timestep of this DOUBLE_SUPPORT phase
                ds_start_idx = i
                for j in range(i - 1, -1, -1):
                    if state_ref[j] == State.DOUBLE_SUPPORT:
                        ds_start_idx = j
                    else:
                        break
                
                current_footstep_total_nb_steps = next_ds_idx - ds_start_idx
            elif cur_state == State.SINGLE_SUPPORT:
                # Wait to next DOUBLE_SUPPORT
                found = False
                next_ds_idx = None
                for j in range(i + 1, n):
                    if state_ref[j] == State.DOUBLE_SUPPORT:
                        next_ds_idx = j
                        found = True
                        break
                if not found:
                    next_ds_idx = n
                
                prev_ds_idx = None
                for j in range(i - 1, -1, -1):
                    if state_ref[j] == State.DOUBLE_SUPPORT:
                        prev_ds_idx = j
                        break
                
                if prev_ds_idx is not None:
                    ds_start_idx = prev_ds_idx
                    for j in range(prev_ds_idx - 1, -1, -1):
                        if state_ref[j] == State.DOUBLE_SUPPORT:
                            ds_start_idx = j
                        else:
                            break
                    current_footstep_total_nb_steps = next_ds_idx - ds_start_idx
                else:
                    current_footstep_total_nb_steps = remaining
            else:
                current_footstep_total_nb_steps = remaining
            
            result[i] = (nb_steps_to_next[i], current_footstep_total_nb_steps)
        
        # Now process the rest backwards
        for i in range(n - 1, 0, -1):
            cur_state = state_ref[i]
            remaining = n - i
            
            if cur_state == State.STANDING:
                # For STANDING: use nb_steps_to_next_state of previous DOUBLE_SUPPORT, or result[0][1] if none
                prev_ds_idx = None
                for j in range(i - 1, -1, -1):
                    if state_ref[j] == State.DOUBLE_SUPPORT:
                        prev_ds_idx = j
                        break
                
                if prev_ds_idx is not None:
                    # Use the nb_steps_to_next_state of the previous DOUBLE_SUPPORT
                    current_footstep_total_nb_steps = nb_steps_to_next[prev_ds_idx]
                else:
                    # No DOUBLE_SUPPORT before, use current_footstep_total_nb_steps of first element
                    current_footstep_total_nb_steps = result[0][1]
                    
            elif cur_state == State.DOUBLE_SUPPORT:
                # Wait to next DOUBLE_SUPPORT (excluding current index)
                found = False
                next_ds_idx = None
                for j in range(i + 1, n):
                    if state_ref[j] == State.DOUBLE_SUPPORT:
                        next_ds_idx = j
                        found = True
                        break
                if not found:
                    next_ds_idx = n
                
                # For DOUBLE_SUPPORT: find the first timestep of this DOUBLE_SUPPORT phase
                # Go backwards to find where this DOUBLE_SUPPORT phase started
                ds_start_idx = i
                for j in range(i - 1, -1, -1):
                    if state_ref[j] == State.DOUBLE_SUPPORT:
                        ds_start_idx = j
                    else:
                        break
                
                # Total = from start of DOUBLE_SUPPORT phase to end of this phase
                # End is either next DOUBLE_SUPPORT or end of array
                current_footstep_total_nb_steps = next_ds_idx - ds_start_idx
                    
            elif cur_state == State.SINGLE_SUPPORT:
                # Wait to next DOUBLE_SUPPORT
                found = False
                next_ds_idx = None
                for j in range(i + 1, n):
                    if state_ref[j] == State.DOUBLE_SUPPORT:
                        next_ds_idx = j
                        found = True
                        break
                if not found:
                    next_ds_idx = n
                
                # For SINGLE_SUPPORT: find the first DOUBLE_SUPPORT that this SINGLE_SUPPORT belongs to
                # Go backwards to find the DOUBLE_SUPPORT that precedes this SINGLE_SUPPORT
                prev_ds_idx = None
                for j in range(i - 1, -1, -1):
                    if state_ref[j] == State.DOUBLE_SUPPORT:
                        prev_ds_idx = j
                        break
                
                if prev_ds_idx is not None:
                    # Find the start of that DOUBLE_SUPPORT phase
                    ds_start_idx = prev_ds_idx
                    for j in range(prev_ds_idx - 1, -1, -1):
                        if state_ref[j] == State.DOUBLE_SUPPORT:
                            ds_start_idx = j
                        else:
                            break
                    # Total = from start of DOUBLE_SUPPORT to end of current SINGLE_SUPPORT phase
                    current_footstep_total_nb_steps = next_ds_idx - ds_start_idx
                else:
                    # No previous DOUBLE_SUPPORT found, use remaining
                    current_footstep_total_nb_steps = remaining
            
            else:
                # Unknown state? Be conservative
                current_footstep_total_nb_steps = remaining
            
            result[i] = (nb_steps_to_next[i], current_footstep_total_nb_steps)
        
        return result

    def generate_com_trajectory_herdt(
        self,
        x_init: np.ndarray,
        y_init: np.ndarray,
        v_ref: np.ndarray,
        state_ref: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        force_time = n_steps // 3
        x_fc = np.array([[0.0]])
        y_fc = np.array([[self.config.foot_spread]])
        foot_side = "right"
        x_airc = x_fc.copy()
        y_airc = y_fc.copy()
        x_fc_hist = [x_fc]
        y_fc_hist = [y_fc]
        current_state = state_ref[0]

        # Pad refs to handle side effects
        last_v = v_ref[-1:]
        pad = np.repeat(last_v, self.config.horizon, axis=0)
        v_ref = np.vstack([v_ref, pad])
        last_s = state_ref[-1:]
        pad_s = np.repeat(last_s, self.config.horizon, axis=0)
        state_ref = np.concatenate([state_ref, pad_s])
        nb_steps_to_next_state = self.find_nb_steps(state_ref)
        # for i in range(len(state_ref)):
        #     print(state_ref[i], nb_steps_to_next_state[i])

        for i in tqdm(range(n_steps-1), desc="Generating COM trajectory"):
            preview_n_steps = self.config.horizon
            x_next, y_next, first_x_footstep, first_y_footstep = self.predict_herdt_joint(
                x_hist[-1],
                y_hist[-1],
                v_ref[i+1:i+1+preview_n_steps, :],
                x_fc_hist[-1],
                y_fc_hist[-1],
                current_state,
                state_ref[i+1:i+1+preview_n_steps],
                preview_n_steps,
                nb_steps_to_next_state[i],
                x_airc,
                y_airc,
                foot_side,
            )
            x_hist.append(x_next)
            y_hist.append(y_next)

            if first_x_footstep is not None:
                x_airc += (1 / nb_steps_to_next_state[i][0]) * (first_x_footstep - x_airc)
            if first_y_footstep is not None:
                y_airc += (1 / nb_steps_to_next_state[i][0]) * (first_y_footstep - y_airc)
            if state_ref[i+1] != current_state and current_state == State.SINGLE_SUPPORT:
                foot_side = "left" if foot_side == "right" else "right"
                x_fc_hist.append(np.array([[float(first_x_footstep)]]))
                y_fc_hist.append(np.array([[float(first_y_footstep)]]))
                y_airc = y_fc_hist[-1]
                x_airc = x_fc_hist[-1]
            else:
                x_fc_hist.append(x_fc_hist[-1])
                y_fc_hist.append(y_fc_hist[-1])
            if self.config.add_force and i == force_time:
                y_hist[-1] = y_hist[-1] - np.array([[0., self.config.dt * self.config.F_ext / self.config.m, 0.]]).T
            if state_ref[i+1] != current_state:
                current_state = state_ref[i+1]
        com_traj = np.array([[x[0, 0], y[0, 0]] for x, y in zip(x_hist, y_hist)])
        foot_hist = np.array([[xf[0, 0], yf[0, 0]] for xf, yf in zip(x_fc_hist, y_fc_hist)])
        return com_traj, np.array(y_hist), foot_hist

    def predict_herdt_joint(
        self,
        x_init: np.ndarray,
        y_init: np.ndarray,
        v_ref: np.ndarray,
        x_fc: np.ndarray,
        y_fc: np.ndarray,
        current_state: State,
        state_ref: np.ndarray,
        nb_steps: int,
        nb_steps_to_next_state: Tuple[int, int],
        x_airc: np.ndarray,
        y_airc: np.ndarray,
        foot_side: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Joint Herdt-style prediction that solves a single QP for both axes.

        Args:
            x_init: 3x1 state for the x axis.
            y_init: 3x1 state for the y axis.
            v_ref: (nb_steps x 2) reference velocities [vx, vy].
            x_fc / y_fc: current foot positions for each axis.
            nb_steps: preview horizon.

        Returns:
            Tuple of (next_x_state, next_y_state, first_x_footstep, first_y_footstep).
        """
        T = self.config.dt
        h, g = self.config.h, self.config.g

        # Shared prediction matrices
        Pzx = np.zeros((nb_steps, 3))
        Pzu = np.zeros((nb_steps, nb_steps))
        Pvs = np.zeros((nb_steps, 3))
        Pvu = np.zeros((nb_steps, nb_steps))

        for i in range(nb_steps):
            Pzx[i, 0] = 1.0
            Pzx[i, 1] = T * (i + 1)
            Pzx[i, 2] = (T**2) / 2.0 * (i + 1) ** 2 - h / g
            Pvs[i, 1] = 1.0
            Pvs[i, 2] = (i + 1) * T
            for j in range(i + 1):
                Pzu[i, j] = (T**3) / 6.0 * (1 + 3 * (i - j) + 3 * (i - j) ** 2) - T * h / g
                Pvu[i, j] = (T**2) / 2.0 * (2 * (i - j) + 1)

        # Support phases
        s = current_state
        l = []
        c = 1
        for state in state_ref:
            if state == s:
                c += 1
            elif s == State.DOUBLE_SUPPORT and state == State.SINGLE_SUPPORT:
                c += 1
            else:
                l.append(c)
                c = 1
            s = state
        l.append(c)

        m = len(l) - 1
        U = np.zeros((nb_steps, m))
        U_c = np.zeros((nb_steps, 1))
        U_c[: l[0], 0] = 1
        n_c = l[0]
        for i, n_f in enumerate(l[1:]):
            U[n_c : n_c + n_f, i] = 1
            n_c += n_f

        alpha = self.config.alpha
        beta = self.config.beta
        gamma = self.config.gamma

        N = nb_steps

        I_N = np.eye(N)

        # Cost for x axis
        Qxx = alpha * I_N + beta * (Pvu.T @ Pvu) + gamma * (Pzu.T @ Pzu)
        Qxf = -gamma * (Pzu.T @ U)
        Qfx = Qxf.T
        Qff = gamma * (U.T @ U)
        Qx = np.block([[Qxx, Qxf], [Qfx, Qff]])
        Qx = 0.5 * (Qx + Qx.T)

        # Cost for y axis (identical structure)
        Qy = Qx.copy()

        # Linear terms for x
        v_pred_x = Pvs @ x_init
        z_pred_x = Pzx @ x_init
        v_ref_x = v_ref[:, 0:1]
        z_ref_x = U_c * x_fc
        e_v_x = v_pred_x - v_ref_x
        e_z_x = z_pred_x - z_ref_x
        p_x_jerk = beta * (Pvu.T @ e_v_x) + gamma * (Pzu.T @ e_z_x)
        p_x_foot = -gamma * (U.T @ e_z_x)
        p_x = np.vstack([p_x_jerk, p_x_foot]).flatten()

        # Linear terms for y
        v_pred_y = Pvs @ y_init
        z_pred_y = Pzx @ y_init
        v_ref_y = v_ref[:, 1:2]
        z_ref_y = U_c * y_fc
        e_v_y = v_pred_y - v_ref_y
        e_z_y = z_pred_y - z_ref_y
        p_y_jerk = beta * (Pvu.T @ e_v_y) + gamma * (Pzu.T @ e_z_y)
        p_y_foot = -gamma * (U.T @ e_z_y)
        p_y = np.vstack([p_y_jerk, p_y_foot]).flatten()

        total_vars = 2 * (N + m)
        Q = np.block(
            [
                [Qx, np.zeros((N + m, N + m))],
                [np.zeros((N + m, N + m)), Qy],
            ]
        )
        Q = 0.5 * (Q + Q.T)
        p = np.concatenate([p_x, p_y])

        u = cp.Variable(total_vars)

        # Constraints!!!!!!!!!!!!!!!!!
        constraints = []

        coef = 0.5
        # ZMP constraints for x
        Zx_nom_x = Pzx @ x_init
        foot_const_x = U_c * float(x_fc)
        A_dx_x = np.hstack([Pzu, -U])
        b_dx_x = np.full((N, 1), coef * self.config.foot_length)
        rhs_upper_x = b_dx_x - Zx_nom_x + foot_const_x
        rhs_lower_x = b_dx_x + Zx_nom_x - foot_const_x
        Aineq_x = np.vstack([A_dx_x, -A_dx_x])
        bineq_x = np.vstack([rhs_upper_x, rhs_lower_x]).flatten()

        # ZMP constraints for y
        Zx_nom_y = Pzx @ y_init
        foot_const_y = U_c * float(y_fc)
        A_dx_y = np.hstack([Pzu, -U])
        b_dx_y = np.full((N, 1), coef * self.config.foot_width)
        rhs_upper_y = b_dx_y - Zx_nom_y + foot_const_y
        rhs_lower_y = b_dx_y + Zx_nom_y - foot_const_y
        Aineq_y = np.vstack([A_dx_y, -A_dx_y])
        bineq_y = np.vstack([rhs_upper_y, rhs_lower_y]).flatten()

        # Embed constraints in full variable space
        A_block_x = np.hstack([Aineq_x, np.zeros((Aineq_x.shape[0], N + m))])
        A_block_y = np.hstack([np.zeros((Aineq_y.shape[0], N + m)), Aineq_y])
        constraints.append(A_block_x @ u <= bineq_x)
        constraints.append(A_block_y @ u <= bineq_y)
        
        # Footsteps constraints
        if m > 0:
            poly_vertices = (
                np.array(self.config.left_foot_polytope)
                if foot_side == "left"
                else np.array(self.config.right_foot_polytope)
            )
            A_poly, b_poly = self._polytope_halfspace(poly_vertices)
            idx_fx = N  # first future footstep x variable
            idx_fy = N + m + N  # first future footstep y variable

            A_full = np.zeros((A_poly.shape[0], total_vars))
            A_full[:, idx_fx] = A_poly[:, 0]
            A_full[:, idx_fy] = A_poly[:, 1]

            # Move current foot position to the RHS: A (f - f_current) <= b
            b_full = b_poly + A_poly @ np.array([[x_fc[0, 0], y_fc[0, 0]]]).T.flatten()
            constraints.append(A_full @ u <= b_full)

        objective = 0.5 * cp.quad_form(u, cp.psd_wrap(Q)) + p @ u
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, polish=False, verbose=False)

        if u.value is None:
            print(f"Joint QP solver failed: {nb_steps_to_next_state[0]} steps remaining to the next footstep")
            # print(f"status: {prob.status} | solver_stats: {prob.solver_stats}")
            u_opt = np.zeros(total_vars)
            if m > 0:
                u_opt[N] = x_airc[0, 0]
                u_opt[N + m + N] = y_airc[0, 0]
        else:
            u_opt = u.value

        first_jerk_x = u_opt[0]
        first_x_footstep = u_opt[N] if m > 0 else None
        first_jerk_y = u_opt[N + m]
        first_y_footstep = u_opt[N + m + N] if m > 0 else None

        x_next = self.A @ x_init + self.B * first_jerk_x
        y_next = self.A @ y_init + self.B * first_jerk_y
        return x_next, y_next, first_x_footstep, first_y_footstep

    def _polytope_halfspace(self, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert ordered convex polygon vertices to half-space form A x <= b.

        Normals point outward; inequalities describe the interior.
        """
        verts = np.asarray(vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 3:
            raise ValueError("Polytope must be array-like of shape (k, 2), k>=3")

        # Use SciPy convex hull to get half-space representation (outward normals)
        hull = ConvexHull(verts)
        equations = hull.equations  # shape (n_facets, 3) with [n1, n2, offset]
        A = equations[:, :2]
        b = -equations[:, 2]
        return A, b