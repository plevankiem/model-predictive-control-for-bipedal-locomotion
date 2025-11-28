from config import MPCConfig, CoPGeneratorConfig
import numpy as np
import cvxpy as cp
from typing import Any, Tuple
from pynamoid import generate_footsteps
from enum import Enum
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm

enum = Enum('State', ['STANDING', 'DOUBLE_SUPPORT', 'SINGLE_SUPPORT'])

class CoPGenerator:
    """
    The role of this class is to generate a viable cop trajectory, to be provided to the ZMPController.
    """
    def __init__(self, config: CoPGeneratorConfig):
        self.ssp_duration = config.ssp_duration
        self.dsp_duration = config.dsp_duration
        self.standing_duration = config.standing_duration
        self.dt = config.dt
        self.distance = config.distance
        self.step_length = config.step_length
        self.foot_spread = config.foot_spread
    
    def generate_cop_trajectory(self):
        footsteps = generate_footsteps(
            distance=self.distance,
            step_length=self.step_length,
            foot_spread=self.foot_spread,
        )
        fig, ax = plt.subplots()
        for contact in footsteps:
            x, y = contact.x, contact.y
            w, h = contact.shape
            rect = plt.Rectangle((x - w/2, y - h/2), w, h, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        X = [contact.x for contact in footsteps]
        Y = [contact.y for contact in footsteps]
        ax.scatter(X, Y, color='r', s=0.2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Footsteps (rectangles centered on contacts)")
        ax.set_aspect('equal')
        plt.savefig('results/footsteps.png')
        plt.close(fig)

        curr_footstep = 1
        state = enum.STANDING
        t = 0.
        next_state_change = self.standing_duration
        z_max, z_min = [], []
        while curr_footstep < len(footsteps):
            if t > next_state_change:
                if state == enum.STANDING and curr_footstep == len(footsteps) - 1:
                    curr_footstep += 1
                elif state == enum.STANDING:
                    state = enum.DOUBLE_SUPPORT
                    next_state_change += self.dsp_duration
                elif state == enum.SINGLE_SUPPORT and curr_footstep + 1 == len(footsteps) - 1:
                    state = enum.DOUBLE_SUPPORT
                    next_state_change += self.dsp_duration
                    curr_footstep += 1
                elif state == enum.SINGLE_SUPPORT:
                    state = enum.DOUBLE_SUPPORT
                    next_state_change += self.dsp_duration
                    curr_footstep += 1
                elif state == enum.DOUBLE_SUPPORT and curr_footstep == len(footsteps) - 1:
                    state = enum.STANDING
                    next_state_change += self.standing_duration
                elif state == enum.DOUBLE_SUPPORT:
                    state = enum.SINGLE_SUPPORT
                    next_state_change += self.ssp_duration
                else:
                    raise ValueError(f"Invalid state: {state}")

            if curr_footstep < len(footsteps):
                if state == enum.STANDING or state == enum.DOUBLE_SUPPORT:
                    footstep0, footstep1 = footsteps[curr_footstep-1], footsteps[curr_footstep]
                    z_max.append([max(footstep0.z_max[0], footstep1.z_max[0]), max(footstep0.z_max[1], footstep1.z_max[1])])
                    z_min.append([min(footstep0.z_min[0], footstep1.z_min[0]), min(footstep0.z_min[1], footstep1.z_min[1])])
                else:
                    z_max.append([footsteps[curr_footstep].z_max[0], footsteps[curr_footstep].z_max[1]])
                    z_min.append([footsteps[curr_footstep].z_min[0], footsteps[curr_footstep].z_min[1]])
            
            t += self.dt

        return np.array(z_max), np.array(z_min)

class ZMPController:
    def __init__(self, config: MPCConfig):
        T = config.dt
        self.config = config
        self.A = np.array([[1., T, T ** 2 / 2.], [0., 1., T], [0., 0., 1.]])
        self.B = np.array([T ** 3 / 6., T ** 2 / 2., T]).reshape((3, 1))
        self.C = np.array([1., 0., -self.config.h / self.config.g])

    def generate_com_trajectory(self, x_init, y_init, z_max, z_min):
        """
        Generate the com trajectory using the linear predictive control.
        Args:
            x_init: The initial x state of the system.
            y_init: The initial y state of the system.
            e: The error of the system. e = Z_ref, which should be provided by the user.

        Returns:
            The com trajectory.
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
        
        for i in tqdm(range(n_steps-1)):
            preview_n_steps = self.config.horizon
            x_hist.append(self.predict(x_hist[-1], preview_n_steps, z_max_extended[i+1:i+1+preview_n_steps, 0:1], z_min_extended[i+1:i+1+preview_n_steps, 0:1]))
            y_hist.append(self.predict(y_hist[-1], preview_n_steps, z_max_extended[i+1:i+1+preview_n_steps, 1:2], z_min_extended[i+1:i+1+preview_n_steps, 1:2]))
        return np.array([[x[0, 0], y[0, 0]] for x, y in zip(x_hist, y_hist)]), np.array(y_hist)

    def predict(self, x_init, nb_steps, z_max, z_min):
        """
        Predict the next state of the system using the linear predictive control.
        Args:
            x_init: The initial state of the system.
            z_max, z_min: Should of shape (n_steps, 1)

        Returns:
            The predicted state of the system.
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
            # S'assurer que les dimensions sont cohérentes (tous en 1D)
            Px_x_init = (Px @ x_init).flatten()
            z_max_flat = z_max.flatten()
            z_min_flat = z_min.flatten()
            b_ineq = np.concatenate([z_max_flat - Px_x_init, -z_min_flat + Px_x_init], axis=0)
            J = cp.Variable(nb_steps)

            # Calculer z_ref et z_pred pour le tracking
            z_ref = (z_max_flat + z_min_flat) / 2
            z_pred = Px_x_init + Pu @ J  # ZMP prédit
            
            # Fonction objectif avec tracking vers z_ref et régularisation
            objective = 0.5 * self.config.Q * cp.sum_squares(z_pred - z_ref) + 0.5 * cp.quad_form(J, H)
            constraints = [A_ineq @ J <= b_ineq]
            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver=cp.OSQP, warm_start=True)

            if J.value is None:
                raise RuntimeError("Le solveur QP n'a pas trouvé de solution (infeasible ou autre).")
            X_burst = np.expand_dims(np.array(J.value), axis=1)
        else:
            z_ref = (z_max + z_min) / 2
            X_burst = - np.linalg.inv(Pu.T @ Pu + self.config.R/self.config.Q * np.eye(nb_steps)) @ Pu.T  @ (Px @ x_init - z_ref)
        result = self.A @ x_init + self.B @ X_burst[0:1, :]

        return result

def main():
    config = CoPGeneratorConfig()
    cop_generator = CoPGenerator(config)
    z_max, z_min = cop_generator.generate_cop_trajectory()
    t = np.arange(z_max.shape[0]) * config.dt

    zmp_config = MPCConfig()
    controller = ZMPController(zmp_config)
    com_trajectory, y_hist = controller.generate_com_trajectory(np.array([[0., 0., 0.]]).T, np.array([[0., 0., 0.]]).T, z_max, z_min)
    C_dot_y_hist = np.tensordot(y_hist[:, :, 0], controller.C, axes=([1], [0]))
    print(com_trajectory.shape)


    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t, y=z_max[:, 1],
        mode='lines',
        name='z_max',
        line=dict(color='red', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=t, y=z_min[:, 1],
        mode='lines',
        name='z_min',
        line=dict(color='blue', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=t, y=C_dot_y_hist,
        mode='lines',
        name='Estimation de z',
        line=dict(color='green', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=t, y=com_trajectory[:, 1],
        mode='lines',
        name='com',
        line=dict(color='black')
    ))

    fig.update_layout(
        title="CoP Limits Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Y Axis",
        legend=dict(x=0, y=1),
        template="plotly_white"
    )

    fig.show()
    
    # Visualisation 3D de la trajectoire du COM
    # Décommenter les lignes suivantes pour activer la visualisation 3D
    from visualize import visualize_com_trajectory_3d, visualize_com_trajectory_static
    
    # Version avec animation (sphère qui suit la trajectoire)
    visualize_com_trajectory_3d(com_trajectory, show_sphere=True, save_animation=False)
    
    # Version statique avec gradient de couleur (alternative)
    # visualize_com_trajectory_static(com_trajectory)


if __name__ == "__main__":
    main()