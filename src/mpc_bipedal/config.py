"""Configuration class for the MPC bipedal locomotion system.

Historically, this project used two separate configuration classes:
`CoPGeneratorConfig` for the Center of Pressure (CoP) generator and
`MPCConfig` for the controller. These have been merged into a single
`MPCConfig` so that one object now contains **all** configuration
parameters for both components.
"""

from dataclasses import dataclass


@dataclass
class MPCConfig:
    """Unified configuration for CoP generation and the MPC controller."""

    # --- CoP generator parameters ---
    ssp_duration: float = 24 * 0.01
    dsp_duration: float = 1 * 0.01
    standing_duration: float = 50 * 0.01
    distance: float = 2.1
    step_length: float = 0.3
    foot_spread: float = 0.1

    # Time step (shared by CoP generator and MPC).
    # If None, it will be computed from the prediction horizon.
    dt: float = None

    # --- MPC parameters ---
    horizon: int = 150
    Q: float = 1.0
    R: float = 1e-6
    S: float = 1.0
    h: float = 0.75
    g: float = 9.81
    m: float = 40.0
    F_ext: float = 400.0
    strict: bool = True
    add_force: bool = True

    # Method selection and Herdt-specific parameters
    method: str = "wieber"
    alpha: float = 1e-6
    beta: float = 1.0
    gamma: float = 1.0
    vx_ref: float = 0.0
    vy_ref: float = 0.0
    foot_length: float = 0.11
    foot_width: float = 0.05
    v_max_x: float = 0.9
    v_max_y: float = 0.5
    # Speed generation strategy for reference velocities
    # "wieber" uses MPC-derived COM trajectory; "classic" uses fixed vx
    speed_generation: str = "classic"
    # Footstep polytopes (relative vertices) for left/right swing
    # The next footstep offset [dx, dy] must lie inside the chosen polytope
    left_foot_polytope: tuple = (
        (-0.1,-0.3),
        (-0.1, -0.4),
        (0.0, -0.4),
        (0.0, -0.2),
        (0.1, -0.17),
        (0.2, -0.13),
        (0.3,-0.1),
        (0.7, -0.05),
        (0.8, -0.05),
        (0.8, -0.3),
        (0.4,-0.35)
    )
    right_foot_polytope: tuple = (
        (-0.1, 0.3),
        (-0.1, 0.4),
        (0.0, 0.4),
        (0.0, 0.2),
        (0.1, 0.17),
        (0.2, 0.13),
        (0.3, 0.1),
        (0.7, 0.05),
        (0.8, 0.05),
        (0.8, 0.3),
        (0.4, 0.35)
    )

    def __post_init__(self):
        """Calculate dt from horizon if not explicitly provided."""
        if self.dt is None:
            self.dt = 1.5 / self.horizon
