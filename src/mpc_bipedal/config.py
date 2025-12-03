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
    dsp_duration: float = 3 * 0.01
    standing_duration: float = 100 * 0.01
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
    gamma: float = 0.0
    vx_ref: float = 0.0
    vy_ref: float = 0.0
    foot_length: float = 0.11
    foot_width: float = 0.05

    def __post_init__(self):
        """Calculate dt from horizon if not explicitly provided."""
        if self.dt is None:
            self.dt = 1.5 / self.horizon
