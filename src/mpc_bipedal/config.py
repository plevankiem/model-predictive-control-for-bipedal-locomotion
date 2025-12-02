"""Configuration classes for the MPC bipedal locomotion system."""

from dataclasses import dataclass

@dataclass
class CoPGeneratorConfig:
    """Configuration for the Center of Pressure (CoP) trajectory generator."""
    ssp_duration: float = 24 * 0.01
    dsp_duration: float = 3 * 0.01
    standing_duration: float = 100 * 0.01
    dt: float = None  # Will be synchronized with MPCConfig.dt
    distance: float = 2.1
    step_length: float = 0.3
    foot_spread: float = 0.1

@dataclass
class MPCConfig:
    """Configuration for the Model Predictive Controller."""
    horizon: int = 150
    Q: float = 1.0
    R: float = 1e-6
    dt: float = None  # Will be calculated from horizon
    h: float = 0.75
    g: float = 9.81
    m: float = 40.0
    F_ext: float = 400.0
    strict: bool = True
    add_force: bool = True
    
    def __post_init__(self):
        """Calculate dt from horizon if not explicitly provided."""
        if self.dt is None:
            self.dt = 1.5 / self.horizon

