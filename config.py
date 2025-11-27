from dataclasses import dataclass

@dataclass
class CoPGeneratorConfig:
    ssp_duration: float = 24 * 0.01
    dsp_duration: float = 3 * 0.01
    standing_duration: float = 50 * 0.01
    dt: float = 0.01
    distance: float = 2.1
    step_length: float = 0.3
    foot_spread: float = 0.1

@dataclass
class MPCConfig:
    horizon: int = 150
    Q: float = 1.0
    R: float = 1e-6
    dt: float = 0.01
    h: float = 0.75
    g: float = 9.81
    strict: bool = True