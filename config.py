from dataclasses import dataclass

@dataclass
class ModelConfig:
    g: float = 9.81
    h: float = 0.8
    dt: float = 0.01

@dataclass
class MPCConfig:
    horizon: int = 20
    Q: float = 1.0
    R: float = 0.01