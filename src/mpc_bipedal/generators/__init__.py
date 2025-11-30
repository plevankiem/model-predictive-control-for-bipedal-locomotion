"""Generators for footsteps and CoP trajectories."""

from .footstep_generator import Contact, generate_footsteps
from .cop_generator import CoPGenerator

__all__ = ['Contact', 'generate_footsteps', 'CoPGenerator']

