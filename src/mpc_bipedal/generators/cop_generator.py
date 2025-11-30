"""Center of Pressure (CoP) trajectory generator."""

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from typing import Tuple
from ..config import CoPGeneratorConfig
from .footstep_generator import generate_footsteps


class State(Enum):
    """Walking state enum."""
    STANDING = 'STANDING'
    DOUBLE_SUPPORT = 'DOUBLE_SUPPORT'
    SINGLE_SUPPORT = 'SINGLE_SUPPORT'


class CoPGenerator:
    """
    Generates a viable CoP (Center of Pressure) trajectory to be provided to the ZMPController.
    """
    
    def __init__(self, config: CoPGeneratorConfig):
        self.ssp_duration = config.ssp_duration
        self.dsp_duration = config.dsp_duration
        self.standing_duration = config.standing_duration
        self.dt = config.dt
        self.distance = config.distance
        self.step_length = config.step_length
        self.foot_spread = config.foot_spread
    
    def generate_cop_trajectory(self, save_footsteps: bool = True, output_dir: str = 'results') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate CoP trajectory from footsteps.
        
        Args:
            save_footsteps: Whether to save the footsteps visualization
            output_dir: Directory to save the footsteps plot
            
        Returns:
            Tuple of (z_max, z_min) arrays defining the CoP bounds
        """
        footsteps = generate_footsteps(
            distance=self.distance,
            step_length=self.step_length,
            foot_spread=self.foot_spread,
        )
        
        if save_footsteps:
            import os
            os.makedirs(output_dir, exist_ok=True)
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
            plt.savefig(f'{output_dir}/footsteps.png')
            plt.close(fig)

        curr_footstep = 1
        state = State.STANDING
        t = 0.
        next_state_change = self.standing_duration
        z_max, z_min = [], []
        
        while curr_footstep < len(footsteps):
            if t > next_state_change:
                if state == State.STANDING and curr_footstep == len(footsteps) - 1:
                    curr_footstep += 1
                elif state == State.STANDING:
                    state = State.DOUBLE_SUPPORT
                    next_state_change += self.dsp_duration
                elif state == State.SINGLE_SUPPORT and curr_footstep + 1 == len(footsteps) - 1:
                    state = State.DOUBLE_SUPPORT
                    next_state_change += self.dsp_duration
                    curr_footstep += 1
                elif state == State.SINGLE_SUPPORT:
                    state = State.DOUBLE_SUPPORT
                    next_state_change += self.dsp_duration
                    curr_footstep += 1
                elif state == State.DOUBLE_SUPPORT and curr_footstep == len(footsteps) - 1:
                    state = State.STANDING
                    next_state_change += self.standing_duration
                elif state == State.DOUBLE_SUPPORT:
                    state = State.SINGLE_SUPPORT
                    next_state_change += self.ssp_duration
                else:
                    raise ValueError(f"Invalid state: {state}")

            if curr_footstep < len(footsteps):
                if state == State.STANDING or state == State.DOUBLE_SUPPORT:
                    footstep0, footstep1 = footsteps[curr_footstep-1], footsteps[curr_footstep]
                    z_max.append([max(footstep0.z_max[0], footstep1.z_max[0]), max(footstep0.z_max[1], footstep1.z_max[1])])
                    z_min.append([min(footstep0.z_min[0], footstep1.z_min[0]), min(footstep0.z_min[1], footstep1.z_min[1])])
                else:
                    z_max.append([footsteps[curr_footstep].z_max[0], footsteps[curr_footstep].z_max[1]])
                    z_min.append([footsteps[curr_footstep].z_min[0], footsteps[curr_footstep].z_min[1]])
            
            t += self.dt

        return np.array(z_max), np.array(z_min)

