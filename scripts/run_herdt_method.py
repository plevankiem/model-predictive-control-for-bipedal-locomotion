#!/usr/bin/env python3
"""
Simple runner for the Herdt MPC pipeline using the speed/state generator.

This script mirrors run_mpc.py but focuses on the Herdt method and relies on
SpeedTrajectoryGenerator to produce velocity references and state timelines.
"""

import sys
import os
import argparse
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.mpc_bipedal.config import MPCConfig
from src.mpc_bipedal.generators.speed_generation import SpeedTrajectoryGenerator
from src.mpc_bipedal.controllers.zmp_controller import ZMPController


def load_config_from_json(config_file: str) -> MPCConfig:
    """Load unified configuration from a JSON file."""
    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    mpc_dict = config_dict.get('mpc', {}).copy()

    # Handle dt: horizon takes precedence. If only dt is provided, recalculate horizon from it.
    # dt will always be calculated from horizon in __post_init__, so we remove dt from dict.
    if 'dt' in mpc_dict:
        dt_value = mpc_dict.pop('dt')
        if 'horizon' not in mpc_dict:
            mpc_dict['horizon'] = int(1.5 / dt_value)

    return MPCConfig(**mpc_dict)


def main():
    parser = argparse.ArgumentParser(description="Run Herdt MPC with speed/state generation.")
    parser.add_argument('--config', type=str, help='Path to JSON config file')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--F-ext', type=float, dest='F_ext', help='External force (N)')
    args = parser.parse_args()

    # Load config
    if args.config and os.path.exists(args.config):
        config = load_config_from_json(args.config)
    else:
        config = MPCConfig()

    # Override external force if provided
    if args.F_ext is not None:
        config.F_ext = args.F_ext
        config.add_force = True

    # Force herdt method
    config.method = "herdt"

    # Print configuration summary
    print("=" * 60)
    print("Herdt MPC configuration")
    print("=" * 60)
    print(f"  Horizon: {config.horizon}")
    print(f"  dt: {config.dt:.4f} s")
    print(f"  Method: {config.method}")
    print(f"  Strict (unused in Herdt): {config.strict}")
    print(f"  External force: {config.F_ext} N | add_force: {config.add_force}")
    print(f"  alpha: {config.alpha} | beta: {config.beta} | gamma: {config.gamma}")
    print()

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate speed references and state timeline
    speed_generator = SpeedTrajectoryGenerator(config)
    v_x, v_y, states = speed_generator.generate_speed_and_state(save_footsteps=False, output_dir=args.output_dir)
    v_ref = np.stack([v_x, v_y], axis=1)
    state_ref = np.array(states)

    # Initialize controller and initial states
    controller = ZMPController(config)
    x_init = np.zeros((3, 1))
    y_init = np.zeros((3, 1))

    print(f"Running Herdt MPC with {len(v_ref)} steps...")
    com_trajectory, y_hist, foot_hist = controller.generate_com_trajectory(
        x_init=x_init,
        y_init=y_init,
        v_ref=v_ref,
        state_ref=state_ref,
    )

    print("Completed Herdt MPC run.")
    print(f"COM trajectory shape: {com_trajectory.shape}")
    print(f"State history shape: {y_hist.shape}")

    # ---- Plot velocity (v_x) and discrete walking states over time ----
    t = np.arange(len(v_x)) * config.dt

    state_to_level = {
        "STANDING": 0,
        "DOUBLE_SUPPORT": 1,
        "SINGLE_SUPPORT": 2,
    }
    state_levels = [state_to_level.get(s.value, -1) for s in states]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=t,
            y=v_x,
            mode='lines',
            name='v_x (m/s)',
            line=dict(color='blue')
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=state_levels,
            mode='lines',
            name='state',
            line=dict(color='orange', dash='dash')
        ),
        secondary_y=True,
    )

    fig.update_yaxes(
        title_text="v_x (m/s)",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="State",
        tickvals=[0, 1, 2],
        ticktext=["STANDING", "DOUBLE_SUPPORT", "SINGLE_SUPPORT"],
        secondary_y=True,
    )

    fig.update_layout(
        xaxis_title="Time (s)",
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.05,
        ),
        template="plotly_white",
        margin=dict(l=60, r=30, t=40, b=60),
    )

    fig.show()

    # ---- Plot CoM y position with footstep y positions over time ----
    t_com = np.arange(com_trajectory.shape[0]) * config.dt
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=t_com,
            y=com_trajectory[:, 1],
            mode='lines',
            name='CoM y',
            line=dict(color='green')
        )
    )
    # Build discrete footsteps as time-interval rectangles (replace continuous foot_hist plot)
    half_W = config.foot_width / 2.0
    # find change indices
    change_indices = [0]
    for k in range(1, len(foot_hist)):
        if not np.allclose(foot_hist[k], foot_hist[k-1]):
            change_indices.append(k)
    change_indices.append(len(foot_hist))

    shapes = []
    for i in range(len(change_indices) - 1):
        start = change_indices[i]
        end = change_indices[i+1]
        yf = foot_hist[start, 1]
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=t_com[start],
                x1=t_com[end-1] if end-1 < len(t_com) else t_com[-1],
                y0=yf - half_W,
                y1=yf + half_W,
                line=dict(color="black"),
                fillcolor="rgba(0,0,0,0.1)",
                layer="below",
            )
        )

    # Estimated CoP (same computation as in wieber path)
    C_dot_y_hist = np.tensordot(y_hist[:, :, 0], controller.C, axes=([1], [0]))
    fig2.add_trace(
        go.Scatter(
            x=t_com,
            y=C_dot_y_hist,
            mode='lines',
            name='CoP est.',
            line=dict(color='orange', dash='dash')
        )
    )
    # Shapes will be added after traces for consistent layering
    fig2.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Y (m)",
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.05,
        ),
        template="plotly_white",
        shapes=shapes,
        margin=dict(l=60, r=30, t=40, b=60),
    )
    fig2.show()

    # ---- Plot CoM x with footstep x rectangles over time (fig3) ----
    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=t_com,
            y=com_trajectory[:, 0],
            mode='lines',
            name='CoM x',
            line=dict(color='green')
        )
    )

    # Build discrete footsteps as time-interval rectangles for x
    shapes_x = []
    for i in range(len(change_indices) - 1):
        start = change_indices[i]
        end = change_indices[i+1]
        xf = foot_hist[start, 0]
        shapes_x.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=t_com[start],
                x1=t_com[end-1] if end-1 < len(t_com) else t_com[-1],
                y0=xf - half_W,  # reuse half_W; only thickness for visualization
                y1=xf + half_W,
                line=dict(color="black"),
                fillcolor="rgba(0,0,0,0.1)",
                layer="below",
            )
        )

    # Estimated CoP in x direction
    C_dot_x_hist = np.tensordot(y_hist[:, :, 0], controller.C, axes=([1], [0]))  # C acts on state; for x use same C?
    # Actually, C is defined for y in controller; to mimic, reuse same computation on x_hist stored in com_trajectory?
    # The original controller stores only y_hist; we approximate CoP x from com_trajectory[:,0]
    # Here we skip CoP x to avoid misuse; plot only footsteps and CoM x.

    fig3.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="X (m)",
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.05,
        ),
        template="plotly_white",
        shapes=shapes_x,
        margin=dict(l=60, r=30, t=40, b=60),
    )
    fig3.show()

if __name__ == "__main__":
    main()