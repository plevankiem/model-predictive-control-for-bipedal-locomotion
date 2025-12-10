#!/usr/bin/env python3
"""
Script to compare runtime performance of strict vs non-strict MPC methods
for different horizon values (N).
"""

import sys
import os
import time
import numpy as np
import plotly.graph_objects as go

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.mpc_bipedal.config import MPCConfig
from src.mpc_bipedal.controllers import ZMPController
from src.mpc_bipedal.generators.cop_generator import State


def _prepare_dummy_bounds(controller: ZMPController, n_steps: int = 100):
    """Create dummy ZMP bounds large enough for any horizon slice."""
    z_max = np.ones((n_steps, 2)) * 0.1
    z_min = np.ones((n_steps, 2)) * -0.1
    z_max_extended = np.vstack([
        z_max,
        np.tile(z_max[-1:, :], (controller.config.horizon, 1))
    ])
    z_min_extended = np.vstack([
        z_min,
        np.tile(z_min[-1:, :], (controller.config.horizon, 1))
    ])
    return z_max_extended, z_min_extended


def measure_wieber_iteration_time(config: MPCConfig, n_steps: int = 100) -> float:
    """Average time for one Wieber MPC step (per axis) using predict_wieber_axis."""
    controller = ZMPController(config)
    x_init = np.zeros((3, 1))
    y_init = np.zeros((3, 1))
    z_max_ext, z_min_ext = _prepare_dummy_bounds(controller, n_steps)
    preview_n_steps = controller.config.horizon

    def _run_once(x_state, y_state):
        x_state = controller.predict_wieber_axis(
            x_state,
            preview_n_steps,
            z_max_ext[1:1+preview_n_steps, 0:1],
            z_min_ext[1:1+preview_n_steps, 0:1],
        )
        y_state = controller.predict_wieber_axis(
            y_state,
            preview_n_steps,
            z_max_ext[1:1+preview_n_steps, 1:2],
            z_min_ext[1:1+preview_n_steps, 1:2],
        )
        return x_state, y_state

    # Warm-up to initialize solver
    x_state, y_state = x_init.copy(), y_init.copy()
    for _ in range(3):
        x_state, y_state = _run_once(x_state, y_state)

    # Timed iterations
    n_iterations = 10
    times = []
    for _ in range(n_iterations):
        x_state, y_state = x_init.copy(), y_init.copy()
        start = time.perf_counter()
        _run_once(x_state, y_state)
        times.append(time.perf_counter() - start)

    return float(np.mean(times))


def measure_herdt_iteration_time(config: MPCConfig, n_steps: int = 100) -> float:
    """Average time for one Herdt MPC step using predict_herdt_joint."""
    controller = ZMPController(config)
    x_init = np.zeros((3, 1))
    y_init = np.zeros((3, 1))

    # Dummy references and states; pad to support horizon slicing
    v_ref = np.zeros((n_steps, 2))
    state_ref = np.array([State.STANDING] * n_steps)
    v_pad = np.repeat(v_ref[-1:], controller.config.horizon, axis=0)
    s_pad = np.repeat(state_ref[-1:], controller.config.horizon, axis=0)
    v_ref_ext = np.vstack([v_ref, v_pad])
    state_ref_ext = np.concatenate([state_ref, s_pad])
    nb_steps_to_next_state = controller.find_nb_steps(state_ref_ext)

    # Initial foot positions (same as controller's Herdt path)
    x_fc = np.array([[0.0]])
    y_fc = np.array([[controller.config.foot_spread]])
    x_airc = x_fc.copy()
    y_airc = y_fc.copy()
    foot_side = "right"
    current_state = state_ref_ext[0]
    preview_n_steps = controller.config.horizon

    def _run_once():
        controller.predict_herdt_joint(
            x_init,
            y_init,
            v_ref_ext[1:1+preview_n_steps, :],
            x_fc,
            y_fc,
            current_state,
            state_ref_ext[1:1+preview_n_steps],
            preview_n_steps,
            nb_steps_to_next_state[0],
            x_airc,
            y_airc,
            foot_side,
        )

    # Warm-up
    for _ in range(3):
        _run_once()

    # Timed iterations
    n_iterations = 10
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _run_once()
        times.append(time.perf_counter() - start)

    return float(np.median(times))


def main():
    """Main function to run the runtime comparison."""
    print("=" * 60)
    print("Runtime Comparison: Wieber (strict vs non-strict) vs Herdt")
    print("=" * 60)
    print()
    
    # Horizon values to test (N)
    horizon_values = np.arange(10, 301, 10)  # From 10 to 300, step 10
    
    # Storage for results
    strict_times = []
    non_strict_times = []
    herdt_times = []
    
    print(f"Testing {len(horizon_values)} horizon values from {horizon_values[0]} to {horizon_values[-1]}...")
    print()
    
    for horizon in horizon_values:
        print(f"Testing horizon N = {horizon}...", end=" ", flush=True)
        
        try:
            # Wieber - strict
            config_strict = MPCConfig(
                horizon=horizon,
                strict=True,
                add_force=False,
                method="wieber",
            )
            time_strict = measure_wieber_iteration_time(config_strict)
            strict_times.append(time_strict)

            # Wieber - non-strict (analytical)
            config_non_strict = MPCConfig(
                horizon=horizon,
                strict=False,
                add_force=False,
                method="wieber",
            )
            time_non_strict = measure_wieber_iteration_time(config_non_strict)
            non_strict_times.append(time_non_strict)

            # Herdt
            config_herdt = MPCConfig(
                horizon=horizon,
                strict=True,   # strict flag unused in Herdt path, keep default
                add_force=False,
                method="herdt",
            )
            time_herdt = measure_herdt_iteration_time(config_herdt)
            herdt_times.append(time_herdt)

            print(
                f"✓ Strict: {time_strict*1000:.3f}ms, "
                f"Non-strict: {time_non_strict*1000:.3f}ms, "
                f"Herdt: {time_herdt*1000:.3f}ms"
            )
        except Exception as e:
            print(f"✗ Error: {e}")
            strict_times.append(np.nan)
            non_strict_times.append(np.nan)
            herdt_times.append(np.nan)
    
    print()
    print("=" * 60)
    print("Plotting results...")
    print("=" * 60)
    
    # Filter out NaN values for plotting
    valid_indices = ~(np.isnan(strict_times) | np.isnan(non_strict_times) | np.isnan(herdt_times))
    horizon_valid = horizon_values[valid_indices]
    strict_valid = np.array(strict_times)[valid_indices] * 1000  # Convert to ms
    non_strict_valid = np.array(non_strict_times)[valid_indices] * 1000  # Convert to ms
    herdt_valid = np.array(herdt_times)[valid_indices] * 1000  # Convert to ms
    
    # Calculate dt values for valid horizons (dt = 1.5 / horizon) and convert to ms
    dt_valid = (1.5 / horizon_valid) * 1000  # Convert to ms
    
    # Create the plot with Plotly
    fig = go.Figure()
    
    # Add strict method trace
    fig.add_trace(go.Scatter(
        x=horizon_valid,
        y=strict_valid,
        mode='lines',
        name='Strict (QP)',
        line=dict(color='#1f77b4', width=3),
        hovertemplate='Horizon: %{x}<br>Time: %{y:.3f} ms<extra></extra>'
    ))
    
    # Add non-strict method trace
    fig.add_trace(go.Scatter(
        x=horizon_valid,
        y=non_strict_valid,
        mode='lines',
        name='Non-strict (Analytical)',
        line=dict(color='#ff7f0e', width=3),
        hovertemplate='Horizon: %{x}<br>Time: %{y:.3f} ms<extra></extra>'
    ))

    # Add Herdt method trace
    fig.add_trace(go.Scatter(
        x=horizon_valid,
        y=herdt_valid,
        mode='lines',
        name='Herdt (Joint QP)',
        line=dict(color='#2ca02c', width=3),
        hovertemplate='Horizon: %{x}<br>Time: %{y:.3f} ms<extra></extra>'
    ))
    
    # Add dt trace on same y-axis
    fig.add_trace(go.Scatter(
        x=horizon_valid,
        y=dt_valid,
        mode='lines',
        name='dt (ms)',
        line=dict(color='#7f7f7f', width=2, dash='dash'),
        hovertemplate='Horizon: %{x}<br>dt: %{y:.3f} ms<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Runtime Comparison: Wieber (strict/non-strict) vs Herdt',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': 'black'}
        },
        xaxis_title='Horizon N',
        yaxis_title='Time per iteration (ms)',
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=600,
        margin=dict(l=60, r=20, t=60, b=50)
    )
    
    # Save the plot as PNG
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_file_png = os.path.join(output_dir, 'runtime_comparison.png')
    
    # Save as static PNG (requires kaleido)
    try:
        fig.write_image(output_file_png, width=1000, height=600, scale=2)
        print(f"✓ Plot saved to: {output_file_png}")
    except Exception as e:
        print(f"✗ Error: Could not save PNG (kaleido may not be installed): {e}")
        print("  Please install kaleido: pip install kaleido")
    
    # Print summary statistics
    print()
    print("=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Strict method:")
    print(f"  Mean: {np.nanmean(strict_times)*1000:.3f} ms")
    print(f"  Std:  {np.nanstd(strict_times)*1000:.3f} ms")
    print(f"  Min:  {np.nanmin(strict_times)*1000:.3f} ms (N={horizon_values[np.nanargmin(strict_times)]})")
    print(f"  Max:  {np.nanmax(strict_times)*1000:.3f} ms (N={horizon_values[np.nanargmax(strict_times)]})")
    print()
    print(f"Non-strict method:")
    print(f"  Mean: {np.nanmean(non_strict_times)*1000:.3f} ms")
    print(f"  Std:  {np.nanstd(non_strict_times)*1000:.3f} ms")
    print(f"  Min:  {np.nanmin(non_strict_times)*1000:.3f} ms (N={horizon_values[np.nanargmin(non_strict_times)]})")
    print(f"  Max:  {np.nanmax(non_strict_times)*1000:.3f} ms (N={horizon_values[np.nanargmax(non_strict_times)]})")
    print()
    print(f"Herdt method:")
    print(f"  Mean: {np.nanmean(herdt_times)*1000:.3f} ms")
    print(f"  Std:  {np.nanstd(herdt_times)*1000:.3f} ms")
    print(f"  Min:  {np.nanmin(herdt_times)*1000:.3f} ms (N={horizon_values[np.nanargmin(herdt_times)]})")
    print(f"  Max:  {np.nanmax(herdt_times)*1000:.3f} ms (N={horizon_values[np.nanargmax(herdt_times)]})")
    print()
    
    # Show the plot
    fig.show()
    
    print("✓ Comparison complete!")


if __name__ == "__main__":
    main()

