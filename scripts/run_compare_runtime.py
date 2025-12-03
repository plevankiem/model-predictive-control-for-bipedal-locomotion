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


def measure_iteration_time(controller: ZMPController, n_steps: int = 100):
    """
    Measure the time for one iteration step in the COM trajectory generation.
    
    Args:
        controller: ZMPController instance
        n_steps: Number of steps for the trajectory (used to create dummy bounds)
    
    Returns:
        Average time per iteration in seconds
    """
    # Create dummy initial states
    x_init = np.array([[0., 0., 0.]]).T
    y_init = np.array([[0., 0., 0.]]).T
    
    # Create dummy z_max and z_min bounds
    z_max = np.ones((n_steps, 2)) * 0.1
    z_min = np.ones((n_steps, 2)) * -0.1
    
    # Extend bounds for preview
    z_max_extended = np.vstack([
        z_max,
        np.tile(z_max[-1:, :], (controller.config.horizon, 1))
    ])
    z_min_extended = np.vstack([
        z_min,
        np.tile(z_min[-1:, :], (controller.config.horizon, 1))
    ])
    
    # Warm up: run a few iterations to initialize solvers
    x_state = x_init.copy()
    y_state = y_init.copy()
    for _ in range(3):
        i = 0
        preview_n_steps = controller.config.horizon
        x_state = controller.predict(
            x_state, preview_n_steps,
            z_max_extended[i+1:i+1+preview_n_steps, 0:1],
            z_min_extended[i+1:i+1+preview_n_steps, 0:1]
        )
        y_state = controller.predict(
            y_state, preview_n_steps,
            z_max_extended[i+1:i+1+preview_n_steps, 1:2],
            z_min_extended[i+1:i+1+preview_n_steps, 1:2]
        )
    
    # Measure time for one iteration (2 predict calls)
    n_iterations = 10  # Average over multiple iterations for better accuracy
    times = []
    
    for _ in range(n_iterations):
        x_state = x_init.copy()
        y_state = y_init.copy()
        i = 0
        
        start_time = time.perf_counter()
        
        preview_n_steps = controller.config.horizon
        x_state = controller.predict(
            x_state, preview_n_steps,
            z_max_extended[i+1:i+1+preview_n_steps, 0:1],
            z_min_extended[i+1:i+1+preview_n_steps, 0:1]
        )
        y_state = controller.predict(
            y_state, preview_n_steps,
            z_max_extended[i+1:i+1+preview_n_steps, 1:2],
            z_min_extended[i+1:i+1+preview_n_steps, 1:2]
        )
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return np.mean(times)


def main():
    """Main function to run the runtime comparison."""
    print("=" * 60)
    print("Runtime Comparison: Strict vs Non-Strict MPC Methods")
    print("=" * 60)
    print()
    
    # Horizon values to test (N)
    horizon_values = np.arange(10, 301, 10)  # From 10 to 300, step 10
    
    # Storage for results
    strict_times = []
    non_strict_times = []
    
    print(f"Testing {len(horizon_values)} horizon values from {horizon_values[0]} to {horizon_values[-1]}...")
    print()
    
    for horizon in horizon_values:
        print(f"Testing horizon N = {horizon}...", end=" ", flush=True)
        
        # Create config with current horizon
        # dt will be calculated as 1.5 / horizon
        config_strict = MPCConfig(
            horizon=horizon,
            strict=True,
            add_force=False  # Disable force for cleaner timing
        )
        config_non_strict = MPCConfig(
            horizon=horizon,
            strict=False,
            add_force=False
        )
        
        # Create controllers
        controller_strict = ZMPController(config_strict)
        controller_non_strict = ZMPController(config_non_strict)
        
        # Measure times
        try:
            time_strict = measure_iteration_time(controller_strict)
            strict_times.append(time_strict)
            
            time_non_strict = measure_iteration_time(controller_non_strict)
            non_strict_times.append(time_non_strict)
            
            print(f"✓ Strict: {time_strict*1000:.3f}ms, Non-strict: {time_non_strict*1000:.3f}ms")
        except Exception as e:
            print(f"✗ Error: {e}")
            strict_times.append(np.nan)
            non_strict_times.append(np.nan)
    
    print()
    print("=" * 60)
    print("Plotting results...")
    print("=" * 60)
    
    # Filter out NaN values for plotting
    valid_indices = ~(np.isnan(strict_times) | np.isnan(non_strict_times))
    horizon_valid = horizon_values[valid_indices]
    strict_valid = np.array(strict_times)[valid_indices] * 1000  # Convert to ms
    non_strict_valid = np.array(non_strict_times)[valid_indices] * 1000  # Convert to ms
    
    # Calculate dt values for valid horizons (dt = 1.5 / horizon) and convert to ms
    dt_valid = (1.5 / horizon_valid) * 1000  # Convert to ms
    
    # Create the plot with Plotly
    fig = go.Figure()
    
    # Add strict method trace
    fig.add_trace(go.Scatter(
        x=horizon_valid,
        y=strict_valid,
        mode='lines+markers',
        name='Strict (QP)',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6, symbol='circle'),
        hovertemplate='Horizon: %{x}<br>Time: %{y:.3f} ms<extra></extra>'
    ))
    
    # Add non-strict method trace
    fig.add_trace(go.Scatter(
        x=horizon_valid,
        y=non_strict_valid,
        mode='lines+markers',
        name='Non-strict (Analytical)',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=6, symbol='square'),
        hovertemplate='Horizon: %{x}<br>Time: %{y:.3f} ms<extra></extra>'
    ))
    
    # Add dt trace on same y-axis
    fig.add_trace(go.Scatter(
        x=horizon_valid,
        y=dt_valid,
        mode='lines+markers',
        name='dt (ms)',
        line=dict(color='#2ca02c', width=2, dash='dash'),
        marker=dict(size=6, symbol='diamond'),
        hovertemplate='Horizon: %{x}<br>dt: %{y:.3f} ms<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Runtime Comparison: Strict vs Non-Strict MPC Methods',
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
    
    # Show the plot
    fig.show()
    
    print("✓ Comparison complete!")


if __name__ == "__main__":
    main()

