#!/usr/bin/env python3
"""
Script to compare resistance to external forces between strict and non-strict MPC methods.
Plots the ZMP bounds and estimated ZMP trajectories for both methods.
"""

import sys
import os
import argparse
import json
import numpy as np
import plotly.graph_objects as go

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.mpc_bipedal.config import CoPGeneratorConfig, MPCConfig
from src.mpc_bipedal.generators import CoPGenerator
from src.mpc_bipedal.controllers import ZMPController


def load_config_from_json(config_file: str):
    """Charge une configuration depuis un fichier JSON."""
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    cop_dict = config_dict.get('cop_generator', {}).copy()
    # Remove dt from cop_generator dict - it will be synchronized from mpc_config
    cop_dict.pop('dt', None)
    cop_config = CoPGeneratorConfig(**cop_dict)
    
    mpc_dict = config_dict.get('mpc', {}).copy()
    
    # Handle dt: horizon takes precedence. If only dt is provided, recalculate horizon from it.
    # dt will always be calculated from horizon in __post_init__, so we remove dt from dict.
    if 'dt' in mpc_dict:
        dt_value = mpc_dict.pop('dt')
        if 'horizon' not in mpc_dict:
            # Recalculate horizon from dt to maintain dt = 1.5 / horizon
            mpc_dict['horizon'] = int(1.5 / dt_value)
        # If both dt and horizon are provided, horizon takes precedence (dt will be recalculated)
    
    mpc_config = MPCConfig(**mpc_dict)
    
    # Synchronize cop_config.dt with mpc_config.dt (dt is calculated from horizon)
    cop_config.dt = mpc_config.dt
    
    return cop_config, mpc_config


def main():
    """Main function to compare resistance between methods."""
    parser = argparse.ArgumentParser(
        description='Compare resistance to external forces between strict and non-strict MPC methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Utiliser la force externe par défaut (400 N)
  python scripts/run_compare_resistance.py

  # Spécifier une force externe personnalisée
  python scripts/run_compare_resistance.py --F-ext 600.0

  # Utiliser un fichier de configuration
  python scripts/run_compare_resistance.py --config configs/default.json --F-ext 500.0
        """
    )
    
    parser.add_argument('--config', type=str, help='Fichier de configuration JSON')
    parser.add_argument('--F-ext', type=float, dest='F_ext', help='Force externe (N)')
    parser.add_argument('--output-dir', type=str, default='results', help='Répertoire de sortie')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Comparison: Strict vs Non-Strict MPC Methods")
    print("=" * 60)
    print()
    
    # Load or create configurations
    if args.config:
        cop_config, mpc_config_base = load_config_from_json(args.config)
    else:
        # Try to load default.json if it exists
        default_config_path = 'configs/default.json'
        if os.path.exists(default_config_path):
            cop_config, mpc_config_base = load_config_from_json(default_config_path)
        else:
            # Use default values
            cop_config = CoPGeneratorConfig()
            mpc_config_base = MPCConfig()
            # Synchronize cop_config.dt with mpc_config_base.dt
            cop_config.dt = mpc_config_base.dt
    
    # Synchronize cop_config.dt with mpc_config_base.dt
    cop_config.dt = mpc_config_base.dt
    
    # Override F_ext if provided
    if args.F_ext is not None:
        mpc_config_base.F_ext = args.F_ext
    
    # Create configurations for both methods
    config_strict = MPCConfig(
        horizon=mpc_config_base.horizon,
        Q=mpc_config_base.Q,
        R=mpc_config_base.R,
        dt=mpc_config_base.dt,
        h=mpc_config_base.h,
        g=mpc_config_base.g,
        m=mpc_config_base.m,
        F_ext=mpc_config_base.F_ext,
        strict=True,
        add_force=True  # Enable force application
    )
    
    config_non_strict = MPCConfig(
        horizon=mpc_config_base.horizon,
        Q=mpc_config_base.Q,
        R=mpc_config_base.R,
        dt=mpc_config_base.dt,
        h=mpc_config_base.h,
        g=mpc_config_base.g,
        m=mpc_config_base.m,
        F_ext=mpc_config_base.F_ext,
        strict=False,
        add_force=True  # Enable force application
    )
    
    print("Configuration:")
    print(f"  Horizon: {config_strict.horizon}")
    print(f"  F_ext: {config_strict.F_ext} N")
    print(f"  dt: {config_strict.dt} s")
    print("=" * 60)
    print()
    
    # Generate CoP trajectory (same for both methods)
    print("Génération de la trajectoire CoP...")
    cop_generator = CoPGenerator(cop_config)
    z_max, z_min = cop_generator.generate_cop_trajectory(output_dir=args.output_dir, save_footsteps=False)
    t = np.arange(z_max.shape[0]) * cop_config.dt
    print(f"✓ Trajectoire CoP générée ({len(t)} pas de temps)\n")
    
    # Generate COM trajectories for both methods
    print("Génération de la trajectoire COM (Strict method)...")
    controller_strict = ZMPController(config_strict)
    x_init = np.array([[0., 0., 0.]]).T
    y_init = np.array([[0., 0., 0.]]).T
    com_trajectory_strict, y_hist_strict = controller_strict.generate_com_trajectory(
        x_init, y_init, z_max, z_min
    )
    print(f"✓ Trajectoire COM générée (Strict)\n")
    
    print("Génération de la trajectoire COM (Non-strict method)...")
    controller_non_strict = ZMPController(config_non_strict)
    com_trajectory_non_strict, y_hist_non_strict = controller_non_strict.generate_com_trajectory(
        x_init, y_init, z_max, z_min
    )
    print(f"✓ Trajectoire COM générée (Non-strict)\n")
    
    # Calculate estimated ZMP (C @ y) for both methods
    C_dot_y_strict = np.tensordot(y_hist_strict[:, :, 0], controller_strict.C, axes=([1], [0]))
    C_dot_y_non_strict = np.tensordot(y_hist_non_strict[:, :, 0], controller_non_strict.C, axes=([1], [0]))
    
    print("=" * 60)
    print("Plotting results...")
    print("=" * 60)
    
    # Create the plot with Plotly
    fig = go.Figure()
    
    # Add z_max and z_min bounds (same for both methods)
    fig.add_trace(go.Scatter(
        x=t, y=z_max[:, 1],
        mode='lines',
        name='z_max',
        line=dict(color='red', dash='dash', width=2),
        hovertemplate='Time: %{x:.3f} s<br>z_max: %{y:.4f} m<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=t, y=z_min[:, 1],
        mode='lines',
        name='z_min',
        line=dict(color='blue', dash='dash', width=2),
        hovertemplate='Time: %{x:.3f} s<br>z_min: %{y:.4f} m<extra></extra>'
    ))
    
    # Add estimated ZMP for strict method
    fig.add_trace(go.Scatter(
        x=t, y=C_dot_y_strict,
        mode='lines',
        name='Estimated ZMP (Strict)',
        line=dict(color='green', width=2),
        hovertemplate='Time: %{x:.3f} s<br>ZMP (Strict): %{y:.4f} m<extra></extra>'
    ))
    
    # Add estimated ZMP for non-strict method
    fig.add_trace(go.Scatter(
        x=t, y=C_dot_y_non_strict,
        mode='lines',
        name='Estimated ZMP (Non-strict)',
        line=dict(color='orange', width=2),
        hovertemplate='Time: %{x:.3f} s<br>ZMP (Non-strict): %{y:.4f} m<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'ZMP Trajectory Comparison (F_ext = {config_strict.F_ext} N)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': 'black'}
        },
        xaxis_title='Time (s)',
        yaxis_title='Y Axis (m)',
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        hovermode='x unified',
        template='plotly_white',
        width=1200,
        height=700,
        margin=dict(l=60, r=20, t=80, b=50)
    )
    
    # Save the plot as PNG
    os.makedirs(args.output_dir, exist_ok=True)
    output_file_png = os.path.join(args.output_dir, 'resistance_comparison.png')
    
    try:
        fig.write_image(output_file_png, width=1200, height=700, scale=2)
        print(f"✓ Plot saved to: {output_file_png}")
    except Exception as e:
        print(f"✗ Error: Could not save PNG (kaleido may not be installed): {e}")
        print("  Please install kaleido: pip install kaleido")
    
    # Show the plot
    fig.show()
    
    print("✓ Comparison complete!")


if __name__ == "__main__":
    main()

