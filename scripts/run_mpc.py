#!/usr/bin/env python3
"""
Script principal pour lancer la simulation MPC de locomotion bipède.
Permet de configurer les paramètres via ligne de commande ou fichier de configuration.
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
from src.mpc_bipedal.utils import visualize_com_trajectory_3d


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


def create_default_config_file(output_file: str):
    """Crée un fichier de configuration par défaut."""
    default_config = {
        "cop_generator": {
            "ssp_duration": 0.24,
            "dsp_duration": 0.03,
            "standing_duration": 1.0,
            "distance": 2.1,
            "step_length": 0.3,
            "foot_spread": 0.1
        },
        "mpc": {
            "horizon": 150,
            "Q": 1.0,
            "R": 1e-6,
            "h": 0.75,
            "g": 9.81,
            "m": 40.0,
            "F_ext": 400.0,
            "strict": True,
            "add_force": True
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(default_config, f, indent=4)
    
    print(f"Fichier de configuration par défaut créé: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Simulation MPC pour la locomotion bipède',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Utiliser les paramètres par défaut
  python scripts/run_mpc.py

  # Utiliser un fichier de configuration
  python scripts/run_mpc.py --config configs/default.json

  # Surcharger certains paramètres
  python scripts/run_mpc.py --distance 3.0 --step-length 0.4 --horizon 200

  # Créer un fichier de configuration par défaut
  python scripts/run_mpc.py --create-config configs/my_config.json
        """
    )
    
    # Options de configuration
    parser.add_argument('--config', type=str, help='Fichier de configuration JSON')
    parser.add_argument('--create-config', type=str, metavar='FILE', 
                       help='Créer un fichier de configuration par défaut')
    
    # Paramètres CoP Generator
    parser.add_argument('--distance', type=float, help='Distance totale à parcourir (m)')
    parser.add_argument('--step-length', type=float, dest='step_length', help='Longueur de chaque pas (m)')
    parser.add_argument('--foot-spread', type=float, dest='foot_spread', help='Espacement latéral des pieds (m)')
    parser.add_argument('--ssp-duration', type=float, dest='ssp_duration', help='Durée phase simple support (s)')
    parser.add_argument('--dsp-duration', type=float, dest='dsp_duration', help='Durée phase double support (s)')
    parser.add_argument('--standing-duration', type=float, dest='standing_duration', help='Durée phase debout (s)')
    parser.add_argument('--dt', type=float, help='Pas de temps (s)')
    
    # Paramètres MPC
    parser.add_argument('--horizon', type=int, help='Horizon de prédiction')
    parser.add_argument('--Q', type=float, help='Poids du tracking (Q)')
    parser.add_argument('--R', type=float, help='Poids de la régularisation (R)')
    parser.add_argument('--h', type=float, help='Hauteur du COM (m)')
    parser.add_argument('--m', type=float, help='Masse du robot (kg)')
    parser.add_argument('--F-ext', type=float, dest='F_ext', help='Force externe (N)')
    parser.add_argument('--strict', action='store_true', default=None, help='Utiliser les contraintes strictes')
    parser.add_argument('--no-strict', action='store_true', dest='no_strict', help='Ne pas utiliser les contraintes strictes')
    parser.add_argument('--add-force', action='store_true', default=None, help='Ajouter la force externe au moment spécifié')
    parser.add_argument('--no-add-force', action='store_true', dest='no_add_force', help='Ne pas ajouter la force externe')
    
    # Options d'exécution
    parser.add_argument('--no-visualization', action='store_true', help='Ne pas afficher les visualisations')
    parser.add_argument('--save-animation', action='store_true', help='Sauvegarder l\'animation 3D')
    parser.add_argument('--output-dir', type=str, default='results', help='Répertoire de sortie')
    
    args = parser.parse_args()
    
    # Créer un fichier de configuration si demandé
    if args.create_config:
        os.makedirs(os.path.dirname(args.create_config) or '.', exist_ok=True)
        create_default_config_file(args.create_config)
        return
    
    # Charger ou créer les configurations
    if args.config:
        cop_config, mpc_config = load_config_from_json(args.config)
    else:
        # Essayer de charger default.json par défaut s'il existe
        default_config_path = 'configs/default.json'
        if os.path.exists(default_config_path):
            cop_config, mpc_config = load_config_from_json(default_config_path)
        else:
            # Sinon utiliser les valeurs par défaut des classes
            cop_config = CoPGeneratorConfig()
            mpc_config = MPCConfig()
    
    # Synchronize cop_config.dt with mpc_config.dt after loading
    cop_config.dt = mpc_config.dt
    
    # Appliquer les arguments de ligne de commande
    if args.distance is not None:
        cop_config.distance = args.distance
    if args.step_length is not None:
        cop_config.step_length = args.step_length
    if args.foot_spread is not None:
        cop_config.foot_spread = args.foot_spread
    if args.ssp_duration is not None:
        cop_config.ssp_duration = args.ssp_duration
    if args.dsp_duration is not None:
        cop_config.dsp_duration = args.dsp_duration
    if args.standing_duration is not None:
        cop_config.standing_duration = args.standing_duration
    # Handle horizon and dt: horizon takes precedence if both are provided
    if args.horizon is not None:
        mpc_config.horizon = args.horizon
        # Recalculate dt from horizon
        mpc_config.dt = 1.5 / mpc_config.horizon
    elif args.dt is not None:
        # If only dt is provided via command line, recalculate horizon to maintain dt = 1.5 / horizon
        mpc_config.horizon = int(1.5 / args.dt)
        mpc_config.dt = 1.5 / mpc_config.horizon
    else:
        # Recalculate dt from horizon to ensure consistency
        mpc_config.dt = 1.5 / mpc_config.horizon
    
    # Synchronize cop_config.dt with mpc_config.dt
    cop_config.dt = mpc_config.dt
    if args.Q is not None:
        mpc_config.Q = args.Q
    if args.R is not None:
        mpc_config.R = args.R
    if args.h is not None:
        mpc_config.h = args.h
    if args.m is not None:
        mpc_config.m = args.m
    if args.F_ext is not None:
        mpc_config.F_ext = args.F_ext
    # Gérer strict/no-strict : appliquer seulement si explicitement fourni
    if args.strict:
        mpc_config.strict = True
    elif args.no_strict:
        mpc_config.strict = False
    # Gérer add_force/no-add-force : appliquer seulement si explicitement fourni
    if args.add_force:
        mpc_config.add_force = True
    elif args.no_add_force:
        mpc_config.add_force = False
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Configuration MPC pour la locomotion bipède")
    print("=" * 60)
    print("\nCoP Generator Config:")
    print(f"  Distance: {cop_config.distance} m")
    print(f"  Step length: {cop_config.step_length} m")
    print(f"  Foot spread: {cop_config.foot_spread} m")
    print(f"  dt: {cop_config.dt} s")
    print("\nMPC Config:")
    print(f"  Horizon: {mpc_config.horizon}")
    print(f"  Q: {mpc_config.Q}")
    print(f"  R: {mpc_config.R}")
    print(f"  h: {mpc_config.h} m")
    print(f"  m: {mpc_config.m} kg")
    print(f"  F_ext: {mpc_config.F_ext} N")
    print(f"  Strict: {mpc_config.strict}")
    print(f"  Add force: {mpc_config.add_force}")
    print("=" * 60)
    print()
    
    # Générer la trajectoire CoP
    print("Génération de la trajectoire CoP...")
    cop_generator = CoPGenerator(cop_config)
    z_max, z_min = cop_generator.generate_cop_trajectory(output_dir=args.output_dir)
    t = np.arange(z_max.shape[0]) * cop_config.dt
    print(f"✓ Trajectoire CoP générée ({len(t)} pas de temps)\n")
    
    # Générer la trajectoire COM
    print("Génération de la trajectoire COM...")
    controller = ZMPController(mpc_config)
    x_init = np.array([[0., 0., 0.]]).T
    y_init = np.array([[0., 0., 0.]]).T
    com_trajectory, y_hist = controller.generate_com_trajectory(x_init, y_init, z_max, z_min)
    print(f"✓ Trajectoire COM générée\n")
    
    # Calculer l'estimation ZMP
    C_dot_y_hist = np.tensordot(y_hist[:, :, 0], controller.C, axes=([1], [0]))
    print(f"Shape de la trajectoire COM: {com_trajectory.shape}\n")
    
    # Visualisation ZMP (toujours affichée comme dans mpc.py original, sauf si --no-visualization)
    if not args.no_visualization:
        print("Affichage du graphique ZMP (CoP Limits Over Time)...")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=t, y=z_max[:, 1],
            mode='lines',
            name='z_max',
            line=dict(color='red', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=t, y=z_min[:, 1],
            mode='lines',
            name='z_min',
            line=dict(color='blue', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=t, y=C_dot_y_hist,
            mode='lines',
            name='Estimation de z',
            line=dict(color='green', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=t, y=com_trajectory[:, 1],
            mode='lines',
            name='com',
            line=dict(color='black')
        ))
        
        fig.update_layout(
            title="CoP Limits Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Y Axis",
            legend=dict(x=0, y=1),
            template="plotly_white"
        )
        
        fig.show()
    
    # Visualisation 3D de la trajectoire du COM (toujours affichée comme dans mpc.py original)
    if not args.no_visualization:
        print("Affichage de la visualisation 3D...")
        visualize_com_trajectory_3d(
            com_trajectory, 
            h=mpc_config.h,
            show_sphere=True, 
            save_animation=args.save_animation,
            output_file=os.path.join(args.output_dir, 'com_trajectory_3d.gif')
        )
    
    print("\n✓ Simulation terminée avec succès!")


if __name__ == "__main__":
    main()

