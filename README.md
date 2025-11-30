# Model Predictive Control for Bipedal Locomotion

Ce projet implémente un contrôleur prédictif (MPC) pour la locomotion bipède basé sur le modèle du pendule inversé linéaire (LIPM).

## 📋 Table des matières

- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Configuration](#configuration)
- [Exemples](#exemples)
- [Documentation](#documentation)

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## 💻 Utilisation

### Utilisation de base

Le script principal `scripts/run_mpc.py` permet de lancer la simulation avec différentes options :

```bash
# Utiliser les paramètres par défaut
python scripts/run_mpc.py

# Ou directement (si exécutable)
./scripts/run_mpc.py
```

### Utilisation avec fichier de configuration

```bash
# Utiliser une configuration prédéfinie
python scripts/run_mpc.py --config configs/default.json

# Utiliser une configuration de marche rapide
python scripts/run_mpc.py --config configs/fast_walk.json

# Utiliser une configuration conservative
python scripts/run_mpc.py --config configs/conservative.json
```

### Personnalisation des paramètres

Vous pouvez surcharger les paramètres via la ligne de commande :

```bash
# Changer la distance et la longueur de pas
python scripts/run_mpc.py --distance 3.0 --step-length 0.4

# Ajuster les paramètres MPC
python scripts/run_mpc.py --horizon 200 --Q 2.0 --R 1e-5

# Modifier les paramètres physiques
python scripts/run_mpc.py --h 0.80 --m 50.0 --F-ext 500.0
```

### Options disponibles

#### Paramètres CoP Generator
- `--distance`: Distance totale à parcourir (m)
- `--step-length`: Longueur de chaque pas (m)
- `--foot-spread`: Espacement latéral des pieds (m)
- `--ssp-duration`: Durée phase simple support (s)
- `--dsp-duration`: Durée phase double support (s)
- `--standing-duration`: Durée phase debout (s)
- `--dt`: Pas de temps (s)

#### Paramètres MPC
- `--horizon`: Horizon de prédiction
- `--Q`: Poids du tracking
- `--R`: Poids de la régularisation
- `--h`: Hauteur du COM (m)
- `--m`: Masse du robot (kg)
- `--F-ext`: Force externe (N)
- `--strict`: Utiliser les contraintes strictes
- `--no-strict`: Ne pas utiliser les contraintes strictes

#### Options d'affichage
- `--no-visualization`: Ne pas afficher les visualisations
- `--save-animation`: Sauvegarder l'animation 3D
- `--output-dir`: Répertoire de sortie (défaut: `results`)
- `--plot-zmp`: Afficher le graphique ZMP

### Créer un fichier de configuration personnalisé

```bash
# Créer un fichier de configuration par défaut
python scripts/run_mpc.py --create-config configs/my_config.json
```

Ensuite, modifiez `configs/my_config.json` selon vos besoins.

## 📁 Structure du projet

```
.
├── src/
│   └── mpc_bipedal/          # Package principal
│       ├── __init__.py
│       ├── config.py         # Classes de configuration
│       ├── models/           # Modèles (LIPM)
│       │   ├── __init__.py
│       │   └── lipm_model.py
│       ├── generators/       # Générateurs (footsteps, CoP)
│       │   ├── __init__.py
│       │   ├── footstep_generator.py
│       │   └── cop_generator.py
│       ├── controllers/      # Contrôleurs (ZMP Controller)
│       │   ├── __init__.py
│       │   └── zmp_controller.py
│       └── utils/            # Utilitaires (visualisation)
│           ├── __init__.py
│           └── visualization.py
├── scripts/
│   └── run_mpc.py           # Script principal d'exécution
├── configs/                  # Fichiers de configuration
│   ├── default.json
│   ├── fast_walk.json
│   └── conservative.json
├── results/                  # Résultats et visualisations
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Configuration

### Format de fichier de configuration

Les fichiers de configuration sont au format JSON :

```json
{
    "cop_generator": {
        "ssp_duration": 0.24,
        "dsp_duration": 0.03,
        "standing_duration": 1.0,
        "dt": 0.01,
        "distance": 2.1,
        "step_length": 0.3,
        "foot_spread": 0.1
    },
    "mpc": {
        "horizon": 150,
        "Q": 1.0,
        "R": 1e-6,
        "dt": 0.01,
        "h": 0.75,
        "g": 9.81,
        "m": 40.0,
        "F_ext": 400.0,
        "strict": true
    }
}
```

### Paramètres importants

#### CoP Generator
- **distance**: Distance totale que le robot doit parcourir (m)
- **step_length**: Longueur moyenne de chaque pas (m)
- **foot_spread**: Distance latérale entre les pieds (m)
- **dt**: Pas de temps pour la simulation (s)
- **ssp_duration**: Durée de la phase de simple support (s)
- **dsp_duration**: Durée de la phase de double support (s)
- **standing_duration**: Durée de la phase debout initiale (s)

#### MPC
- **horizon**: Nombre de pas de temps dans l'horizon de prédiction
- **Q**: Poids du terme de tracking (erreur par rapport à la référence ZMP)
- **R**: Poids du terme de régularisation (pénalise les grandes accélérations)
- **h**: Hauteur du centre de masse (COM) (m)
- **m**: Masse du robot (kg)
- **F_ext**: Force externe appliquée à mi-parcours (N)
- **strict**: Utiliser les contraintes strictes (QP) ou solution analytique

## 📊 Exemples

### Exemple 1 : Simulation par défaut

```bash
python scripts/run_mpc.py
```

### Exemple 2 : Marche longue distance

```bash
python scripts/run_mpc.py --distance 5.0 --step-length 0.35 --horizon 250
```

### Exemple 3 : Robot plus lourd avec pas plus courts

```bash
python scripts/run_mpc.py --m 60.0 --step-length 0.25 --foot-spread 0.12
```

### Exemple 4 : Simulation sans visualisation (pour débogage)

```bash
python scripts/run_mpc.py --no-visualization --output-dir results/debug
```

### Exemple 5 : Sauvegarder l'animation

```bash
python scripts/run_mpc.py --save-animation --output-dir results/animations
```

## 📚 Documentation

### Classes principales

#### `CoPGenerator`
Génère une trajectoire viable du Centre de Pression (CoP) à partir des footsteps.

#### `ZMPController`
Implémente le contrôleur ZMP basé sur le MPC pour générer la trajectoire du Centre de Masse (COM).

#### `LIPMModel`
Modèle du pendule inversé linéaire (Linear Inverted Pendulum Model).

### Visualisations

Le projet génère plusieurs visualisations :

1. **Graphique des footsteps** (`results/footsteps.png`) : Vue de dessus des points de contact
2. **Graphique ZMP** : Évolution temporelle des limites ZMP et de la trajectoire COM (si `--plot-zmp`)
3. **Visualisation 3D** : Trajectoire 3D du COM avec animation optionnelle

## 🔧 Développement

Pour contribuer au projet :

1. Installer les dépendances de développement
2. Créer une branche pour votre fonctionnalité
3. Suivre les conventions de code Python (PEP 8)
4. Ajouter des tests si nécessaire

## 📝 Notes

- Le solveur QP utilise `cvxpy` avec le solveur `OSQP`
- Les visualisations utilisent `plotly` et `matplotlib`
- La simulation peut prendre quelques secondes selon l'horizon et le nombre de pas

## 📄 Licence

[Spécifier votre licence ici]

## 👤 Auteur

[Votre nom]

