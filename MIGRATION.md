# Guide de migration

Ce projet a été réorganisé pour une meilleure architecture. Voici comment migrer votre code.

## Nouvelle structure

Le code a été réorganisé dans le package `src/mpc_bipedal/` avec une structure modulaire :

- `src/mpc_bipedal/config.py` - Classes de configuration
- `src/mpc_bipedal/models/` - Modèles (LIPM)
- `src/mpc_bipedal/generators/` - Générateurs de trajectoires
- `src/mpc_bipedal/controllers/` - Contrôleurs MPC
- `src/mpc_bipedal/utils/` - Utilitaires (visualisation)

## Migration des imports

### Ancien code
```python
from config import MPCConfig, CoPGeneratorConfig
from model import LIPMModel
from pynamoid import generate_footsteps
from visualize import visualize_com_trajectory_3d
```

### Nouveau code
```python
from mpc_bipedal.config import MPCConfig, CoPGeneratorConfig, ModelConfig
from mpc_bipedal.models import LIPMModel
from mpc_bipedal.generators import generate_footsteps, CoPGenerator
from mpc_bipedal.utils import visualize_com_trajectory_3d
```

## Utilisation du script CLI

Au lieu d'exécuter directement `mpc.py`, utilisez maintenant :

```bash
python scripts/run_mpc.py
```

Voir le README principal pour toutes les options disponibles.

## Fichiers obsolètes

Les fichiers suivants à la racine sont maintenant obsolètes mais sont conservés pour compatibilité :
- `mpc.py` - Utilisez `scripts/run_mpc.py` à la place
- `model.py` - Déplacé vers `src/mpc_bipedal/models/lipm_model.py`
- `pynamoid.py` - Déplacé vers `src/mpc_bipedal/generators/footstep_generator.py`
- `visualize.py` - Déplacé vers `src/mpc_bipedal/utils/visualization.py`
- `config.py` (racine) - Déplacé vers `src/mpc_bipedal/config.py` (inclut maintenant ModelConfig)

Ces fichiers peuvent être supprimés après migration complète.

