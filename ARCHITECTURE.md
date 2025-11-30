# Architecture du projet

## Vue d'ensemble

Ce projet implémente un contrôleur prédictif (MPC) pour la locomotion bipède basé sur le modèle du pendule inversé linéaire (LIPM).

## Structure des modules

### `src/mpc_bipedal/config.py`
Contient les classes de configuration utilisées dans tout le projet :
- `CoPGeneratorConfig` : Configuration pour la génération de la trajectoire CoP
- `ModelConfig` : Configuration pour le modèle LIPM
- `MPCConfig` : Configuration pour le contrôleur MPC

### `src/mpc_bipedal/models/`
**lipm_model.py** : Implémentation du modèle Linear Inverted Pendulum Model (LIPM)
- État du système : position, vitesse, accélération
- Matrices de dynamique : A, B, C
- Méthodes : `step()`, `get_zmp()`

### `src/mpc_bipedal/generators/`
**footstep_generator.py** : Génération de la séquence de footsteps
- Classe `Contact` : Représente un point de contact
- Fonction `generate_footsteps()` : Génère la séquence de pas

**cop_generator.py** : Génération de la trajectoire du Centre de Pression (CoP)
- Classe `CoPGenerator` : Génère les limites z_max et z_min
- Gère les transitions entre phases (STANDING, DOUBLE_SUPPORT, SINGLE_SUPPORT)

### `src/mpc_bipedal/controllers/`
**zmp_controller.py** : Contrôleur ZMP basé sur MPC
- Classe `ZMPController` : Implémente le contrôle prédictif
- Méthode `predict()` : Résout le problème d'optimisation QP ou analytique
- Méthode `generate_com_trajectory()` : Génère la trajectoire complète du COM

### `src/mpc_bipedal/utils/`
**visualization.py** : Utilitaires de visualisation
- `visualize_com_trajectory_3d()` : Visualisation 3D animée ou statique
- `visualize_com_trajectory_static()` : Visualisation statique avec gradient de couleur

## Flux d'exécution

1. **Initialisation** : Chargement de la configuration (fichier JSON ou paramètres CLI)
2. **Génération des footsteps** : Création de la séquence de points de contact
3. **Génération CoP** : Calcul des limites z_max/z_min à partir des footsteps
4. **MPC** : Résolution du problème d'optimisation pour chaque pas de temps
5. **Visualisation** : Affichage des trajectoires et résultats

## Algorithmes

### Modèle LIPM
Le modèle utilise les équations du pendule inversé linéaire :
- Dynamique : `x_{k+1} = A * x_k + B * u_k`
- ZMP : `zmp = C * x_k`

### Contrôleur MPC
Le contrôleur résout à chaque pas un problème d'optimisation :
- **Objectif** : Minimiser l'erreur de tracking ZMP et la régularisation
- **Contraintes** : ZMP doit rester dans les limites (z_min, z_max)
- **Mode strict** : Résolution QP avec contraintes strictes (OSQP)
- **Mode non-strict** : Solution analytique sans contraintes strictes

## Paramètres clés

### Paramètres physiques
- `h` : Hauteur du COM (m)
- `m` : Masse du robot (kg)
- `g` : Gravité (m/s²)
- `F_ext` : Force externe appliquée (N)

### Paramètres de marche
- `distance` : Distance totale (m)
- `step_length` : Longueur de pas (m)
- `foot_spread` : Espacement latéral (m)
- `dt` : Pas de temps (s)

### Paramètres MPC
- `horizon` : Horizon de prédiction
- `Q` : Poids du tracking
- `R` : Poids de la régularisation
- `strict` : Mode avec contraintes strictes

## Extensibilité

Le projet est conçu pour être facilement extensible :

1. **Nouveaux modèles** : Ajouter dans `models/` en suivant l'interface de `LIPMModel`
2. **Nouveaux générateurs** : Ajouter dans `generators/` pour d'autres types de trajectoires
3. **Nouveaux contrôleurs** : Ajouter dans `controllers/` pour d'autres stratégies de contrôle
4. **Nouveaux utilitaires** : Ajouter dans `utils/` pour d'autres fonctions auxiliaires

## Dépendances

- **numpy** : Calculs numériques
- **cvxpy** : Optimisation convexe (solveur QP)
- **matplotlib** : Visualisation
- **plotly** : Graphiques interactifs
- **tqdm** : Barres de progression

