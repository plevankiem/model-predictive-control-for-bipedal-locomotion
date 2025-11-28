import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from config import MPCConfig
import matplotlib.animation as animation


def visualize_com_trajectory_3d(com_trajectory_xy, h=None, show_sphere=True, save_animation=False, output_file='com_trajectory_3d.gif'):
    """
    Visualise la trajectoire du COM en 3D.
    
    Args:
        com_trajectory_xy: Array numpy de shape (n_steps, 2) contenant les coordonnées [x, y] du COM
        h: Hauteur du COM (si None, utilise la valeur par défaut de MPCConfig)
        show_sphere: Si True, affiche une sphère animée suivant la trajectoire
        save_animation: Si True, sauvegarde une animation
        output_file: Nom du fichier pour sauvegarder l'animation
    """
    if h is None:
        config = MPCConfig()
        h = config.h
    
    # Ajouter la coordonnée z = h pour toutes les positions
    n_steps = com_trajectory_xy.shape[0]
    com_trajectory_3d = np.zeros((n_steps, 3))
    com_trajectory_3d[:, :2] = com_trajectory_xy
    com_trajectory_3d[:, 2] = h
    
    # Créer la figure 3D
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Tracer la trajectoire complète comme une ligne
    ax.plot(com_trajectory_3d[:, 0], 
            com_trajectory_3d[:, 1], 
            com_trajectory_3d[:, 2], 
            'b-', linewidth=2, alpha=0.6, label='Trajectoire du COM')
    
    # Marquer le point de départ
    ax.scatter(com_trajectory_3d[0, 0], 
               com_trajectory_3d[0, 1], 
               com_trajectory_3d[0, 2], 
               color='green', s=100, marker='o', label='Départ')
    
    # Marquer le point d'arrivée
    ax.scatter(com_trajectory_3d[-1, 0], 
               com_trajectory_3d[-1, 1], 
               com_trajectory_3d[-1, 2], 
               color='red', s=100, marker='s', label='Arrivée')
    
    # Ajuster les limites des axes pour avoir des proportions correctes
    x_range = com_trajectory_3d[:, 0].max() - com_trajectory_3d[:, 0].min()
    y_range = com_trajectory_3d[:, 1].max() - com_trajectory_3d[:, 1].min()
    z_range = com_trajectory_3d[:, 2].max() - com_trajectory_3d[:, 2].min()
    
    # Établir des limites avec un peu de marge
    margin = 0.1
    x_min = com_trajectory_3d[:, 0].min() - margin * x_range
    x_max = com_trajectory_3d[:, 0].max() + margin * x_range
    y_min = com_trajectory_3d[:, 1].min() - margin * y_range
    y_max = com_trajectory_3d[:, 1].max() + margin * y_range
    z_min = 0
    z_max = com_trajectory_3d[:, 2].max() + margin * z_range
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    # Définir les échelles égales pour tous les axes (vraie échelle)
    ax.set_box_aspect([x_max - x_min, y_max - y_min, z_max - z_min])
    
    # Labels et titre
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'Trajectoire 3D du COM (hauteur = {h:.2f} m)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    
    # Ajouter une grille pour meilleure lisibilité
    ax.grid(True, alpha=0.3)
    
    # Animation avec sphère
    if show_sphere or save_animation:
        # Rayon de la sphère (petite sphère)
        sphere_radius = 0.02
        
        # Créer la sphère initiale
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = sphere_radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = sphere_radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Position initiale de la sphère
        sphere_x = com_trajectory_3d[0, 0]
        sphere_y = com_trajectory_3d[0, 1]
        sphere_z = com_trajectory_3d[0, 2]
        
        sphere_surface = ax.plot_surface(x_sphere + sphere_x, 
                                         y_sphere + sphere_y, 
                                         z_sphere + sphere_z,
                                         color='red', alpha=0.8, shade=True)
        
        def animate(frame):
            # Mettre à jour la position de la sphère
            ax.clear()
            
            # Retracer la trajectoire complète
            ax.plot(com_trajectory_3d[:, 0], 
                    com_trajectory_3d[:, 1], 
                    com_trajectory_3d[:, 2], 
                    'b-', linewidth=2, alpha=0.6, label='Trajectoire du COM')
            
            # Tracer la partie déjà parcourue en couleur différente
            if frame > 0:
                ax.plot(com_trajectory_3d[:frame+1, 0], 
                        com_trajectory_3d[:frame+1, 1], 
                        com_trajectory_3d[:frame+1, 2], 
                        'r-', linewidth=2.5, alpha=0.8, label='Trajectoire parcourue')
            
            # Position actuelle de la sphère
            sphere_x = com_trajectory_3d[frame, 0]
            sphere_y = com_trajectory_3d[frame, 1]
            sphere_z = com_trajectory_3d[frame, 2]
            
            # Dessiner la sphère
            ax.plot_surface(x_sphere + sphere_x, 
                           y_sphere + sphere_y, 
                           z_sphere + sphere_z,
                           color='red', alpha=0.9, shade=True)
            
            # Points de départ et d'arrivée
            ax.scatter(com_trajectory_3d[0, 0], 
                       com_trajectory_3d[0, 1], 
                       com_trajectory_3d[0, 2], 
                       color='green', s=100, marker='o', label='Départ')
            ax.scatter(com_trajectory_3d[-1, 0], 
                       com_trajectory_3d[-1, 1], 
                       com_trajectory_3d[-1, 2], 
                       color='orange', s=100, marker='s', label='Arrivée')
            
            # Réappliquer les limites et labels
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            # Réappliquer les échelles égales
            ax.set_box_aspect([x_max - x_min, y_max - y_min, z_max - z_min])
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_zlabel('Z (m)', fontsize=12)
            ax.set_title(f'Trajectoire 3D du COM (hauteur = {h:.2f} m) - Frame {frame}/{n_steps-1}', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Créer l'animation
        # Utiliser un sous-échantillonnage pour rendre l'animation plus fluide
        step = max(1, n_steps // 200)  # Environ 200 frames max
        frames_indices = list(range(0, n_steps, step))
        if frames_indices[-1] != n_steps - 1:
            frames_indices.append(n_steps - 1)
        
        anim = animation.FuncAnimation(fig, animate, frames=frames_indices, 
                                      interval=50, repeat=True, blit=False)
        
        if save_animation:
            print(f"Sauvegarde de l'animation dans {output_file}...")
            anim.save(output_file, writer='pillow', fps=20)
            print(f"Animation sauvegardée avec succès!")
        else:
            plt.show()
    else:
        plt.show()
    
    return fig, ax


def visualize_com_trajectory_static(com_trajectory_xy, h=None):
    """
    Visualise la trajectoire du COM en 3D de manière statique (sans animation).
    
    Args:
        com_trajectory_xy: Array numpy de shape (n_steps, 2) contenant les coordonnées [x, y] du COM
        h: Hauteur du COM (si None, utilise la valeur par défaut de MPCConfig)
    """
    if h is None:
        config = MPCConfig()
        h = config.h
    
    # Ajouter la coordonnée z = h pour toutes les positions
    n_steps = com_trajectory_xy.shape[0]
    com_trajectory_3d = np.zeros((n_steps, 3))
    com_trajectory_3d[:, :2] = com_trajectory_xy
    com_trajectory_3d[:, 2] = h
    
    # Créer la figure 3D
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Tracer la trajectoire avec un gradient de couleur pour montrer la progression
    points = com_trajectory_3d.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, n_steps)
    lc = Line3DCollection(segments, cmap='viridis', norm=norm, linewidth=2)
    lc.set_array(np.arange(n_steps-1))
    line = ax.add_collection3d(lc)
    
    # Marquer le point de départ
    ax.scatter(com_trajectory_3d[0, 0], 
               com_trajectory_3d[0, 1], 
               com_trajectory_3d[0, 2], 
               color='green', s=150, marker='o', label='Départ', zorder=5)
    
    # Marquer le point d'arrivée
    ax.scatter(com_trajectory_3d[-1, 0], 
               com_trajectory_3d[-1, 1], 
               com_trajectory_3d[-1, 2], 
               color='red', s=150, marker='s', label='Arrivée', zorder=5)
    
    # Ajuster les limites des axes
    margin = 0.1
    x_range = com_trajectory_3d[:, 0].max() - com_trajectory_3d[:, 0].min()
    y_range = com_trajectory_3d[:, 1].max() - com_trajectory_3d[:, 1].min()
    z_range = com_trajectory_3d[:, 2].max() - com_trajectory_3d[:, 2].min()
    
    x_min = com_trajectory_3d[:, 0].min() - margin * x_range
    x_max = com_trajectory_3d[:, 0].max() + margin * x_range
    y_min = com_trajectory_3d[:, 1].min() - margin * y_range
    y_max = com_trajectory_3d[:, 1].max() + margin * y_range
    z_min = 0
    z_max = com_trajectory_3d[:, 2].max() + margin * z_range
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    # Définir les échelles égales pour tous les axes (vraie échelle)
    ax.set_box_aspect([x_max - x_min, y_max - y_min, z_max - z_min])
    
    # Labels et titre
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'Trajectoire 3D du COM (hauteur = {h:.2f} m)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    
    # Ajouter une colorbar pour montrer la progression temporelle
    cbar = fig.colorbar(line, ax=ax, shrink=0.5, aspect=20, pad=0.1)
    cbar.set_label('Progression temporelle', rotation=270, labelpad=20)
    
    # Ajouter une grille
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


if __name__ == "__main__":
    # Exemple d'utilisation - nécessite d'exécuter d'abord mpc.py pour obtenir com_trajectory
    print("Pour utiliser ce script:")
    print("1. Soit l'importer dans mpc.py et appeler visualize_com_trajectory_3d(com_trajectory)")
    print("2. Soit sauvegarder com_trajectory dans un fichier numpy et le charger ici")
    print("\nExemple d'utilisation depuis mpc.py:")
    print("  from visualize import visualize_com_trajectory_3d")
    print("  visualize_com_trajectory_3d(com_trajectory, show_sphere=True)")
