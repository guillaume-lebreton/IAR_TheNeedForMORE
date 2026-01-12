import numpy as np
import matplotlib.pyplot as plt
from algos_rl import V_MORE

"""
    fonctions utils pour faires des plots
"""


###### Fonctions de courbes ######
def sliding_cumulative_reward(all_rewards, window_size):
    """ 
        Retourne la récompense cumulée glissante sur une fenêtre de taille window_size 
        all_rewards : shape (K, nb_episodes, nb_steps)
    """
    sliding = np.cumsum(all_rewards, axis=2) #on fais la somme cumulée sur l'axe des steps 
    sliding[:, :, window_size:] = sliding[:, :, window_size:] - sliding[:, :, :-window_size]
    return sliding

def plot_cumulative_sliding_reward(sliding_cumulative_rewards, window_size, eps, lr, nb_episodes, nb_criteres, nameAlgo="standard"):
    """
        Affiche la courbe de la récompense cumulée glissante
        sliding_cumulative_rewards : shape (K, nb_episodes, nb_steps)
    """
    mean_rewards = np.mean(sliding_cumulative_rewards, axis=1) #shape (K, nb_steps)
    std_rewards = np.std(sliding_cumulative_rewards, axis=1)   #shape (K, nb_steps)

    plt.figure()
    for k in range(nb_criteres):
        plt.plot(mean_rewards[k], label=f"r {k}")
        plt.fill_between(range(mean_rewards.shape[1]), mean_rewards[k]-std_rewards[k], mean_rewards[k]+std_rewards[k], alpha=0.2)
        plt.xlabel("t")
        plt.ylabel(f"Sliding Cumulative reward T={window_size}")
        plt.title(f"Separate objectives: {nameAlgo}\n epsilon={eps}, lr={lr}, nb_episodes={nb_episodes}")

    plt.grid()
    plt.legend()
    plt.show()

def plot_evolution_w(evolution_w0, evolution_w0_discret, nb_episodes, nb_steps, eps, lr, pas_discr):
    """
        Affiche l'évolution du poids w0 au cours des épisodes et des steps
        evolution_w0 : shape (nb_episodes, nb_steps)
    """
    plt.figure()
    mean_w0 = np.mean(evolution_w0, axis=0)
    std_w0 = np.std(evolution_w0, axis=0)

    mean_w0_discret = np.mean(evolution_w0_discret, axis=0)
    std_w0_discret = np.std(evolution_w0_discret, axis=0)

    plt.plot(mean_w0, label="w0 avant discrétisation", color='blue')
    #plt.fill_between(range(nb_steps), mean_w0 - std_w0, mean_w0 + std_w0, alpha=0.2, color='blue')


    plt.plot(mean_w0_discret, label="w0 discretisé", color='orange')
    #plt.fill_between(range(nb_steps), mean_w0_discret - std_w0_discret, mean_w0_discret + std_w0_discret, alpha=0.2, color='orange')

    
    plt.xlabel("t")
    plt.ylabel("Evolution de w0")
    plt.title(f"Evolution de w0 au cours des épisodes\n epsilon={eps}, lr={lr}, nb_episodes={nb_episodes}, pas_discretisation={pas_discr}")
    plt.grid()
    plt.legend()
    plt.show()

def temps_passe(x):
    """ Retourne un np array où chaque position i contient le nombre de pas consécutifs passés dans l'état jusqu'à sa sortie """
    y = np.zeros_like(x)
    i = 0
    while i < len(x):
        if x[i] == 0:
            i += 1
            continue
        cpt = 0
        debut = i
        while i < len(x) and x[i] != 0:
            cpt += 1
            i += 1
        y[debut:i] = cpt
    return y

def plot_space_traversal(all_state_space_traversal, nb_episodes, nb_steps, eps, lr, nb_state):

    """ Afffiche la courbe du de combien de temps consécutif l'agent reste dans un etat """

    l = [[] for _ in range(nb_state)]
    for s in range(nb_state):
        for episode in range(nb_episodes):
            # sumcum = np.cumsum(all_state_space_traversal[episode][s])
            # l[s].append(sumcum)

            # l[s].append(all_state_space_traversal[episode][s])

            y = temps_passe(all_state_space_traversal[episode][s])
            l[s].append(y)


    l = np.array(l) #shape (nb_states, nb_episodes, nb_steps)
    mean_stay = np.mean(l, axis=1) #shape (nb_states, nb_steps)
    std_stay = np.std(l, axis=1)

    plt.figure()
    for s in range(nb_state):
        plt.plot(mean_stay[s], label=f"s = {s}")
        plt.fill_between(range(mean_stay.shape[1]), mean_stay[s]-std_stay[s], mean_stay[s]+std_stay[s], alpha=0.2)

    plt.xlabel("t")
    plt.ylabel("Average stay length")
    plt.title(f"State space traversal\n epsilon={eps}, lr={lr}, nb_episodes={nb_episodes}")
    plt.grid()
    plt.legend()
    plt.show()

def plot_global_3x3(mdp, results, nb_episodes, nb_steps, window_size, K, eps=None, lr=None):
    """
    Figure 3x3:
      - Lignes: standard / objective switching / MORE
      - Colonnes:
          (1) State space traversal
          (2) Sliding cumulative reward
          (3) V_MORE
    """

    algo_order = ["standard", "objective switching", "MORE"]
    col_titles = [
        "State space traversal (average consecutive stay)",
        f"Sliding cumulative reward (window={window_size})",
        "V_MORE",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex="col")
    # fig.suptitle("Comparaison globale (3 algos × 3 métriques)", fontsize=16)

    # Titres UNIQUEMENT sur la première ligne
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title)

    nb_state = len(mdp.states)

    for i, algo in enumerate(algo_order):
        if algo not in results:
            for j in range(3):
                axes[i, j].axis("off")
            continue

        # Nom de l'algo UNIQUEMENT dans la première colonne
        axes[i, 0].text(
            -0.15, 0.5, algo,
            transform=axes[i, 0].transAxes,
            fontsize=12,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center"
        )

        all_rewards = results[algo]["all_rewards"]
        all_state_space_traversal = results[algo]["all_state_space_traversal"]

        if not isinstance(all_state_space_traversal, list):
            all_state_space_traversal = list(all_state_space_traversal)

        # =========================================================
        # COLONNE 1 : STATE SPACE TRAVERSAL
        # =========================================================
        ax = axes[i, 0]

        l = [[] for _ in range(nb_state)]
        for s in range(nb_state):
            for episode in range(nb_episodes):
                # sumcum = np.cumsum(all_state_space_traversal[episode][s])
                # l[s].append(sumcum)
                y = temps_passe(all_state_space_traversal[episode][s])
                l[s].append(y)
                
        l = np.array(l)  # (nb_states, nb_episodes, nb_steps)
        mean_stay = np.mean(l, axis=1)
        std_stay = np.std(l, axis=1)

        for s in range(nb_state):
            ax.plot(mean_stay[s], label=f"s={s}")
            ax.fill_between(
                range(mean_stay.shape[1]),
                mean_stay[s] - std_stay[s],
                mean_stay[s] + std_stay[s],
                alpha=0.2
            )

        ax.grid(True)
        if i == 0:
            ax.legend(ncol=2, fontsize=8)

        if i == 2:
            ax.set_xlabel("t")
        else:
            ax.set_xlabel("")

        ax.set_ylabel("")

        # =========================================================
        # COLONNE 2 : SLIDING CUMULATIVE REWARD
        # =========================================================
        ax = axes[i, 1]

        sliding = sliding_cumulative_reward(all_rewards, window_size=window_size)

        if sliding.ndim == 3:
            mean_sliding = sliding.mean(axis=1)
            std_sliding = sliding.std(axis=1)
        elif sliding.ndim == 2:
            mean_sliding = sliding
            std_sliding = None
        else:
            raise ValueError("Format sliding_cumulative_reward inattendu")

        for k in range(K):
            ax.plot(mean_sliding[k], label=f"r{k}")
            if std_sliding is not None:
                ax.fill_between(
                    range(mean_sliding.shape[1]),
                    mean_sliding[k] - std_sliding[k],
                    mean_sliding[k] + std_sliding[k],
                    alpha=0.2
                )

        ax.grid(True)

        if i == 2:
            ax.set_xlabel("t")
        else:
            ax.set_xlabel("")

        ax.set_ylabel("")

        if i == 0:
            ax.legend(fontsize=8)

        # =========================================================
        # COLONNE 3 : V_MORE
        # =========================================================
        ax = axes[i, 2]

        vmore = V_MORE(sliding)
        ax.plot(vmore)
        ax.grid(True)

        if i == 2:
            ax.set_xlabel("t")
        else:
            ax.set_xlabel("")

        ax.set_ylabel("")

    plt.tight_layout()
    plt.show()
