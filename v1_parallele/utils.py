import numpy as np
import matplotlib.pyplot as plt




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
