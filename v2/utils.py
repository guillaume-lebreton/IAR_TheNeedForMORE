import numpy as np
import matplotlib.pyplot as plt

"""
    fonctions utils pour faires des plots
"""


###### Fonctions de courbes ######

def exponential_moving_average(x, beta=0.001, scale=1000):
    ema = []
    value = 0.0
    for v in x:
        value = (1 - beta) * value + beta * v
        ema.append(scale * value)
    return np.array(ema)


###### Avg stay length curves ######

def avg_stay_length_curves(all_state,n_states=3,beta=0.01,scale=1):

    all_curves = {s: [] for s in range(n_states)}

    for states in all_state:
        states = np.asarray(states)
        T = len(states)

        # courbes finales pour CET épisode
        curves = {s: np.ones(T) for s in range(n_states)}
        last_value = {s: 1 for s in range(n_states)}

        t = 0
        while t < T:
            s = states[t]
            start = t

            # détecter la fin du séjour
            while t < T and states[t] == s:
                t += 1

            if t != T:
                length = t - start
                last_value[s] = length
            else:
                length = t - start
                last_value[s] = max(last_value[s], length)

            # remplir sur toute la durée du séjour
            for k in range(start, t):
                for s2 in range(n_states):
                    curves[s2][k] = last_value[s2]

        # EMA finale (COMME AVANT)
        avg_curves = {s: exponential_moving_average(curves[s], beta=beta, scale=scale) for s in curves}

        # stocker cet épisode
        for s in range(n_states):
            all_curves[s].append(avg_curves[s])

    return all_curves


def plot_state_space_traversal(all_curves, method):
    plt.figure()

    colors = ['red', 'black', 'blue']
    for s in all_curves:
        for i, curve in enumerate(all_curves[s]):
            if i == 0:
                plt.plot(curve, color=colors[s], label=f"{s=}", alpha=0.6, linewidth=0.75)
            else:
                plt.plot(curve, color=colors[s], alpha=0.6, linewidth=0.75)

    plt.xlabel("t")
    plt.ylabel("Average stay length")
    plt.title("State space traversal : " + method)
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()

###### Courbes rewards ######

def prepare_objective_curves(all_rewards, beta=0.001, scale=1000):
    r0_curves = []
    r1_curves = []

    for ep in range(all_rewards.shape[1]):
        r0 = all_rewards[0, ep]
        r1 = all_rewards[1, ep]

        r0_smooth = exponential_moving_average(r0, beta=beta, scale=scale)
        r1_smooth = exponential_moving_average(r1, beta=beta, scale=scale)

        r0_curves.append(r0_smooth)
        r1_curves.append(r1_smooth)

    return r0_curves, r1_curves

# 2eme colonne = EMA de chaque reward
def plot_objective_curves(r0_curves, r1_curves, method):
    plt.figure()
    for i in range(len(r0_curves)):
        if i == 0:
            plt.plot(r0_curves[i], label="r0", color='red', alpha=0.7, linewidth=0.75)
            plt.plot(r1_curves[i], label="r1", color='blue', alpha=0.7, linewidth=0.75)
        else:
            plt.plot(r0_curves[i], color='red', alpha=0.7, linewidth=0.75)
            plt.plot(r1_curves[i], color='blue', alpha=0.7, linewidth=0.75)
    plt.xlabel("t")
    plt.ylabel("Sliding cumulative reward (T=1000)")
    plt.title("Separate objectives :" + method)
    plt.legend(loc="upper right")
    plt.axhline(0, color='black')
    plt.grid()
    plt.show()

#3eme colonne = V_MORE
def vmore_curves(r0_curves, r1_curves):
    """
    r0_curves, r1_curves : listes (une par run) de tableaux 1D (length T)
    retourne : liste de courbes y(t) = -log( exp(-r0(t)) + exp(-r1(t)) )
    """
    V_more = -(np.exp(-r0_curves) + np.exp(-r1_curves))
    return -np.log(-V_more)*0.1


def plot_MORE_curve(r0_curves, r1_curves, method):
    plt.figure()
    for i in range(len(r0_curves)):
        V_more_curve = vmore_curves(r0_curves[i], r1_curves[i])
        plt.plot(V_more_curve, color='green', alpha=0.7, linewidth=0.75)
    plt.xlabel("t")
    plt.ylabel("V-MORE value")
    plt.title("V-MORE objective :" + method)
    plt.axhline(0, color='black')
    plt.grid()
    plt.show()
