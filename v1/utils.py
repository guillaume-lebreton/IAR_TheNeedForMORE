import numpy as np
import matplotlib.pyplot as plt


# Définit les fonctions pour les courbes de la fig5

def exponential_moving_average(x, beta=0.001, scale=1000):
    ema = []
    value = 0.0
    for v in x:
        value = (1 - beta) * value + beta * v
        ema.append(scale * value)
    return np.array(ema)

def moving_average(x, window=100):
    return np.convolve(x, np.ones(window)/window, mode='same')

# region Courbes State space traversal

def avg_stay_length_curves(states, n_states=3):

    states = np.asarray(states)
    T = len(states)

    # courbes finales
    curves = {s: np.ones(T) for s in range(n_states)}

    # dernière valeur connue pour chaque état
    last_value = {s: 1 for s in range(n_states)}

    t = 0
    while t < T:
        s = states[t]
        start = t

        # détecter la fin du séjour
        while t < T and states[t] == s:
            t += 1

        if t != T:
            length = t - start  # longueur du séjour
            last_value[s] = length
        else:
            length = t - start  # longueur du séjour
            last_value[s] = max(last_value[s], length)


        # remplir sur toute la durée du séjour
        for k in range(start, t):
            for s2 in range(n_states):
                curves[s2][k] = last_value[s2]

    avg_curves = {s : exponential_moving_average(curves[s], beta=0.01, scale=1) for s in curves}

    return avg_curves



def plot_state_space_traversal(avg_stay, method):

    plt.figure()
    for i in range(len(avg_stay[0])):
        if i == 0:
            plt.plot(avg_stay[0][i], label="state 0", color='red', alpha=0.7, linewidth=0.75)
            plt.plot(avg_stay[1][i], label="state 1", color='black', alpha=0.7, linewidth=0.75)
            plt.plot(avg_stay[2][i], label="state 2", color='blue', alpha=0.7, linewidth=0.75)
        else:
            plt.plot(avg_stay[0][i], color='red', alpha=0.7, linewidth=0.75)
            plt.plot(avg_stay[1][i], color='black', alpha=0.7, linewidth=0.75)
            plt.plot(avg_stay[2][i], color='blue', alpha=0.7, linewidth=0.75)

    plt.xlabel("t")
    plt.ylabel("Average stay length")
    plt.title("State space traversal :" + method)
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()

# endregion

# region Courbes rewards

def prepare_objective_curves(r0, r1, beta=0.001, scale=1000):
    r0_smooth = exponential_moving_average(r0, beta, scale)
    r1_smooth = exponential_moving_average(r1, beta, scale)
    return r0_smooth, r1_smooth

def V_MORE_curve(r0_smooth, r1_smooth):
    V_more = -(np.exp(-r0_smooth) + np.exp(-r1_smooth))
    return -np.log(-V_more)*0.1

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

# 3eme colonne = V-MORE des coubres de la 2eme colonne
def plot_MORE_curve(r0_curves, r1_curves, method):
    plt.figure()
    for i in range(len(r0_curves)):
        V_more_curve = V_MORE_curve(r0_curves[i], r1_curves[i])
        plt.plot(V_more_curve, color='green', alpha=0.7, linewidth=0.75)
    plt.xlabel("t")
    plt.ylabel("V-MORE value")
    plt.title("V-MORE objective :" + method)
    plt.axhline(0, color='black')
    plt.grid()
    plt.show()
