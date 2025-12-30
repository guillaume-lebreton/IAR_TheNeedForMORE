import numpy as np
import matplotlib.pyplot as plt
import time
from utils import prepare_objective_curves, plot_objective_curves, plot_MORE_curve, plot_state_space_traversal, exponential_moving_average, avg_stay_length_curves

# variables globales
STEPS = 20_000
ALPHA = 0.2
GAMMA = 0.9
GAMMA_PAST = 0.99
SIGMA = 0.4
EPSILON_START = 0.5
HALF_STEPS = 2000

# region env
class Env:
    def __init__(self):
        self.state = 0  # état initial

        # rewards[state] = (r0, r1)
        self.rewards = {
            0: np.array([+0.1,   -0.09]),
            1: np.array([-0.001, -0.001]),
            2: np.array([-0.018, +0.02])
        }

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        # Transitions
        if self.state == 0:
            if action == 1:
                self.state = 1
        elif self.state == 1:
            if action == 0:
                self.state = 0
            else:
                self.state = 2
        elif self.state == 2:
            if action == 0:
                self.state = 1

        reward = self.rewards[self.state]

        return self.state, reward
    
# endregion

# region Q-learning

def init_q_table(n_states=3, n_actions=2):
    return np.zeros((n_states, n_actions))

def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    return np.argmax(Q[state])

def train_linear_qlearning(
    env,
    steps=STEPS,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon_start=EPSILON_START):

    Q = init_q_table()
    state = env.reset()

    epsilon = epsilon_start
    decay_every = HALF_STEPS

    history = {
        "state": [],
        "r0": [],
        "r1": []
    }

    for t in range(steps):
        if t % decay_every == 0 and t > 0:
            epsilon = epsilon * 0.5

        action = epsilon_greedy(Q, state, epsilon)
        next_state, reward_vec = env.step(action)

        r = reward_vec.sum()

        # Q-learning update
        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (r + gamma * np.max(Q[next_state]))

        # logging
        history["state"].append(next_state)
        history["r0"].append(reward_vec[0])
        history["r1"].append(reward_vec[1])

        state = next_state

    return Q, history

# endregion

# region switch 
def train_objective_switching(
    env,
    steps=STEPS,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon_start=EPSILON_START
):
    Q0 = init_q_table()
    Q1 = init_q_table()

    state = env.reset()
    epsilon = epsilon_start

    R0, R1 = 0.0, 0.0  # récompenses cumulées

    history = {
        "state": [],
        "r0": [],
        "r1": [],
        "obj": []
    }

    decay_every = HALF_STEPS

    for t in range(steps):
        if t % decay_every == 0 and t > 0:
            epsilon *= 0.5

        # Choix de l'objectif à optimiser
        if R0 < R1:
            Q_active = Q0
            obj = 0
        else:
            Q_active = Q1
            obj = 1

        action = epsilon_greedy(Q_active, state, epsilon)
        next_state, reward_vec = env.step(action)

        # Mise à jour des deux Q
        Q0[state, action] = (1 - alpha) * Q0[state, action] + alpha * (reward_vec[0] + gamma * np.max(Q0[next_state]))

        Q1[state, action] = (1 - alpha) * Q1[state, action] + alpha * (reward_vec[1] + gamma * np.max(Q1[next_state]))

        # Accumulation
        R0 += reward_vec[0]
        R1 += reward_vec[1]

        history["state"].append(next_state)
        history["r0"].append(reward_vec[0])
        history["r1"].append(reward_vec[1])
        history["obj"].append(obj)

        state = next_state

    return Q0, Q1, history

# endregion

# region Q-MORE

class LocalLinearModel:
    """
    Modèle linéaire local :
    - entrée : w ∈ R^2
    - sortie : Q ∈ R^2
    """

    def __init__(self, sigma=SIGMA):
        self.sigma = sigma
        self.w = []  # w_i
        self.Q = []  # Q_i (vecteur à 2 composantes)

    def add_sample(self, w, q):
        self.w.append(np.array(w, dtype=float))
        self.Q.append(np.array(q, dtype=float))

    def predict(self, w):
        """
        w : array shape (2,)
        retourne : array shape (2,) -> Q^{LIN}(s,a,w)
        """

        # Aucun point stocké
        if len(self.w) == 0:
            return np.zeros(2)

        W = np.asarray(self.w)      # (N, 2)
        Q = np.asarray(self.Q)      # (N, 2)

        # distances euclidiennes dans l'espace des poids
        dists = np.linalg.norm(W - w, axis=1)

        # noyau local (local linear regression)
        weights = np.exp(- (dists / self.sigma) ** 2)

        # X = [w0, w1, 1]
        X = np.hstack([W, np.ones((len(W), 1))])   # (N, 3)
        x_query = np.append(w, 1.0)                # (3,)

        # régression linéaire locale
        WX = X * weights[:, None]
        XtWX = X.T @ WX

        WQ = Q * weights[:, None]
        XtWY = X.T @ WQ

        # stabilité numérique
        beta = np.linalg.pinv(XtWX) @ XtWY          # (3, 2)

        # prédiction locale
        Q_pred = x_query @ beta                     # (2,)

        return Q_pred
    
def Q_MORE(w, Q_LIN):
    return - np.dot(w, np.exp(-Q_LIN))

def update_w(w, r, gamma=GAMMA_PAST):
    w_new = (w**gamma) * np.exp(-r)
    w_new /= np.sum(w_new)
    return w_new
    
def greedy_more(models, s, w):
    q_values = []
    for a in range(2):
        Q_lin = models[s][a].predict(w)
        q_values.append(Q_MORE(w, Q_lin))
    return np.argmax(q_values)

# def init_more_models(states=3, actions=2):
#     models = []
#     for s in range(states):
#         for a in range(actions):
#             models.append(LocalLinearModel())
#     return models

def train_q_more(
    env,
    steps=STEPS,
    alpha=ALPHA,
    gamma=GAMMA,
    gamma_past=GAMMA_PAST,
    epsilon=EPSILON_START,
    decay_every=HALF_STEPS,
):
    models = {s: {a: LocalLinearModel() for a in range(2)} for s in range(3)}
    w = np.array([0.5, 0.5])
    s = 0  # état initial

    history = {
        "state": [],
        "r0": [],
        "r1": [],
        "w0": [],
        "w1": []
    }

    for t in range(steps):
        
        if t%1000 == 0 and t>0:
            print("step:",t," poids:", w)

        if t > 0 and t % decay_every == 0:
            epsilon *= 0.5

        # Choix de l'action
        if np.random.random() < epsilon:
            a = np.random.randint(0, 2)
        else:
            a = greedy_more(models, s, w)

        # Exécution de l'action
        s_next, r = env.step(a)

        # MAJ poids
        w_next = update_w(w, r, gamma_past)

        # Action cible a* (greedy MORE)
        a_star = greedy_more(models, s_next, w_next)
        Q_target = models[s_next][a_star].predict(w_next)
        Q_updated = r + gamma * Q_target

        models[s][a].add_sample(w, Q_updated)

        # logging
        history["state"].append(s_next)
        history["r0"].append(r[0])
        history["r1"].append(r[1])
        history["w0"].append(w_next[0])
        history["w1"].append(w_next[1])

        # Avancement
        s = s_next
        w = w_next

    return models, history

# endregion

# region Q-MORE disctretisé

# endregion

# Training

# Q, history = train_linear_qlearning(Env())
# Q0, Q1, history_switch = train_objective_switching(Env())
# QMORE, history_more = train_q_more(Env())

# Q, history = train_linear_qlearning(Env())
# r0_curve, r1_curve = prepare_objective_curves(history["r0"], history["r1"])
# plot_objective_curves([r0_curve], [r1_curve], "standard")
# plot_MORE_curve([r0_curve], [r1_curve], "standard")

# Q0, Q1, history_switch = train_objective_switching(Env())
# r0_curve_switch, r1_curve_switch = prepare_objective_curves(history_switch["r0"], history_switch["r1"])
# plot_objective_curves([r0_curve_switch], [r1_curve_switch], "switch")
# plot_MORE_curve([r0_curve_switch], [r1_curve_switch], "switch")


# region Courbes multiples

def multiplot(nb_exp=10):
    print("--"*15)
    print("Simulation et analyse de ",nb_exp," épisodes pour les 3 méthodes")
    print("--"*15)

    Qlearning_curves = []
    Switch_curves = []
    More_curves = []
    Qlearning_stays = {0: [], 1: [], 2: []}
    Switch_stays = {0: [], 1: [], 2: []}
    More_stays = {0: [], 1: [], 2: []}

    for i in range(nb_exp):
        print("--"*15)
        print("Training n°", i+1)
        print("--"*15)

        print("Qlearning...")
        Q, history = train_linear_qlearning(Env())
        r0_curve, r1_curve = prepare_objective_curves(history["r0"], history["r1"])
        Qlearning_curves.append((r0_curve, r1_curve))
        avg_stay = avg_stay_length_curves(history["state"])
        for s in range(3):
            Qlearning_stays[s].append(avg_stay[s])

        print("Switch...")
        Q0, Q1, history_switch = train_objective_switching(Env())
        r0_curve_switch, r1_curve_switch = prepare_objective_curves(history_switch["r0"], history_switch["r1"])
        Switch_curves.append((r0_curve_switch, r1_curve_switch))
        avg_stay_switch = avg_stay_length_curves(history_switch["state"])
        for s in range(3):
            Switch_stays[s].append(avg_stay_switch[s])

        print("MORE...")
        QMORE, history_more = train_q_more(Env())
        r0_curve_more, r1_curve_more = prepare_objective_curves(history_more["r0"], history_more["r1"])
        More_curves.append((r0_curve_more, r1_curve_more))

        avg_stay_more = avg_stay_length_curves(history_more["state"])
        for s in range(3):
            More_stays[s].append(avg_stay_more[s])

        print("Fin de l'episode n°", i+1)


    # séparer en 2 listes r0 et r1
    r0_curves_qlearning = [r0 for r0, r1 in Qlearning_curves]
    r1_curves_qlearning = [r1 for r0, r1 in Qlearning_curves]
    r0_curves_switch = [r0 for r0, r1 in Switch_curves]
    r1_curves_switch = [r1 for r0, r1 in Switch_curves]
    r0_curves_more = [r0 for r0, r1 in More_curves]
    r1_curves_more = [r1 for r0, r1 in More_curves]

    # plots
    plot_state_space_traversal(Qlearning_stays, "standard")
    plot_objective_curves(r0_curves_qlearning, r1_curves_qlearning, "standard")
    plot_MORE_curve(r0_curves_qlearning, r1_curves_qlearning, "standard")

    plot_state_space_traversal(Switch_stays, "switch")
    plot_objective_curves(r0_curves_switch, r1_curves_switch, "switch")
    plot_MORE_curve(r0_curves_switch, r1_curves_switch, "switch")

    plot_state_space_traversal(More_stays, "MORE")
    plot_objective_curves(r0_curves_more, r1_curves_more, "MORE")
    plot_MORE_curve(r0_curves_more, r1_curves_more, "MORE")

multiplot(nb_exp=10)

# endregion




