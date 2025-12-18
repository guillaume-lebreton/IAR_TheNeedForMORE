import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
from torch import nn
from utils import *
from tqdm import tqdm

class MDP:
    def __init__(self, states, init_state, actions, transitions, rewards, discount_factor):
        self.states = states
        self.init_state = init_state
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.discount_factor = discount_factor

    def get_current_state(self):
        return self.init_state
    
    def get_next_state(self, current_state, action):
        if (current_state, action) not in self.transitions:
            raise ValueError(f"Paire ({current_state}, {action}) non valide")
        return self.transitions[(current_state, action)]
    
    def get_reward(self, state):
        if state not in self.rewards:
            raise ValueError(f"Etat {state} non valide")
        return self.rewards[state]
    
    def step(self, current_state, action):
        next_state = self.get_next_state(current_state, action)
        reward = self.get_reward(next_state)
        return next_state, reward

    def q_init(self):
        q = np.zeros((len(self.states), len(self.actions)))
        return q
    
    
mdp = MDP(
    states=[0, 1, 2],
    init_state=0,
    actions=[0, 1],
    transitions={
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 0,
        (1, 1): 2,
        (2, 0): 1,
        (2, 1): 2,
    },
    rewards={
        0: (0.1, -0.09),
        1: (-0.001, -0.001),
        2: (-0.018, 0.02)
    },
    discount_factor=0.9
)

PAS_DISCR = 0.001
EPS = 0.5
LR = 0.2
GAMMA_PASTE = 0.99
NB_EPISODES = 10
NB_STEPS = 20000
K = 2 # nombre d'observations (rewards) par état
WINDOW_SIZE = 1000

# W = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
W = np.arange(0, 1.01, PAS_DISCR)  # discretisation des poids w0 dans [0,1] avec un pas de 0.01
q_lin = np.zeros((len(mdp.states), len(W), len(mdp.actions), K)) # q(s, w, a) -> K sorties
print("W:", W)
print("q_lin shape:", q_lin.shape)
print(q_lin[0,0,0])

print(K,"Critères")
print(NB_EPISODES, "nb episodes")
print(NB_STEPS, "nb steps")



def choose_action_greedy(state, q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(q[state]))
    return np.argmax(q[state])

def Q_more(s, idxw, a, w0):
    res = w0 * np.exp(-q_lin[s, idxw, a,0]) + (1 - w0) * np.exp(-q_lin[s, idxw, a,1])
    return -res

def update_w(w0, rewards, discount_factor):
    w0_old = w0
    w0 = (w0 ** discount_factor) * np.exp(-np.array(rewards[0]))
    w1 = ((1 - w0_old) ** discount_factor) * np.exp(-np.array(rewards[1]))
    w0 = w0 / (w0 + w1)  # normalisation
    if w0 + w1 <= 0:
        raise ValueError("Somme des poids <= 0 dans update_w")
    return w0


def discretize_w(w, verbose=False):
    """ retourne le poids dans W le plus proche de w et son index """
    if verbose:
        print("discretize_w input:", w)
        print("W:", W)
    diff = np.abs(w - W[None, :])
    idx = np.argmin(diff, axis=1)
    if verbose:
        print("idx proche:", idx)
    return W[idx], idx

def train_q_more(
    mdp,
    nb_episodes,
    gamma,
    nb_steps, 
    epsilon= 0.5, #pour epsilon-greedy
    alpha=0.01,
    verbose=False
):
    evolution_w0 = np.zeros((nb_episodes, nb_steps))  # pour stocker l'évolution du poids w0
    evolution_w0_discret = np.zeros((nb_episodes, nb_steps))  # pour stocker l'évolution du poids w0 discretisé

    all_reward_MORE = np.zeros((K, nb_episodes, nb_steps))  # taille: K x NB_EPISODES x NB_STEPS
    all_state_space_traversal = np.array([])


    for episode in tqdm(range(nb_episodes), desc="Episodes"):

        state = mdp.get_current_state()
        w_0 = 0.5  # poids initiaux -> w_1 = 1 - w_0 
        w_0, idx_w0 = discretize_w(w_0, verbose=verbose)
        if verbose:
            print("\n\n")
            print(f"Début Episode {episode} avec poids init de [w_0={w_0}, w_1={1 - w_0}]")

        # ## INIT ##
        # # dico pour stocker pour chaque etats le moment ou on l'a visité pendant l'episode
        state_space_traversal = {}
        for key in mdp.states:
            state_space_traversal[key] = np.array([])


        for t in range(nb_steps):
            if verbose:
                print("\nstep", t)
                print(f"poids:[w_0={w_0}, w_1={1 - w_0}] idx_w: {idx_w0} état: {state}")

            #################### choix de l'action #################### 
            if np.random.rand() < epsilon:
                a = np.random.choice(len(mdp.actions))
                if verbose:
                    print("action aléatoire choisie:", a)
            else:
                qv = []
                for action in mdp.actions:
                    qv.append(Q_more(state, idx_w0, action, w_0))
                q_values = np.array(qv)
                a = np.argmax(q_values)
                if verbose:
                    print("action opt choisie:", a)
                    print("q_values: pour chaque action selon chaque critère")
                    print(q_values)


            
            if verbose:
                print("q_lin state, idx_w0, a:")
                print(q_lin[state, idx_w0, a])

            ##################### transition environnement ####################
            state_next, reward = mdp.step(state, a)
            if verbose:
                print("transition vers état:", state_next, " avec reward:", reward)
            for k in range(K):
                all_reward_MORE[k][episode][t] = reward[k]

            # MAJ des poids
            new_w0 = update_w(w_0, reward, GAMMA_PASTE)
            evolution_w0[episode][t] = new_w0 # poids avant discretisation
            new_w0, new_idx_w0 = discretize_w(new_w0, verbose=verbose)
            evolution_w0_discret[episode][t] = new_w0 # poids discretisé
            
            if verbose:
                print("update des poids:")
                print("w0' =", new_w0, ", idx_w0 =", new_idx_w0, ", reward =", reward)
           

            # choix a* pour l'etat suivant
            q_values_next = np.array([Q_more(state_next, new_idx_w0, action, new_w0) for action in mdp.actions])
            a_next_star = np.argmax(q_values_next)

            # MAJ Q_MORE
            for k in range(K):
                target = reward[k] + gamma * q_lin[state_next, new_idx_w0, a_next_star, k]
                q_lin[state, idx_w0, a, k] = (1-alpha) * q_lin[state, idx_w0, a, k] + alpha * target

            
            # # stockage des états visités
            for key in mdp.states:
                if key == state:
                    state_space_traversal[key] = np.append(state_space_traversal[key], 1)
                else:
                    state_space_traversal[key] = np.append(state_space_traversal[key], 0)

            

            
            state = state_next
            w_0 = new_w0
            idx_w0 = new_idx_w0

        all_state_space_traversal = np.append(all_state_space_traversal, copy.deepcopy(state_space_traversal))
        epsilon = max(epsilon * 0.99, 0.01)  # diminution epsilon par épisode

    if verbose:
        print("Fin training Q_MORE")
    return q_lin, all_reward_MORE, evolution_w0, evolution_w0_discret, all_state_space_traversal


def train_q_standard(mdp, nb_episodes, gamma, nb_steps, epsilon, alpha, nb_criteres, verbose=False):
    
    all_reward_standard = np.zeros((nb_criteres, nb_episodes, nb_steps))  # taille: K x NB_EPISODES x NB_STEPS
    all_state_space_traversal = np.array([])


    for episode in tqdm(range(nb_episodes), desc="Episodes"): 
        if verbose:
            print("Episode", episode)
        q = mdp.q_init()
        state = mdp.get_current_state()

        ## INIT ##
        # dico pour stocker pour chaque etats le moment ou on l'a visité pendant l'episode
        state_space_traversal = {}
        for key in mdp.states:
            state_space_traversal[key] = np.array([])

        for t in range(NB_STEPS):
            
            a = choose_action_greedy(state, q, epsilon=epsilon)
            state_next, reward = mdp.step(state, a)

            # Approche classique: somme pondérée des récompenses
            standard(reward, state, a, state_next, q)

            # # stockage des états visités
            for key in mdp.states:
                if key == state:
                    state_space_traversal[key] = np.append(state_space_traversal[key], 1)
                else:
                    state_space_traversal[key] = np.append(state_space_traversal[key], 0)

            # stockage des récompenses
            for k in range(nb_criteres):
                all_reward_standard[k][episode][t] = reward[k] # reward obtenue             

            state = state_next


        all_state_space_traversal = np.append(all_state_space_traversal, copy.deepcopy(state_space_traversal))

        if verbose:
            print("Q-values à la fin de l'épisode:")
            print(q)
            # print("\n\n")
            # print("etats visités pendant l'épisode:")
            # print(state_space_traversal)

        epsilon = max(epsilon * 0.99, 0.01)  # diminution epsilon par épisode

    return q, all_reward_standard, all_state_space_traversal


def train_q_objective_switching(mdp, nb_episodes, gamma, nb_steps, epsilon, alpha, nb_criteres, verbose=False):
    all_reward_objective_switching = np.zeros((K, nb_episodes, nb_steps))  # taille: K x NB_EPISODES x NB_STEPS
    all_state_space_traversal = np.array([])


    for episode in tqdm(range(nb_episodes), desc="Episodes"): 
        if verbose:
            print("Episode", episode)
        q = mdp.q_init()
        state = mdp.get_current_state()

        # ## INIT ##
        # # dico pour stocker pour chaque etats le moment ou on l'a visité pendant l'episode
        state_space_traversal = {}
        for key in mdp.states:
            state_space_traversal[key] = np.array([])

        # Pour objective switching
        sum_cum_r_0 = 0
        sum_cum_r_1 = 0


        for t in range(nb_steps):
            
            a = choose_action_greedy(state, q, epsilon=epsilon)
            state_next, reward = mdp.step(state, a)

            # Pour objective switching
            r_0 = weighted_sum(reward, weights=[1,0])
            r_1 = weighted_sum(reward, weights=[0,1])
            sum_cum_r_0 += r_0
            sum_cum_r_1 += r_1

            # Approche objective switching
            objective_switching(reward, state, a, state_next, q, sum_cum_r_0, sum_cum_r_1, r_0, r_1)

            # # stockage des états visités
            for key in mdp.states:
                if key == state:
                    state_space_traversal[key] = np.append(state_space_traversal[key], 1)
                else:
                    state_space_traversal[key] = np.append(state_space_traversal[key], 0)

            # stockage des récompenses
            for k in range(K):
                all_reward_objective_switching[k][episode][t] = reward[k] # reward obtenue             

            state = state_next


        all_state_space_traversal = np.append(all_state_space_traversal, copy.deepcopy(state_space_traversal))
        epsilon = max(epsilon * 0.99, 0.01)  # diminution epsilon par épisode

        if verbose:
            print("Q-values à la fin de l'épisode:")
            print(q)
            # print("\n\n")
            # print("etats visités pendant l'épisode:")
            # print(state_space_traversal)

    return q, all_reward_objective_switching, all_state_space_traversal


# vecteurs pour stocker les états visités et les récompenses obtenues pour affichage graphique
# all_state_space_traversal = np.array([])
all_rewards = [[] for _ in range(K)]
for k in range(K):
    all_rewards[k] = [[0]for _ in range(NB_EPISODES)] # taille: K x NB_EPISODES

print("ici")
print(all_rewards)


def weighted_sum(rewards, weights=[1,1]):
    """ calcule la somme pondérée des rewards """
    reward = 0
    for r, w in zip(rewards, weights):
        reward += r * w
    return reward

def Q_learning(state, action, reward, next_state, q, lr, discount_factor):
    """ met a jours la q-table selon l'algorithme Q-learning """
    next_a = choose_action_greedy(next_state, q, epsilon=0) #action qui maximise Q(st+1)
    delta = (reward + discount_factor * q[next_state][next_a] - q[state][action])
    q[state][action] = q[state][action] + lr * delta  
    return q

def V_MORE(all_rewards_sliding_cum):
    """ calcul - log (-V^more)
        all_rewards_sliding_cum : shape (K, nb_episodes, nb_steps)
    """
    mean_rewards = np.mean(all_rewards_sliding_cum, axis=1) #shape (K, nb_steps)
    return -np.log(np.exp(-mean_rewards).sum(axis=0))  # somme sur les critères

def standard(reward, state, a, state_next, q):
    #Approche classique: somme pondérée des récompenses
    sum_reward = weighted_sum(reward)
    q = Q_learning(state, a, sum_reward, state_next, q, LR, mdp.discount_factor)

def objective_switching(reward, state, a, state_next, q, sum_cum_r_0, sum_cum_r_1, r_0, r_1):
    #Approche objective switching
    if sum_cum_r_0 <= sum_cum_r_1:
        q = Q_learning(state, a, r_0, state_next, q, LR, mdp.discount_factor)
    else:
        q = Q_learning(state, a, r_1, state_next, q, LR, mdp.discount_factor)




"""
for episode in range(NB_EPISODES): 
    print("Episode", episode)
    q = mdp.q_init()
    state = mdp.get_current_state()

    ## INIT ##
    # dico pour stocker pour chaque etats le moment ou on l'a visité pendant l'episode
    state_space_traversal = {}
    for key in mdp.states:
        state_space_traversal[key] = np.array([])

    # Pour objective switching
    sum_cum_r_0 = 0
    sum_cum_r_1 = 0


    for t in range(NB_STEPS):
        
        a = choose_action_greedy(state, q, epsilon=EPS)
        state_next, reward = mdp.step(state, a)

        # Pour objective switching
        r_0 = weighted_sum(reward, weights=[1,0])
        r_1 = weighted_sum(reward, weights=[0,1])
        sum_cum_r_0 += r_0
        sum_cum_r_1 += r_1

        # Approche objective switching
        # objective_switching(reward, state, a, state_next, q, sum_cum_r_0, sum_cum_r_1, r_0, r_1)

        # Approche classique: somme pondérée des récompenses
        # standard(reward, state, a, state_next, q)

        # stockage des états visités
        for key in mdp.states:
            if key == state:
                state_space_traversal[key] = np.append(state_space_traversal[key], 1)
            else:
                state_space_traversal[key] = np.append(state_space_traversal[key], 0)

        # stockage des récompenses
        for k in range(K):
            all_rewards[k][episode].append(reward[k]) # reward obtenue 
            #all_rewards[k][episode].append(all_rewards[k][episode][-1] + reward[k]) # somme des rewards
        

        state = state_next


    all_state_space_traversal = np.append(all_state_space_traversal, copy.deepcopy(state_space_traversal))

    print("Q-values à la fin de l'épisode:")
    print(q)
    print("\n\n")
    print("etats visités pendant l'épisode:")
    print(state_space_traversal)
"""

# print("Tous les états visités pendant tous les épisodes:")
# print(all_state_space_traversal)
# print(len(all_state_space_traversal))

# ### Affichage des longueurs de séjour par temps et par état ###
# l = [[] for _ in range(len(mdp.states))]
# for s in range(len(mdp.states)):
   
#     for episode in range(NB_EPISODES):
#         sumcum = np.cumsum(all_state_space_traversal[episode][s])
#         l[s].append(sumcum)

# l = np.array(l) #shape (nb_states, nb_episodes, nb_steps)
# print(l)





# ### Affichage des longueurs moyennes de séjour par temps et par état ###

# mean_l = np.mean(l, axis=1) #shape (nb_states, nb_steps) moyenne sur les episodes
# std_l = np.std(l, axis=1)   #shape (nb_states, nb_steps) ecart type sur les episodes

# plt.figure()
# for s in range(len(mdp.states)):
#     plt.plot(mean_l[s], label=f"s = {s}")
#     plt.fill_between(range(mean_l.shape[1]), mean_l[s]-std_l[s], mean_l[s]+std_l[s], alpha=0.2)
#     plt.xlabel("t")
#     plt.ylabel("Average stay length")
#     plt.title("State space traversal: standard")

# plt.grid()
# plt.legend()
# plt.show()

def plot_space_traversal(all_state_space_traversal, nb_episodes, nb_steps, eps, lr):

    l = [[] for _ in range(len(mdp.states))]
    for s in range(len(mdp.states)):
        for episode in range(nb_episodes):
            sumcum = np.cumsum(all_state_space_traversal[episode][s])
            l[s].append(sumcum)

    l = np.array(l) #shape (nb_states, nb_episodes, nb_steps)
    mean_stay = np.mean(l, axis=1)
    std_stay = np.std(l, axis=1)

    plt.figure()
    for s in range(len(mdp.states)):
        plt.plot(mean_stay[s], label=f"s = {s}")
        plt.fill_between(range(mean_stay.shape[1]), mean_stay[s]-std_stay[s], mean_stay[s]+std_stay[s], alpha=0.2)

    plt.xlabel("t")
    plt.ylabel("Average stay length")
    plt.title(f"State space traversal\n epsilon={eps}, lr={lr}, nb_episodes={nb_episodes}")
    plt.grid()
    plt.legend()
    plt.show()






if __name__ == "__main__":

# ################################ Standard Q-learning ################################
#     print("\n######## Standard Q-learning ########")
#     q_standard, all_rewards_standard, all_state_space_traversal = train_q_standard(mdp, nb_episodes=NB_EPISODES, gamma=mdp.discount_factor, nb_steps=NB_STEPS, epsilon=EPS, alpha=LR, nb_criteres=K, verbose=False)
#     print("all_state_space_traversal:")
#     print(all_state_space_traversal)
    
#     sliding_cumulative_rewards_STANDARD = sliding_cumulative_reward(all_rewards_standard, window_size=WINDOW_SIZE)

#     plot_cumulative_sliding_reward(
#         sliding_cumulative_rewards_STANDARD,
#         window_size=WINDOW_SIZE,
#         eps=EPS,
#         lr=LR,
#         nb_episodes=NB_EPISODES,
#         nb_criteres=K
#     )

#     v_more_standard = V_MORE(sliding_cumulative_rewards_STANDARD)

#     plot_space_traversal(all_state_space_traversal, NB_EPISODES, NB_STEPS, EPS, LR)

#     plt.plot(v_more_standard)
#     plt.title("V^MORE: standard")
#     plt.xlabel("t")
#     plt.ylabel("-log(-Vmore)")
#     plt.grid()
#     plt.show()
#     print("Q valeurs apprises:")
#     print(q_standard)
#     print("\n\n")


# ################################ Objective Switching ################################
#     print("######## Objective Switching ########")
#     q_objective_switching, all_rewards_objective_switching, all_state_space_traversal_objective_switching = train_q_objective_switching(mdp, nb_episodes=NB_EPISODES, gamma=mdp.discount_factor, nb_steps=NB_STEPS, epsilon=EPS, alpha=LR, nb_criteres=K, verbose=False)
    
#     sliding_cumulative_rewards_OBJECTIVE_SWITCHING = sliding_cumulative_reward(all_rewards_objective_switching, window_size=WINDOW_SIZE)
#     plot_cumulative_sliding_reward(
#         sliding_cumulative_rewards_OBJECTIVE_SWITCHING,
#         window_size=WINDOW_SIZE,
#         eps=EPS,
#         lr=LR,
#         nb_episodes=NB_EPISODES,
#         nb_criteres=K
#     )

#     v_more_objective_switching = V_MORE(sliding_cumulative_rewards_OBJECTIVE_SWITCHING)

#     plot_space_traversal(all_state_space_traversal_objective_switching, NB_EPISODES, NB_STEPS, EPS, LR)

#     plt.plot(v_more_objective_switching)
#     plt.title("V^MORE: objective switching")
#     plt.xlabel("t")
#     plt.ylabel("-log(-Vmore)")
#     plt.grid()
#     plt.show()
#     print("Q valeurs apprises:")
#     print(q_objective_switching)
#     print("\n\n")

################################ MORE ################################
    print("######## MORE ########")
    qlin, all_rewards_MORE, evolution_w0, evolution_w0_discret, all_state_space_traversal = train_q_more(mdp, nb_episodes=NB_EPISODES, gamma=mdp.discount_factor, nb_steps=NB_STEPS, epsilon=EPS, alpha=LR, verbose=False)
    
    plot_evolution_w(evolution_w0, evolution_w0_discret, NB_EPISODES, NB_STEPS, EPS, LR, PAS_DISCR)
    
    sliding_cumulative_rewards = sliding_cumulative_reward(all_rewards_MORE, window_size=WINDOW_SIZE)
    plot_cumulative_sliding_reward(
        sliding_cumulative_rewards,
        window_size=WINDOW_SIZE,
        eps=EPS,
        lr=LR,
        nb_episodes=NB_EPISODES,
        nb_criteres=K
    )
    v_more = V_MORE(sliding_cumulative_rewards)

    plot_space_traversal(all_state_space_traversal, NB_EPISODES, NB_STEPS, EPS, LR)

    plt.plot(v_more)
    plt.title("V^MORE: MORE")
    plt.xlabel("t")
    plt.ylabel("-log(-Vmore)")
    plt.grid()
    plt.show()

    for idx_w in range(len(W)):
        print("-----------------")
        print(f"Q_MORE pour w0 = {W[idx_w]}")
        for s in mdp.states:
            for a in mdp.actions:
                print(f"s={s}, a={a} -> Q_MORE:", Q_more(s, idx_w, a, W[idx_w]))
    print("fin training Q_MORE")


# Faire une heatmap des Q-values apprises par MORE pour chaque état et action en fonction de w0
import seaborn as sns
for s in mdp.states:
    for a in mdp.actions:
        q_more_values = []
        for idx_w in range(len(W)):
            q_more_values.append(Q_more(s, idx_w, a, W[idx_w]))
        plt.figure()
        sns.heatmap(np.array(q_more_values).reshape(1, -1), xticklabels=np.round(W,2), yticklabels=[f"s={s}, a={a}"], cmap="viridis")
        plt.title(f"Q_MORE values for state {s} and action {a} vs w0")
        plt.xlabel("w0")
        plt.ylabel("Q_MORE")
        plt.show()
