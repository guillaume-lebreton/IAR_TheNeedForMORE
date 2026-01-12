import numpy as np
import copy
from tqdm import tqdm


# from mdp import mdp_standard, mdp_satiete, mdp_batterie


def weighted_sum(rewards, weights=[1,1]):
    """ calcule la somme pondérée des rewards """
    reward = 0
    for r, w in zip(rewards, weights):
        reward += r * w
    return reward

def V_MORE(all_rewards_sliding_cum):
    """ calcul - log (-V^more)
        all_rewards_sliding_cum : shape (K, nb_episodes, nb_steps)
    """
    mean_rewards = np.mean(all_rewards_sliding_cum, axis=1) #shape (K, nb_steps)
    return -np.log(np.exp(-mean_rewards).sum(axis=0))  # somme sur les critères

class RL:
    def __init__(self, mdp, gamma, alpha, epsilon_init):
        self.mdp = mdp
        self.gamma = gamma # discount factor -> 0.9 selon papier
        self.alpha = alpha # learning rate -> 0.2 selon papier
        self.epsilon_init = epsilon_init # pour epsilon-greedy -> 0.5 selon papier puis reduis de moitié tous les 2000 steps
      
    def choose_action_greedy(self, state, q, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(len(q[state]))
        return np.argmax(q[state])
    
    def Q_learning_update(self, state, action, reward, next_state, q):
        """ met a jours la q-table selon l'algorithme Q-learning """
        next_a = np.argmax(q[next_state]) # action qui maximise Q(st+1)
        delta = (reward + self.gamma * q[next_state][next_a]) - q[state][action] # TD error
        q[state][action] = q[state][action] + self.alpha * delta  
        return q

class QLearningStandard(RL):
    def __init__(self, mdp, gamma, alpha, epsilon_init, reward_weights):
        super().__init__(mdp, gamma, alpha, epsilon_init)
        self.reward_weights = reward_weights

    def train(self, nb_episodes, nb_steps, nb_criteres, verbose=False, diminution_epsilon_steps=2000):
        # nb_steps: nombre de steps par épisode -> 20.000 selon papier
        
        all_reward_standard = np.zeros((nb_criteres, nb_episodes, nb_steps), dtype=float)  # taille: K x NB_EPISODES x NB_STEPS
        all_state_space_traversal = []

        for episode in tqdm(range(nb_episodes), desc="Episodes"): 
            if verbose:
                print("Episode", episode)
            q = self.mdp.q_init()
            state = self.mdp.reset()
            epsilon = self.epsilon_init  # reset epsilon au début de chaque épisode

            ## INIT ##
            # dico pour stocker pour chaque etats le moment ou on l'a visité pendant l'episode
            state_space_traversal = {}
            for key in self.mdp.states:
                state_space_traversal[key] = np.array([])

            for t in range(nb_steps):
                
                a = self.choose_action_greedy(state, q, epsilon)
                state_next, reward = self.mdp.step(state, a)

                # Approche classique: somme pondérée des récompenses
                sum_reward = weighted_sum(reward, weights=self.reward_weights)

                self.Q_learning_update(state, a, sum_reward, state_next, q)

                # # stockage des états visités
                for key in self.mdp.states:
                    if key == state:
                        if len(state_space_traversal[key]) == 0:
                            state_space_traversal[key] = np.append(state_space_traversal[key], 1)
                        else:
                            state_space_traversal[key] = np.append(state_space_traversal[key],state_space_traversal[key][-1]+1)
                    else:
                        state_space_traversal[key] = np.append(state_space_traversal[key], 0)

                # stockage des récompenses
                for k in range(nb_criteres):
                    all_reward_standard[k][episode][t] = reward[k] # reward obtenue             

                state = state_next
                if diminution_epsilon_steps > 0 and t > 0 and t % diminution_epsilon_steps == 0:
                    epsilon = epsilon/2  # diminution epsilon par épisode
                    # epsilon = epsilon * 0.99  # -> donne de meilleurs résultats pour le calcule de VMORE

            all_state_space_traversal.append(copy.deepcopy(state_space_traversal))

            if verbose:
                print("Q-values à la fin de l'épisode:")
                print(q)

        return q, all_reward_standard, np.array(all_state_space_traversal)
 
class QLearningObjectiveSwitching(RL):
    def __init__(self, mdp, gamma, alpha, epsilon_init):
        super().__init__(mdp, gamma, alpha, epsilon_init)

    def train(self, nb_episodes, nb_steps, nb_criteres, verbose=False, diminution_epsilon_steps=2000):
        all_reward_objective_switching = np.zeros((nb_criteres, nb_episodes, nb_steps), dtype=float)  # taille: K x NB_EPISODES x NB_STEPS
        all_state_space_traversal = []

        for episode in tqdm(range(nb_episodes), desc="Episodes"): 
            if verbose:
                print("Episode", episode)
            q = self.mdp.q_init()
            state = self.mdp.reset()
            epsilon = self.epsilon_init  # reset epsilon au début de chaque épisode

            ## INIT ##
            # dico pour stocker pour chaque etats le moment ou on l'a visité pendant l'episode
            state_space_traversal = {}
            for key in self.mdp.states:
                state_space_traversal[key] = np.array([])

            # somme cumulée des récompenses pour chaque critère : Pour pouvoir faire le switching
            sum_cum_r0 = 0
            sum_cum_r1 = 0

            for t in range(nb_steps):
                
                a = self.choose_action_greedy(state, q, epsilon)
                state_next, reward = self.mdp.step(state, a)

                # Calcul des sommes cumulées des récompenses pour chaque critère
                r_0 = weighted_sum(reward, weights=[1,0])
                r_1 = weighted_sum(reward, weights=[0,1])
                sum_cum_r0 += r_0
                sum_cum_r1 += r_1

                # Choix du critère à optimiser selon les sommes cumulées
                if sum_cum_r0 <= sum_cum_r1:
                    r = r_0
                else:
                    r = r_1
                self.Q_learning_update(state, a, r, state_next, q)


                # # stockage des états visités
                for key in self.mdp.states:
                    if key == state:
                        if len(state_space_traversal[key]) == 0:
                            state_space_traversal[key] = np.append(state_space_traversal[key], 1)
                        else:
                            state_space_traversal[key] = np.append(state_space_traversal[key],state_space_traversal[key][-1]+1)
                    else:
                        state_space_traversal[key] = np.append(state_space_traversal[key], 0)

                # stockage des récompenses
                for k in range(nb_criteres):
                    all_reward_objective_switching[k][episode][t] = reward[k] # reward obtenue             

                state = state_next
                if diminution_epsilon_steps > 0 and t > 0 and t % diminution_epsilon_steps == 0:
                    epsilon = epsilon/2 # diminution epsilon par épisode
                    # epsilon = epsilon * 0.99  # -> donne de meilleurs résultats pour le calcule de VMORE

            all_state_space_traversal.append(copy.deepcopy(state_space_traversal))

            if verbose:
                print("Q-values à la fin de l'épisode:")
                print(q)

        return q, all_reward_objective_switching, np.array(all_state_space_traversal)

class QLearningMORE(RL):
    def __init__(self, mdp, gamma, alpha, epsilon_init, gamma_paste, W):
        super().__init__(mdp, gamma, alpha, epsilon_init)
        self.gamma_paste = gamma_paste
        self.W = W  # discretisation des poids w0 dans [0,1]
        
    def Q_more(self, s, idxw, a, w0):
        """ calcule la Q-value selon MORE 
            Q_MORE(s,idxw,a) = - (w0*exp(-Q0) + (1-w0)*exp(-Q1))
        """
        q0 = self.q_lin[s, idxw, a,0]
        q1 = self.q_lin[s, idxw, a,1]
        res =  w0 * np.exp(-q0) + (1 - w0) * np.exp(-q1)
        return -res

    def discretize_w(self, w, verbose=False):
        """ retourne le poids dans W le plus proche de w et son index """
        if verbose:
            print("discretize_w input:", w)
            print("W:", self.W)
        diff = np.abs(w - self.W[None, :])
        idx = np.argmin(diff, axis=1)
        if verbose:
            print("idx proche:", idx)
        return self.W[idx], idx # valeur discrétisée et index dans W

    def choose_action(self, state, idx_w0, w0, epsilon, verbose=False):
        """ choisit l'action selon epsilon-greedy 
            etat augmenté avec idx_w0
        """
        if np.random.rand() < epsilon:
            a = np.random.choice(len(self.mdp.actions))
            if verbose:
                print("action aléatoire choisie:", a)
            return a
        qv = []
        for action in self.mdp.actions:
            qv.append(self.Q_more(state, idx_w0, action, w0))
        q_values = np.array(qv)
        a = np.argmax(q_values)
        if verbose:
            print("action opt choisie:", a)
            print("q_values: pour chaque action selon chaque critère")
            print(q_values)
        return a

    def update_w(self, w0, rewards):
        """ update de w0 selon MORE (normalisé)"""
        w0_old = w0
        w0 = (w0_old ** self.gamma_paste) * np.exp(-np.array(rewards[0]))
        w1 = ((1 - w0_old) ** self.gamma_paste) * np.exp(-np.array(rewards[1]))
        w0 = w0 / (w0 + w1)  # normalisation
        if w0 + w1 <= 0:
            raise ValueError("Somme des poids <= 0 dans update_w")
        
        return w0 # nouveau poids w0 normalisé entre [0,1]

    def train(self, nb_episodes, nb_steps, nb_criteres, verbose=False, diminution_epsilon_steps=2000, w0_init=0.5):
        
        self.q_lin = np.zeros((len(self.mdp.states), len(self.W), len(self.mdp.actions), nb_criteres)) # q(s, w, a) -> K sorties
        print("q_lin shape:", self.q_lin.shape)

        evolution_w0 = np.zeros((nb_episodes, nb_steps))  # pour stocker l'évolution du poids w0
        evolution_w0_discret = np.zeros((nb_episodes, nb_steps))  # pour stocker l'évolution du poids w0 discretisé
  
        all_reward_MORE = np.zeros((nb_criteres, nb_episodes, nb_steps))  # taille: K x NB_EPISODES x NB_STEPS
        all_state_space_traversal = []

        for episode in tqdm(range(nb_episodes), desc="Episodes"):
            state = self.mdp.reset()
            epsilon = self.epsilon_init  # reset epsilon au début de chaque épisode

            w_0 = w0_init  # poids initial
            w_0, idx_w0 = self.discretize_w(w_0, verbose=verbose)

            if verbose:
                print("\n\n")
                print(f"Début Episode {episode}: w_0={w_0}, w_1={1 - w_0}, idx_w0={idx_w0}, epsilon={epsilon}")

            # # dico pour stocker pour chaque etats le moment ou on l'a visité pendant l'episode
            state_space_traversal = {}
            for key in self.mdp.states:
                state_space_traversal[key] = np.array([])

            for t in range(nb_steps):
                if verbose:
                    print("\nstep", t)
                    print(f"w_0={w_0}, w_1={1 - w_0}, idx_w0={idx_w0}, epsilon={epsilon}")


                #################### choix de l'action #################### 
                a = self.choose_action(state, idx_w0, w_0, epsilon, verbose=verbose)            
                if verbose:
                    print("self.q_lin state, idx_w0, a:")
                    print(self.q_lin[state, idx_w0, a])

                ##################### transition environnement ####################
                state_next, reward = self.mdp.step(state, a)
                if verbose:
                    print("transition vers état:", state_next, " avec reward:", reward)
                
                for k in range(nb_criteres):
                    all_reward_MORE[k][episode][t] = reward[k] # reward obtenue pour chaque critère

                ##################### mise à jour des poids ####################
                # calcul poids continu w0
                new_w0 = self.update_w(w_0, reward) 
                evolution_w0[episode][t] = new_w0 # poids avant discretisation

                # discretisation du poids w0
                new_w0, new_idx_w0 = self.discretize_w(new_w0, verbose=verbose) 
                evolution_w0_discret[episode][t] = new_w0 # poids discretisé
                
                if verbose:
                    print("update des poids:")
                    print("w0' =", new_w0, ", idx_w0 =", new_idx_w0, ", reward =", reward)


                ##################### choix a* pour l'etat suivant ####################
                q_values_next = np.array([self.Q_more(state_next, new_idx_w0, action, new_w0) for action in self.mdp.actions])
                a_next_star = np.argmax(q_values_next)

                ###################### mise à jour de la Q-table ####################
                for k in range(nb_criteres):
                    target = reward[k] + self.gamma * self.q_lin[state_next, new_idx_w0, a_next_star, k]
                    self.q_lin[state, idx_w0, a, k] = (1 - self.alpha) * self.q_lin[state, idx_w0, a, k] + self.alpha * target

                # # stockage des états visités
                for key in self.mdp.states:
                    if key == state:
                        if len(state_space_traversal[key]) == 0:
                            state_space_traversal[key] = np.append(state_space_traversal[key], 1)
                        else:
                            state_space_traversal[key] = np.append(state_space_traversal[key] ,state_space_traversal[key][-1]+1)
                    
                    else:
                        state_space_traversal[key] = np.append(state_space_traversal[key], 0)
        
                state = state_next
                w_0 = new_w0
                idx_w0 = new_idx_w0

            
                
                if diminution_epsilon_steps > 0 and t > 0 and t % diminution_epsilon_steps == 0:
                    epsilon = epsilon/2  # diminution epsilon par épisode
                    # epsilon = epsilon * 0.99  # -> donne de meilleurs résultats pour le calcule de VMORE

            all_state_space_traversal.append(copy.deepcopy(state_space_traversal))

        if verbose:
            print("*$*$*$*$*$*$ Fin training Q_MORE *$*$*$*$*$*$")

        return self.q_lin, all_reward_MORE, evolution_w0, evolution_w0_discret, np.array(all_state_space_traversal)