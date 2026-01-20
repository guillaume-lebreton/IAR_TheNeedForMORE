import random
import numpy as np
import copy
from tqdm import tqdm


""""
    Algorithmes de Reinforcement Learning en plusieurs classes:
    - QLearningStandard : Q-learning avec somme pondérée des récompenses
    - QLearningObjectiveSwitching : Q-learning avec switching d'objectif
    - QLearningMORE : Q-learning avec approche MORE

    Tous les algorithmes héritent de la classe RL qui contient les fonctions communes:
    Tous les algorithmes possèdent une fonction train()
"""

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
        q[state, action] = (1 - self.alpha) * q[state, action] + self.alpha * (reward + self.gamma * np.max(q[next_state]))
        return q

class QLearningStandard(RL):
    def __init__(self, mdp, gamma, alpha, epsilon_init, reward_weights):
        super().__init__(mdp, gamma, alpha, epsilon_init)
        self.reward_weights = reward_weights

    def train(self, nb_episodes, nb_steps, nb_criteres, verbose=False, diminution_epsilon_steps=2000):
        # nb_steps: nombre de steps par épisode -> 20.000 selon papier
        
        all_reward_standard = np.zeros((nb_criteres, nb_episodes, nb_steps), dtype=float)  # taille: K x NB_EPISODES x NB_STEPS
        all_state_space_traversal = []
        all_state = []

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

            states = []

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

                #Stockage de l'etat visité
                states.append(state)

                # stockage des récompenses
                for k in range(nb_criteres):
                    all_reward_standard[k][episode][t] = reward[k] # reward obtenue             

                state = state_next
                if diminution_epsilon_steps > 0 and t > 0 and t % diminution_epsilon_steps == 0:
                    epsilon = epsilon/2  # diminution epsilon par épisode
                    # epsilon = epsilon * 0.99  # -> donne de meilleurs résultats pour le calcule de VMORE

            all_state_space_traversal.append(copy.deepcopy(state_space_traversal))
            all_state.append(states)

            if verbose:
                print("Q-values à la fin de l'épisode:")
                print(q)

        return q, all_reward_standard, all_state

class QLearningObjectiveSwitching(RL):
    def __init__(self, mdp, gamma, alpha, epsilon_init):
        super().__init__(mdp, gamma, alpha, epsilon_init)

    def train(self, nb_episodes, nb_steps, nb_criteres, verbose=False, diminution_epsilon_steps=2000):
        all_reward_objective_switching = np.zeros((nb_criteres, nb_episodes, nb_steps), dtype=float)  # taille: K x NB_EPISODES x NB_STEPS
        all_state = []

        for episode in tqdm(range(nb_episodes), desc="Episodes"): 
            if verbose:
                print("Episode", episode)
            Q0 = self.mdp.q_init()
            Q1 = self.mdp.q_init()
            state = self.mdp.reset()
            epsilon = self.epsilon_init  # reset epsilon au début de chaque épisode

            states = []

            # somme cumulée des récompenses pour chaque critère : Pour pouvoir faire le switching
            sum_cum_r0 = 0
            sum_cum_r1 = 0

            for t in range(nb_steps):

                if sum_cum_r0 < sum_cum_r1:
                    q, obj = Q0, 0
                else:
                    q, obj = Q1, 1
                
                a = self.choose_action_greedy(state, q, epsilon)
                state_next, reward = self.mdp.step(state, a)

                # rewards
                r0 = reward[0]
                r1 = reward[1]

                sum_cum_r0 += r0
                sum_cum_r1 += r1

                # MAJ de chaque Q-table avec SON reward
                self.Q_learning_update(state, a, r0, state_next, Q0)
                self.Q_learning_update(state, a, r1, state_next, Q1)

                #Stockage de l'etat visité
                states.append(state)

                # stockage des récompenses
                for k in range(nb_criteres):
                    all_reward_objective_switching[k][episode][t] = reward[k] # reward obtenue             

                state = state_next
                if diminution_epsilon_steps > 0 and t > 0 and t % diminution_epsilon_steps == 0:
                    epsilon = epsilon/2 # diminution epsilon par épisode
                    # epsilon = epsilon * 0.99  # -> donne de meilleurs résultats pour le calcule de VMORE

            all_state.append(states)

            if verbose:
                print("Q-values à la fin de l'épisode:")
                print(q)

        return (Q0, Q1), all_reward_objective_switching, all_state


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

    # def discretize_w(self, w, verbose=False):
    #     """ retourne le poids dans W le plus proche de w et son index """
    #     if verbose:
    #         print("discretize_w input:", w)
    #         print("W:", self.W)
    #     diff = np.abs(w - self.W[None, :])
    #     idx = np.argmin(diff, axis=1)
    #     if verbose:
    #         print("idx proche:", idx)
    #     return self.W[idx], idx # valeur discrétisée et index dans W
    
    def discretize_w(self, w, verbose=False):
        """ retourne le poids dans W le plus proche de w et son index """
        if verbose:
            print("discretize_w input:", w)
            print("W:", self.W)
        w = float(w)
        idx = int(np.argmin(np.abs(self.W - w)))
        return float(self.W[idx]), idx # valeur discrétisée et index dans W

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
        self.q_lin += 1e-3 * np.random.randn(*self.q_lin.shape) #bruit initial
        print("q_lin shape:", self.q_lin.shape)

        evolution_w0 = np.zeros((nb_episodes, nb_steps))  # pour stocker l'évolution du poids w0
        evolution_w0_discret = np.zeros((nb_episodes, nb_steps))  # pour stocker l'évolution du poids w0 discretisé
  
        all_reward_MORE = np.zeros((nb_criteres, nb_episodes, nb_steps))  # taille: K x NB_EPISODES x NB_STEPS
        all_state_space_traversal = []
        all_state = []

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

            states = []

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

                #Stockage de l'etat visité
                states.append(state)
        
                state = state_next
                w_0 = new_w0
                idx_w0 = new_idx_w0
            
                
                if diminution_epsilon_steps > 0 and t > 0 and t % diminution_epsilon_steps == 0:
                    epsilon = epsilon/2  # diminution epsilon par épisode
                    # epsilon = epsilon * 0.99  # -> donne de meilleurs résultats pour le calcule de VMORE

            all_state_space_traversal.append(copy.deepcopy(state_space_traversal))
            all_state.append(states)

        if verbose:
            print("*$*$*$*$*$*$ Fin training Q_MORE *$*$*$*$*$*$")

        return self.q_lin, all_reward_MORE, evolution_w0, evolution_w0_discret, all_state
    

# MORE avec LLR
class LocalLinearModel:
    """
    LLR pour approximer Q_LIN(s, w0, a)
    - entrée : w0
    - sortie : (Q0, Q1)
    """
    def __init__(self, radius=0.4, max_points=100):
        self.radius = radius
        self.max_points = max_points
        self.W0 = []   # w0_i
        self.Q = []    # Q_i 

    def add_sample(self, w0, q, merge_radius=0.02):
        w0 = float(w0)
        q = np.array(q, dtype=float)

        self.W0.append(w0)
        self.Q.append(q)

        if len(self.W0) > self.max_points:
            self.W0.pop(0)
            self.Q.pop(0)

    def predict(self, w0):
        """
        Standard Locally Linear Regression (LLR)
        - entrée : w0 
        - sortie : Q_LIN(s, w0, a) 
        """
        if len(self.W0) < 2:
            return np.zeros(2)

        w0 = float(w0)
        W0 = np.array(self.W0)          
        Q  = np.vstack(self.Q)         

        # distances locales en weight-space (1D)
        d = np.abs(W0 - w0)
        mask = d <= self.radius

        # aucun voisin local → plus proche voisin
        if not np.any(mask):
            return Q[np.argmin(d)]
              
        Qloc = Q[mask]               
        dloc = d[mask]            

        # noyau gaussien
        weights = np.exp(-(dloc / self.radius) ** 2)
        weights /= np.sum(weights)                  

        return weights @ Qloc
 
class QLearningMORE_LLR(RL):
    """
    MORE-Q Learning avec LLR
    """
    def __init__(self, mdp, gamma, alpha, epsilon_init, gamma_paste, radius=0.4, max_points=100):
        super().__init__(mdp, gamma, alpha, epsilon_init)
        self.gamma_paste = gamma_paste
        self.radius = radius
        self.max_points = max_points
        self.models = None   # 6 LLR : (s,a)

    def Q_more(self, q_lin, w0):
        """
        Q_MORE(s,w,a) = - ( w0*exp(-Q0) + (1-w0)*exp(-Q1) )
        """
        q0, q1 = q_lin
        return - (w0 * np.exp(-q0) + (1 - w0) * np.exp(-q1))
        
    def update_w(self, w0, reward):
        """
        maj w0
        """
        r0, r1 = reward
        a0 = (w0 ** self.gamma_paste) * np.exp(-r0)
        a1 = ((1 - w0) ** self.gamma_paste) * np.exp(-r1)
        w0 = a0 / (a0 + a1)
        return w0

    def greedy_more_action(self, state, w0):
        qvals = []
        for a in self.mdp.actions:
            q_lin = self.models[state][a].predict(w0)
            qvals.append(self.Q_more(q_lin, w0))
        return int(np.argmax(qvals))

    def train(self, nb_episodes, nb_steps, nb_criteres, verbose=False,
              diminution_epsilon_steps=2000, w0_init=0.5):

        # init des 6 LLR
        self.models = {
            s: {a: LocalLinearModel(self.radius, self.max_points)
                for a in self.mdp.actions}
            for s in self.mdp.states
        }

        all_rewards = np.zeros((2, nb_episodes, nb_steps))
        all_states = []
        all_w0 = np.zeros((nb_episodes, nb_steps))

        for ep in tqdm(range(nb_episodes), desc="MORE-LLR"):
            state = self.mdp.reset()
            epsilon = self.epsilon_init
            w0 = float(w0_init)
            states = []

            for t in range(nb_steps):

                if verbose:
                    print("\nstep", t)
                    print(f"w_0={w0}, w_1={1 - w0}, epsilon={epsilon}")

                if diminution_epsilon_steps > 0 and t > 0 and t % diminution_epsilon_steps == 0:
                    epsilon *= 0.5
                    print(f"{t=}, w_0={w0}, w_1={1 - w0}, epsilon={epsilon}")

                # choix action
                if np.random.rand() < epsilon:
                    a = np.random.choice(self.mdp.actions)
                else:
                    a = self.greedy_more_action(state, w0)

                # transition
                next_state, reward = self.mdp.step(state, a)
                all_rewards[0, ep, t] = reward[0]
                all_rewards[1, ep, t] = reward[1]

                # update w
                w0_next = self.update_w(w0, reward)
                all_w0[ep, t] = w0_next

                # action optimale suivante
                a_star = self.greedy_more_action(next_state, w0_next)

                # TD update vectoriel
                q_lin_cur = self.models[state][a].predict(w0)
                q_lin_next = self.models[next_state][a_star].predict(w0_next)
                target = np.array(reward) + self.gamma * q_lin_next
                q_updated = (1 - self.alpha) * q_lin_cur + self.alpha * target

                # apprentissage LLR
                self.models[state][a].add_sample(w0, q_updated)

                states.append(state)
                state = next_state
                w0 = w0_next

            all_states.append(states)

        return self.models, all_rewards, all_states, all_w0
                         
