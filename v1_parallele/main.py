from algos_rl import *
from mdp import mdp_standard, mdp_satiete, mdp_batterie, mdp_setpoint
import numpy as np
import matplotlib.pyplot as plt
from utils import *


PAS_DISCR = 0.001 # Pour MORE
EPS = 0.5
LR = 0.2 
GAMMA_PASTE = 0.99 # Pour MORE
NB_EPISODES = 10
NB_STEPS = 20000
K = 2 # nombre d'observations (rewards) par état
WINDOW_SIZE = 1000
GAMMA = 0.9

def main(qlearning_standard=False, qlearning_objective_switching=False, qlearning_more=False, mdp_choice="standard", plot_all=False):
    print(K,"Critères")
    print(NB_EPISODES, "nb episodes")
    print(NB_STEPS, "nb steps")

    if mdp_choice == "standard":
        mdp = mdp_standard
        print("MDP standard choisi")
    elif mdp_choice == "satiete":
        mdp = mdp_satiete
        print("MDP satiete choisi")
    elif mdp_choice == "batterie":
        mdp = mdp_batterie
        print("MDP batterie choisi")
    elif mdp_choice == "setpoint":
        mdp = mdp_setpoint
        print("MDP setpoint choisi")
    else:
        raise ValueError("mdp_choice doit être 'standard', 'satiete', 'batterie' ou 'setpoint'")

####################################### QLearning - Standard #######################################
    if qlearning_standard:
        print("*************************** QLearning - Standard ***************************")
        q_learning_standard = QLearningStandard(mdp=mdp, gamma=GAMMA, alpha=LR, epsilon_init=EPS, reward_weights=[1,1])
        q_standard, all_rewards_standard, all_state_space_traversal_standard = q_learning_standard.train(nb_episodes=NB_EPISODES, nb_steps=NB_STEPS, nb_criteres=K, verbose=False)

        if plot_all:
            results = {}
            results["standard"] = {
                "q": q_standard,
                "all_rewards": all_rewards_standard,
                "all_state_space_traversal": all_state_space_traversal_standard
            }
        else :
            sliding_cumulative_rewards_STANDARD = sliding_cumulative_reward(all_rewards_standard, window_size=WINDOW_SIZE)
            plot_cumulative_sliding_reward(
                sliding_cumulative_rewards_STANDARD,
                window_size=WINDOW_SIZE,
                eps=EPS,
                lr=LR,
                nb_episodes=NB_EPISODES,
                nb_criteres=K
            )

            v_more_standard = V_MORE(sliding_cumulative_rewards_STANDARD)
            
            plot_space_traversal(all_state_space_traversal_standard, NB_EPISODES, NB_STEPS, EPS, LR, len(mdp.states))

            plt.plot(v_more_standard)
            plt.title("V^MORE: standard")
            plt.xlabel("t")
            plt.ylabel("-log(-Vmore)")
            plt.grid()
            plt.show()
            print("Q valeurs apprises:")
            print(q_standard)
            print("\n\n")

####################################### Objective Switching #######################################
    if qlearning_objective_switching:
        print("*************************** Objective Switching ***************************")
        obj_switching = QLearningObjectiveSwitching(mdp=mdp, gamma=GAMMA, alpha=LR, epsilon_init=EPS)
        q_objective_switching, all_rewards_objective_switching, all_state_space_traversal_objective_switching = obj_switching.train(nb_episodes=NB_EPISODES, nb_steps=NB_STEPS, nb_criteres=K, verbose=False)
        
        if plot_all:
            results["objective switching"] = {
                "q": q_objective_switching,
                "all_rewards": all_rewards_objective_switching,
                "all_state_space_traversal": all_state_space_traversal_objective_switching
            }
        else :
            sliding_cumulative_rewards_OBJECTIVE_SWITCHING = sliding_cumulative_reward(all_rewards_objective_switching, window_size=WINDOW_SIZE)
            plot_cumulative_sliding_reward(
                sliding_cumulative_rewards_OBJECTIVE_SWITCHING,
                window_size=WINDOW_SIZE,
                eps=EPS,
                lr=LR,
                nb_episodes=NB_EPISODES,
                nb_criteres=K,
                nameAlgo="objective switching"
            )

            v_more_objective_switching = V_MORE(sliding_cumulative_rewards_OBJECTIVE_SWITCHING)

            plot_space_traversal(all_state_space_traversal_objective_switching, NB_EPISODES, NB_STEPS, EPS, LR, len(mdp.states))

            plt.plot(v_more_objective_switching)
            plt.title("V^MORE: objective switching")
            plt.xlabel("t")
            plt.ylabel("-log(-Vmore)")
            plt.grid()
            plt.show()
            print("Q valeurs apprises:")
            print(q_objective_switching)
            print("\n\n")

####################################### MORE #######################################
    if qlearning_more:
        print("*************************** QLearning - MORE ***************************")
        W = np.arange(0, 1.01, PAS_DISCR)  # discretisation des poids w0 dans [0,1] avec un pas de PAS_DISCR
        print("W:", W)
        q_learning_more = QLearningMORE(mdp=mdp, gamma=GAMMA, alpha=LR, epsilon_init=EPS, gamma_paste=GAMMA_PASTE, W=W)
        q_more, all_rewards_more, evolution_w0, evolution_w0_discret, all_state_space_traversal_more = q_learning_more.train(nb_episodes=NB_EPISODES, nb_steps=NB_STEPS, nb_criteres=K, verbose=False)

        if plot_all:
            results["MORE"] = {
                "q": q_more,
                "all_rewards": all_rewards_more,
                "all_state_space_traversal": all_state_space_traversal_more
            }
        else :
            sliding_cumulative_rewards_MORE = sliding_cumulative_reward(all_rewards_more, window_size=WINDOW_SIZE)

            plot_cumulative_sliding_reward(
                sliding_cumulative_rewards_MORE,
                window_size=WINDOW_SIZE,
                eps=EPS,
                lr=LR,
                nb_episodes=NB_EPISODES,
                nb_criteres=K,
                nameAlgo="MORE"
            )

            v_more_more = V_MORE(sliding_cumulative_rewards_MORE)

            plot_space_traversal(all_state_space_traversal_more, NB_EPISODES, NB_STEPS, EPS, LR, len(mdp.states))

            plt.plot(v_more_more)
            plt.title("V^MORE: MORE")
            plt.xlabel("t")
            plt.ylabel("-log(-Vmore)")
            plt.grid()
            plt.show()
            print("Q valeurs apprises:")
            print(q_more)
            print("\n\n")

        for idx_w in range(len(q_learning_more.W)):
            print("-----------------")
            print(f"Q_MORE pour w0 = {q_learning_more.W[idx_w]}")
            for s in q_learning_more.mdp.states:
                for a in q_learning_more.mdp.actions:
                    print(f"s={s}, a={a} -> Q_MORE:", q_learning_more.Q_more(s, idx_w, a, q_learning_more.W[idx_w]))


        # n_states = len(q_learning_more.mdp.states)
        # n_actions = len(q_learning_more.mdp.actions)
        # n_w = len(q_learning_more.W)

        # # Q_values[s_idx, a_idx, w_idx]
        # Q_values = np.zeros((n_states, n_actions, n_w))

        # for s in (q_learning_more.mdp.states):
        #     for a in (q_learning_more.mdp.actions):
        #         for k, w0 in enumerate(q_learning_more.W):
        #             Q_values[s, a, k] = q_learning_more.Q_more(s, k, a, w0)

        # # indice de la meilleure action : argmax sur l'axe action
        # best_actions = np.argmax(Q_values, axis=1)   # shape: (n_states, n_w)
        # print(np.unique(best_actions))

        # plt.imshow(best_actions, aspect='auto', origin='lower')
        # plt.colorbar(label='Action optimale')

        # plt.yticks(range(n_states), [f"s={s}" for s in mdp.states])
        # plt.xlabel("w0")
        # plt.ylabel("Etat")
        # plt.title("Meilleure action par état en fonction de w0")
        # plt.tight_layout()
        # plt.show()

    if plot_all:
        plot_global_3x3(mdp=mdp,
                        results=results,
                        nb_episodes=NB_EPISODES,
                        nb_steps=NB_STEPS,
                        window_size=WINDOW_SIZE,
                        K=K,
                        eps=EPS,
                        lr=LR
                    )

if __name__ == "__main__":
    main(qlearning_standard=True, 
         qlearning_objective_switching=True, 
         qlearning_more=True,
         mdp_choice="setpoint",
         plot_all=True
         )
    