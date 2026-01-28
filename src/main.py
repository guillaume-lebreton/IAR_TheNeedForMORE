import numpy as np
import matplotlib.pyplot as plt
import time

from algos_rl import (
    QMORE,
    QLearningStandard,
    QLearningObjectiveSwitching,
    QMORE_DISCR_TAB
)

from mdp import mdp_standard, mdp_satiete, mdp_batterie, mdp_setpoint
from utils import (
    avg_stay_length_curves,
    prepare_objective_curves,
    plot_all
)

# ======================================================
# PARAMÈTRES GLOBAUX
# ======================================================
NB_EPISODES = 10
NB_STEPS = 20000

GAMMA = 0.9
ALPHA = 0.2
EPSILON = 0.5

K = 2  # nombre d'objectifs

# MORE
GAMMA_PASTE = 0.99
PAS_DISCR = 0.0001


# ======================================================
# CHOIX DU MDP
# ======================================================
def choose_mdp(name="standard"):
    if name == "standard":
        return mdp_standard
    elif name == "satiete":
        return mdp_satiete
    elif name == "batterie":
        return mdp_batterie
    elif name == "setpoint":
        return mdp_setpoint
    else:
        raise ValueError("MDP inconnu")


# ======================================================
# PIPELINE COMMUN DE PLOT
# ======================================================
def run_and_plot(algo_name, algo, mdp_name, verbose=False):
    print("=" * 60)
    print(f"Running {algo_name}")
    print("=" * 60)

    start_time = time.perf_counter()

    outputs = algo.train(
        nb_episodes=NB_EPISODES,
        nb_steps=NB_STEPS,
        nb_criteres=K,
        verbose=verbose
    )

    exec_time = time.perf_counter() - start_time

    # Déballage selon l'algo
    if algo_name == "MORE":
        _, all_rewards, all_states = outputs
    else:
        _, all_rewards, all_states = outputs

    # ============================
    # COLONNE 1 — STATE TRAVERSAL
    # ============================
    stays = avg_stay_length_curves(all_states)
    # plot_state_space_traversal(stays, algo_name)

    # ============================
    # COLONNE 2 — REWARDS (EMA)
    # ============================
    r0_curves, r1_curves = prepare_objective_curves(
        all_rewards,
        beta=0.001
    )
    # plot_objective_curves(r0_curves, r1_curves, algo_name)

    # ============================
    # COLONNE 3 — V_MORE
    # ============================
    # plot_MORE_curve(r0_curves, r1_curves, algo_name)

    #Plot les 3 dans une seule fenetre
    path = "../plots/mdp_" + mdp_name + "/" + algo_name
    plot_all(stays, r0_curves, r1_curves, algo_name, mdp_name, exec_time, path, show=False)


# ======================================================
# MAIN
# ======================================================
def main(mdp_name,standard=True, switch=True, more=True, more_discr=True):

    mdp = choose_mdp(mdp_name)
    print("MDP choisi :", mdp_name)

    # -----------------------------
    # Q-LEARNING STANDARD
    # -----------------------------
    if standard:
        q_standard = QLearningStandard(
            mdp=mdp,
            gamma=GAMMA,
            alpha=ALPHA,
            epsilon_init=EPSILON,
            reward_weights=[1, 1]
        )
        run_and_plot("standard", q_standard, mdp_name)

    # -----------------------------
    # OBJECTIVE SWITCHING
    # -----------------------------
    if switch:
        q_switch = QLearningObjectiveSwitching(
            mdp=mdp,
            gamma=GAMMA,
            gamma_past=GAMMA_PASTE,
            alpha=ALPHA,
            epsilon_init=EPSILON
        )
        run_and_plot("switch", q_switch, mdp_name)

    # -----------------------------
    # MORE-Q LLR
    # -----------------------------
    if more:
        more_llr = QMORE(
            mdp=mdp,
            gamma=GAMMA,
            alpha=ALPHA,
            epsilon_init=EPSILON,
            gamma_paste=GAMMA_PASTE
        )
        run_and_plot("MORE", more_llr, mdp_name, verbose=True)
    
    # -----------------------------
    # MORE-Q DISCRÉTISATION TAB
    # -----------------------------
    if more_discr:
        # W = np.arange(0, 1.01, PAS_DISCR)  # discretisation des poids w0 dans [0,1] avec un pas de PAS_DISCR
        W = np.arange(1e-3, 1.0, PAS_DISCR)  # exclut 0 et 1
        print("W:", W)
        more_discr = QMORE_DISCR_TAB(
            mdp=mdp,
            gamma=GAMMA,
            alpha=ALPHA,
            epsilon_init=EPSILON,
            gamma_paste=GAMMA_PASTE,
            W=W
        )
        algo_name = "MORE_discr"
        run_and_plot(algo_name, more_discr, mdp_name, verbose=False)

if __name__ == "__main__":
    mdp_name = "setpoint"  # standard, satiete, batterie, setpoint
    main(mdp_name, standard=True, switch=True, more=False, more_discr=True)
