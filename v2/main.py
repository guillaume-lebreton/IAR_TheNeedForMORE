import numpy as np
import matplotlib.pyplot as plt

from algos_rl import (
    QLearningStandard,
    QLearningObjectiveSwitching,
    QLearningMORE,
    QLearningMORE_LLR
)

from mdp import mdp_standard, mdp_satiete, mdp_batterie, mdp_setpoint
from utils import (
    avg_stay_length_curves,
    plot_state_space_traversal,
    prepare_objective_curves,
    plot_objective_curves,
    plot_MORE_curve,
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
PAS_DISCR = 0.001


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
def run_and_plot(algo_name, algo):
    print("=" * 60)
    print(f"Running {algo_name}")
    print("=" * 60)

    outputs = algo.train(
        nb_episodes=NB_EPISODES,
        nb_steps=NB_STEPS,
        nb_criteres=K,
        verbose=False
    )

    # Déballage selon l'algo
    if algo_name == "MORE":
        _, all_rewards, all_states, _ = outputs
    else:
        _, all_rewards, all_states = outputs

    # ============================
    # COLONNE 1 — STATE TRAVERSAL
    # ============================
    stays = avg_stay_length_curves(all_states)
    plot_state_space_traversal(stays, algo_name)

    # ============================
    # COLONNE 2 — REWARDS (EMA)
    # ============================
    r0_curves, r1_curves = prepare_objective_curves(
        all_rewards,
        beta=0.001
    )
    plot_objective_curves(r0_curves, r1_curves, algo_name)

    # ============================
    # COLONNE 3 — V_MORE
    # ============================
    plot_MORE_curve(r0_curves, r1_curves, algo_name)


# ======================================================
# MAIN
# ======================================================
def main():
    mdp = choose_mdp("standard")
    print("MDP choisi :", mdp)

    # -----------------------------
    # Q-LEARNING STANDARD
    # -----------------------------
    # q_standard = QLearningStandard(
    #     mdp=mdp,
    #     gamma=GAMMA,
    #     alpha=ALPHA,
    #     epsilon_init=EPSILON,
    #     reward_weights=[1, 1]
    # )
    # run_and_plot("standard", q_standard)

    # -----------------------------
    # OBJECTIVE SWITCHING
    # -----------------------------
    # q_switch = QLearningObjectiveSwitching(
    #     mdp=mdp,
    #     gamma=GAMMA,
    #     alpha=ALPHA,
    #     epsilon_init=EPSILON
    # )
    # run_and_plot("switch", q_switch)

    # -----------------------------
    # MORE-Q
    # -----------------------------
    # W = np.arange(0, 1.01, PAS_DISCR)

    # q_more = QLearningMORE(
    #     mdp=mdp,
    #     gamma=GAMMA,
    #     alpha=ALPHA,
    #     epsilon_init=EPSILON,
    #     gamma_paste=GAMMA_PASTE,
    #     W=W
    # )
    # run_and_plot("MORE", q_more)

    # -----------------------------
    # MORE-Q LLR
    # -----------------------------
    more_llr = QLearningMORE_LLR(
        mdp=mdp,
        gamma=GAMMA,
        alpha=ALPHA,
        epsilon_init=EPSILON,
        gamma_paste=GAMMA_PASTE
    )
    run_and_plot("MORE", more_llr)

if __name__ == "__main__":
    main()
