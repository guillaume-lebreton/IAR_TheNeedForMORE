from utils import *
from MORE import *
import matplotlib.pyplot as plt


if __name__ == "__main__":

    mdp_sasiete = MDP(
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
    discount_factor=0.9,
    sasiete=True,
    augmentation_faim=0.001,
    diminution_faim=0.2,

    augmentation_soif=0.01,
    diminution_soif=0.1
    )

    mdp = mdp_sasiete

################################ Standard Q-learning ################################
    print("\n######## Standard Q-learning ########")
    q_standard, all_rewards_standard, all_state_space_traversal = train_q_standard(mdp, nb_episodes=NB_EPISODES, gamma=mdp.discount_factor, nb_steps=NB_STEPS, epsilon=EPS, alpha=LR, nb_criteres=K, verbose=False)
    print("all_state_space_traversal:")
    print(all_state_space_traversal)
    
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

    plot_space_traversal(all_state_space_traversal, NB_EPISODES, NB_STEPS, EPS, LR)

    plt.plot(v_more_standard)
    plt.title("V^MORE: standard")
    plt.xlabel("t")
    plt.ylabel("-log(-Vmore)")
    plt.grid()
    plt.show()
    print("Q valeurs apprises:")
    print(q_standard)
    print("\n\n")

    plt.imshow(q_standard, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Q-values heatmap: standard")
    plt.xlabel("Actions")
    plt.ylabel("Etats")
    plt.show()


