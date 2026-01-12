import numpy as np


class MDP:
    def __init__(self, states, init_state, actions, transitions, rewards, discount_factor):
        self.states = states
        self.init_state = init_state
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.discount_factor = discount_factor


    def reset(self):
        return self.init_state
    
    def get_next_state(self, current_state, action):
        if (current_state, action) not in self.transitions:
            raise ValueError(f"Paire ({current_state}, {action}) non valide")
        return self.transitions[(current_state, action)]
    
    def get_reward(self, state):
        """ Retourne la récompense associée à l'état 'state' en prenant en compte la satiété si activée """
        if state not in self.rewards:
            raise ValueError(f"Etat {state} non valide")
        r0, r1 = self.rewards[state]
        return (r0, r1)
        

    def step(self, current_state, action):
        next_state = self.get_next_state(current_state, action)
        reward = self.get_reward(next_state)
        return next_state, reward

    def q_init(self):
        q = np.zeros((len(self.states), len(self.actions)))
        return q
    
class MDP_Satiete(MDP):
    """ MDP imitant le fait que :
        - plus l'agent est rassasié, moins il a de récompense en mangeant/buvant
        - plus l'agent est affamé/assoiffé, plus sont gain en mangeant/buvant augmente
    """
    def __init__(self, states, init_state, actions, transitions, rewards, discount_factor, augmentation_faim=0.1, augmentation_soif=0.1, diminution_faim=0.1, diminution_soif=0.1):
        super().__init__(states, init_state, actions, transitions, rewards, discount_factor)
        self.augmentation_faim = augmentation_faim
        self.augmentation_soif = augmentation_soif
        self.diminution_faim = diminution_faim
        self.diminution_soif = diminution_soif
        self.niveau_faim = 1 # (1=tres faim, 0= rassasié) appartient à [0,1]
        self.niveau_soif = 1 # (1=tres soif, 0= rassasié) appartient à [0,1]

    def reset(self):
        self.niveau_faim = 1
        self.niveau_soif = 1
        return self.init_state
    
    def get_reward(self, state):
        r0, r1 = super().get_reward(state)

        # application de la satiété
        r0 = r0 * self.niveau_faim
        r1 = r1 * self.niveau_soif

        if state == 0: 
            self.niveau_faim = max(0, self.niveau_faim - self.diminution_faim) # l'agent vient de manger -> moins faim 
        else:
            self.niveau_faim = min(1, self.niveau_faim + self.augmentation_faim) # l'agent n'a pas mangé -> plus faim

        if state == 2:
            self.niveau_soif = max(0, self.niveau_soif - self.diminution_soif) # l'agent vient de boire -> moins soif
        else:
            self.niveau_soif = min(1, self.niveau_soif + self.augmentation_soif) # l'agent n'a pas bu -> plus soif

        return (r0, r1)

class MDP_Batterie(MDP):
    """
        MDP immitant une batterie qui se décharge et se recharge.
        Lorsque la batterie est pleine, l'agent ne reçoit plus de récompense en mangeant/buvant
        Lorsque la batterie est vide, l'agent reçoit la récompense complète en mangeant/buvant
    """
    def __init__(self, states, init_state, actions, transitions, rewards, discount_factor, augmentation_faim=0.1, augmentation_soif=0.1, diminution_faim=0.1, diminution_soif=0.1, Rmax=2):
        super().__init__(states, init_state, actions, transitions, rewards, discount_factor)
        self.augmentation_faim = augmentation_faim
        self.augmentation_soif = augmentation_soif
        self.diminution_faim = diminution_faim
        self.diminution_soif = diminution_soif
        self.Rmax = Rmax
        self.niveau_faim_batterie = 0 # niveau de batterie associé à la faim (0=vide, Rmax=plein)
        self.niveau_soif_batterie = 0  # niveau de batterie associé à la soif (0=vide, Rmax=plein)


    def reset(self):
        self.niveau_faim_batterie = 0
        self.niveau_soif_batterie = 0
        return self.init_state
    
    def get_reward(self, state):
        r0, r1 = super().get_reward(state)

        if self.niveau_faim_batterie >= self.Rmax:
            r0 = 0
        if self.niveau_soif_batterie >= self.Rmax:
            r1 = 0

        if state == 0: 
            self.niveau_faim_batterie = min(self.niveau_faim_batterie + self.augmentation_faim, self.Rmax) # l'agent vient de manger -> recharge batterie faim
        else:
            self.niveau_faim_batterie = max(self.niveau_faim_batterie - self.diminution_faim, 0) # l'agent n'a pas mangé -> le niveau de batterie faim diminue

        if state == 2:
            self.niveau_soif_batterie = min(self.niveau_soif_batterie + self.augmentation_soif, self.Rmax) # l'agent vient de boire -> recharge batterie soif
        else:
            self.niveau_soif_batterie = max(self.niveau_soif_batterie - self.diminution_soif, 0) # l'agent n'a pas bu -> le niveau de batterie soif diminue

        return (r0, r1)

class MDP_Setpoint(MDP_Batterie):
    """ MDP imitant le fait que l'agent doit maintenir ses niveaux de faim et de soif autour d'une valeur prédéfinie (setpoint)
        Plus il s'en éloigne, plus la récompense diminue.
    """
    def __init__(self, states, init_state, actions, transitions, rewards, discount_factor, augmentation_faim=0.1, augmentation_soif=0.1, diminution_faim=0.1, diminution_soif=0.1, Rmax=2, setpoint_faim=1, setpoint_soif=1, tolerance=0.01):
        super().__init__(states, init_state, actions, transitions, rewards, discount_factor, augmentation_faim, augmentation_soif, diminution_faim, diminution_soif, Rmax)
        self.setpoint_faim = setpoint_faim
        self.setpoint_soif = setpoint_soif
        self.tolerance = tolerance


    def penalite(self, niveau_batterie):
        """ Calcule une pénalité en fonction de la distance au setpoint """
        distance = abs(niveau_batterie - self.setpoint_faim)
        if distance < self.tolerance:
            return 1  # pas de pénalité si dans la tolérance
        penalite = distance / self.Rmax 
        return penalite

    def get_reward(self, state):
        r0, r1 = super().get_reward(state)
        # ajustement de la récompense en fonction de la distance au setpoint
        
        r0 *= (1 - self.penalite(self.niveau_faim_batterie))
        r1 *= (1 - self.penalite(self.niveau_soif_batterie))

        return (r0, r1)


mdp_standard = MDP(
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

mdp_satiete = MDP_Satiete(
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
    augmentation_faim=0.001, 
    augmentation_soif=0.001, 
    diminution_faim=0.01, 
    diminution_soif=0.01
)

mdp_batterie = MDP_Batterie(
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
    augmentation_faim=0.1, 
    augmentation_soif=0.1, 
    diminution_faim=0.1, 
    diminution_soif=0.1,
    Rmax=2
)

mdp_setpoint = MDP_Setpoint(
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
    augmentation_faim=0.1, 
    augmentation_soif=0.1, 
    diminution_faim=0.1, 
    diminution_soif=0.1,
    Rmax=2,
    setpoint_faim=1.90,
    setpoint_soif=1.90
)



