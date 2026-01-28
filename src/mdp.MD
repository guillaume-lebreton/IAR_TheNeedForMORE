# Explication MDP:

* MDP classe de base
    - l'agent reçoit toujours la récompense associée à l'état dans lequel il se trouve
---

* MDP satiete:
    - l'agent démarre avec un niveau de faim et de soif maximal à 1
        - ```niveau_faim = 1```
        - ```niveau_soif = 1```
        
    - Lorsque l'agent mange (etat 0):
        - son niveau de faim diminue de ```diminution_faim``` jusqu'à un minimum de 0
        - son niveau de soif augmente de ```augmentation_soif``` jusqu'à un maximum de 1
    - inversement, lorsque l'agent boit (etat 1):
        - son niveau de soif diminue de ```diminution_soif``` jusqu'à un minimum de 0
        - son niveau de faim augmente de ```augmentation_faim``` jusqu'à un maximum de 1

    - La recompense reçue par l'agent est alors une multiplication de la récompense de base par le niveau de faim ou de soif:
        - en mangeant: ```reward = r0 * niveau_faim```
        - en buvant: ```reward = r1 * niveau_soif```

    - Ainsi, plus l'agent est rassasié en nourriture (niveau proche de 0), moins il reçoit de récompense en mangeant, et plus il est affamé (niveau proche de 1), plus il reçoit de récompense en mangeant. Même principe pour la soif en buvant.
---
* MDP batterie:
    - l'agent démarre avec deux batterie vide à 0
        - ```niveau_faim_batterie = 0```
        - ```niveau_soif_batterie = 0```
        
    - À chaque états (manger ou boire), le niveau de la batterie associée augmente de ```augmentation_faim``` ou ```augmentation_soif``` jusqu'à un maximum de ```Rmax=2``` (par exemple)



    - Ainsi, plus le niveau de batterie est bas, moins l'agent reçoit de récompense en mangeant ou en buvant, et plus le niveau de batterie est élevé, plus il reçoit de récompense.

    - Tant que ```niveau_faim_batterie``` est dans l'intervalle **[0, Rmax[**, l'agent reçoit la recompense standard de nourriture en mangeant(etat 0), equivalent pour la boisson (etat 1) avec ```niveau_soif_batterie```.

    - Si le niveau de batterie est à ```Rmax```, la récompense reçue par l'agent est alors de 0, l'agent ne reçoit plus de récompense dans cette état si sa batterie est pleine.
---
* MDP setpoint:
    - identique aux MDP batterie, mais l'agent doit maintenir ses niveaux de faim et de soif autour d'une valeur prédéfinie (setpoint)

    - Plus il s'en éloigne, plus la récompense diminue.
    - la reward obtenue en etat 0 est:
        - ```r0 = r0 * (1 - penalite(niveau_batterie_faim))``` 
        - ```r1 = r1 * (1 - penalite(niveau_batterie_soif))```

    - où la fonction penalite est définie comme:
        - ```penalite(niveau_batterie) = dist(setpoint, niveau_batterie) / Rmax ```   -> **[0, 1]** car dist < Rmax
        - avec **dist** la **valeur absolue** entre le **niveau de batterie** **(entre 0 et Rmax)** et le **setpoint**

    - plus la distance entre le ``niveau de batterie`` et le ```setpoint``` est grande, plus la pénalité est élevée (pénalité proche de 1) donc on multiplie la reward par (1 - pénalité) qui signifie :
        - si ```penalite``` est proche de 1 (niveau de batterie loin du setpoint), la reward devient très faible voire nulle
        - si la distance est proche de 0 (niveau de batterie proche du setpoint), la pénalité est proche de 0 donc on multiplie la reward par (1 - penalité) proche de 1, donc la reward reste élevée.