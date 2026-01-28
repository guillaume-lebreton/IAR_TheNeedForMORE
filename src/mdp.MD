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
    - l'agent démarre avec deux batteries vides à 0
        - ```niveau_faim_batterie = 0```
        - ```niveau_soif_batterie = 0```
        
    - À chaque états (manger ou boire), le niveau de la batterie associée augmente de ```augmentation_faim``` ou ```augmentation_soif``` jusqu'à un maximum de ```Rmax=2``` (par exemple)


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


## Environnements (MDP)

### MDP de base (`MDP`)
- **États** : {0, 1, 2}
- **Actions** : {0, 1}
- La dynamique est définie par `transitions[(state, action)]`.
- À chaque pas, l’agent reçoit la **récompense associée à l’état d’arrivée** :
  \[
  (s_{t+1}, r_{t+1}) = \text{step}(s_t, a_t), \quad r_{t+1} = R(s_{t+1})
  \]
- La récompense est **bi-objectif** : `reward = (r0, r1)`.

---

### MDP avec satiété (`MDP_Satiete`)
Idée : la récompense “manger/boire” diminue si l’agent est déjà rassasié.

- Variables internes (réinitialisées à chaque épisode) :
  - `niveau_faim ∈ [0,1]` avec `niveau_faim = 1` au départ (très faim)
  - `niveau_soif ∈ [0,1]` avec `niveau_soif = 1` au départ (très soif)

- Modification des récompenses :
  - `r0 ← r0 * niveau_faim`
  - `r1 ← r1 * niveau_soif`

- Mise à jour des niveaux (selon l’état atteint) :
  - Si `state == 0` (manger) :
    - `niveau_faim = max(0, niveau_faim - diminution_faim)`
    - `niveau_soif = min(1, niveau_soif + augmentation_soif)`
  - Si `state == 2` (boire) :
    - `niveau_soif = max(0, niveau_soif - diminution_soif)`
    - `niveau_faim = min(1, niveau_faim + augmentation_faim)`
  - Sinon (état 1), les niveaux remontent légèrement (selon les paramètres) car l’agent ne mange/boit pas.

Interprétation :  
- plus `niveau_faim` est petit (rassasié), plus la reward liée à la nourriture diminue ;  
- plus `niveau_soif` est petit, plus la reward liée à la boisson diminue.


---

### MDP batterie (`MDP_Batterie`)
Idée : une récompense “sature” quand une ressource (batterie) est pleine.

- Variables internes (réinitialisées à chaque épisode) :
  - `niveau_faim_batterie ∈ [0, Rmax]`, initialisé à 0
  - `niveau_soif_batterie ∈ [0, Rmax]`, initialisé à 0

- Saturation des récompenses :
  - si `niveau_faim_batterie >= Rmax` alors `r0 = 0`
  - si `niveau_soif_batterie >= Rmax` alors `r1 = 0`

- Mise à jour des batteries (selon l’état atteint) :
  - Si `state == 0` (manger) :
    - `niveau_faim_batterie = min(Rmax, niveau_faim_batterie + augmentation_faim)`
  - Sinon :
    - `niveau_faim_batterie = max(0, niveau_faim_batterie - diminution_faim)`
  - Si `state == 2` (boire) :
    - `niveau_soif_batterie = min(Rmax, niveau_soif_batterie + augmentation_soif)`
  - Sinon :
    - `niveau_soif_batterie = max(0, niveau_soif_batterie - diminution_soif)`

Interprétation : quand une batterie est pleine, l’action correspondante “ne rapporte plus rien” (reward coupée à 0).

---

### MDP setpoint (`MDP_Setpoint`)

Idée : au lieu de vouloir “charger à fond”, l’agent doit maintenir chaque batterie **autour d’un setpoint** (point d’équilibre).

- Identique à `MDP_Batterie`, mais la récompense est réduite selon l’écart au setpoint :

r0 = r0 * (1 - penalite(niveau_faim_batterie))  
r1 = r1 * (1 - penalite(niveau_soif_batterie))

- Pénalité :

penalite(x) =
- 0 si |x - setpoint| < tolerance  
- |x - setpoint| / Rmax sinon  

- Avec Rmax = 2 et niveau_batterie ∈ [0, Rmax], on a toujours :
  - penalite(x) ∈ [0, 1]

Interprétation :  
- proche du setpoint → pénalité ≈ 0 → reward conservée  
- éloigné du setpoint → pénalité → 1 → reward fortement réduite  

Intuition :  
L’agent ne cherche plus à maximiser la charge des batteries, mais à stabiliser ses niveaux internes autour d’une valeur cible.  


