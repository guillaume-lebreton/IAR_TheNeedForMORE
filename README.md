# The Need for MORE: Need Systems as Non-Linear Multi-Objective Reinforcement Learning

Projet de reproduction et d'analyse de la Figure 5 du papier de Rolf et al. (2020) sur les methodes MORE en apprentissage par renforcement multi-objectifs.

## Objectifs
- Reproduire et analyser la Figure 5 (comparaison des strategies et des performances).
- Tester la robustesse des methodes dans des environnements avec saturation.

## Méthodes implémentées
- Q-learning standard (somme ponderee des recompenses)
- Objective Switching (Q-learning a objectif alternatif)
- Q-MORE (version LLR)
- Q-MORE avec discretisation des poids

## Environnements (MDP)
Le projet propose 4 environnements simples a 3 etats / 2 actions :
- `standard` (décrit dans le papier)
- `satiete` (reward depend de la faim/soif)
- `batterie` (reward sature quand la batterie est pleine)
- `setpoint` (reward penalise l'écart à un setpoint)

Un resume plus detaille est disponible dans `src/mdp.MD`.

## Installation
Prerequis :
- Python >= 3.9
- Bibliotheques : `numpy`, `matplotlib`, `tqdm`

Installation rapide :
```bash
pip install numpy matplotlib tqdm
```

## Utilisation
Le point d'entree est `src/main.py`.

1) Choisir le MDP a evaluer :
```python
mdp_name = "setpoint"  # standard, satiete, batterie, setpoint
```

2) Activer/desactiver les algorithmes :
```python
main(mdp_name, standard=True, switch=True, more=False, more_discr=True)
```

3) Lancer :
```bash
python src/main.py
```

Les figures sont enregistrees dans `plots/mdp_<nom>/`.

## Resultats
Chaque execution produit une figure composee de 3 colonnes :
1) State space traversal
2) Rewards (EMA) pour r0 / r1
3) V-MORE

Les images générées sont sauvegardées dans `plots/` (une sous-dossier par MDP).

## Paramètres principaux
Dans `src/main.py` :
- `NB_EPISODES`, `NB_STEPS`
- `GAMMA`, `ALPHA`, `EPSILON`
- `GAMMA_PASTE`, `PAS_DISCR`

Ajustez ces valeurs pour reproduire la figure ou explorer d'autres regimes.

## Structure du projet
```text
README.md
Rolf2020-MORE-ICDL.pdf
src/
  algos_rl.py    # Algorithmes (standard, switch, MORE, MORE discretise)
  mdp.py         # Definition des MDP
  mdp.MD         # Explication des MDP
  main.py        # Point d'entree : entrainement + generation de figures
  utils.py       # Fonctions de plot et courbes
plots/
  mdp_standard/
  mdp_satiete/
  mdp_batterie/
  mdp_setpoint/
```


## Reference
Rolf, M. et al. (2020). *The Need for MORE: Need Systems as Non-Linear Multi-Objective Reinforcement Learning*.
