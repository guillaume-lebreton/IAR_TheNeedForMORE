# The Need For MORE: : Need Systems as Non-Linear Multi-Objective Reinforcement Learning

## Objectif

L’objectif principal du projet est de **reproduire et analyser la Figure 5 du papier**, qui compare les stratégies et performances de différentes approches multi-objectifs dans un environnement simple.

Puis de tester la robustesse des méthodes implémenter dans des environnements avec saturation

## Méthodes implémenter
- Qlearning standard
- Objetcive switching
- QMore


## Structure
```text
algo_rl.py    #Implémentation des algorithmes
mdp.py        #Implémentation des mdp avec saturation
mdp.md        #Définition des différents mdp
main.py       #Fichier éxécutable pour tester les différentes méthode et générer les graphiques
utils.py      #Fonction nécessaire à la création des graphes
plots/        #Graphique des performance sur les différents mdp
```

## Prérequis

- Python ≥ 3.9  
- Librairies Python :
  - numpy  
  - matplotlib  