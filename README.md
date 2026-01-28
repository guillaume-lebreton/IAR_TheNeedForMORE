# Pick-and-Deliver_Traveling-Salesman-Problem

Ce projet est réalisé dans le cadre du module **MAO (Méthodes d’Approximation et d’Optimisation)**.  
Il porte sur l’étude et la résolution du **PD-TSP (Pick and Delivery Traveling Salesman Problem)**, une variante du problème du voyageur de commerce.

Le PD-TSP modélise des systèmes logistiques où un véhicule :
- transporte initialement un ensemble d’objets à livrer,
- collecte de nouveaux objets dans les différentes villes,
- doit visiter chaque ville exactement une fois,
- subit un coût de déplacement dépendant du poids transporté

L’objectif est de déterminer une tournée partant et revenant au dépôt qui maximise:   
<p align="center">
<strong>le profit total collecté moins le coût total du parcours</strong>
</p>

## Contenu
- deux heuristiques gloutonnes (Delivery First et Late Pickup)
- une méthode itérative améliorante par recherche locale
- une modélisation exacte en PLNE résolue avec Gurobi


## Structure
```text
data/             # Données utilisées pour la génération des instances
results/          # Graphiques et résultats obtenus
src/
├── instance.py          # Instances et génération des données
├── glouton.py           # Heuristiques gloutonnes
├── local_search.py      # Méthodes de recherche locale
├── PLNE.py              # Modèle PLNE (Gurobi)
├── solution.py          # Représentation des solutions
├── test_comparaison.py  # Tests et comparaison des méthodes
├── solutions.py         # gestion des solutions
├── distributionGenerator.py  # génération des poids et profits
└── utils.py             # Fonctions utilitaires
```

## Prérequis

- Python ≥ 3.9  
- Librairies Python :
  - numpy  
  - matplotlib  
  - scipy  
  - gurobipy (uniquement pour la PLNE)

**Remarque**  
La résolution PLNE nécessite une **licence Gurobi valide**.  
Les heuristiques et la recherche locale fonctionnent sans Gurobi.

## Utilisation

### Génération d'une instance
La création d'instance est implémenter dans **instance.py**

On a 2 possibilités pour la création d'instance :
 - Instance tiré du site (https://hhperez.webs.ull.es/PDsite/index.html), stocké dans data/class2/, qui donne les coordonnées des villes ainsi que la demande des clients.  
*read_file(path)* -> dans utils.py

 - Instance alétaoire à partir du fichier data/villes.csv  
 *create_instance(n, dim)* avec n le nombre de villes et dim le nombre de comodités différentes

 class **InstancePD_TSP**(demand_dim, nb_customers, coords, capacity, demands, poids, profits)  
 Si poids et profits à None ils sont généré aléatoirement à l'aide de **distributionGenerator.py**


        python instance.py 

création de 2 instance, avec n=10 et d=5 avec les 2 méthodes

---

### Méthode Gloutonne
Les 2 méthodes gloutonnes sont implémenter dans **glouton.py**

- *delivery_first_pondere(instance, quadratique=False)*
- *late_pickup(instance, quadratique=False)*

Si quadratique est à False on utilise la version linéaire des coûts sinon on utilise la version quadratique

        python glouton.py

Exécute les 2 méthodes avec la version des coûts linéaire sur la même instance(n=10, dim=5) et affiche les résultats

---

### Recherche locale

---

## PLNE

*plne(instance)* exécute la méthode de résolution exacte à l'aide de Gurobi sur une instance

        python PLNE.py

Exécute la méthode sur une instance aléatoire(n=10,dim=5) et affiche les résultats

## Tests et comparaisons
