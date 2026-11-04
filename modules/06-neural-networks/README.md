# Module 06 — Fondements des réseaux de neurones

**Niveau :** intermédiaire · **Durée :** 6–8 h · **Prérequis :** modules 02, 04 et 05 ; dérivées partielles et produit matriciel.

## Objectifs et notions

Suivre un exemple à travers un perceptron et un MLP ; distinguer poids, biais, activation et loss ; calculer les dimensions de la forward pass ; expliquer la rétropropagation par règle de chaîne ; comparer SGD, momentum et Adam ; interpréter effets du batch size, learning rate, régularisation, dropout et batch normalization.

Pour un lot `X` de forme `(batch, features)`, une couche dense produit `Z = X @ W + b` avec `W: (features, units)`, `b: (units,)`, puis `A = activation(Z)`. Un MLP peut représenter des frontières non linéaires, au coût d’un réglage plus délicat.

## Activité et limites

Calculer manuellement un passage avant sur deux neurones, puis comparer différences finies et gradient analytique sur une petite loss. Le notebook complet NumPy et la comparaison des optimiseurs figurent dans la suite de travail (`implementation-status.md`). Les poids ne sont pas une explication causale et une accuracy élevée ne prouve pas l’absence de biais.
