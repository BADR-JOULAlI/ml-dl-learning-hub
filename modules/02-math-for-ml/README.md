# Module 02 — Mathématiques utiles pour le ML

**Niveau :** débutant à intermédiaire · **Durée :** 5–6 h · **Prérequis :** fonctions et opérations sur nombres ; dérivées vues au lycée utiles, mais révisées ici.

## Objectifs

Interpréter un vecteur comme une liste de coordonnées/variables, calculer un produit scalaire, relier la dérivée à une pente locale, calculer la moyenne quadratique et expliquer une mise à jour de gradient descent. Savoir où ces idées se trouvent dans les formes NumPy.

## Formules de travail

Pour `n` observations et une prédiction linéaire \(\hat y_i = wx_i+b\), la perte MSE est

\[J(w,b)=\frac{1}{n}\sum_{i=1}^n(wx_i+b-y_i)^2.\]

Le gradient indique la direction de croissance la plus rapide. On le soustrait pour faire baisser la perte :

\[w \leftarrow w-\eta\frac{2}{n}\sum_i(\hat y_i-y_i)x_i,\qquad
b \leftarrow b-\eta\frac{2}{n}\sum_i(\hat y_i-y_i).\]

`eta` (`learning_rate`) règle la taille du pas. Trop petite, la progression est lente ; trop grande, la loss peut osciller ou diverger.

## Activité

Exécuter [02_linear_algebra_probability.ipynb](02_linear_algebra_probability.ipynb) pour les vecteurs, matrices, covariance et Bayes, puis [01_gradient_descent.ipynb](01_gradient_descent.ipynb). Modifier la learning rate et observer `loss_history`. Refaire une mise à jour à la main sur deux points.

## Après le module

Vous pouvez expliquer la règle de chaîne intuitivement et lire une multiplication de dimensions `(n, p) @ (p, q) -> (n, q)`. Valeurs propres, Bayes et optimisation multidimensionnelle sont introduits par intuition, mais leur traitement avancé reste à approfondir.
