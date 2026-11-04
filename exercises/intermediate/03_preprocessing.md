# Exercices — Prétraitement et fuite

## Easy — 15 min

Chargez Iris, affichez les dimensions, types, statistiques et fréquence de chaque classe. Indiquez la cible et l’unité d’observation.

## Medium — 30 min

Ajoutez une variable catégorielle dérivée d’une feature, introduisez des valeurs manquantes, puis construisez une `ColumnTransformer` avec imputation, standardisation et OneHotEncoder. Entraînez un modèle après un split stratifié.

## Hard — 35 min

Créez un exemple où la distribution test est décalée. Comparez médiane calculée sur train seul et sur toutes les lignes. Écrivez une assertion prouvant que les transformations du pipeline sont ajustées dans le fold, et expliquez pourquoi « scaler avant split » est incorrect.

Correction : [`03_preprocessing_solutions.ipynb`](03_preprocessing_solutions.ipynb).
