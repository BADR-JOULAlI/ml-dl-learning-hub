# Projet — Estimer un prix immobilier pédagogique

**Question :** à partir d’attributs simulés d’un logement, comparer une prédiction moyenne à une forêt d’arbres.

## Important sur les données

Le générateur [make_housing_data](../../src/learning_hub/preprocessing.py) produit un jeu synthétique déterministe (600 lignes par défaut). Les prix sont construits par une formule définie dans le code. Ils ne représentent aucun marché et ne doivent pas être utilisés pour estimer un vrai bien. Aucun fichier de données n’est téléchargé ou committé.

## Exécution

Installer le dépôt comme décrit dans [le guide](../../docs/getting-started.md), puis depuis la racine :

```bash
python -m learning_hub
```

Le script crée un split train/test reproductible, ajuste imputation, scaling et one-hot encoding dans chaque pipeline, puis compare un `DummyRegressor` au `RandomForestRegressor`. Les métriques affichées sont MAE (unités de prix simulées), RMSE et R².

Le notebook [analysis.ipynb](analysis.ipynb) ajoute EDA, graphiques de résidus, métriques par district et conclusions guidées.

## Travail demandé

1. Décrire les features et leurs unités. Vérifier les valeurs manquantes et la distribution de prix.
2. Écrire avant le résultat une hypothèse sur le modèle qui gagnera.
3. Comparer le baseline et la forêt sur le même split.
4. Tracer résidus contre prédictions, lire les erreurs extrêmes et calculer MAE dans chaque district.
5. Modifier une décision à la fois (par exemple taille de forêt ou profondeur) via validation croisée.
6. Résumer les limites du générateur et proposer une source réelle avec licence à vérifier si vous souhaitez étendre le projet.

## Conclusions attendues

Ne concluez pas qu’une forêt est le meilleur estimateur de prix en général si elle gagne sur une séparation synthétique. Le résultat vérifie surtout que les étapes du projet s’enchaînent sans fuite et que le pipeline accepte des données brutes.
