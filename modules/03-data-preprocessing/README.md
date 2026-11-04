# Module 03 — Comprendre et préparer les données

**Niveau :** débutant à intermédiaire · **Durée :** 8–10 h · **Prérequis :** module 01 et notions de split.

## Objectifs

Produire un audit EDA concis ; séparer train/test avant de calculer médianes, moyenne ou catégories ; choisir un encodage et une mise à l’échelle ; assembler les étapes dans `Pipeline` et `ColumnTransformer` ; détecter une fuite de données ; interpréter l’effet d’un choix de preprocessing.

## Concepts et risques

Une transformation ajustée est un modèle lui aussi : la médiane, l’échelle ou le vocabulaire doivent être appris seulement depuis le train. Si le même ensemble sert à choisir beaucoup de variantes puis à annoncer le score final, le résultat est optimiste. Stratifier conserve approximativement la proportion des classes pour une classification ordinaire, mais une séparation temporelle ou par groupe est préférable si elle correspond au problème.

## Étapes

1. Examiner lignes, colonnes, types, identifiants, doublons, distribution de la cible et valeurs manquantes.
2. Définir l’unité d’observation et la cible avant de transformer.
3. Séparer train/test ; utiliser validation ou CV pour choisir.
4. Comparer `SimpleImputer`, `StandardScaler`, `OneHotEncoder` et les modèles dans un pipeline.
5. Documenter toutes les colonnes exclues et pourquoi.

Exécuter [01_pipeline_sans_fuite.ipynb](01_pipeline_sans_fuite.ipynb) et [02_quality_and_features.ipynb](02_quality_and_features.ipynb), puis [l’exercice sur le preprocessing](../../exercises/intermediate/03_preprocessing.md). Le premier notebook montre en schéma comment une pipeline apprend ses statistiques sur train, puis réutilise ces paramètres pour validation, test et inférence.

## Résultat et pièges

Le livrable est un pipeline entraînable sur données brutes et une courte expérience comparant au moins deux préparations avec le même split. Les outliers ne sont pas automatiquement des erreurs ; ne pas supprimer des valeurs test en fonction de leurs labels. L’imputation apprise à l’avance, les doublons entre splits et les variables calculées après l’événement prédit sont des causes fréquentes de fuite.
