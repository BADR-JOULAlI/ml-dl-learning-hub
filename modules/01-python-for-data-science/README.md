# Module 01 — Python pour la Data Science

**Niveau :** débutant · **Durée :** 5–6 h · **Prérequis :** variables, boucles, fonctions et imports simples.

## Objectifs et compétences

À l’issue du module, vous savez inspecter une forme NumPy, filtrer et agréger un DataFrame Pandas, lire un CSV, éviter une boucle Python inutile sur un calcul numérique et tracer une relation avec des axes annotés. Compétences : indexation, types, fonctions, tableaux, groupby et visualisation.

## Notions

Un tableau `X` de forme `(n, p)` contient `n` observations et `p` variables. `X[:, 0]` sélectionne la première colonne. Le **broadcasting** permet des opérations élément par élément si les dimensions sont égales ou si l’une vaut 1 ; il n’implique pas une multiplication matricielle. Pour cela, `@` suit l’algèbre linéaire.

## Parcours pratique

1. Exécuter [01_numpy_pandas.ipynb](01_numpy_pandas.ipynb).
2. Faire les exercices Easy/Medium/Hard dans [`exercises`](../../exercises/beginner/01_python.md).
3. Comparer une moyenne par groupe à un graphique correctement légendé.

Pandas lit CSV et JSON avec `read_csv` et `read_json`. Pour Parquet, installer l’extension optionnelle `python -m pip install -e ".[parquet]"`, puis utiliser `read_parquet`/`to_parquet`. Ce format conserve les types et compresse efficacement les tables ; le moteur Arrow est isolé pour garder l’installation de base légère.

## Durée, limites, résultat

Compter 90 min sur le notebook et 3–4 h sur les exercices. Un DataFrame ne dispense pas de vérifier les unités, valeurs manquantes ou types. **Résultat attendu :** un petit rapport contenant un tableau descriptif, un graphique et une observation vérifiable.
