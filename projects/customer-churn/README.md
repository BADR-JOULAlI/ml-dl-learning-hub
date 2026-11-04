# Projet — Prédire une attrition synthétique

## Question et limites

Construire un classifieur de démonstration à partir d’ancienneté, montant mensuel, nombre d’appels au support et type de contrat. Les labels sont tirés d’un mécanisme artificiel visible dans [`make_churn_data`](../../src/learning_hub/preprocessing.py). Ils n’estiment aucun taux d’attrition réel et ne doivent pas guider des décisions sur de vrais clients.

## Étapes

1. Créer des données reproductibles avec `make_churn_data(random_state=42)`.
2. Décrire proportions et distributions pour chaque classe.
3. Garder les quatre colonnes comme features ; la cible n’existe pas dans `X`.
4. Créer un split stratifié avant de fitter les transformations.
5. Comparer une baseline majorité à une pipeline imputation/standardisation/encodage + régression logistique.
6. Mesurer precision, recall, F1 et Average Precision ; comparer les seuils après l’entraînement et expliciter le coût d’erreur choisi.
7. Examiner les cas mal classés et écrire ce qui ne peut être appris de données simulées.

Le générateur est exécutable en Python ; la correction du protocole apparaît aussi dans [l’exercice preprocessing](../../exercises/intermediate/03_preprocessing_solutions.ipynb). Une extension plus avancée est le service local dans [end-to-end-ml](../end-to-end-ml/README.md).

Le notebook [analysis.ipynb](analysis.ipynb) suit le projet complet, et `python projects/customer-churn/train.py` exécute l’évaluation dans le terminal.
