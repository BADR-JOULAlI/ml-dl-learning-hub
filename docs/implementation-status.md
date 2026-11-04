# État d’implémentation

Mis à jour le 4 novembre 2026.

## Parcours disponible

- Documentation de démarrage, prérequis, feuille de route, références, contributions et installation.
- Modules 00 à 05 : cours, notebooks principaux, exercices et quiz pour le parcours débutant/intermédiaire.
- Modules 06 à 09 : notes de cours et premiers notebooks exécutables pour rétropropagation NumPy, entraînement PyTorch, CNN et TF-IDF. Les modules 02 à 04 incluent aussi algèbre/probabilité, audit de données et classification logistique faite avec NumPy.
- Modules 10 à 12 : briefs de projets, un projet prix complet, un parcours churn avec entraînement, deux notebooks d’analyse tabulaire, une activité CNN, une activité sentiment, et un projet FastAPI sérialisable avec Dockerfile et MLflow local optionnel.
- Modules 00 à 05 : exercices Easy/Medium/Hard et notebooks de correction séparés. Les modules avancés ont leurs propositions d’exercices, mais leurs notebooks d’exercices et corrections ne sont pas encore produits.
- Tests du code Python du package, smoke test optionnel pour l’API, vérification des liens et validation/exécution de notebooks via scripts dédiés.
- CI : Ruff et tests unitaires du parcours de base ; les notebooks et tests optionnels ne tournent pas encore dans GitHub Actions.

## Travail restant

1. Ajouter des exercices et corrections sous forme de notebooks pour les modules 06–12.
2. Ajouter une étude Transformer/Hugging Face, une visualisation Grad-CAM, une activité de segmentation et transfer learning.
3. Vérifier les chemins MLflow optionnels et l’image Docker avec une installation propre où Docker est disponible.
4. Faire tourner les notebooks dans des environnements séparés minimal, DL et MLOps, puis ajouter un contrôle CI notebook ciblé si le coût reste raisonnable.
5. Développer les quiz des modules avancés et une galerie de résultats obtenus par des apprenants.

Les tâches de suite sont prêtes à reprendre dans [`docs/backlog.md`](backlog.md).

Ces éléments ne sont pas présentés comme terminés. Le plan actionnable est regroupé dans [les issues proposées](../.github/ISSUE_TEMPLATE/).
