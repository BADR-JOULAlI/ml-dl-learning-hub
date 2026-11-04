# État d’implémentation

Mis à jour le 4 novembre 2026.

## Parcours disponible

- Documentation de démarrage, prérequis, feuille de route, références, contributions et installation.
- Modules 00 à 05 : cours, notebooks principaux, exercices et quiz pour le parcours débutant/intermédiaire.
- Modules 06 à 09 : notebooks enrichis sur rétropropagation NumPy, entraînement PyTorch, CNN et TF-IDF. Les modules 02 à 04 incluent algèbre/probabilité, audit de données, classification logistique NumPy, K-Means, DBSCAN et PCA.
- Modules 10 à 12 : briefs de projets, un projet prix complet, un parcours churn avec entraînement, deux notebooks d’analyse tabulaire, une activité CNN, une activité sentiment, un notebook MLOps local sur les expériences et la dérive des entrées, et un projet FastAPI sérialisable avec Dockerfile et MLflow local optionnel.
- Modules 00 à 05 : exercices Easy/Medium/Hard et notebooks de correction séparés. Les modules avancés ont leurs propositions d’exercices, mais leurs notebooks d’exercices et corrections ne sont pas encore produits.
- Tests du code Python du package, smoke test optionnel pour l’API, vérification des liens et validation/exécution de notebooks via scripts dédiés.
- CI : Ruff, tests unitaires, validation structurelle des notebooks et liens locaux. L’exécution complète des notebooks ne tourne pas encore dans GitHub Actions.

## Exécution vérifiée le 4 novembre 2026

La commande `python scripts/validate_notebooks.py --execute` a exécuté **28 notebooks sur 28**, de la première à la dernière cellule, avec Python 3.12 sur CPU sous Windows. Le total inclut les deux TP NLP historiques, les notebooks d’exercices et solutions, les projets tabulaires et le notebook MLOps. Chaque notebook contient au moins une visualisation calculée par le code. Aucune sortie d’exécution n’est enregistrée dans les fichiers `.ipynb`; les sorties Reuters ne sont donc pas publiées. Les **103 liens Markdown locaux** ont été vérifiés.

Les gros modèles Word2Vec, GloVe et FastText des TP historiques restent des cellules volontairement commentées : leurs téléchargements et leurs résultats ne sont pas couverts par cette exécution. Le chemin MLflow optionnel et l’image Docker restent à vérifier dans un environnement propre avec les services nécessaires.

## Travail restant

1. Ajouter des exercices et corrections sous forme de notebooks pour les modules 06–12.
2. Ajouter une étude Transformer/Hugging Face, une visualisation Grad-CAM, une activité de segmentation et transfer learning.
3. Vérifier les chemins MLflow optionnels et l’image Docker avec une installation propre où Docker est disponible.
4. Répéter l’exécution dans des environnements séparés minimal, DL et MLOps, puis ajouter un contrôle CI notebook ciblé si le coût reste raisonnable.
5. Développer les quiz des modules avancés et une galerie de résultats obtenus par des apprenants.

Les tâches de suite sont prêtes à reprendre dans [`docs/backlog.md`](backlog.md).

Ces éléments ne sont pas présentés comme terminés. Le plan actionnable est détaillé dans le [backlog](backlog.md).
