# Contribuer

Merci de proposer des corrections, exemples ou leçons. Le dépôt est pédagogique : toute contribution doit permettre à un lecteur de comprendre ce que fait le code et pourquoi.

## Workflow

1. Ouvrir une issue si le changement dépasse une correction locale. Décrire le public, le problème pédagogique et les critères de réussite.
2. Créer une branche courte depuis la version à jour, par exemple `lesson/precision-recall` ou `fix/pipeline-leakage`.
3. Faire un changement limité et documenter les choix. Ne pas ajouter de données ou de poids de modèle volumineux.
4. Exécuter les vérifications : `python -m pytest`, `ruff check src tests scripts`, `ruff format --check src tests scripts`, `python scripts/validate_notebooks.py` et `python scripts/validate_links.py`.
5. Pour un notebook, redémarrer le noyau, exécuter toutes les cellules depuis la racine, effacer les sorties inutiles et vérifier les liens/chemins.
6. Ouvrir une pull request avec le modèle fourni : objectifs, changements, vérifications, limites connues.

## Conventions

- Langue d’enseignement principale : français ; noms de variables, code, noms de fichiers et docstrings Python : anglais.
- Un notebook enseigne une question précise. Cellules courtes, graines explicites, formes des tableaux visibles et interprétation après les graphiques. Chaque notebook contient au moins une visualisation originale qui aide à expliquer une notion : titre, axes, unités, légende si nécessaire et commentaire sur ce que la figure prouve ou ne prouve pas.
- Un notebook d’étudiant pose le problème et les critères de réussite, mais ne révèle pas la solution complète. Mettre la correction dans un fichier séparé.
- Les exemples commencent par un baseline ; ils séparent données d’entraînement/validation/test avant d’apprendre les transformations.
- Les métriques sont choisies selon l’usage et les conséquences des erreurs, pas uniquement parce qu’elles sont disponibles.
- Ajouter des références pour les données et idées spécifiques. Ne pas importer du texte, du code ou des figures de cours tiers sans vérifier leur licence et ajouter l’attribution requise.
- Code formaté avec Ruff, ligne cible de 100 caractères, tests sous `tests/`.

## Proposer une nouvelle leçon

Indiquer le prérequis, le résultat observable et la durée estimée. Relier intuition et équation, donner un exemple numérique minuscule, une implémentation, une visualisation ou une expérience, une interprétation, une limite, puis des exercices easy/medium/hard. Une leçon n’est complète que si l’installation et le notebook sont reproductibles de haut en bas.

## Revue du contenu

La personne qui relit vérifie la cohérence de la notation et des formes, les hypothèses, l’absence de fuite de données, les usages de métriques, la licence des matériaux, l’accessibilité des figures et les commandes de reproduction. Toute affirmation de performance doit préciser jeu de données, split, métrique et graine. Exécuter `python scripts/validate_notebooks.py --execute --path chemin/du/notebook.ipynb` depuis la racine ; la commande n’enregistre pas les sorties. Ne pas committer de texte Reuters généré dans les sorties des TP NLP historiques.
