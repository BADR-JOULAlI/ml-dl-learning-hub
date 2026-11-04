# Module 05 — Entraînement, validation et évaluation

**Niveau :** intermédiaire · **Durée :** 5–6 h · **Prérequis :** classification et régression du module 04.

## Objectifs

Distinguer entraînement et inférence ; diagnostiquer underfitting/overfitting par train vs validation ; choisir une métrique selon les coûts d’erreurs ; utiliser stratified cross-validation pour comparer des réglages ; analyser quelques erreurs avant de déclarer une amélioration.

## Métriques

Pour la classe positive, `precision = TP/(TP+FP)` répond à « parmi les alertes, combien sont correctes ? ». `recall = TP/(TP+FN)` répond à « parmi les positifs réels, combien ont été trouvés ? ». F1 est leur moyenne harmonique ; il ne donne pas un coût universel. ROC-AUC évalue le classement sur plusieurs seuils ; PR-AUC est souvent plus informative quand la classe positive est rare. En régression, MAE est interprétable en unités de la cible, RMSE pénalise davantage les grosses erreurs.

## Évaluation honnête

Le test final reste scellé jusqu’à ce que les choix soient arrêtés. `GridSearchCV` ou `RandomizedSearchCV` s’applique à une pipeline entière ; les transformations apprises sont réajustées dans chaque fold. Les folds doivent respecter groupes ou temps si les observations ne sont pas indépendantes.

## Activité et résultat

Exécuter [01_classification.ipynb](01_classification.ipynb). Comparer seuils sur un jeu de test illustratif, afficher matrice de confusion et lire des cas mal classés. Vous livrez une métrique principale justifiée, une métrique secondaire et une limite de l’expérience.
