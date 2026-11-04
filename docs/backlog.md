# Backlog proposé

Chaque bloc ci-dessous peut être copié dans une issue GitHub. Priorité recommandée : P1 avant de présenter les modules avancés comme un parcours complet.

## [P1] Ajouter exercices et corrections aux modules 06 à 12

**Contexte :** les briefs avancés existent dans `exercises/advanced/06-12.md`, mais les exercices corrigés au format notebook couvrent seulement les modules 01 à 05.

**Critères d’acceptation :** créer un notebook de pratique et une correction séparée pour chaque module 06–12 ; chaque paire contient une tâche easy, medium et hard ; les notebooks sont exécutables ; la correction explicite hypothèses, formes et interprétation ; aucun notebook étudiant ne révèle toute sa correction.

## [P1] Vérifier les chemins optionnels MLOps en environnement propre

**Contexte :** le code local couvre entraînement, sérialisation, MLflow facultatif, API FastAPI et Dockerfile ; l’image n’a pas pu être construite sans daemon Docker actif dans l’environnement de développement.

**Critères d’acceptation :** installation propre `.[mlops]`, exécution sans MLflow puis avec tracking local, création d’artefact, smoke test API sans puis avec artefact, build/launch Docker lorsque le daemon est disponible, commandes testées inscrites dans le README.

## [P2] Ajouter Transfer Learning et Grad-CAM

**Contexte :** le module vision enseigne une CNN sur images 8×8 et documente les limites ; il ne comprend pas encore Transfer Learning ni interprétation Grad-CAM.

**Critères d’acceptation :** notebook facultatif CPU/Colab avec dataset et licence identifiés, baseline, modèle pré-entraîné, Grad-CAM correctement associé à la classe choisie, analyse d’erreurs et note mémoire/GPU.

## [P2] Créer un parcours Transformer sans API payante

**Contexte :** NLP couvre prétraitement et TF-IDF ; Transformer et Hugging Face sont au programme mais n’ont pas encore leur notebook.

**Critères d’acceptation :** expliquer token ids, embeddings, position, Q/K/V, masque et logits ; afficher les shapes ; comparer un baseline TF-IDF à un modèle léger accessible sans compte payant ; noter licence du corpus et modèle ; évaluer les erreurs, latence et empreinte mémoire.

## [P2] Ajouter une comparaison Random Forest et Gradient Boosting

**Contexte :** le module 04 couvre maintenant K-Means, DBSCAN et PCA dans un notebook exécutable. Random Forest et Gradient Boosting figurent dans le cours, mais ne disposent pas encore d’une expérience dédiée avec validation croisée et analyse d’erreurs.

**Critères d’acceptation :** notebook qui compare Random Forest et Gradient Boosting à une baseline simple, avec même split, validation croisée sur train, visualisation des erreurs, coût d’entraînement/inférence et limites des importances de variables.

## [P3] Exécuter les notebooks dans CI avec groupes ciblés

**Contexte :** CI actuelle vérifie lint et tests Python ; les notebooks sont validés localement par script mais ne sont pas exécutés sur chaque pull request.

**Critères d’acceptation :** job CPU court sur notebooks 01–05 et projets légers ; job optionnel ou manuel pour PyTorch ; cache maîtrisé ; échec rendu lisible avec chemin/cellule fautive ; temps total mesuré.
