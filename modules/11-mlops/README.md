# Module 11 — Fondamentaux MLOps

**Niveau :** avancé · **Durée :** 6–8 h · **Prérequis :** modules 04–05 et bases de Git ; Docker facultatif au premier passage.

## Objectifs

Décrire une expérience par données, code, paramètres et métriques ; versionner le code et connaître les limites de versionner les données ; sérialiser et recharger une pipeline ; exposer une prédiction FastAPI ; construire une image Docker ; tester contrat et invariants ; expliquer data drift et model drift.

Un modèle servi est une dépendance logicielle qui a besoin de son preprocessing, schéma d’entrée, version de dépendances et observabilité. Un endpoint disponible ne garantit pas que la qualité est maintenue. Surveiller d’abord le volume, latence, erreurs, valeurs manquantes et distributions ; une mesure de performance requiert souvent des labels différés.

## Activité et statut

L’exemple de bout en bout est dans [projects/end-to-end-ml](../../projects/end-to-end-ml/README.md). Démarrer par exécuter le pipeline local, tracer paramètres et scores avec MLflow, puis lancer l’API en local. Docker, CI de test API et exercice de drift complet restent à finaliser.

## Limites

Le parcours ne requiert aucun cloud ni service privé. Les choix de registry, orchestration et monitoring dépendent de contraintes d’équipe ; ne pas déployer un modèle avant de définir sécurité, seuils de service et procédure de retrait.
