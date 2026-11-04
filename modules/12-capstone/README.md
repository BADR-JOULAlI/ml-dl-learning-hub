# Module 12 — Capstone d’équipe

**Niveau :** avancé · **Durée :** 12–20 h · **Prérequis :** modules 03–05 et 11 ; équipe de 2–5.

## Mission

Choisir une cible accessible et licite, construire un pipeline de données reproductible, comparer une baseline à un modèle, expliquer les erreurs, exposer une interface d’inférence et proposer des signaux de monitoring. Un dataset léger et sans informations sensibles est recommandé.

```mermaid
flowchart LR
  A[Collecte et provenance] --> B[Validation des données]
  B --> C[EDA et preprocessing]
  C --> D[Entraînement et expérience]
  D --> E[Évaluation et revue des erreurs]
  E --> F[API ou batch]
  F --> G[Signaux et plan de suivi]
```

## Rôles en équipe

- **Data steward** : provenance, licence, schéma, qualité et split.
- **Model owner** : baseline, pipeline, tuning et artefact.
- **Platform owner** : commande reproductible, API, tests, Docker facultatif.
- **Reviewer/documenter** : erreurs, risques, README et revue croisée.

Dans une petite équipe, chaque rôle peut être partagé et les personnes tournent à mi-parcours.

## Livrables

README avec l’usage prévu, l’origine et les limites des données ; code d’entraînement reproductible ; protocole de validation ; tableau de métriques ; rapport d’erreurs ; artefact générable, API locale ou batch ; tests ; schéma d’architecture ; plan de monitoring précisant le signal, fréquence, seuil indicatif, propriétaire et action.

## Barème mentor — 100 points

| Critère | Points | Preuve attendue |
|---|---:|---|
| Problème, usages et provenance | 10 | Cible compréhensible et données permises |
| Validation et preprocessing sans fuite | 20 | Split justifié et transformations dans pipeline |
| Baseline, expérimentation et reproductibilité | 20 | Graine, config, métriques comparables |
| Évaluation et analyse d’erreurs | 20 | Métriques adaptées, incertitude et exemples |
| Livraison et tests | 15 | Commandes fonctionnelles, schéma d’entrée documenté |
| Monitoring, risques et communication | 15 | Limites, signaux/action et rapport clair |

On peut commencer par un projet tabulaire CPU. Aucun cloud ni gain de métrique minimal n’est imposé ; la qualité du raisonnement compte davantage que la taille du modèle.
