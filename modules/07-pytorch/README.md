# Module 07 — Entraîner avec PyTorch

**Niveau :** intermédiaire à avancé · **Durée :** 6–8 h · **Prérequis :** module 06 et programmation Python de base ; groupe optionnel `dl`.

## Objectifs

Lire les shapes d’un tensor, placer modèle et batch sur le même device, définir `nn.Module`, séparer étapes d’entraînement et d’évaluation, calculer gradients via autograd, choisir un loss adapté, sauvegarder `state_dict` et documenter graines et checkpoint.

## Boucle canonique

Une itération d’entraînement suit `model.train()` → forward → loss → `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`. En évaluation, `model.eval()` et `torch.inference_mode()` désactivent les comportements d’entraînement et le suivi des gradients. Le data loader mélange généralement train mais pas validation/test.

## Activité

Créer un tensor CPU, vérifier shapes, calculer une dérivée simple via autograd, puis entraîner un MLP sur un petit jeu fourni par `scikit-learn`. La version détaillée exécutable et les recettes CUDA/Colab sont à compléter ; voir l’état du dépôt avant de traiter ce module comme terminé.

## Limites

La reproducibilité GPU peut varier selon matériel et noyaux. Une boucle plus courte mais explicite facilite d’abord le diagnostic. Le modèle sauvegardé doit être accompagné de la structure, des transformations et des classes attendues.
