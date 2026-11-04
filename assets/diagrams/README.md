# Schémas originaux du parcours

Les diagrammes sont écrits en Mermaid pour pouvoir être révisés avec le cours et rester lisibles sans image binaire. Ils sont simplifiés pour l’enseignement ; les axes ou paramètres absents sont explicités dans les leçons.

## Préparation et séparation

```mermaid
flowchart TD
  A[Définir observation, cible et usage] --> B[Contrôles de qualité]
  B --> C[Split train / validation / test]
  C --> D[Fit imputer, scaler, encoder sur train]
  D --> E[Fit estimateur sur train transformé]
  E --> F[Choix sur validation ou CV]
  F --> G{Choix figé ?}
  G -- non --> F
  G -- oui --> H[Évaluer une fois sur test]
```

## Un pas de gradient

```mermaid
flowchart LR
  A[Paramètres w,b] --> B[Prédiction y_hat = wx + b]
  B --> C[Loss J(w,b)]
  C --> D[Gradient dJ/dw, dJ/db]
  D --> E[Soustraire learning_rate × gradient]
  E --> A
```

Le gradient donne la pente de la loss dans l’espace des paramètres. Un pas trop grand peut passer de l’autre côté du minimum ; le notebook trace la loss pour le cas étudié.

## Forward pass et rétropropagation

```mermaid
flowchart LR
  X[Features X: batch × features] --> Z1[Z1 = X W1 + b1]
  Z1 --> A1[Activation A1]
  A1 --> Z2[Z2 = A1 W2 + b2]
  Z2 --> L[Loss]
  L -. règle de chaîne .-> G2[Gradient couche 2]
  G2 -. rétropropager .-> G1[Gradient couche 1]
```

## CNN

```mermaid
flowchart LR
  I[Image: channels × hauteur × largeur] --> C[Convolution: filtres appris]
  C --> A[Activation]
  A --> P[Pooling: réduire la résolution]
  P --> F[Aplatir ou agréger]
  F --> O[Logits par classe]
```

## Transformer encoder simplifié

```mermaid
flowchart LR
  T[Tokens] --> E[Token embeddings + position]
  E --> Q[Queries]
  E --> K[Keys]
  E --> V[Values]
  Q --> S[Scores Q Kᵀ / √d]
  K --> S
  S --> W[Softmax: poids d'attention]
  W --> M[Combinaison pondérée de V]
  V --> M
  M --> R[Résiduel + normalisation]
  R --> F[MLP positionnel]
  F --> Z[Représentations contextualisées]
```

Cette vue omet les têtes multiples, masques, couches répétées et détails de normalisation ; le schéma décrit la circulation d’information, pas un modèle complet.

## Cycle MLOps

```mermaid
flowchart LR
  A[Données versionnées] --> B[Validation]
  B --> C[Pipeline d'entraînement]
  C --> D[Expériences et artefact]
  D --> E[Revue et déploiement]
  E --> F[Prédictions et journaux]
  F --> G[Monitoring et feedback]
  G --> B
```

Le feedback peut être différé ; surveiller des entrées ne remplace pas la mesure de performance quand les vrais labels arrivent.
