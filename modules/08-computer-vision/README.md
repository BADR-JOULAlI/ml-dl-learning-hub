# Module 08 — Vision par ordinateur

**Niveau :** avancé · **Durée :** 6–8 h · **Prérequis :** modules 06–07, groupe `dl` et notions de tenseurs.

## Objectifs

Représenter une image en `(channels, height, width)` ; appliquer une convolution 2D ; calculer taille de sortie avec stride/padding ; interpréter un feature map et pooling ; entraîner et évaluer un CNN ; distinguer augmentation, transfert d’apprentissage et interprétation.

Pour un noyau de taille `K`, stride `S` et padding `P`, une dimension spatiale de sortie est \(\lfloor(H+2P-K)/S\rfloor+1\). Les canaux de sortie correspondent aux filtres appris. Une convolution partage ses poids spatialement, ce qui réduit les paramètres par rapport à une couche dense sur pixels.

## Activité

Le projet [classification d’images](../../projects/image-classification/README.md) utilise le dataset digits 8×8 de scikit-learn, inclus sans téléchargement. Comparer baseline logistique sur pixels et petit CNN, afficher matrice de confusion et exemples d’erreur.

## Statut et limites

Le notebook CNN complet, augmentation robuste, transfer learning, Grad-CAM et segmentation sont à compléter. Le dataset miniature sert à enseigner la boucle, pas à estimer la qualité en conditions réelles.
