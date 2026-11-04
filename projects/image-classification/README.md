# Projet — Chiffres manuscrits en 8×8

Utiliser `sklearn.datasets.load_digits`, un dataset léger intégré et sans téléchargement. Chaque exemple est une image grise 8×8 dont les intensités vont de 0 à 16. Il s’agit d’un exercice de boucle d’apprentissage, pas d’un test de reconnaissance d’écriture déployable.

## Étapes de projet

1. Visualiser quelques images et vérifier la distribution des labels.
2. Faire un split stratifié, en gardant le test final scellé.
3. Entraîner une régression logistique sur pixels aplatis comme baseline.
4. Suivre le notebook [01_cnn_digits](../../modules/08-computer-vision/01_cnn_digits.ipynb), ou le reconstruire sous forme de script : tenseurs `(n,1,8,8)`, Conv2d, pooling, logits et CrossEntropyLoss.
5. Mesurer accuracy et matrice de confusion ; afficher des erreurs.
6. Changer le learning rate ou nombre d’epochs via validation, jamais sur test.

Dépendance : `python -m pip install -e ".[dl]"`. Exécutable sur CPU. Le CNN est un point d’entrée ; augmentation, transfer learning, segmentation et Grad-CAM restent hors du brief courant.
