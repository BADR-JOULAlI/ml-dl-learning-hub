# Module 04 — Machine Learning classique

**Niveau :** intermédiaire · **Durée :** 8–10 h · **Prérequis :** modules 01–03 et lecture d’une formule de gradient.

## Objectifs

Relier la forme d’un problème à régression, classification ou clustering ; établir une baseline ; implémenter une régression linéaire à une variable avec NumPy ; utiliser une pipeline scikit-learn ; décrire les hypothèses et compromis de modèles usuels.

## Notions essentielles

La régression linéaire choisit des coefficients en minimisant une perte. La régression logistique transforme une combinaison linéaire avec la fonction sigmoïde \(\sigma(z)=1/(1+e^{-z})\) en probabilité. KNN dépend de la distance et donc de l’échelle ; arbres et forêts segmentent l’espace par règles ; SVM cherche une marge ; boosting ajoute des apprenants successifs ; Naive Bayes suppose une indépendance conditionnelle simplificatrice. K-Means minimise une inertie intra-cluster et nécessite le choix de `k`. PCA tourne vers les directions de variance, sans tenir compte de la cible.

Le coût dépend du nombre d’observations, de variables et des paramètres : KNN stocke les données et calcule des distances au moment de prédire ; une forêt peut augmenter coût mémoire et latence ; les modèles linéaires donnent un baseline rapide et lisible.

## Mise en pratique

Exécuter [01_regression.ipynb](01_regression.ipynb), [02_logistic_classification.ipynb](02_logistic_classification.ipynb) et [03_unsupervised_clustering_pca.ipynb](03_unsupervised_clustering_pca.ipynb). Comparer les algorithmes faits avec NumPy et ceux de scikit-learn ; pour l’apprentissage non supervisé, observer comment l’échelle, la densité et la projection PCA modifient la lecture des groupes.

## Résultat et limites

Vous savez tracer résidu vs prédiction et expliquer pourquoi un algorithme supérieur sur un split ne l’est pas forcément sur une autre population. Le cours n’affirme pas que le modèle le plus complexe est le meilleur ; hypothèses, taille, latence et coût d’erreur doivent guider le choix.
