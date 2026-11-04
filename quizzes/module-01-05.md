# Quiz de compréhension — Modules 01 à 05

Faites le quiz sans exécuter de code. Les réponses sont après les questions.

1. Pour un tableau de forme `(120, 6)`, combien y a-t-il d’observations et de variables ? **Réponse :** 120 observations, 6 variables.
2. Quelle différence entre `X @ w` et `X * w` ? **Réponse :** produit matriciel/scalaire versus multiplication élément par élément avec broadcasting.
3. Que représente une dérivée dans la descente de gradient ? **Réponse :** variation locale de la loss selon un paramètre et direction de pente.
4. Pourquoi ne calcule-t-on pas la médiane sur train + test ? **Réponse :** la transformation incorporerait de l’information du test.
5. Quel traitement s’adapte naturellement à une variable catégorielle nominale ? **Réponse :** encodage tel que OneHotEncoder, au sein de la pipeline.
6. Quand un scaler est-il souvent essentiel ? **Réponse :** pour les modèles fondés sur distance ou régularisés, notamment KNN/SVM/logistique ; l’effet dépend de l’algorithme.
7. À quoi sert le stratified split ? **Réponse :** préserver approximativement les proportions de classes dans chaque partition.
8. Quelle métrique répond à « combien de positifs réels ai-je trouvés ? » **Réponse :** recall.
9. F1 inclut-il les vrais négatifs directement ? **Réponse :** non, F1 combine precision et recall.
10. Pourquoi garder le test final scellé pendant le tuning ? **Réponse :** éviter de choisir indirectement des paramètres en fonction de ce test et d’annoncer un score optimiste.

Pour chaque réponse, donnez un exemple d’application où le choix pourrait différer.
