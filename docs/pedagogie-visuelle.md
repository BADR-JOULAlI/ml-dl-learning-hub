# Lire et construire les visualisations du cours

Une figure répond à une question précise. Avant de regarder un graphe, nommez les variables représentées, leurs unités et le jeu de données. Après le graphe, écrivez une phrase sur ce que vous observez et une phrase sur la limite de cette observation.

| Notion | Figure utile | Ce qu’il faut lire | Limite à mentionner |
|---|---|---|---|
| Broadcasting et produit matriciel | Grille de valeurs et formes `(n,p)` | Quelle dimension correspond aux observations ? | Deux formes compatibles ne rendent pas l’opération pertinente. |
| Gradient descent | Loss selon les epochs ou trajectoire des paramètres | Le pas converge-t-il, oscille-t-il ou diverge-t-il ? | La loss train ne mesure pas la généralisation. |
| Preprocessing | Distribution avant/après et parts manquantes | Qu’a modifié l’imputation ou le scaling ? | Une figure descriptive sur test ne doit pas servir à ajuster la transformation. |
| Classification | Frontière et matrice de confusion | Où se situent FP et FN ? | Le résultat dépend du seuil et du split. |
| Évaluation | Precision/recall selon le seuil | Quel coût d’erreur change avec le seuil ? | Le seuil se choisit sur validation, pas sur test. |
| Réseau de neurones | Loss et surface de probabilité sur un problème jouet | La représentation non linéaire apparaît-elle ? | Un jeu synthétique minuscule ne mesure pas la robustesse. |
| PyTorch | Courbes train et validation | L’écart augmente-t-il ? | Une courbe lisse seule ne garantit pas l’absence de fuite. |
| Convolution | Image, noyau et feature map | Quelle structure le filtre met-il en évidence ? | Un filtre fabriqué n’est pas une explication d’un CNN entraîné. |
| TF-IDF | Heatmap documents × termes | Quels termes reçoivent un poids élevé ? | Une statistique lexicale ne capture pas tout le sens. |
| Monitoring | Distribution de référence et lot récent | Le profil des entrées a-t-il changé ? | Data drift n’est pas automatiquement model drift. |

Les exemples sont originaux et générés à l’exécution avec Matplotlib. Les notebooks utilisent des jeux intégrés, synthétiques ou téléchargés à l’exécution selon leur notice. Évitez les captures d’écran fixes : un apprenant doit pouvoir changer un paramètre et reproduire la figure.

## Rituel d’étude

1. Prédire qualitativement la forme de la figure avant de lancer la cellule.
2. Exécuter, puis nommer une observation visible.
3. Changer **un seul** paramètre, par exemple le pas du gradient ou le seuil de classification.
4. Comparer les figures sur le même split et avec la même graine quand c’est pertinent.
5. Répondre : « Qu’est-ce que la figure ne peut pas établir ? »

Cette approche s’inspire de la pratique par projets et des contrôles fréquents du [curriculum Microsoft](https://github.com/microsoft/ML-For-Beginners), des [notebooks reproductibles de Géron](https://github.com/ageron/handson-ml3), de l’apprentissage par expérimentation décrit par [fastbook](https://github.com/fastai/fastbook), du workflow PyTorch explicite de [Daniel Bourke](https://github.com/mrdbourke/pytorch-deep-learning) et de la progression expérience → monitoring du [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp). Le texte, les données synthétiques et les figures de ce dépôt sont rédigés indépendamment.
