# ML & Deep Learning Learning Hub

Un parcours francophone, pratique et progressif pour comprendre les systèmes de Machine Learning : explorer des données, construire des modèles, mesurer leurs limites, puis rendre une expérience reproductible.

Le dépôt combine cours, notebooks, exercices et petits projets. Le chemin débutant (`modules/01` à `modules/05`) est le premier parcours complet et exécutable. Les modules avancés sont en cours d’extension ; leur état réel est suivi dans [`docs/implementation-status.md`](docs/implementation-status.md).

## Objectifs pédagogiques

- Relier intuition, équations, code, visualisation et interprétation.
- Savoir préparer les données sans fuite d’information.
- Comparer un baseline à un modèle plus élaboré avec une évaluation adaptée.
- Comprendre les bases des réseaux de neurones, de PyTorch et du déploiement.
- Documenter les expériences pour qu’une autre personne puisse les reproduire.

## À qui s’adresse ce parcours ?

Aux étudiantes, étudiants et autodidactes ayant déjà vu les bases de Python. Les modules 00 et 01 aident à combler les prérequis. Le parcours se suit seul ou en club, avec des exercices à trois niveaux et des propositions de projets en équipe.

## Prérequis

- Python élémentaire : variables, conditions, boucles, fonctions.
- Un terminal et un éditeur (VS Code recommandé, mais pas obligatoire).
- Les notions mathématiques sont introduites au fil des exemples ; algèbre et dérivées de niveau lycée facilitent le module 02.

Voir [prérequis détaillés](docs/prerequisites.md) et [installation](docs/getting-started.md).

## Installation rapide

Python 3.11 ou 3.12 est recommandé. Depuis la racine :

```bash
python -m venv .venv
# Windows PowerShell : .venv\Scripts\Activate.ps1
# macOS/Linux : source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
jupyter lab
```

Les dépendances PyTorch et MLOps sont séparées : `python -m pip install -e ".[dl,mlops]"`. Pour CUDA, choisir la commande qui correspond à sa machine sur le [sélecteur officiel PyTorch](https://pytorch.org/get-started/locally/). Le parcours de base fonctionne sur CPU. Voir [toutes les options d’installation](docs/getting-started.md).

## Parcours recommandé — 14 semaines

| Semaines | Module | Niveau | Durée indicative | Résultat attendu |
|---|---|---|---:|---|
| 0 | [00 · Démarrer](modules/00-getting-started/README.md) | Débutant | 2–3 h | Environnement et méthode de travail prêts |
| 1 | [01 · Python pour la data](modules/01-python-for-data-science/README.md) | Débutant | 5–6 h | Manipuler des tableaux et un DataFrame |
| 2 | [02 · Mathématiques utiles](modules/02-math-for-ml/README.md) | Débutant → intermédiaire | 5–6 h | Relier gradient et mise à jour des paramètres |
| 3–4 | [03 · Données et preprocessing](modules/03-data-preprocessing/README.md) | Débutant → intermédiaire | 8–10 h | Construire une préparation sans fuite |
| 5–6 | [04 · Machine Learning classique](modules/04-classical-machine-learning/README.md) | Intermédiaire | 8–10 h | Comparer régression et classification |
| 7 | [05 · Évaluation](modules/05-model-evaluation/README.md) | Intermédiaire | 5–6 h | Choisir une métrique et expliquer les erreurs |
| 8 | [06 · Réseaux de neurones](modules/06-neural-networks/README.md) | Intermédiaire | 6–8 h | Comprendre un petit réseau et son gradient |
| 9 | [07 · PyTorch](modules/07-pytorch/README.md) | Intermédiaire → avancé | 6–8 h | Écrire une boucle d’entraînement explicite |
| 10 | [08 · Vision](modules/08-computer-vision/README.md) | Avancé | 6–8 h | Entraîner un CNN sur les chiffres 8×8 |
| 11 | [09 · NLP](modules/09-nlp/README.md) | Intermédiaire → avancé | 6–8 h | Comparer TF-IDF et représentations apprises |
| 12 | [10 · Projets](modules/10-practical-projects/README.md) | Tous niveaux | 8–12 h | Compléter un projet guidé |
| 13 | [11 · MLOps](modules/11-mlops/README.md) | Avancé | 6–8 h | Servir un modèle reproductible |
| 14 | [12 · Capstone](modules/12-capstone/README.md) | Avancé | 12–20 h | Livrer un petit système de bout en bout |

Les durées incluent les notebooks et exercices, pas un projet de portfolio complet. On peut étaler le rythme sur 16 semaines.

## Navigation par module

Les notebooks de chaque module indiquent objectifs, intuition, équations, démonstration, interprétation, limites et exercices. Quand une section n’est pas encore développée, le statut le précise.

| Module | Sujet | Ressource principale |
|---|---|---|
| 00 | Installation, Git, notebooks, CPU/GPU | [Guide](modules/00-getting-started/README.md) |
| 01 | Python, NumPy, Pandas et graphiques | [Cours](modules/01-python-for-data-science/README.md) · [Notebook](modules/01-python-for-data-science/01_numpy_pandas.ipynb) |
| 02 | Vecteurs, probabilités, dérivées, gradient descent | [Cours](modules/02-math-for-ml/README.md) · [Algèbre](modules/02-math-for-ml/02_linear_algebra_probability.ipynb) · [Gradient](modules/02-math-for-ml/01_gradient_descent.ipynb) |
| 03 | EDA, qualité, valeurs manquantes, encodage, fuite | [Cours](modules/03-data-preprocessing/README.md) · [Pipeline](modules/03-data-preprocessing/01_pipeline_sans_fuite.ipynb) · [Qualité et features](modules/03-data-preprocessing/02_quality_and_features.ipynb) |
| 04 | Régression, classification, modèles et frontières | [Cours](modules/04-classical-machine-learning/README.md) · [Régression](modules/04-classical-machine-learning/01_regression.ipynb) · [Classification](modules/04-classical-machine-learning/02_logistic_classification.ipynb) |
| 05 | Métriques, validation, erreurs | [Cours](modules/05-model-evaluation/README.md) · [Notebook](modules/05-model-evaluation/01_classification.ipynb) |
| 06 | MLP et rétropropagation | [Cours](modules/06-neural-networks/README.md) |
| 07 | Tensors et entraînement PyTorch | [Cours](modules/07-pytorch/README.md) |
| 08 | CNN et vision par ordinateur | [Cours](modules/08-computer-vision/README.md) |
| 09 | TF-IDF, sentiment et Transformers | [Cours](modules/09-nlp/README.md) |
| 10 | Projets guidés | [Cours](modules/10-practical-projects/README.md) |
| 11 | Expériences, API, Docker et monitoring | [Cours](modules/11-mlops/README.md) |
| 12 | Projet d’équipe de bout en bout | [Brief et barème](modules/12-capstone/README.md) |

## Projets

- [Prédiction de prix immobiliers](projects/house-price-prediction/README.md) — pipeline de régression sur données synthétiques documentées.
- [Attrition client](projects/customer-churn/README.md) — classification binaire synthétique et analyse des erreurs.
- [Classification d’images](projects/image-classification/README.md) — chiffres manuscrits de `scikit-learn` (8×8, inclus dans la bibliothèque).
- [Analyse de sentiment](projects/sentiment-analysis/README.md) — petit corpus pédagogique écrit pour le dépôt ; ne représente pas une population réelle.
- [Service ML de bout en bout](projects/end-to-end-ml/README.md) — entraînement, sérialisation et API locale.

## Exercices et évaluations

Les exercices sont séparés des solutions et répartis par niveau sous [`exercises/`](exercises/). Les mini-quiz sont dans [`quizzes/`](quizzes/). Les corrections expliquent le raisonnement, et pas seulement le résultat attendu.

## Lancer un notebook

Depuis la racine du dépôt, lancer `jupyter lab`, puis ouvrir le notebook ciblé. Les chemins relatifs sont calculés depuis la racine. Les notebooks des modules 01 à 05 n’ont besoin que des dépendances de base. Les modules PyTorch nécessitent le groupe `dl`. Une alternative Colab est documentée dans le guide de chaque module ; aucun compte payant ni secret n’est requis.

## Contribuer

Les contributions sont les bienvenues en français ou en anglais. Lire [CONTRIBUTING.md](CONTRIBUTING.md), les [règles de comportement](CODE_OF_CONDUCT.md), puis utiliser les modèles d’issue et de pull request dans `.github/`.

## Références pédagogiques

Ce dépôt est original. Il s’inspire des approches pédagogiques publiques des projets listés dans [docs/references.md](docs/references.md), sans copier leurs notebooks, textes ou figures. Attribution, licences observées et limites de réutilisation y sont détaillées.

Les [diagrammes pédagogiques](assets/diagrams/README.md) sont maintenus comme Mermaid source.

## Licence

Les nouveaux supports et le code source de ce dépôt sont publiés sous licence MIT (voir [LICENSE](LICENSE)). Les données intégrées ou récupérées et les fichiers antérieurs non créés pour ce dépôt peuvent avoir des conditions distinctes ; consulter leurs notices respectives.
