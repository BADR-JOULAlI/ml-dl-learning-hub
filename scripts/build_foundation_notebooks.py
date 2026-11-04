"""Build the linear algebra, data quality, and scratch logistic lessons."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write(path: str, contents: list[tuple[str, str]]) -> None:
    cells = []
    for index, (kind, source) in enumerate(contents):
        entry = {"cell_type": kind, "id": f"foundation-{index:03d}", "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            entry.update({"execution_count": None, "outputs": []})
        cells.append(entry)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11+"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


write(
    "modules/02-math-for-ml/02_linear_algebra_probability.ipynb",
    [
        ("markdown", "# Algèbre linéaire et probabilités en pratique\n\n**Objectifs.** Lire vecteurs et matrices comme des formes de données, calculer produit scalaire et distance, interpréter une paire propre, résumer une variable aléatoire et appliquer Bayes. Durée : 75–100 min."),
        ("markdown", "## 1. Vecteurs, normes et distances\n\nUn vecteur peut encoder les mesures d’une observation. Le produit scalaire combine deux vecteurs coordonnée par coordonnée. La norme L2 mesure la longueur ; la distance euclidienne mesure l’écart. KNN utilise une distance, d’où l’importance de mettre les variables sur des échelles comparables."),
        ("code", "import numpy as np\n\nobservation = np.array([3.0, 4.0])\nweights = np.array([0.6, 0.8])\nother = np.array([0.0, 4.0])\nprint('shape:', observation.shape)\nprint('dot product:', observation @ weights)\nprint('L2 norm:', np.linalg.norm(observation))\nprint('distance:', np.linalg.norm(observation - other))"),
        ("markdown", "## 2. Matrices et produit matriciel\n\nSi `A` a la forme `(n,p)` et `B` la forme `(p,q)`, alors `A @ B` a la forme `(n,q)`. En ML, les lignes sont souvent les exemples, tandis que colonnes sont variables ou unités cachées."),
        ("code", "A = np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 1.0]])\nB = np.array([[2.0, 0.0], [1.0, 4.0]])\nprint('A:', A.shape, 'B:', B.shape, 'A @ B:', (A @ B).shape)\nprint(A @ B)"),
        ("markdown", "## 3. Valeurs propres : directions qui restent alignées\n\nPour certaines matrices carrées `A`, un vecteur propre `v` conserve sa direction lors de la multiplication : `A @ v = lambda * v`. La valeur `lambda` donne l’étirement dans cette direction. PCA s’appuie sur les directions de variance d’une matrice de covariance."),
        ("code", "cov_like = np.array([[2.0, 1.0], [1.0, 2.0]])\neigenvalues, eigenvectors = np.linalg.eigh(cov_like)\nprint('eigenvalues:', eigenvalues.round(3))\nprint('eigenvectors are columns:\\n', eigenvectors.round(3))\nfor index in range(len(eigenvalues)):\n    vector = eigenvectors[:, index]\n    print(np.allclose(cov_like @ vector, eigenvalues[index] * vector))"),
        ("markdown", "## 4. Moyenne, variance, covariance et corrélation\n\nLa moyenne résume le centre, la variance la dispersion autour du centre. La covariance décrit un changement conjoint mais son échelle dépend des unités ; la corrélation normalisée reste entre -1 et 1. Corrélation ne signifie pas causalité."),
        ("code", "rng = np.random.default_rng(19)\nx = rng.normal(10, 2, size=500)\ny = 0.7 * x + rng.normal(0, 1.4, size=500)\nprint('mean/std x:', round(x.mean(), 2), round(x.std(), 2))\nprint('covariance matrix:\\n', np.cov(np.column_stack([x, y])).round(2))\nprint('correlation:', round(float(np.corrcoef(x, y)[0, 1]), 3))"),
        ("markdown", "## 5. Bayes et base rates\n\nBayes inverse une probabilité conditionnelle : \\(P(A|B)=P(B|A)P(A)/P(B)\\). Exemple fictif : un signal rare a une prévalence de 1 %, un test détecte 95 % des cas et produit 10 % de faux positifs chez les non-cas. Parmi les alertes, la proportion de vrais cas dépend aussi de cette prévalence."),
        ("code", "prevalence = 0.01\nsensitivity = 0.95\nfalse_positive_rate = 0.10\ntrue_positive_rate = sensitivity * prevalence\nall_positive_rate = true_positive_rate + false_positive_rate * (1 - prevalence)\nposterior = true_positive_rate / all_positive_rate\nprint('probability of condition after positive signal:', round(posterior, 3))"),
        ("markdown", "## Interprétation et limites\n\nUne sensibilité élevée ne garantit pas une valeur prédictive positive élevée lorsque le phénomène est rare. L’exemple est mathématique et n’est pas un conseil médical. Les matrices, distributions et formules doivent être adaptées au plan d’échantillonnage réel.\n\nÀ réviser ensuite : [`01_gradient_descent.ipynb`](01_gradient_descent.ipynb) pour le gradient et la loss. Exercices : [`02_gradients.ipynb`](../../exercises/intermediate/02_gradients.ipynb)."),
    ],
)


write(
    "modules/03-data-preprocessing/02_quality_and_features.ipynb",
    [
        ("markdown", "# Audit de données, variables et classes déséquilibrées\n\n**Objectifs.** Repérer doublons et trous, quantifier une valeur atypique sans la supprimer automatiquement, créer une feature explicable, préserver le ratio des classes et comparer les modèles avec une métrique adaptée. Durée : 90 min."),
        ("markdown", "## 1. Un jeu synthétique à auditer\n\nLes observations sont générées artificiellement. Nous injectons des doublons, quelques valeurs manquantes et une valeur extrême pour pratiquer les contrôles. La cible ne doit pas être utilisée comme feature."),
        ("code", "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom learning_hub.preprocessing import make_churn_data\n\nX, y = make_churn_data(n_samples=800, random_state=24)\nraw = X.assign(churn=y)\nraw = pd.concat([raw, raw.iloc[:3]], ignore_index=True)\nraw.loc[raw.index[12:30:4], 'monthly_charge'] = np.nan\nraw.loc[40, 'monthly_charge'] = 650.0\nprint('shape:', raw.shape)\nprint('duplicates:', raw.duplicated().sum())\nprint('missing:\\n', raw.isna().sum())\nprint('target balance:\\n', raw.churn.value_counts(normalize=True).round(3))"),
        ("markdown", "## 2. Nettoyer avant le split et décider des outliers\n\nLes lignes dupliquées peuvent traverser train et test si on split avant déduplication. Ici nous enlevons uniquement des lignes strictement identiques et gardons le premier exemplaire. Une valeur extrême est un signal à examiner : elle peut être vraie, erronée ou issue d’une autre unité."),
        ("code", "clean = raw.drop_duplicates().copy()\nprint('rows removed:', len(raw) - len(clean))\nq1, q3 = clean['monthly_charge'].quantile([0.25, 0.75])\niqr = q3 - q1\nlower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr\noutliers = clean.loc[(clean.monthly_charge < lower) | (clean.monthly_charge > upper)]\nprint('IQR review bounds:', round(lower, 1), round(upper, 1), '| flagged rows:', len(outliers))\nclean['monthly_charge'].plot(kind='box', title='Valeurs mensuelles simulées'); plt.show()"),
        ("markdown", "## 3. Split stratifié et feature engineering\n\nOn garde la cible à part, puis on sépare de manière stratifiée. Une transformation déterministe de colonnes déjà disponibles peut être créée avant split ; les paramètres appris (médiane, moyenne, variance, vocabulaire) restent dans la pipeline. `calls_per_month` convertit les appels observés sur 90 jours en taux mensuel."),
        ("code", "from sklearn.model_selection import train_test_split\n\nlabels = clean.pop('churn')\nclean['calls_per_month'] = clean['support_calls_90d'] / 3.0\nx_train, x_test, y_train, y_test = train_test_split(clean, labels, stratify=labels, test_size=0.25, random_state=24)\nprint('positive fractions train/test:', round(y_train.mean(), 3), round(y_test.mean(), 3))\nprint('feature columns:', list(x_train.columns))"),
        ("markdown", "## 4. Preprocessing, baseline et déséquilibre\n\nAccuracy peut être élevée en prédisant toujours la classe majoritaire. Average Precision mesure le compromis precision–recall sur plusieurs seuils. Toutes les transformations et la sélection de variables sont ajustées dans le pipeline à l’intérieur des folds."),
        ("code", "from sklearn.dummy import DummyClassifier\nfrom sklearn.feature_selection import SelectKBest, f_classif\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import average_precision_score, classification_report\nfrom sklearn.model_selection import StratifiedKFold, cross_val_score\nfrom sklearn.pipeline import Pipeline\nfrom learning_hub.preprocessing import make_tabular_preprocessor\n\nnumeric = ['tenure_months', 'monthly_charge', 'support_calls_90d', 'calls_per_month']\ncategorical = ['contract']\npreprocess = make_tabular_preprocessor(numeric, categorical)\nbaseline = DummyClassifier(strategy='most_frequent').fit(x_train, y_train)\nmodel = Pipeline([('preprocess', preprocess), ('select', SelectKBest(f_classif, k=5)), ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))])\nfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)\nscores = cross_val_score(model, x_train, y_train, cv=folds, scoring='average_precision')\nmodel.fit(x_train, y_train)\nprobability = model.predict_proba(x_test)[:, 1]\nprint('majority baseline accuracy:', round(baseline.score(x_test, y_test), 3))\nprint('CV average precision mean/std:', round(scores.mean(), 3), round(scores.std(), 3))\nprint(classification_report(y_test, probability >= 0.5, zero_division=0))\nprint('test average precision:', round(average_precision_score(y_test, probability), 3))"),
        ("markdown", "## 5. Interprétation, leakage et limites\n\nIQR bounds calculés sur toute la table sont descriptifs ; pour clipper en production, borner à partir du train et appliquer exactement la même règle à validation/test et futures données. La déduplication dépend de l’identifiant métier et du moment de collecte. La feature `calls_per_month` n’ajoute aucune nouvelle observation ; elle réexprime la mesure. Les labels synthétiques ne représentent aucun vrai client."),
        ("markdown", "## Suite et exercices\n\nÉvaluez l’effet d’une autre règle de missingness, comparez `class_weight=None` à `balanced`, inspectez les faux négatifs, et séparez par client si plusieurs lignes décrivent le même client. Ne choisissez pas le seuil avec le test. Corrélations et importances ne prouvent pas de causalité."),
    ],
)


write(
    "modules/04-classical-machine-learning/02_logistic_classification.ipynb",
    [
        ("markdown", "# Classification : logistic regression à la main et frontières\n\n**Objectifs.** Dériver et coder une logistic regression binaire, visualiser une frontière linéaire, comparer à KNN, arbre, SVM RBF et Naive Bayes sur le même split. Durée : 90–120 min."),
        ("markdown", "## 1. Un problème non linéaire en deux dimensions\n\nLes deux caractéristiques artificielles forment deux croissants. La logistic regression combine des variables linéairement puis passe par sigmoid ; une seule frontière droite ne peut donc pas suivre parfaitement les deux courbes."),
        ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.datasets import make_moons\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\n\nX, y = make_moons(n_samples=600, noise=0.22, random_state=11)\nx_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=11)\nscaler = StandardScaler().fit(x_train)\nx_train_scaled = scaler.transform(x_train)\nx_test_scaled = scaler.transform(x_test)\nplt.scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], c=y_train, cmap='coolwarm', s=15)\nplt.xlabel('feature 1 (standardized)'); plt.ylabel('feature 2 (standardized)'); plt.title('Training observations')\nplt.show()"),
        ("markdown", "## 2. Implémentation NumPy et gradients\n\nPour logits `z = X @ w + b`, la probabilité est `sigmoid(z)`. La cross-entropy binaire mène au gradient `X.T @ (p-y)/n` et `mean(p-y)` pour le biais. La standardisation est ajustée sur train seulement pour stabiliser l’optimisation."),
        ("code", "def sigmoid(logits):\n    return 1 / (1 + np.exp(-np.clip(logits, -30, 30)))\n\nweight = np.zeros(x_train_scaled.shape[1])\nbias = 0.0\nlearning_rate = 0.12\nlosses = []\nfor _ in range(2_000):\n    probability = sigmoid(x_train_scaled @ weight + bias)\n    clipped = np.clip(probability, 1e-8, 1 - 1e-8)\n    losses.append(float(-np.mean(y_train * np.log(clipped) + (1 - y_train) * np.log(1 - clipped))))\n    error = probability - y_train\n    weight -= learning_rate * (x_train_scaled.T @ error) / len(y_train)\n    bias -= learning_rate * error.mean()\nprobability_test = sigmoid(x_test_scaled @ weight + bias)\nprediction_test = (probability_test >= 0.5).astype(int)\nprint('weights:', weight.round(3), 'bias:', round(float(bias), 3), 'final train loss:', round(losses[-1], 3))\nprint('held-out accuracy:', round(float((prediction_test == y_test).mean()), 3))"),
        ("markdown", "## 3. Implémentations de bibliothèque et compromis\n\nLe même split permet une comparaison pédagogique. Scaling est dans chaque pipeline et s’entraîne uniquement sur le train. Les coûts sont des ordres de grandeur dépendant de l’algorithme, implémentation et données."),
        ("code", "from sklearn.linear_model import LogisticRegression\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.svm import SVC\nfrom sklearn.naive_bayes import GaussianNB\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.metrics import accuracy_score, f1_score\n\nmodels = {\n    'logistic': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),\n    'knn k=15': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),\n    'decision tree': DecisionTreeClassifier(max_depth=5, random_state=11),\n    'rbf svm': make_pipeline(StandardScaler(), SVC(C=1.0, gamma='scale')),\n    'GaussianNB': GaussianNB(),\n}\nfor name, estimator in models.items():\n    estimator.fit(x_train, y_train)\n    predicted = estimator.predict(x_test)\n    print(name, 'accuracy=', round(accuracy_score(y_test, predicted), 3), 'F1=', round(f1_score(y_test, predicted), 3))"),
        ("markdown", "## 4. Visualiser les frontières\n\nLa couleur de fond indique la classe prédite en chaque point du plan. Le contour n’est qu’une projection 2D ; un modèle avec plus de variables n’aurait pas un unique graphique équivalent."),
        ("code", "from matplotlib.colors import ListedColormap\n\nmesh_x, mesh_y = np.meshgrid(np.linspace(x_train[:, 0].min() - 0.4, x_train[:, 0].max() + 0.4, 180), np.linspace(x_train[:, 1].min() - 0.4, x_train[:, 1].max() + 0.4, 180))\nmesh = np.c_[mesh_x.ravel(), mesh_y.ravel()]\nfig, axes = plt.subplots(2, 2, figsize=(10, 8))\nfor ax, name in zip(axes.flat, ['logistic', 'knn k=15', 'decision tree', 'rbf svm']):\n    prediction_grid = models[name].predict(mesh).reshape(mesh_x.shape)\n    ax.contourf(mesh_x, mesh_y, prediction_grid, alpha=0.25, cmap='coolwarm')\n    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=10, cmap='coolwarm')\n    ax.set_title(name)\nplt.tight_layout(); plt.show()"),
        ("markdown", "## 5. Hypothèses, complexité et limites\n\n| Algorithme | Idée/hypothèse utile | Limite pratique fréquente |\n|---|---|---|\n| Logistic regression | frontière linéaire en features | nécessite feature engineering pour courbure |\n| KNN | voisins semblables ont des labels proches | sensible à l’échelle et coûteux à prédire (`O(n·p)`) |\n| Decision tree | règles de seuil successives | profondeur excessive sur-apprend |\n| SVM RBF | marge avec frontière non linéaire | scaling et choix d’hyperparamètres ; kernel peut coûter cher en `n` |\n| Gaussian Naive Bayes | features conditionnellement indépendantes, vraisemblance gaussienne | hypothèse souvent simplificatrice |\n\nUne seule accuracy et une graine ne classent pas les modèles universellement. Pour continuer : inclure CV, calibration, Random Forest et Gradient Boosting ; faire un clustering ou PCA sans cible dans un notebook ultérieur. Exercices : [`04_models.ipynb`](../../exercises/intermediate/04_models.ipynb)."),
    ],
)


if __name__ == "__main__":
    print("Generated three foundational notebooks.")
