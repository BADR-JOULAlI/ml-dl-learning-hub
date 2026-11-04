"""Create course notebooks from reviewable source cells in this file."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def write_notebook(relative_path: str, cells: list[dict]) -> None:
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:03d}"
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11+"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = ROOT / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


write_notebook(
    "modules/01-python-for-data-science/01_numpy_pandas.ipynb",
    [
        markdown("# NumPy et Pandas : des observations aux graphiques\n\n**Objectifs.** Lire les dimensions d’un tableau, appliquer le broadcasting, résumer un DataFrame et tracer une relation avec des unités. Durée : 60–90 min.\n\nNous suivons ici des mesures de plantes entièrement inventées pour l’exercice. Chaque ligne représente une plante, chaque colonne une variable mesurée."),
        markdown("## 1. Une table devient un tableau numérique\n\nPour `X.shape == (n, p)`, `n` est le nombre d’observations et `p` le nombre de variables. Une ligne contient les caractéristiques d’une plante."),
        code("import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nrng = np.random.default_rng(12)\nheights_cm = rng.normal(45, 8, size=80).clip(20, 75)\nwidths_cm = 0.42 * heights_cm + rng.normal(0, 3, size=80)\nX = np.column_stack([heights_cm, widths_cm])\nX.shape, X[:3]") ,
        markdown("## 2. Broadcasting et produit matriciel\n\nLe broadcasting applique une opération élément par élément lorsque les dimensions sont compatibles. Le produit scalaire additionne les produits coordonnée par coordonnée. Ici, un modèle linéaire calcule une prédiction par ligne : `(n, 2) @ (2,) -> (n,)`."),
        code("center = X.mean(axis=0)\nscale = X.std(axis=0)\nX_scaled = (X - center) / scale\nweights = np.array([0.7, 0.3])\nscores = X_scaled @ weights\nprint('shape de X :', X.shape)\nprint('moyennes après centrage :', X_scaled.mean(axis=0).round(10))\nprint('shape des scores :', scores.shape)") ,
        markdown("## 3. Passer au DataFrame\n\nPandas donne des noms aux variables et des opérations pratiques pour les inspecter. L’inspection ne remplace pas une décision sur la qualité des mesures."),
        code("plants = pd.DataFrame({'height_cm': heights_cm, 'width_cm': widths_cm})\nplants['shape_ratio'] = plants['width_cm'] / plants['height_cm']\nprint(plants.head())\nprint('Dimensions :', plants.shape)\nprint('Valeurs manquantes :\\n', plants.isna().sum())\nprint('Résumé :\\n', plants.describe().round(2))"),
        markdown("## 4. Lire un fichier sans dépendance externe\n\nEn production, le chemin et l’encodage doivent être documentés. Une cellule `StringIO` montre ici l’interface de lecture CSV sans créer de fichier temporaire."),
        code("from io import StringIO\n\nsample_csv = 'species,height_cm\\nA,41.2\\nB,53.8\\nA,48.1\\n'\nsmall_table = pd.read_csv(StringIO(sample_csv))\nsmall_table, small_table.groupby('species')['height_cm'].mean()"),
        markdown("### JSON et Parquet\n\nUn tableau JSON peut se lire avec `read_json`. Parquet conserve schéma et types, et convient aux tables larges ou partitionnées. Pour lire/écrire Parquet, installer le moteur optionnel `python -m pip install -e \".[parquet]\"`. Exemple à exécuter après cette installation :\n\n```python\nsmall_table.to_parquet('sample.parquet', index=False)\nrestored = pd.read_parquet('sample.parquet')\n```"),
        code("json_text = '[{\"species\": \"A\", \"height_cm\": 41.2}, {\"species\": \"B\", \"height_cm\": 53.8}]'\njson_table = pd.read_json(StringIO(json_text))\njson_table"),
        markdown("## 5. Visualiser et interpréter\n\nLes axes, unités et titre rendent le graphique lisible hors du notebook. Une association visible ne prouve ni causalité ni utilité prédictive."),
        code("fig, ax = plt.subplots(figsize=(6, 4))\nax.scatter(plants['height_cm'], plants['width_cm'], alpha=0.7)\nax.set(title='Dimensions simulées des plantes', xlabel='Hauteur (cm)', ylabel='Largeur (cm)')\nax.grid(alpha=0.2)\nplt.show()"),
        markdown("## 6. Expérience et limites\n\nLe ratio `width/height` permet une comparaison relative, mais il peut devenir instable si le dénominateur est nul ou très petit. Ici les hauteurs sont strictement positives. Modifiez la dispersion du bruit ci-dessous et comparez la corrélation ; le résultat dépend de la graine."),
        code("for noise_level in [1, 5, 12]:\n    simulated_width = 0.42 * heights_cm + rng.normal(0, noise_level, size=len(heights_cm))\n    print(noise_level, round(float(np.corrcoef(heights_cm, simulated_width)[0, 1]), 3))"),
        markdown("## À retenir et exercices\n\n- Vérifier `shape`, types et unités avant de calculer.\n- `axis=0` résume chaque colonne ; `axis=1` chaque observation.\n- Le broadcasting ne remplace pas un produit matriciel.\n- Les données simulées valident le code, pas une conclusion sur des plantes réelles.\n\nExercices séparés : [`01_python.ipynb`](../../exercises/beginner/01_python.ipynb)."),
    ],
)


write_notebook(
    "modules/02-math-for-ml/01_gradient_descent.ipynb",
    [
        markdown("# Gradient descent : ajuster une droite\n\n**Objectifs.** Calculer une loss, relier pente et dérivée, coder une mise à jour de gradient et visualiser l’effet du pas. Durée : 60–90 min.\n\nOn cherche les paramètres `w` et `b` d’une droite `y_hat = w*x + b`. Les données ci-dessous sont fabriquées selon `y = 3x + 2` avec un bruit léger."),
        markdown("## 1. Loss quadratique\n\nPour `n` points, \\(J(w,b)=\\frac{1}{n}\\sum_i(wx_i+b-y_i)^2\\). La loss est grande lorsque les prédictions s’éloignent des observations. Le but est d’en trouver un minimum, pas d’obtenir une pente de zéro."),
        code("import numpy as np\nimport matplotlib.pyplot as plt\n\nrng = np.random.default_rng(3)\nx = np.linspace(-2, 2, 80)\ny = 3 * x + 2 + rng.normal(0, 0.25, size=x.size)\nplt.scatter(x, y, s=18)\nplt.xlabel('x'); plt.ylabel('y')\nplt.title('Exemples observés')\nplt.show()"),
        markdown("## 2. Dérivées partielles\n\nAvec `error = prediction - y`, on obtient \\(\\partial J/\\partial w = 2\\,mean(error\\cdot x)\\) et \\(\\partial J/\\partial b = 2\\,mean(error)\\). Les formes NumPy sont des scalaires : les deux paramètres sont partagés par toutes les observations."),
        code("def mse(weight, bias, features, targets):\n    predictions = weight * features + bias\n    return np.mean((predictions - targets) ** 2)\n\nweight, bias = 0.0, 0.0\nerror = weight * x + bias - y\ngrad_weight = 2 * np.mean(error * x)\ngrad_bias = 2 * np.mean(error)\nprint('J(0, 0) =', round(float(mse(weight, bias, x, y)), 3))\nprint('gradient (w, b) =', round(float(grad_weight), 3), round(float(grad_bias), 3))"),
        markdown("## 3. Implémenter une mise à jour\n\nSoustraire une fraction du gradient déplace les paramètres vers une région de loss plus faible. `learning_rate` est cette fraction. On enregistre la loss avant chaque mise à jour pour examiner l’apprentissage."),
        code("learning_rate = 0.08\nepochs = 300\nweight, bias = 0.0, 0.0\nloss_history = []\nfor _ in range(epochs):\n    prediction = weight * x + bias\n    error = prediction - y\n    loss_history.append(float(np.mean(error ** 2)))\n    grad_weight = 2 * np.mean(error * x)\n    grad_bias = 2 * np.mean(error)\n    weight -= learning_rate * grad_weight\n    bias -= learning_rate * grad_bias\nprint(f'w={weight:.3f}, b={bias:.3f}, final MSE={loss_history[-1]:.4f}')"),
        markdown("## 4. Visualiser la convergence\n\nUne courbe descendante indique que cet entraînement réduit la loss sur ces points. Cela ne garantit pas une bonne généralisation, ni que toute learning rate fonctionnera."),
        code("fig, axes = plt.subplots(1, 2, figsize=(10, 4))\naxes[0].plot(loss_history)\naxes[0].set(xlabel='Epoch', ylabel='MSE', title='Loss pendant l’entraînement')\naxes[1].scatter(x, y, s=18, label='Données')\naxes[1].plot(x, weight * x + bias, color='darkorange', label='Droite ajustée')\naxes[1].legend(); axes[1].set(xlabel='x', ylabel='y', title='Prédiction finale')\nplt.tight_layout(); plt.show()"),
        markdown("## 5. Expérimenter avec le pas\n\nUn pas trop grand peut osciller ou diverger. Cette cellule mesure la loss après 40 mises à jour pour plusieurs taux. Si la valeur devient non finie, la descente a divergé."),
        code("for rate in [0.01, 0.2, 0.8, 1.5]:\n    w, b = 0.0, 0.0\n    for _ in range(40):\n        err = w * x + b - y\n        w -= rate * 2 * np.mean(err * x)\n        b -= rate * 2 * np.mean(err)\n    print(f'learning_rate={rate:>4}: MSE={mse(w, b, x, y):.3g}')"),
        markdown("## 6. Interpréter et relier au code\n\n`x` et `y` ont la forme `(80,)`. `prediction` et `error` gardent cette forme. Le gradient agrège les 80 contributions par `mean`, ce qui produit un nombre pour chaque paramètre. Les paramètres approchent 3 et 2, mais le bruit et la longueur finie du calcul empêchent une égalité parfaite.\n\n**Limites :** MSE pénalise les grosses erreurs au carré ; les variables à des échelles très différentes peuvent rendre le choix du pas difficile ; les minima locaux et dimensions multiples rendent les problèmes réels plus complexes."),
        markdown("## À retenir et exercices\n\nGradient descent est une méthode numérique générale, pas une garantie automatique. Comparez avec la solution analytique simple, dessinez une mise à jour sur papier, puis modifiez la graine. Exercices : [`02_gradients.ipynb`](../../exercises/intermediate/02_gradients.ipynb)."),
    ],
)


write_notebook(
    "modules/03-data-preprocessing/01_pipeline_sans_fuite.ipynb",
    [
        markdown("# Construire un pipeline sans fuite de données\n\n**Objectifs.** Inspecter des colonnes mixtes, séparer les observations avant l’ajustement, imputer et encoder dans une pipeline, et comparer deux traitements de l’échelle. Durée : 90 min.\n\nNous utilisons le dataset breast cancer fourni avec scikit-learn. Les variables et labels sont conservés localement ; la colonne catégorielle et les valeurs manquantes sont des transformations pédagogiques simulées."),
        markdown("## 1. Comprendre les lignes, colonnes et labels\n\n`X` représente les caractéristiques disponibles au moment de la prédiction ; `y` la cible. Le dataset intégré est fourni pour l’apprentissage, pas pour tirer une conclusion clinique."),
        code("import numpy as np\nimport pandas as pd\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split\n\ndata = load_breast_cancer(as_frame=True)\nX = data.data[['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']].copy()\ny = data.target\nX['radius_band'] = np.where(X['mean radius'] < 15, 'small', 'large')\nrng = np.random.default_rng(42)\nmissing_rows = rng.choice(X.index, size=25, replace=False)\nX.loc[missing_rows, 'mean texture'] = np.nan\nprint('X:', X.shape, '| y:', y.shape)\nprint(X.dtypes)\nprint('Taux manquant :\\n', X.isna().mean().round(3))"),
        markdown("## 2. Séparer avant d’apprendre les transformations\n\nLe split stratifié préserve approximativement les proportions des classes. La pipeline apprend ses statistiques uniquement quand on appelle `fit` sur `X_train`. Le jeu test reste de côté jusqu’à l’évaluation."),
        code("x_train, x_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.25, stratify=y, random_state=42\n)\nprint('train :', x_train.shape, '| test :', x_test.shape)\nprint('proportion classe 1 train/test :', round(y_train.mean(), 3), round(y_test.mean(), 3))"),
        markdown("## 3. Expliquer les étapes du preprocessing\n\nLes valeurs numériques sont imputées par leur médiane train puis standardisées. La catégorie est imputée par le mode train et transformée en colonnes binaires. L’encodeur ignore une catégorie jamais vue au fit plutôt que d’échouer."),
        code("from sklearn.compose import ColumnTransformer\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score\n\nnumeric_columns = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']\ncategorical_columns = ['radius_band']\npreprocess = ColumnTransformer([\n    ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_columns),\n    ('category', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('one_hot', OneHotEncoder(handle_unknown='ignore'))]), categorical_columns),\n])\npipeline = Pipeline([('preprocess', preprocess), ('classifier', LogisticRegression(max_iter=2_000))])\npipeline.fit(x_train, y_train)\nprint('Accuracy test :', round(accuracy_score(y_test, pipeline.predict(x_test)), 3))"),
        markdown("## 4. Où se cacherait une fuite ?\n\nSi une médiane est calculée sur toutes les lignes avant le split, une statistique du test influence le modèle. Le petit exemple ci-dessous compare les médianes ; les valeurs test sont volontairement décalées pour rendre le problème visible. Le pipeline ci-dessus évite cette fuite en apprenant l’imputeur sur train."),
        code("demo = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 100.0, 120.0]})\ntrain_demo, test_demo = demo.iloc[:4], demo.iloc[4:]\nprint('Médiane train :', train_demo['value'].median())\nprint('Médiane calculée illégalement sur train + test :', demo['value'].median())\nprint('Médiane test :', test_demo['value'].median())"),
        markdown("## 5. Expérience : l’échelle compte-t-elle ?\n\nLa régression logistique régularisée compare des coefficients et peut être sensible à l’échelle. Le même train/test et la même valeur de `C` sont utilisés ; seul le scaling change."),
        code("from sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import make_pipeline\n\nscaled_model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=2_000))\nunscaled_model = make_pipeline(SimpleImputer(strategy='median'), LogisticRegression(max_iter=2_000))\nfor name, model in [('avec StandardScaler', scaled_model), ('sans scaler', unscaled_model)]:\n    model.fit(x_train[numeric_columns], y_train)\n    print(name, round(accuracy_score(y_test, model.predict(x_test[numeric_columns])), 3))"),
        markdown("## 6. Conclusions, limites et suite\n\nUn score n’est pas suffisant : on inspectera aussi les classes, les faux positifs/faux négatifs, le split et la stabilité en validation croisée. Le dataset a des observations apparentées et ne remplace pas une étude clinique. La catégorie `radius_band` est dérivée d’une feature et n’apporte pas d’information indépendante.\n\n**À retenir :** split d’abord ; fit des imputers/scalers/encoders ensuite, uniquement via la pipeline sur train ; test une seule fois pour le rapport final. Les choix de données manquantes, outliers et variables temporelles nécessitent une justification métier."),
        markdown("## Exercices\n\nEcrivez une pipeline avec un seuil de catégorie différent et mesurez les changements ; comparez imputation moyenne et médiane ; vérifiez que les catégories inconnues ne font pas échouer la prédiction. [`03_preprocessing.ipynb`](../../exercises/intermediate/03_preprocessing.ipynb)."),
    ],
)


write_notebook(
    "modules/04-classical-machine-learning/01_regression.ipynb",
    [
        markdown("# Régression linéaire : de la formule à scikit-learn\n\n**Objectifs.** Ajuster une droite par gradient descent, comparer à `LinearRegression`, utiliser un dataset de régression intégré, et interpréter un résidu. Durée : 90 min."),
        markdown("## 1. Intuition et modèle\n\nLa régression prédit une quantité numérique. Pour un seul attribut, `y_hat = w*x+b` ; `w` est la pente et `b` l’intercept. MSE choisit les coefficients en pénalisant les grandes erreurs au carré."),
        code("import numpy as np\nimport matplotlib.pyplot as plt\n\nrng = np.random.default_rng(8)\nx = np.linspace(0, 10, 100)\ny = 4.5 * x + 7 + rng.normal(0, 3, x.size)\nplt.scatter(x, y, s=18); plt.xlabel('Heures de pratique'); plt.ylabel('Score simulé')\nplt.title('Données synthétiques'); plt.show()"),
        markdown("## 2. Calculer une régression à la main\n\nLa dérivée de MSE indique comment modifier chacun des paramètres. On part de `w=0,b=0`, puis on répète la mise à jour. L’implémentation ci-dessous n’utilise aucune librairie ML."),
        code("weight, bias = 0.0, 0.0\nlearning_rate = 0.01\nlosses = []\nfor _ in range(800):\n    prediction = weight * x + bias\n    error = prediction - y\n    losses.append(np.mean(error ** 2))\n    grad_w = 2 * np.mean(error * x)\n    grad_b = 2 * np.mean(error)\n    weight -= learning_rate * grad_w\n    bias -= learning_rate * grad_b\nprint(f'pente={weight:.2f}; intercept={bias:.2f}; MSE={losses[-1]:.2f}')"),
        markdown("## 3. Visualiser et inspecter les résidus\n\nUn résidu est `observed - predicted`. Une structure courbe dans les résidus peut indiquer que la droite manque une relation non linéaire ; une forme en éventail peut indiquer une variance non constante."),
        code("predictions = weight * x + bias\nresiduals = y - predictions\nfig, axes = plt.subplots(1, 2, figsize=(10, 4))\naxes[0].scatter(x, y, s=16); axes[0].plot(x, predictions, color='darkorange')\naxes[0].set(xlabel='x', ylabel='y', title='Droite ajustée')\naxes[1].scatter(predictions, residuals, s=16); axes[1].axhline(0, color='black', linewidth=1)\naxes[1].set(xlabel='Prédiction', ylabel='Résidu', title='Diagnostic des erreurs')\nplt.tight_layout(); plt.show()"),
        markdown("## 4. Comparer à l’implémentation de bibliothèque\n\n`scikit-learn` facilite la composition mais n’enlève pas la nécessité de choisir un split, des variables et des métriques. Nous reproduisons la comparaison sur les mêmes points."),
        code("from sklearn.linear_model import LinearRegression\n\nmodel = LinearRegression().fit(x.reshape(-1, 1), y)\nprint('pente/intercept scikit-learn :', model.coef_[0].round(3), model.intercept_.round(3))\nprint('différence avec la descente :', round(weight - model.coef_[0], 4), round(bias - model.intercept_, 4))"),
        markdown("## 5. Tester sur un problème tabulaire intégré\n\nLe dataset diabetes est distribué avec scikit-learn. Un split reproductible donne un score de démonstration ; le nombre dépend du split. L’évaluation plus soigneuse par cross-validation arrive au module 05."),
        code("from sklearn.datasets import load_diabetes\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n\nfeatures, target = load_diabetes(return_X_y=True, as_frame=True)\nx_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)\nregressor = LinearRegression().fit(x_train, y_train)\nprediction = regressor.predict(x_test)\nprint('MAE :', round(mean_absolute_error(y_test, prediction), 2))\nprint('RMSE :', round(mean_squared_error(y_test, prediction) ** 0.5, 2))\nprint('R² :', round(r2_score(y_test, prediction), 3))"),
        markdown("## Limites, conclusion et exercice\n\nUne relation linéaire peut mal représenter des effets non linéaires, interactions ou mesures biaisées. Un coefficient n’est causal que sous des hypothèses bien plus fortes. Comparez au baseline `DummyRegressor`, et testez une transformation polynomial sans regarder le test pour choisir son degré.\n\nExercices et correction séparés : [`04_models.ipynb`](../../exercises/intermediate/04_models.ipynb)."),
    ],
)


write_notebook(
    "modules/05-model-evaluation/01_classification.ipynb",
    [
        markdown("# Évaluer une classification sans se limiter à l’accuracy\n\n**Objectifs.** Construire une baseline, lire une confusion matrix, relier précision/rappel au seuil, et comparer des modèles sans toucher au test pendant le tuning. Durée : 75–100 min."),
        markdown("## 1. Une classification binaire et sa baseline\n\nLe dataset `breast_cancer` est intégré à scikit-learn. Dans cette leçon on interprète la classe `1` telle que fournie ; ce choix est pédagogique et n’a aucune portée de diagnostic. On commence par un split stratifié."),
        code("import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.dummy import DummyClassifier\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score)\nfrom sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n\ndata = load_breast_cancer(as_frame=True)\nX, y = data.data, data.target\nx_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=21)\nbaseline = DummyClassifier(strategy='most_frequent').fit(x_train, y_train)\nmodel = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=2_000))\nmodel.fit(x_train, y_train)\nprint('Baseline accuracy :', round(accuracy_score(y_test, baseline.predict(x_test)), 3))\nprint('Modèle accuracy :', round(accuracy_score(y_test, model.predict(x_test)), 3))"),
        markdown("## 2. Confusion matrix et métriques\n\n`TN, FP, FN, TP` signifient vrais négatifs, faux positifs, faux négatifs et vrais positifs. La précision parle des alertes produites, le rappel des positifs trouvés. Quelle erreur coûte le plus dépend de l’usage."),
        code("test_prediction = model.predict(x_test)\ncm = confusion_matrix(y_test, test_prediction)\nConfusionMatrixDisplay(cm, display_labels=data.target_names).plot(cmap='Blues', values_format='d')\nplt.title('Comptes sur le test'); plt.show()\nprint('precision :', round(precision_score(y_test, test_prediction), 3))\nprint('recall    :', round(recall_score(y_test, test_prediction), 3))\nprint('F1        :', round(f1_score(y_test, test_prediction), 3))"),
        markdown("## 3. Le seuil modifie les erreurs\n\nLa régression logistique retourne des scores probabilistes. Le seuil 0.5 est conventionnel, pas magique. Le modifier change les alertes, donc FP et FN ; le test ne doit pas être utilisé pour choisir ce seuil. Cette visualisation est descriptive et ne propose pas de seuil clinique."),
        code("probability = model.predict_proba(x_test)[:, 1]\nfor threshold in [0.3, 0.5, 0.7]:\n    prediction_at_threshold = (probability >= threshold).astype(int)\n    print(f'seuil={threshold:.1f} precision={precision_score(y_test, prediction_at_threshold):.3f} recall={recall_score(y_test, prediction_at_threshold):.3f}')\nprecision_values, recall_values, thresholds = precision_recall_curve(y_test, probability)\nplt.plot(recall_values, precision_values)\nplt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Courbe precision–recall (split de démonstration)')\nplt.grid(alpha=0.2); plt.show()\nprint('Average precision :', round(average_precision_score(y_test, probability), 3))"),
        markdown("## 4. Validation croisée pour comparer\n\nLa cross-validation utilise plusieurs partitions train/validation, sans toucher au test final. La pipeline entière est ajustée dans chaque fold, donc l’imputation et le scaling n’aperçoivent pas la validation de ce fold."),
        code("cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)\nfold_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring='average_precision')\nprint('AP par fold :', fold_scores.round(3))\nprint('Moyenne ± écart-type :', round(fold_scores.mean(), 3), '±', round(fold_scores.std(), 3))"),
        markdown("## 5. Examiner les erreurs et les limites\n\nUne métrique résume un groupe d’exemples. Avant d’agir, examiner cas mal classés, calibrage, provenance et représentativité des données. Les mesures du même dataset peuvent être dépendantes ; une stratification simple ne corrige pas toutes les formes de dépendance. ROC-AUC ou PR-AUC évaluent classement, pas le coût réel ni calibration."),
        code("error_table = x_test.copy()\nerror_table['actual'] = y_test\nerror_table['predicted'] = test_prediction\nerror_table['probability_class_1'] = probability\nmisclassified = error_table[error_table['actual'] != error_table['predicted']]\nprint('Nombre d’erreurs :', len(misclassified))\nmisclassified[['actual', 'predicted', 'probability_class_1']].head()"),
        markdown("## À retenir et exercices\n\nPrévoir une règle de décision, mesurer les erreurs associées, puis écrire la métrique principale et les coûts qu’elle ne capture pas. Garder le test final hors du réglage. Exercices : [`05_evaluation.ipynb`](../../exercises/intermediate/05_evaluation.ipynb)."),
    ],
)


exercise_sets = [
    (
        "01_python",
        "beginner",
        "Python, NumPy et Pandas",
        "Créez un tableau de 12 mesures ; calculez moyenne et écart-type sans boucle ; placez-le dans un DataFrame et groupez par catégorie ; tracez les observations avec titres et unités.",
        "01_python_solutions",
        """import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nrng = np.random.default_rng(4)\nvalues = rng.normal(10, 2, 12)\n# Easy: calculate mean and standard deviation with NumPy.\n# Medium: build a DataFrame with category labels and aggregate each group.\n# Hard: draw a labeled plot and describe one limitation.\n""",
        """import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nrng = np.random.default_rng(4)\nvalues = rng.normal(10, 2, 12)\ncategories = np.array(['A', 'B'] * 6)\nprint('mean/std:', values.mean().round(2), values.std().round(2))\nframe = pd.DataFrame({'category': categories, 'measurement': values})\nprint(frame.groupby('category')['measurement'].agg(['mean', 'count']))\nframe.boxplot(column='measurement', by='category')\nplt.title('Mesures simulées par groupe'); plt.suptitle(''); plt.ylabel('Unités arbitraires'); plt.show()\n# The sample is small and simulated; a difference is not evidence about real groups.\n""",
    ),
    (
        "02_gradients",
        "intermediate",
        "Gradient descent",
        "Calculez un produit scalaire ; estimez une dérivée par différence finie ; entraînez une droite sur y=2x-1 ; comparez convergence pour deux learning rates.",
        "02_gradients_solutions",
        """import numpy as np\nx = np.array([1.0, 2.0, 3.0])\ny = 2 * x - 1\n# Easy: compute the dot product of x and a vector of weights.\n# Medium: estimate d/dw MSE(w, b) with a centered finite difference.\n# Hard: compare two gradient-descent learning rates and report final loss.\n""",
        """import numpy as np\nx = np.array([1.0, 2.0, 3.0]); y = 2 * x - 1\nprint('dot product:', x @ np.array([0.2, 0.3, 0.5]))\ndef loss(w, b): return np.mean((w * x + b - y) ** 2)\nepsilon = 1e-5; w, b = 0.4, 0.2\nfinite_difference = (loss(w + epsilon, b) - loss(w - epsilon, b)) / (2 * epsilon)\nanalytic = 2 * np.mean((w * x + b - y) * x)\nprint('finite difference / analytic:', finite_difference, analytic)\nfor rate in [0.01, 0.15]:\n    w, b = 0.0, 0.0\n    for _ in range(120):\n        error = w * x + b - y\n        w -= rate * 2 * np.mean(error * x); b -= rate * 2 * np.mean(error)\n    print(rate, w, b, loss(w, b))\n""",
    ),
    (
        "03_preprocessing",
        "intermediate",
        "Preprocessing sans fuite",
        "Construisez un split stratifié ; introduisez des valeurs manquantes ; ajustez imputer et scaler dans une pipeline ; vérifiez qu’une catégorie inconnue est tolérée ; comparez à l’imputation globale fautive.",
        "03_preprocessing_solutions",
        """import pandas as pd\nfrom sklearn.datasets import load_iris\nfrom sklearn.model_selection import train_test_split\n# Easy: inspect columns, class counts and missing-value counts.\n# Medium: add a categorical column and build an imputer/encoder/scaler pipeline.\n# Hard: show why a statistic computed before splitting leaks test information.\n""",
        """import numpy as np\nimport pandas as pd\nfrom sklearn.datasets import load_iris\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\nfrom sklearn.linear_model import LogisticRegression\ndata = load_iris(as_frame=True); X = data.data.copy(); y = data.target\nX['size_group'] = np.where(X['sepal length (cm)'] > 5.8, 'large', 'small')\nX.loc[X.index[::17], 'sepal width (cm)'] = np.nan\nprint(X.dtypes); print(y.value_counts(normalize=True).round(2))\nx_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=3)\nnumeric = list(data.feature_names); categorical = ['size_group']\nprep = ColumnTransformer([('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric), ('cat', make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore')), categorical)])\nmodel = make_pipeline(prep, LogisticRegression(max_iter=1000)).fit(x_train, y_train)\nprint('test accuracy:', model.score(x_test, y_test))\nunknown = x_test.iloc[:1].copy(); unknown['size_group'] = 'unseen'\nprint('unknown category accepted:', model.predict(unknown))\ndemo_train = pd.Series([1., 2., 4., 5.]); demo_test = pd.Series([100., 120.])\nprint('train median:', demo_train.median(), '| illegal global median:', pd.concat([demo_train, demo_test]).median())\n""",
    ),
    (
        "04_models",
        "intermediate",
        "Régression et baseline",
        "Écrivez une fonction de MSE ; entraînez une droite avec gradient descent ; comparez à LinearRegression ; utilisez un baseline DummyRegressor et expliquez pourquoi un split test n’est pas du tuning.",
        "04_models_solutions",
        """import numpy as np\n# Easy: define MSE for arrays of targets and predictions.\n# Medium: fit a line with gradient descent and compare its coefficients to LinearRegression.\n# Hard: compare against DummyRegressor on a held-out split and inspect residuals.\n""",
        """import numpy as np\nfrom sklearn.dummy import DummyRegressor\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import mean_absolute_error\nrng = np.random.default_rng(17); x = np.linspace(0, 5, 120); y = 2.4 * x + 1 + rng.normal(0, 0.5, len(x))\ndef mse(target, prediction): return np.mean((target - prediction) ** 2)\nw, b = 0., 0.\nfor _ in range(300):\n    error = w * x + b - y\n    w -= 0.01 * 2 * np.mean(error * x); b -= 0.01 * 2 * np.mean(error)\nreference = LinearRegression().fit(x[:, None], y)\nprint('scratch:', w, b, '| library:', reference.coef_[0], reference.intercept_)\nx_train, x_test, y_train, y_test = train_test_split(x[:, None], y, random_state=9)\nfor estimator in [DummyRegressor(strategy='mean'), LinearRegression()]:\n    estimator.fit(x_train, y_train); pred = estimator.predict(x_test)\n    print(type(estimator).__name__, 'MAE:', round(mean_absolute_error(y_test, pred), 3), 'MSE:', round(mse(y_test, pred), 3))\n""",
    ),
    (
        "05_evaluation",
        "intermediate",
        "Métriques de classification",
        "Calculer precision et recall depuis TP/FP/FN ; choisir une métrique selon le coût d’erreur ; obtenir des probabilités ; utiliser stratified cross-validation ; examiner des erreurs.",
        "05_evaluation_solutions",
        """import numpy as np\n# Easy: compute precision, recall and F1 from TP=18, FP=7, FN=5.\n# Medium: evaluate two thresholds on held-out probability scores.\n# Hard: compare mean and spread from stratified cross-validation.\n""",
        """import numpy as np\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix\ntp, fp, fn = 18, 7, 5\nprecision = tp / (tp + fp); recall = tp / (tp + fn)\nprint('precision, recall, F1:', precision, recall, 2 * precision * recall / (precision + recall))\ndata = load_breast_cancer(as_frame=True); X, y = data.data, data.target\na, b, ya, yb = train_test_split(X, y, stratify=y, random_state=11)\nmodel = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=2000)).fit(a, ya)\nprob = model.predict_proba(b)[:, 1]\nfor threshold in [0.3, 0.5, 0.7]:\n    pred = (prob >= threshold).astype(int)\n    print(threshold, precision_score(yb, pred), recall_score(yb, pred), f1_score(yb, pred), confusion_matrix(yb, pred).tolist())\ncv = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)\nscores = cross_val_score(model, X, y, cv=cv, scoring='average_precision')\nprint('AP mean/std:', scores.mean().round(3), scores.std().round(3))\n""",
    ),
]

for slug, level, title, prompt, solution_slug, student_code, solution_code in exercise_sets:
    directory = "beginner" if level == "beginner" else "intermediate"
    write_notebook(
        f"exercises/{directory}/{slug}.ipynb",
        [
            markdown(f"# Exercice — {title}\n\nNiveau : {level}. Faites les tâches sans consulter la correction. Les cellules restent exécutables avant que vous les complétiez ; remplacez les commentaires TODO par votre code."),
            markdown(f"## Énoncé\n\n{prompt}\n\nNotez votre réponse et interprétation en Markdown après chaque résultat."),
            code(student_code),
            markdown("## Vérification personnelle\n\nPouvez-vous expliquer les formes des tableaux, la comparaison choisie et une limite de vos résultats ?"),
        ],
    )
    write_notebook(
        f"exercises/{directory}/{solution_slug}.ipynb",
        [
            markdown(f"# Correction — {title}\n\nCette correction montre un raisonnement possible. Les chiffres peuvent varier si vous modifiez les données ou la graine."),
            code(solution_code),
            markdown("## Interprétation\n\nReliez les nombres affichés à la question. Une autre solution peut convenir si elle respecte les contraintes de split, de métrique et de reproductibilité."),
        ],
    )


write_notebook(
    "modules/06-neural-networks/01_mlp_numpy.ipynb",
    [
        markdown("# Un petit réseau de neurones avec NumPy\n\n**Objectifs.** Calculer une forward pass, relier les dimensions, calculer une loss binaire et implémenter la rétropropagation avec la règle de chaîne. Durée : 90 min.\n\nLe XOR est un petit problème synthétique non linéaire : une droite ne sépare pas ses classes correctement. Le réseau suivant est une démonstration pédagogique, pas une architecture à réutiliser en production."),
        markdown("## 1. Formes et calcul avant\n\nPour `X (n, 2)`, `W1 (2, h)`, `b1 (h,)`, on a `Z1 = X @ W1 + b1` de forme `(n,h)`. Après tanh, `A1` conserve la forme. `W2 (h,1)` donne un logit `(n,1)`, converti en probabilité par sigmoid."),
        code("import numpy as np\nimport matplotlib.pyplot as plt\n\nX = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)\ny = np.array([[0], [1], [1], [0]], dtype=float)\nrng = np.random.default_rng(3)\nhidden_units = 4\nW1 = rng.normal(0, 0.6, size=(2, hidden_units)); b1 = np.zeros((1, hidden_units))\nW2 = rng.normal(0, 0.6, size=(hidden_units, 1)); b2 = np.zeros((1, 1))\nprint('X:', X.shape, 'W1:', W1.shape, 'W2:', W2.shape)"),
        markdown("## 2. Forward pass et loss\n\nLa sigmoid transforme un logit en nombre de 0 à 1. La binary cross-entropy pénalise les probabilités incompatibles avec le label. On clippe uniquement dans le logarithme pour éviter `log(0)` en virgule flottante."),
        code("def sigmoid(value):\n    return 1 / (1 + np.exp(-value))\n\ndef forward(features, w1, b1, w2, b2):\n    z1 = features @ w1 + b1\n    a1 = np.tanh(z1)\n    logits = a1 @ w2 + b2\n    probabilities = sigmoid(logits)\n    return z1, a1, logits, probabilities\n\ndef binary_cross_entropy(target, probability):\n    clipped = np.clip(probability, 1e-8, 1 - 1e-8)\n    return -np.mean(target * np.log(clipped) + (1 - target) * np.log(1 - clipped))\n\n_, hidden, logits, probability = forward(X, W1, b1, W2, b2)\nprint('hidden/output:', hidden.shape, probability.shape)\nprint('initial BCE:', round(float(binary_cross_entropy(y, probability)), 4))"),
        markdown("## 3. Rétropropagation\n\nAvec sigmoid + cross-entropy, le gradient du logit vaut `(probability - target) / n`. Les gradients de `W2,b2` se calculent ensuite. La règle de chaîne transmet le signal à la couche précédente ; pour tanh, `d tanh(z)/dz = 1 - tanh(z)^2`.\n\nOn applique les mises à jour avec gradient descent batch."),
        code("learning_rate = 0.7\nloss_history = []\nfor epoch in range(5_000):\n    z1, a1, _, p = forward(X, W1, b1, W2, b2)\n    loss_history.append(float(binary_cross_entropy(y, p)))\n    delta2 = (p - y) / len(X)\n    grad_w2 = a1.T @ delta2\n    grad_b2 = delta2.sum(axis=0, keepdims=True)\n    delta1 = (delta2 @ W2.T) * (1 - a1 ** 2)\n    grad_w1 = X.T @ delta1\n    grad_b1 = delta1.sum(axis=0, keepdims=True)\n    W2 -= learning_rate * grad_w2; b2 -= learning_rate * grad_b2\n    W1 -= learning_rate * grad_w1; b1 -= learning_rate * grad_b1\nprint('final BCE:', round(loss_history[-1], 4))\nprint('probabilities:', forward(X, W1, b1, W2, b2)[-1].round(3).ravel())"),
        markdown("## 4. Visualiser et interpréter\n\nLa perte doit décroître pour ces paramètres et ce problème. Le résultat dépend de l’initialisation ; entraînez plusieurs graines pour mesurer cette variabilité. XOR n’est pas un benchmark représentatif."),
        code("plt.plot(loss_history)\nplt.xlabel('Epoch'); plt.ylabel('Binary cross-entropy')\nplt.title('Loss du MLP NumPy'); plt.grid(alpha=0.2); plt.show()\nprint('prédictions binaires:', (forward(X, W1, b1, W2, b2)[-1] >= 0.5).astype(int).ravel())"),
        markdown("## 5. Ce que font les frameworks\n\nAutograd construit les dérivées à partir des opérations de `forward`. Un framework gère aussi calcul GPU, batchs, initialisations, optimizers et états ; la dérivée sous-jacente reste la règle de chaîne. La boucle explicite de ce notebook est gardée pour comprendre ce que l’outil automatise.\n\n**Limites :** paramètres non convexes, sensibilité aux graines et à l’échelle, loss qui peut diverger si le pas est trop grand. Dropout et batch normalization sont des notions à étudier après la boucle simple."),
        markdown("## Exercice\n\nAjoutez un point bruité, essayez 2 neurones cachés puis 8 et comparez 5 graines. Rapportez moyenne de la loss finale et variance. Une seule réussite ne suffit pas à conclure sur la robustesse."),
    ],
)


write_notebook(
    "modules/07-pytorch/01_training_loop.ipynb",
    [
        markdown("# Une boucle PyTorch lisible\n\n**Objectifs.** Vérifier formes et device, obtenir un gradient via autograd, organiser `Dataset`/`DataLoader`, entraîner puis évaluer explicitement. Durée : 60–90 min. Dépendance optionnelle : `pip install -e '.[dl]'`. L’exemple fonctionne sur CPU."),
        markdown("## 1. Tensors, opérations et autograd\n\nUn tensor porte valeurs, forme, dtype et device. Ici `x` est un scalaire. `loss.backward()` applique la règle de chaîne et remplit `weight.grad`; l’optimiseur fera ensuite la mise à jour."),
        code("import torch\n\ntorch.set_num_threads(1)\ntorch.manual_seed(5)\nweight = torch.tensor(2.0, requires_grad=True)\nx = torch.tensor(3.0)\nloss = (weight * x - 7) ** 2\nloss.backward()\nprint('loss:', loss.item(), '| dloss/dweight:', weight.grad.item())"),
        markdown("## 2. Données et dimensions\n\nLa cible suit une relation synthétique `y = 2.5*x - 0.7 + bruit`. Chaque mini-lot a `features: (batch,1)` et `targets: (batch,1)`. Le dernier lot peut être plus petit si le nombre d’exemples n’est pas divisible par batch size."),
        code("from torch import nn\nfrom torch.utils.data import DataLoader, TensorDataset\n\nrng = torch.Generator().manual_seed(8)\nfeatures = torch.linspace(-2, 2, 240).unsqueeze(1)\ntargets = 2.5 * features - 0.7 + 0.15 * torch.randn(features.shape, generator=rng)\ndataset = TensorDataset(features, targets)\nloader = DataLoader(dataset, batch_size=32, shuffle=True, generator=rng)\nprint('features:', features.shape, 'targets:', targets.shape, 'batches:', len(loader))"),
        markdown("## 3. Modèle, loss et optimizer\n\n`nn.Module` regroupe les paramètres appris. `MSELoss` convient à une cible numérique. SGD applique `parameter -= learning_rate * gradient`. Chaque epoch repasse sur les données d’entraînement."),
        code("model = nn.Linear(in_features=1, out_features=1)\nloss_fn = nn.MSELoss()\noptimizer = torch.optim.SGD(model.parameters(), lr=0.08)\n\nloss_history = []\nfor epoch in range(80):\n    model.train()\n    batch_losses = []\n    for batch_features, batch_targets in loader:\n        prediction = model(batch_features)\n        loss = loss_fn(prediction, batch_targets)\n        optimizer.zero_grad()\n        loss.backward()\n        optimizer.step()\n        batch_losses.append(loss.item())\n    loss_history.append(sum(batch_losses) / len(batch_losses))\nprint('weight/bias:', model.weight.item(), model.bias.item())\nprint('last train MSE:', round(loss_history[-1], 4))"),
        markdown("## 4. Évaluation et inference\n\n`eval()` met le module en mode évaluation. `inference_mode()` évite de construire un graphe de gradient inutile. Un dataset test devrait être distinct dans une vraie tâche ; ici les points de démonstration servent uniquement à voir les prédictions apprises."),
        code("model.eval()\nwith torch.inference_mode():\n    probe = torch.tensor([[-1.0], [0.0], [1.0]])\n    print('x / prediction:', list(zip(probe.ravel().tolist(), model(probe).ravel().tolist())))\n\nimport matplotlib.pyplot as plt\nplt.plot(loss_history); plt.xlabel('Epoch'); plt.ylabel('Train MSE')\nplt.title('Boucle PyTorch'); plt.grid(alpha=0.2); plt.show()"),
        markdown("## 5. Sauvegarde, CPU/GPU et reproductibilité\n\nEn pratique, enregistrer `model.state_dict()` et documenter les dimensions d’entrée, transformations, versions, seed et split. Pour GPU, sélectionner `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`, puis transférer modèle et lots. Les calculs GPU et certains noyaux peuvent varier entre machines."),
        code("import io\n\nbuffer = io.BytesIO()\ntorch.save(model.state_dict(), buffer)\nbuffer.seek(0)\nrestored = nn.Linear(1, 1)\nrestored.load_state_dict(torch.load(buffer, weights_only=True))\nprint('restored prediction:', restored(torch.tensor([[0.0]])).item())"),
        markdown("## À retenir et exercice\n\nLa séquence essentielle est mode train → forward → loss → zero_grad → backward → optimizer step. En validation : mode eval + inference_mode. Pour aller plus loin, séparez train/validation, ajoutez un DataLoader de validation, comparez Adam à SGD et tracez les deux courbes. Ce mini-problème ne démontre pas une performance de généralisation."),
    ],
)


write_notebook(
    "modules/08-computer-vision/01_cnn_digits.ipynb",
    [
        markdown("# Premier CNN sur des chiffres 8×8\n\n**Objectifs.** Lire les dimensions image/canal, calculer une taille de convolution, entraîner un petit CNN et inspecter les erreurs. Durée : 90 min. Dépendances optionnelles : `.[dl]`. Données : digits intégrés à scikit-learn, pas de téléchargement."),
        markdown("## 1. Représentation et séparation\n\nLes images niveaux de gris sont de forme `(n, 8, 8)` ; PyTorch Conv2d attend `(batch, channels, height, width)`, donc `(n,1,8,8)`. Normaliser les pixels par 16, valeur maximale de ce dataset."),
        code("import numpy as np\nimport torch\nfrom torch import nn\nfrom torch.utils.data import DataLoader, TensorDataset\nfrom sklearn.datasets import load_digits\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\n\ntorch.set_num_threads(1)\ntorch.manual_seed(10)\ndigits = load_digits()\nX = torch.tensor(digits.images[:, None, :, :] / 16.0, dtype=torch.float32)\ny = torch.tensor(digits.target, dtype=torch.long)\nx_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=10)\ntrain_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)\nprint('images:', tuple(X.shape), '| labels:', tuple(y.shape))\npixel_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))\npixel_model.fit(x_train.numpy().reshape(len(x_train), -1), y_train.numpy())\npixel_accuracy = pixel_model.score(x_test.numpy().reshape(len(x_test), -1), y_test.numpy())\nprint('pixel logistic-regression baseline accuracy:', round(pixel_accuracy, 3))"),
        markdown("## 2. Dimensions de la convolution\n\nAvec `kernel=3`, `padding=1`, `stride=1`, la formule retourne `floor((8+2-3)/1)+1=8`. Une couche de 8 filtres a la forme de sortie `(batch,8,8,8)`. Pooling 2×2 réduit chaque dimension spatiale par deux."),
        code("model = nn.Sequential(\n    nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n    nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n    nn.Flatten(), nn.Linear(16 * 2 * 2, 10)\n)\nprint('logits shape:', tuple(model(x_train[:5]).shape))"),
        markdown("## 3. Entraîner et évaluer\n\n`CrossEntropyLoss` attend des logits bruts de forme `(batch, 10)` et des labels entiers. Softmax est incorporé à la loss ; ne pas appliquer une softmax avant celle-ci."),
        code("optimizer = torch.optim.Adam(model.parameters(), lr=0.01)\nloss_fn = nn.CrossEntropyLoss()\nfor epoch in range(6):\n    model.train(); total_loss = 0.0\n    for images, labels in train_loader:\n        optimizer.zero_grad()\n        logits = model(images)\n        loss = loss_fn(logits, labels)\n        loss.backward(); optimizer.step()\n        total_loss += loss.item() * len(labels)\n    print(f'epoch {epoch + 1}: mean train loss={total_loss / len(train_loader.dataset):.3f}')\nmodel.eval()\nwith torch.inference_mode():\n    test_logits = model(x_test)\n    test_prediction = test_logits.argmax(dim=1)\n    test_accuracy = (test_prediction == y_test).float().mean().item()\nprint('held-out accuracy:', round(test_accuracy, 3))"),
        markdown("## 4. Inspecter des erreurs\n\nUne moyenne ne montre pas quelles classes se confondent. Afficher quelques images mal classées aide à vérifier les entrées et à formuler des hypothèses."),
        code("import matplotlib.pyplot as plt\nfrom sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix\n\nerrors = torch.where(test_prediction != y_test)[0][:6]\nfig, axes = plt.subplots(1, max(1, len(errors)), figsize=(12, 2.5))\nfor ax, index in zip(np.atleast_1d(axes), errors):\n    ax.imshow(x_test[index, 0], cmap='gray', vmin=0, vmax=1)\n    ax.set_title(f'true={y_test[index].item()}, pred={test_prediction[index].item()}'); ax.axis('off')\nplt.tight_layout(); plt.show()\nConfusionMatrixDisplay(confusion_matrix(y_test.numpy(), test_prediction.numpy())).plot(cmap='Blues')\nplt.show()"),
        markdown("## 5. Interprétation et limites\n\nLe dataset est petit, monochrome, centré et basse résolution ; ce score n’est pas transférable à des photos du monde réel. Pour continuer : comparer à une régression logistique sur pixels, changer une seule augmentation et vérifier quelles classes gagnent/perdent. Transfer learning, Grad-CAM et segmentation ne sont pas couverts par ce notebook d’introduction."),
    ],
)


write_notebook(
    "modules/09-nlp/01_sentiment_tfidf.ipynb",
    [
        markdown("# Classer un sentiment avec TF-IDF\n\n**Objectifs.** Décrire une tâche texte, comparer un vocabulaire aux tokens, ajuster TF-IDF dans une pipeline train-only et inspecter les erreurs. Durée : 75 min. Le micro-corpus est écrit pour illustrer le protocole ; il ne représente pas une population ni un usage client réel."),
        markdown("## 1. Définir le problème\n\nChaque ligne est un texte court et un label binaire positif/négatif. Le modèle ne comprend pas l’ironie ni le contexte ; nous voulons d’abord tester si le pipeline fonctionne. Les doublons et formulations artificielles limitent la portée du résultat."),
        code("import pandas as pd\nfrom sklearn.dummy import DummyClassifier\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import classification_report, confusion_matrix\n\ntexts = [\n 'excellent service and kind staff', 'the meal was fresh and delicious', 'fast delivery and great quality', 'friendly team solved my issue',\n 'wonderful experience and helpful support', 'the product works perfectly', 'clear instructions and easy setup', 'I am happy with this purchase',\n 'terrible service and rude staff', 'the meal was cold and tasteless', 'slow delivery and poor quality', 'unhelpful team ignored my issue',\n 'awful experience and frustrating support', 'the product broke immediately', 'confusing instructions and difficult setup', 'I regret this purchase',\n 'great quality but the box was damaged', 'helpful support but delivery was slow', 'fresh food and friendly staff', 'poor product and confusing manual',\n 'fast shipping and perfect item', 'rude response and long wait', 'kind team and useful solution', 'cold food arrived late'\n]\nlabels = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0]\nframe = pd.DataFrame({'text': texts, 'sentiment': labels})\nprint(frame.groupby('sentiment').size())\nframe.head()"),
        markdown("## 2. Split avant l’apprentissage du vocabulaire\n\n`TfidfVectorizer` découvre son vocabulaire et ses fréquences documentaires pendant `fit`. Il doit être dans la pipeline pour que les mots du test ne soient pas appris à l’avance."),
        code("text_train, text_test, y_train, y_test = train_test_split(\n    frame['text'], frame['sentiment'], test_size=0.25, stratify=frame['sentiment'], random_state=12\n)\ntext_baseline = DummyClassifier(strategy='most_frequent').fit(text_train.to_frame(), y_train)\ntext_model = make_pipeline(\n    TfidfVectorizer(ngram_range=(1, 2), min_df=1),\n    LogisticRegression(C=1.0, max_iter=1_000)\n)\ntext_model.fit(text_train, y_train)\nprediction = text_model.predict(text_test)\nprint('majority baseline accuracy:', text_baseline.score(text_test.to_frame(), y_test))\nprint(classification_report(y_test, prediction, zero_division=0))\nprint('confusion matrix:', confusion_matrix(y_test, prediction).tolist())"),
        markdown("## 3. Inspecter les coefficients et erreurs\n\nLe coefficient mesure une association dans ce corpus et ce modèle régularisé ; ce n’est pas une explication causale. Des n-grams familiers peuvent masquer que le test est petit."),
        code("vectorizer = text_model.named_steps['tfidfvectorizer']\nclassifier = text_model.named_steps['logisticregression']\nweights = pd.Series(classifier.coef_[0], index=vectorizer.get_feature_names_out())\nprint('termes positifs associés :\\n', weights.nlargest(8).round(2))\nprint('termes négatifs associés :\\n', weights.nsmallest(8).round(2))\nerror_review = pd.DataFrame({'text': text_test, 'actual': y_test, 'predicted': prediction})\nerror_review[error_review['actual'] != error_review['predicted']]"),
        markdown("## 4. Expérience et limites\n\nComparez unigrammes à unigrammes+bigrammes en validation croisée, en plaçant chaque `TfidfVectorizer` dans une pipeline. Ne choisissez pas à partir du test. Tokenization avancée, embeddings, LSTM, self-attention et Transformers/Hugging Face sont les suites du module et nécessitent une leçon distincte."),
    ],
)


if __name__ == "__main__":
    print(f"Generated notebooks beneath {ROOT}.")
