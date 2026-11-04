"""Build complete, runnable tabular project notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def cell(kind: str, source: str, index: int) -> dict:
    result = {"cell_type": kind, "metadata": {}, "id": f"project-{index:03d}"}
    result["source"] = source.splitlines(keepends=True)
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def write(path: str, cells: list[tuple[str, str]]) -> None:
    notebook = {
        "cells": [cell(kind, source, index) for index, (kind, source) in enumerate(cells)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11+"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / path).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


write(
    "projects/house-price-prediction/analysis.ipynb",
    [
        ("markdown", "# Projet complet — Régression de prix synthétiques\n\nCe notebook suit question → EDA → split → baseline → modèle → évaluation → analyse d’erreurs. Les données sont simulées par une équation dans le générateur ; les résultats ne sont pas représentatifs d’un marché. Durée : 2–3 h."),
        ("markdown", "## 1. Générer et comprendre le tableau\n\nUne ligne représente un logement simulé. `price` est construit par le générateur et ne doit pas être interprété comme un prix réel."),
        ("code", "import pandas as pd\nimport matplotlib.pyplot as plt\nfrom learning_hub.preprocessing import make_housing_data\n\nfeatures, target = make_housing_data(n_samples=600, random_state=42)\ndata = features.assign(price=target)\nprint(data.shape); print(data.dtypes); print(data.isna().sum()); print(data.describe(include='all').round(2))\ndata.head()"),
        ("markdown", "## 2. EDA : distributions et relations\n\nCherchez des incohérences de formes et d’échelles. Un graphique décrit les données générées ; il ne prouve pas que ces relations existent dans une ville réelle."),
        ("code", "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\naxes[0].hist(data.price, bins=25, color='slateblue')\naxes[0].set(title='Prix synthétique', xlabel='Unités de prix', ylabel='Nombre')\naxes[1].scatter(data.area_m2, data.price, alpha=0.45)\naxes[1].set(title='Surface et prix simulés', xlabel='Surface (m²)', ylabel='Unités de prix')\nplt.tight_layout(); plt.show()"),
        ("markdown", "## 3. Séparer train/test avant le fit\n\nLe test final simule les observations non utilisées pour choisir les paramètres. Preprocessing et estimateur sont réunis dans une pipeline."),
        ("code", "from sklearn.dummy import DummyRegressor\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.pipeline import Pipeline\nfrom learning_hub.preprocessing import make_tabular_preprocessor\n\nnumeric = ['area_m2', 'bedrooms', 'age_years', 'distance_to_center_km']\ncategorical = ['district']\nx_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)\nmodels = {\n    'mean baseline': Pipeline([('prep', make_tabular_preprocessor(numeric, categorical)), ('model', DummyRegressor())]),\n    'random forest': Pipeline([('prep', make_tabular_preprocessor(numeric, categorical)), ('model', RandomForestRegressor(n_estimators=60, min_samples_leaf=2, random_state=42, n_jobs=1))]),\n}\nresults = {}\nfor name, model in models.items():\n    model.fit(x_train, y_train)\n    prediction = model.predict(x_test)\n    results[name] = prediction\n    print(name, {'MAE': round(mean_absolute_error(y_test, prediction), 1), 'RMSE': round(mean_squared_error(y_test, prediction) ** 0.5, 1), 'R2': round(r2_score(y_test, prediction), 3)})"),
        ("markdown", "## 4. Lire les erreurs du meilleur candidat\n\nLa forêt exploite la relation non linéaire créée par le générateur ; la comparaison répond d’abord à « ce code suit-il le signal défini ? ». MAE et RMSE sont en unités du prix simulé."),
        ("code", "best_prediction = results['random forest']\nresiduals = y_test - best_prediction\nfig, axes = plt.subplots(1, 2, figsize=(10, 4))\naxes[0].scatter(best_prediction, residuals, alpha=0.5); axes[0].axhline(0, color='black')\naxes[0].set(xlabel='Prédiction', ylabel='Résidu', title='Résidus sur test')\nerror_by_district = x_test[['district']].copy()\nerror_by_district['absolute_error'] = abs(residuals)\nerror_by_district.boxplot(column='absolute_error', by='district', ax=axes[1])\naxes[1].set(title='Erreur par district (synthétique)', ylabel='Erreur absolue'); plt.suptitle(''); plt.tight_layout(); plt.show()\nerror_by_district.groupby('district').absolute_error.agg(['median', 'mean', 'count']).round(1)"),
        ("markdown", "## 5. Conclusion et pistes\n\nLa forêt devrait battre le baseline moyen sur le split fixe car la cible a été fabriquée à partir des variables. Cela ne valide pas les hypothèses métier ni l’estimation de biens réels.\n\nPistes : validation croisée répétée, baseline linéaire, comparaison des résultats par district, intervalle d’incertitude, vraie source de données avec licence vérifiée, et analyse de dérive. N’ajustez pas les hyperparamètres à partir de ce test."),
    ],
)


write(
    "projects/customer-churn/analysis.ipynb",
    [
        ("markdown", "# Projet complet — Attrition synthétique\n\nQuestion : prédire un label binaire simulé à partir d’informations disponibles aujourd’hui. Ce corpus synthétique n’est pas une mesure du comportement de vrais abonnés. Durée : 2–3 h."),
        ("markdown", "## 1. Explorer les lignes et la cible\n\nLes labels sont tirés avec du bruit à partir d’ancienneté, tarif, appels et contrat. Il n’existe pas de colonne cible cachée dans `X`."),
        ("code", "import pandas as pd\nimport matplotlib.pyplot as plt\nfrom learning_hub.preprocessing import make_churn_data\n\nfeatures, target = make_churn_data(n_samples=1_000, random_state=31)\ndata = features.assign(churn=target)\nprint(data.head()); print(data.dtypes); print(data.isna().sum())\nprint('proportion positive:', round(data.churn.mean(), 3))\nprint(data.groupby('churn')[['tenure_months', 'monthly_charge', 'support_calls_90d']].mean().round(1))"),
        ("markdown", "## 2. Baseline et split stratifié\n\nLa classe majoritaire est un niveau de comparaison. Les données sont séparées avant que l’imputeur, scaler ou encodeur apprenne quoi que ce soit."),
        ("code", "from sklearn.dummy import DummyClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import (average_precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_auc_score)\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.pipeline import Pipeline\nfrom learning_hub.preprocessing import make_tabular_preprocessor\n\nx_train, x_test, y_train, y_test = train_test_split(features, target, stratify=target, test_size=0.25, random_state=31)\nbaseline = DummyClassifier(strategy='most_frequent').fit(x_train, y_train)\npipeline = Pipeline([('prep', make_tabular_preprocessor(['tenure_months','monthly_charge','support_calls_90d'], ['contract'])), ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))]).fit(x_train, y_train)\nprobability = pipeline.predict_proba(x_test)[:, 1]\nprediction = (probability >= 0.5).astype(int)\nprint('baseline accuracy:', round(baseline.score(x_test, y_test), 3))\nprint('Average Precision:', round(average_precision_score(y_test, probability), 3))\nprint('ROC-AUC:', round(roc_auc_score(y_test, probability), 3))\nprint(classification_report(y_test, prediction, zero_division=0))"),
        ("markdown", "## 3. Erreurs, seuil et conséquences\n\nLa matrice de confusion et la PR curve montrent les compromis du seuil. Un changement de seuil doit suivre une conséquence d’usage définie, et se choisir dans la validation, pas sur le test final."),
        ("code", "ConfusionMatrixDisplay(confusion_matrix(y_test, prediction), display_labels=['retained','churn']).plot(cmap='Blues', values_format='d'); plt.show()\nprecision, recall, thresholds = precision_recall_curve(y_test, probability)\nplt.plot(recall, precision); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Test PR curve (descriptive only)'); plt.show()\nerrors = x_test.copy(); errors['actual'] = y_test; errors['predicted'] = prediction; errors['probability'] = probability\nerrors.loc[errors.actual != errors.predicted].head(10)"),
        ("markdown", "## 4. Conclusion, risques et suite\n\nLa classification synthétique permet de vérifier les shapes, le split stratifié, le preprocessing, les scores et l’analyse d’erreurs. Les métriques décrivent uniquement ce générateur et ce split.\n\nAvant toute application réelle, définir consentement et minimisation des données, vérifier représentativité et biais, mesurer calibration, fixer une règle de décision avec les parties prenantes, puis prévoir feedback différé et monitoring. Ne pas agir sur des personnes avec ce modèle jouet."),
    ],
)


if __name__ == "__main__":
    print("Generated two end-to-end tabular project notebooks.")
