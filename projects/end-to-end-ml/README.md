# Mini-projet MLOps — modèle d’attrition avec API locale

Ce projet génère un dataset synthétique, ajuste une pipeline scikit-learn, sérialise l’artefact, puis l’expose via FastAPI. Il n’appelle aucun service cloud. Les labels simulés ne représentent pas l’attrition réelle.

## Entraîner

```bash
python -m pip install -e ".[mlops]"
python projects/end-to-end-ml/train.py --output artifacts/churn_pipeline.joblib
```

En option, ajouter `--mlflow` pour ouvrir une expérience locale `mlruns/` ; aucune inscription ni clé API n’est demandée.

## Lancer l’API

```bash
python -m uvicorn serve:app --app-dir projects/end-to-end-ml --reload
```

Endpoints : `GET /health` et `POST /predict` avec `tenure_months`, `monthly_charge`, `support_calls_90d`, `contract`. La page interactive locale est `/docs`.

## Tests de bout en bout manuel

1. Entraîner et vérifier qu’un artefact est généré sous `artifacts/`.
2. Démarrer le serveur dans un terminal.
3. Envoyer deux requêtes de prédiction et vérifier que le résultat est une probabilité entre 0 et 1.
4. Redémarrer le serveur sans artefact ; l’API doit signaler qu’elle n’est pas prête plutôt que retourner une prédiction fictive.

## Docker

Une image Docker minimale est fournie. Depuis la racine :

```bash
docker build -f projects/end-to-end-ml/Dockerfile -t ml-hub-churn .
docker run --rm -p 8000:8000 -v "${PWD}/artifacts:/app/artifacts" ml-hub-churn
```

L’entraînement, MLflow, la sérialisation et le contrat API servent de démonstration pédagogique. Authentification, gestion de secrets, tests de charge, alertes et stratégie réelle de réentraînement ne sont pas fournis.
