"""Train a reproducible synthetic churn demo and save a complete pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from learning_hub.preprocessing import make_churn_data, make_tabular_preprocessor


def train(output: Path, random_state: int = 42, log_mlflow: bool = False) -> dict[str, float]:
    """Fit a churn pipeline and optionally log parameters and metrics to MLflow."""
    X, y = make_churn_data(n_samples=1_500, random_state=random_state)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )
    pipeline = Pipeline(
        steps=[
            (
                "preprocess",
                make_tabular_preprocessor(
                    ["tenure_months", "monthly_charge", "support_calls_90d"], ["contract"]
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1_000, class_weight="balanced")),
        ]
    )
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "f1_at_0_5": float(f1_score(y_test, predictions)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output)

    if log_mlflow:
        import mlflow

        mlflow.set_experiment("synthetic-churn-demo")
        with mlflow.start_run():
            mlflow.log_params({"random_state": random_state, "train_rows": len(x_train)})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(output), artifact_path="model")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/churn_pipeline.joblib"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow", action="store_true", help="Log the local run to MLflow")
    arguments = parser.parse_args()
    print(train(arguments.output, arguments.seed, arguments.mlflow))


if __name__ == "__main__":
    main()
