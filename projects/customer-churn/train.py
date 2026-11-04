"""Train and evaluate a reproducible synthetic churn baseline."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from learning_hub.preprocessing import make_churn_data, make_tabular_preprocessor


def main() -> None:
    features, target = make_churn_data(n_samples=1_000, random_state=31)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, stratify=target, random_state=31
    )
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    pipeline = Pipeline(
        [
            (
                "preprocess",
                make_tabular_preprocessor(
                    ["tenure_months", "monthly_charge", "support_calls_90d"], ["contract"]
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1_000, class_weight="balanced")),
        ]
    ).fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    prediction = (probabilities >= 0.5).astype(int)
    print("Majority baseline:", baseline.score(x_test, y_test))
    print("Average precision:", average_precision_score(y_test, probabilities))
    print("ROC-AUC:", roc_auc_score(y_test, probabilities))
    print("Confusion matrix [true rows, predicted columns]:\n", confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction, zero_division=0))
    errors = x_test.copy()
    errors["actual"] = y_test
    errors["prediction"] = prediction
    errors["probability"] = probabilities
    print("Misclassified examples:\n", errors.loc[errors.actual != errors.prediction].head(10))
    print("Feature-only schema:", list(features.columns))
    print("These synthetic scores describe only the chosen generator and split.")


if __name__ == "__main__":
    main()
