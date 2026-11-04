"""Small, reproducible tabular ML experiments used by the project guides."""

from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from learning_hub.preprocessing import make_housing_data, make_tabular_preprocessor


def run_housing_experiment(random_state: int = 42) -> dict[str, dict[str, float]]:
    """Compare a mean predictor with a leakage-safe random forest."""
    features, target = make_housing_data(random_state=random_state)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=random_state
    )
    models = {
        "baseline_mean": Pipeline(
            [
                (
                    "preprocess",
                    make_tabular_preprocessor(
                        ["area_m2", "bedrooms", "age_years", "distance_to_center_km"],
                        ["district"],
                    ),
                ),
                ("model", DummyRegressor(strategy="mean")),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "preprocess",
                    make_tabular_preprocessor(
                        ["area_m2", "bedrooms", "age_years", "distance_to_center_km"], ["district"]
                    ),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=60,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }
    scores = {}
    for name, pipeline in models.items():
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        scores[name] = {
            "mae": float(mean_absolute_error(y_test, prediction)),
            "rmse": float(mean_squared_error(y_test, prediction) ** 0.5),
            "r2": float(r2_score(y_test, prediction)),
        }
    return scores
