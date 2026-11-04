"""Leakage-safe tabular preprocessing examples."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_tabular_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    """Create fit-on-training preprocessing for numeric and categorical columns."""
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ],
        remainder="drop",
    )


def make_housing_data(n_samples: int = 600, random_state: int = 42) -> tuple[pd.DataFrame, NDArray]:
    """Generate a deterministic, explicitly synthetic housing-style dataset."""
    if n_samples < 20:
        raise ValueError("n_samples must be at least 20")
    rng = np.random.default_rng(random_state)
    area = rng.uniform(25, 220, size=n_samples)
    bedrooms = np.clip(np.rint(area / 38 + rng.normal(0, 0.8, n_samples)), 0, 6)
    age = rng.uniform(0, 90, size=n_samples)
    distance = rng.uniform(0.2, 35, size=n_samples)
    district = rng.choice(["central", "north", "south"], size=n_samples, p=[0.35, 0.35, 0.30])
    district_effect = (
        pd.Series(district).map({"central": 75_000, "north": 22_000, "south": 0}).to_numpy()
    )
    noise = rng.normal(0, 18_000, size=n_samples)
    price = 1_450 * area + 9_000 * bedrooms - 500 * age - 1_050 * distance + district_effect + noise
    frame = pd.DataFrame(
        {
            "area_m2": area,
            "bedrooms": bedrooms,
            "age_years": age,
            "distance_to_center_km": distance,
            "district": district,
        }
    )
    # Missingness is injected after defining the target to exercise imputation.
    missing_rows = rng.choice(n_samples, size=max(1, n_samples // 40), replace=False)
    frame.loc[missing_rows, "age_years"] = np.nan
    return frame, price


def make_churn_data(n_samples: int = 1_000, random_state: int = 42) -> tuple[pd.DataFrame, NDArray]:
    """Generate synthetic subscription data for demonstrating binary classification."""
    if n_samples < 40:
        raise ValueError("n_samples must be at least 40")
    rng = np.random.default_rng(random_state)
    tenure = rng.integers(1, 73, size=n_samples)
    monthly_charge = rng.normal(72, 24, size=n_samples).clip(15, 180)
    support_calls = rng.poisson(1.2, size=n_samples).clip(0, 8)
    contract = rng.choice(["monthly", "annual", "two_year"], size=n_samples, p=[0.55, 0.3, 0.15])
    logit = (
        -1.0
        - 0.035 * tenure
        + 0.022 * (monthly_charge - 70)
        + 0.42 * support_calls
        + np.select([contract == "monthly", contract == "annual"], [0.8, -0.35], default=-0.8)
    )
    probability = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, probability)
    frame = pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charge": monthly_charge,
            "support_calls_90d": support_calls,
            "contract": contract,
        }
    )
    return frame, churned
