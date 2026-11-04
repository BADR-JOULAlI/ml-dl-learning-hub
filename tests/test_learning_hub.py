import numpy as np

from learning_hub.experiments import run_housing_experiment
from learning_hub.linear_regression import LinearRegressorGD
from learning_hub.preprocessing import make_churn_data, make_housing_data


def test_linear_regression_recovers_simple_line() -> None:
    x = np.linspace(-2, 2, 100)
    y = 3 * x + 2
    model = LinearRegressorGD(learning_rate=0.05, epochs=1_000).fit(x, y)
    assert abs(model.weight_ - 3) < 1e-5
    assert abs(model.bias_ - 2) < 1e-5
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_synthetic_data_is_reproducible_and_has_expected_shape() -> None:
    features_a, target_a = make_housing_data(n_samples=80, random_state=7)
    features_b, target_b = make_housing_data(n_samples=80, random_state=7)
    assert features_a.equals(features_b)
    assert np.array_equal(target_a, target_b)
    assert features_a.shape == (80, 5)
    assert features_a["age_years"].isna().any()


def test_churn_data_contains_no_target_column() -> None:
    features, target = make_churn_data(n_samples=100, random_state=5)
    assert "churned" not in features.columns
    assert set(np.unique(target)) == {0, 1}


def test_housing_experiment_beats_mean_baseline_on_rmse() -> None:
    scores = run_housing_experiment()
    assert scores["random_forest"]["rmse"] < scores["baseline_mean"]["rmse"]
