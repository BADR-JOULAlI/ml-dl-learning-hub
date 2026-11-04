"""A tiny linear regression implementation for teaching gradient descent."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class LinearRegressorGD:
    """Fit one-feature linear regression by minimizing mean squared error."""

    def __init__(self, learning_rate: float = 0.05, epochs: int = 1_000) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_: float | None = None
        self.bias_: float | None = None
        self.loss_history_: list[float] = []

    def fit(self, x: ArrayLike, y: ArrayLike) -> LinearRegressorGD:
        """Learn a slope and intercept from one-dimensional observations."""
        features = np.asarray(x, dtype=float).reshape(-1)
        targets = np.asarray(y, dtype=float).reshape(-1)
        if features.size == 0 or features.shape != targets.shape:
            raise ValueError("x and y must have the same non-zero length")
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError("x and y must contain only finite values")

        weight = 0.0
        bias = 0.0
        count = features.size
        self.loss_history_ = []
        for _ in range(self.epochs):
            predictions = weight * features + bias
            errors = predictions - targets
            self.loss_history_.append(float(np.mean(errors**2)))
            weight -= self.learning_rate * float(2 * np.mean(errors * features))
            bias -= self.learning_rate * float(2 * np.sum(errors) / count)

        self.weight_ = weight
        self.bias_ = bias
        return self

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        """Predict target values after fit."""
        if self.weight_ is None or self.bias_ is None:
            raise RuntimeError("Call fit before predict")
        features = np.asarray(x, dtype=float)
        return self.weight_ * features + self.bias_
