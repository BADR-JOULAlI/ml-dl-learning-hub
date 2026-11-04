"""Run the synthetic housing project from the repository root."""

from __future__ import annotations

import json

from learning_hub.experiments import run_housing_experiment

if __name__ == "__main__":
    print(json.dumps(run_housing_experiment(), indent=2))
