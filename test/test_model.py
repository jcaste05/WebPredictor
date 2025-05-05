import logging
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

from model.tab_regressor import SkLearnTabRegressor
from model.utils import load_model

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "tab_regressor_model",
    [
        SkLearnTabRegressor(),
        SkLearnTabRegressor(covariables=["x0"], target="target"),
    ],
)
def test_tab_regressor(tab_regressor_model):
    logger.info(
        "Testing TabRegressor with base regressor: %s",
        tab_regressor_model.base_regressor,
    )
    # Testing data
    X_np, y_np = make_regression(
        n_samples=1000, n_features=5, noise=0.1, random_state=42
    )
    X = pd.DataFrame(X_np, columns=[f"x{i}" for i in range(X_np.shape[1])])
    y = pd.DataFrame(y_np, columns=["target"])

    # Training and predicting
    tab_regressor_model.fit(X, y)
    preds = tab_regressor_model.predict(X)

    # Comparing with naive regressor model
    model_mse = mean_squared_error(y, preds)
    mean_prediction = np.full_like(y.values, y.mean().values)
    baseline_mse = mean_squared_error(y, mean_prediction)
    logger.info(
        "MSE of model: %s | MSE of naive regressor: %s", model_mse, baseline_mse
    )

    assert (
        model_mse < baseline_mse
    ), f"Model {tab_regressor_model.__class__.__name__} failed to improve over naive regressor."

    # Testing save and load
    os.makedirs("./tmp", exist_ok=True)
    tmp_path = "./tmp/model.joblib"
    tab_regressor_model.save(tmp_path)
    loaded_model = load_model(tmp_path)
    # Verify predictions still being the same
    loaded_preds = loaded_model.predict(X)
    assert loaded_preds.equals(
        preds
    ), "Predictions are not the same after saving and loading."
    os.remove(tmp_path)
