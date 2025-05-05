from typing import Any, Dict, List, Optional, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from typing_extensions import Self

from model import BaseModel

JSON = Union[str, int, float, bool, None, Dict[str, "JSON"], List["JSON"]]


class TabRegressor(BaseModel):

    def __init__(
        self, covariables: Optional[List[str]] = None, target: Optional[str] = None
    ):
        self.covariables = covariables
        self.target = target

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> Self:
        if self.covariables is None:
            self.covariables = X.columns.tolist()
        if self.target is None:
            self.target = y.columns.tolist()[0]

        return self._fit(X[self.covariables], y[self.target])

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame) -> Self:
        raise NotImplementedError()

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        yhat = self._predict(X[self.covariables])
        pred = pd.DataFrame(
            {
                f"{self.target}_hat": yhat,
            }
        )
        return pred

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError()

    def get_tab_regressor_metadata(self) -> JSON:
        return dict(covariables=self.covariables, target=self.target)

    def save(self, filepath: str) -> str:
        raise NotImplementedError()

    @classmethod
    def load(cls, filepath: str) -> "TabRegressor":
        raise NotImplementedError()


class SkLearnTabRegressor(TabRegressor):
    def __init__(
        self,
        base_regressor: Optional[Any] = lgb.LGBMRegressor(verbose=-1),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_regressor = base_regressor

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame) -> Self:
        self.base_regressor.fit(X, y)
        return self

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.squeeze(self.base_regressor.predict(X))

    def save(self, filepath: str) -> str:
        metadata = self.get_tab_regressor_metadata()
        metadata.update(
            dict(_class=self.__class__.__name__, _module=self.__class__.__module__)
        )
        print(metadata)
        joblib.dump(
            {
                "base_regressor": self.base_regressor,
                "metadata": metadata,
            },
            filepath,
        )
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "SkLearnTabRegressor":
        model_data = joblib.load(filepath)
        kwargs = dict(
            base_regressor=model_data["base_regressor"],
            covariables=model_data["metadata"]["covariables"],
            target=model_data["metadata"]["target"],
        )
        return cls(**kwargs)
