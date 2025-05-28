import logging
import os
import tempfile

import numpy as np
import pandas as pd
from nicegui import ui

from model.tab_regressor import SkLearnTabRegressor

logging.basicConfig(
    level=os.getenv("WEB_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def handle_upload(e, data, select_target):
    logger.info("Uploading file")
    with tempfile.NamedTemporaryFile(
        dir="./tmp", delete=True, suffix=".csv", mode="w+b"
    ) as tmp:
        tmp.write(e.content.read())
        tmp.seek(0)
        data["df"] = pd.read_csv(tmp)
    select_target.options = data["df"].columns.tolist()
    select_target.update()
    data["evaluated"] = False


def predict_df(data, target_column):
    logger.info("Predicting data frame")
    if target_column is None:
        logger.info("No target column selected")
        ui.notify("Please select a target column")
        return

    # Split data
    df = data["df"].copy()
    Xy_train = df[df[target_column].notna()]
    X_train = Xy_train.drop(columns=[target_column], inplace=False)
    y_train = Xy_train[[target_column]]
    X_test = (df[df[target_column].isna()]).drop(columns=[target_column], inplace=False)

    # Training model
    logger.info("Creating and training model")
    model = SkLearnTabRegressor()
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    mse_train = np.mean(
        np.square(pred_train[f"{target_column}_hat"].values - y_train[target_column].values)
    )
    sd_train = y_train[target_column].std()
    logger.info(f"MSE train: {mse_train}")
    logger.info(f"SD train: {sd_train}")

    # Predict
    pred_test = model.predict(X_test)
    df.loc[df[target_column].isna(), target_column] = pred_test[
        f"{target_column}_hat"
    ].values
    data["df"] = df
    data["evaluated"] = True
    ui.notify(f"Predicted {len(X_test)} rows")


def handle_download(data):
    logger.info("Downloading data frame")
    if data["df"] is None or not data["evaluated"]:
        logger.info("Data frame was not evaluated or uploaded before")
        ui.notify(
            "Please upload a file first, select a target column and press predict button"
        )
        return
    with tempfile.NamedTemporaryFile(
        dir="./tmp", delete=False, suffix=".csv", mode="w+b"
    ) as tmp:
        data["df"].to_csv(tmp, index=False)
        tmp.seek(0)
        ui.notify(f"Downloading predicted data frame")
        ui.download(tmp.name, f"predicted_data.csv")


@ui.page("/")
def main():
    data = dict(df=None, evaluated=False)
    with ui.column().classes("fixed-center"):
        ui.label("Upload your CSV file and select your target column:")
        ui.upload(on_upload=lambda e: handle_upload(e, data, select_target)).props(
            "accept=.csv"
        ).classes("max-w-full")
        with ui.row():
            select_target = ui.select(
                [], label="Select target column", clearable=True, with_input=True
            )
            ui.button("Predict", on_click=lambda: predict_df(data, select_target.value))
        ui.button("Download CSV predicted file", on_click=lambda: handle_download(data))


if __name__ in ["__main__", "__mp_main__"]:
    os.makedirs("./tmp", exist_ok=True)
    ui.run(title="WebPredictor")
