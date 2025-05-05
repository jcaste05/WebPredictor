import importlib

import joblib


def load_model(model_path: str):
    model_data = joblib.load(model_path)
    _module = model_data["metadata"]["_module"]
    _class = model_data["metadata"]["_class"]
    model_module = importlib.import_module(_module)
    model_class = getattr(model_module, _class)
    model = model_class.load(model_path)
    return model
