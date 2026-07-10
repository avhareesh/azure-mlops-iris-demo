import json
import logging
import os
from pathlib import Path

import joblib
import pandas as pd


FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def init():
    global model

    logging.info("Starting init()...")

    model_dir = os.getenv("AZUREML_MODEL_DIR")
    logging.info(f"AZUREML_MODEL_DIR = {model_dir}")

    if model_dir is None:
        raise RuntimeError("AZUREML_MODEL_DIR environment variable is not set.")

    model_dir_path = Path(model_dir)

    logging.info("Listing model directory contents:")
    for path in model_dir_path.rglob("*"):
        logging.info(str(path))

    model_path = model_dir_path / "model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = joblib.load(model_path)

    logging.info("Model loaded successfully.")


def run(raw_data):
    try:
        logging.info("Received request.")

        payload = json.loads(raw_data)

        input_df = pd.DataFrame(payload["data"])
        input_df = input_df[FEATURE_COLUMNS]

        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df).max(axis=1)

        results = []

        for prediction, confidence in zip(predictions, probabilities):
            results.append(
                {
                    "prediction": int(prediction),
                    "prediction_label": CLASS_NAMES[int(prediction)],
                    "confidence": float(confidence),
                }
            )

        return {
            "predictions": results
        }

    except Exception as ex:
        logging.exception("Scoring failed.")
        return {
            "error": str(ex)
        }