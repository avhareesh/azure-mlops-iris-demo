import json
import logging
import os
from pathlib import Path

import joblib


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

    if not model_dir:
        raise RuntimeError("AZUREML_MODEL_DIR environment variable is not set.")

    model_root = Path(model_dir)

    logging.info("Listing files under AZUREML_MODEL_DIR:")
    for path in model_root.rglob("*"):
        logging.info(str(path))

    model_files = list(model_root.rglob("model.pkl"))

    if not model_files:
        raise FileNotFoundError(
            f"model.pkl was not found anywhere under AZUREML_MODEL_DIR: {model_root}"
        )

    model_path = model_files[0]
    logging.info(f"Loading model from: {model_path}")

    model = joblib.load(model_path)

    logging.info("Model loaded successfully.")


def run(raw_data):
    try:
        logging.info("Received request.")

        payload = json.loads(raw_data)

        rows = payload["data"]

        input_rows = []
        for row in rows:
            input_rows.append(
                [
                    float(row["sepal_length"]),
                    float(row["sepal_width"]),
                    float(row["petal_length"]),
                    float(row["petal_width"]),
                ]
            )

        predictions = model.predict(input_rows)
        probabilities = model.predict_proba(input_rows).max(axis=1)

        results = []

        for prediction, confidence in zip(predictions, probabilities):
            prediction_int = int(prediction)

            results.append(
                {
                    "prediction": prediction_int,
                    "prediction_label": CLASS_NAMES[prediction_int],
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