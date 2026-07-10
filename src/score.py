import json
import logging
import os

import joblib
import pandas as pd

try:
    from azureml.ai.monitoring import Collector
except Exception:
    Collector = None


FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def init():
    global model
    global inputs_collector
    global outputs_collector

    model_dir = os.getenv("AZUREML_MODEL_DIR", "model")
    model_path = os.path.join(model_dir, "model.pkl")

    model = joblib.load(model_path)

    if Collector:
        inputs_collector = Collector(name="model_inputs")
        outputs_collector = Collector(name="model_outputs")
    else:
        inputs_collector = None
        outputs_collector = None

    logging.info("Model loaded successfully.")


def run(raw_data):
    try:
        payload = json.loads(raw_data)

        input_df = pd.DataFrame(payload["data"])
        input_df = input_df[FEATURE_COLUMNS]

        context = None
        if inputs_collector:
            context = inputs_collector.collect(input_df)

        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df).max(axis=1)

        output_df = pd.DataFrame(
            {
                "prediction": predictions,
                "prediction_label": [CLASS_NAMES[int(p)] for p in predictions],
                "confidence": probabilities,
            }
        )

        if outputs_collector:
            outputs_collector.collect(output_df, context)

        return {
            "predictions": output_df.to_dict(orient="records")
        }

    except Exception as ex:
        logging.exception("Scoring failed.")
        return {
            "error": str(ex)
        }