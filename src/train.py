import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-output", type=str, default="model")
    args = parser.parse_args()

    output_dir = Path(args.model_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    iris = load_iris(as_frame=True)

    X = iris.data.copy()
    X.columns = FEATURE_COLUMNS
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, output_dir / "model.pkl")

    metadata = {
        "model_type": "RandomForestClassifier",
        "framework": "scikit-learn",
        "accuracy": accuracy,
        "features": FEATURE_COLUMNS,
        "class_names": list(iris.target_names),
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to: {output_dir}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, target_names=iris.target_names))


if __name__ == "__main__":
    main()