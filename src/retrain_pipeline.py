import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.monitor_drift import check_data_drift

MODEL_NAME = "IrisRandomForest"

def train_and_register_model():
"""Train the Iris model and register it with MLflow."""

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.2,
    random_state=42,
)

mlflow.set_experiment("iris_classification")

with mlflow.start_run(run_name="automated_retrain_run"):

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_param("trigger", "data_drift")
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name=MODEL_NAME,
    )

    print(
        f"Retraining completed. New Model Accuracy: {accuracy:.4f}"
    )

    print(
        f"Updated version registered to MLflow Model Registry: "
        f"{MODEL_NAME}"
    )

    return model, accuracy


def run_pipeline():
"""Check for data drift and retrain when drift is detected."""

print("Checking production data for drift...")

drift_detected = check_data_drift()

if drift_detected:
    print(
        "Drift threshold breached! "
        "Triggering automated retraining..."
    )

    return train_and_register_model()

print(
    "No significant drift detected. "
    "Retraining skipped."
)

return None


def automated_continuous_training():
"""Backward-compatible entry point."""

return run_pipeline()


if name == "main":
automated_continuous_training()
