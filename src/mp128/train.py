"""Train MP128 PM2.5 regressors, track them in MLflow, and register champion."""
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# conftest.py adds src/ to sys.path for pytest; this fallback also supports
# running the file as a module from the repository root.
try:
    from mp128.cleaning import clean_train
except ImportError:
    from src.mp128.cleaning import clean_train


ROOT = Path(__file__).resolve().parents[2]


def load_params() -> dict:
    with (ROOT / "params.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_pipeline(estimator, numeric_columns: list[str], categorical_columns: list[str]):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", estimator),
    ])


def train() -> dict:
    config = load_params()
    data_config = config["data"]
    model_config = config["model"]

    train_path = ROOT / data_config["train_path"]
    if not train_path.is_file():
        raise FileNotFoundError(
            f"{train_path} is missing. Follow the repository README to place "
            "the supplied training CSV at data/raw/train.csv."
        )

    # Hash the original bytes before cleaning so the run records exact provenance.
    data_sha256 = hashlib.sha256(train_path.read_bytes()).hexdigest()
    raw = pd.read_csv(train_path)
    cleaned = clean_train(raw)

    target_column = data_config["target_column"]
    if target_column not in cleaned.columns:
        raise ValueError(f"Missing target column: {target_column}")
    cleaned = cleaned.dropna(subset=[target_column]).copy()

    # Identifiers and date are not model features.
    excluded = {target_column, "record_id", "date"}
    feature_columns = [column for column in cleaned.columns if column not in excluded]
    X = cleaned[feature_columns]
    y = pd.to_numeric(cleaned[target_column], errors="raise")

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    if not feature_columns:
        raise ValueError("No feature columns remain after excluding identifiers and target")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=int(model_config.get("random_state", 42))
    )

    seed = int(model_config.get("random_state", 42))
    candidates = [
        ("ridge_alpha_1", Ridge(alpha=1.0)),
        ("random_forest_200", RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1
        )),
        ("gradient_boosting_150", GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=2, random_state=seed
        )),
    ]

    tracking_uri = model_config["tracking_uri"]
    experiment_name = model_config["experiment_name"]
    registered_model_name = model_config["registered_model_name"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    results = []
    for setting, estimator in candidates:
        pipeline = make_pipeline(estimator, numeric_columns, categorical_columns)
        with mlflow.start_run(run_name=setting) as run:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_valid)
            mae = float(mean_absolute_error(y_valid, predictions))

            mlflow.log_params({
                "setting": setting,
                "estimator": estimator.__class__.__name__,
                "target_column": target_column,
                "random_state": seed,
                "validation_size": 0.2,
            })
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_metric("mae_val", mae)

            signature = infer_signature(X_train, pipeline.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(5),
                serialization_format="cloudpickle",
            )
            results.append({
                "run_id": run.info.run_id,
                "setting": setting,
                "mae": mae,
            })

    best = min(results, key=lambda result: result["mae"])
    model_uri = f"runs:/{best['run_id']}/model"
    registered = mlflow.register_model(model_uri, registered_model_name)

    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_registered_model_alias(
        name=registered_model_name,
        alias="champion",
        version=registered.version,
    )

    champion = mlflow.sklearn.load_model(f"models:/{registered_model_name}@champion")
    export_path = ROOT / "models" / "model.joblib"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, export_path)

    # Fill only the tracking keys, preserving all Stage 1 and monitoring findings.
    findings_path = ROOT / "reports" / "findings.yaml"
    with findings_path.open(encoding="utf-8") as file:
        findings = yaml.safe_load(file)
    findings.setdefault("tracking", {}).update({
        "experiment_name": experiment_name,
        "best_run_id": best["run_id"],
        "registered_model_name": registered_model_name,
        "champion_version": int(registered.version),
    })
    with findings_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(findings, file, sort_keys=False)

    return {
        "experiment_name": experiment_name,
        "best_run_id": best["run_id"],
        "best_mae": best["mae"],
        "registered_model_name": registered_model_name,
        "champion_version": int(registered.version),
        "runs": results,
    }


if __name__ == "__main__":
    print(train())
