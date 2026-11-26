"""Stage 4: retrain v2 on train + first 70% of later.csv, compare to v1 on
the last 30%, and apply the client's promotion rule."""
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

try:
    from mp128.cleaning import clean_train
    from mp128.train import load_params, make_pipeline
except ImportError:
    from src.mp128.cleaning import clean_train
    from src.mp128.train import load_params, make_pipeline

ROOT = Path(__file__).resolve().parents[2]
PROMOTION_IMPROVEMENT = 0.08  # v2 must be at least 8% lower MAE than v1


def _feature_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    excluded = {target_column, "record_id", "date"}
    return [c for c in df.columns if c not in excluded]


def run_retrain_and_promote() -> dict:
    config = load_params()
    data_config = config["data"]
    model_config = config["model"]
    target_column = data_config["target_column"]

    tracking_uri = model_config["tracking_uri"]
    registered_model_name = model_config["registered_model_name"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(model_config["experiment_name"])
    client = MlflowClient(tracking_uri=tracking_uri)

    # --- data ---
    train_path = ROOT / data_config["train_path"]
    later_path = ROOT / data_config["later_path"]
    data_sha256 = hashlib.sha256(train_path.read_bytes()).hexdigest()

    train_raw = pd.read_csv(train_path)
    train_clean = clean_train(train_raw).dropna(subset=[target_column])

    later = pd.read_csv(later_path)
    later["date"] = pd.to_datetime(later["date"])
    later = later.sort_values("date").reset_index(drop=True)

    split_idx = int(len(later) * 0.70)
    later_first70 = later.iloc[:split_idx]
    later_last30 = later.iloc[split_idx:]

    feature_columns = _feature_columns(train_clean, target_column)
    numeric_columns = train_clean[feature_columns].select_dtypes(include="number").columns.tolist()
    categorical_columns = [c for c in feature_columns if c not in numeric_columns]

    # --- v1: current champion. Loaded from the exported joblib, not the
    # MLflow artifact store --- that store is a local directory (mlruns/),
    # gitignored per the handbook, so it isn't guaranteed to exist here.
    champion_v1 = joblib.load(ROOT / "models" / "model.joblib")
    v1_preds = champion_v1.predict(later_last30[feature_columns])
    mae_v1_on_check = float(mean_absolute_error(later_last30[target_column], v1_preds))

    # --- v2: retrain on cleaned train + first 70% of later.csv ---
    v2_train_df = pd.concat([train_clean, later_first70], ignore_index=True)
    X_v2 = v2_train_df[feature_columns]
    y_v2 = v2_train_df[target_column]

    pipeline_v2 = make_pipeline(Ridge(alpha=1.0), numeric_columns, categorical_columns)

    with mlflow.start_run(run_name="v2_retrain_train_plus_later70") as run:
        pipeline_v2.fit(X_v2, y_v2)
        v2_preds_check = pipeline_v2.predict(later_last30[feature_columns])
        mae_v2_on_check = float(mean_absolute_error(later_last30[target_column], v2_preds_check))

        mlflow.log_params({
            "setting": "v2_retrain_train_plus_later70",
            "estimator": "Ridge",
            "target_column": target_column,
            "n_train_rows": len(v2_train_df),
        })
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_metric("mae_val", mae_v2_on_check)
        mlflow.log_metric("mae_v1_on_check", mae_v1_on_check)
        mlflow.log_metric("mae_v2_on_check", mae_v2_on_check)

        signature = infer_signature(X_v2, pipeline_v2.predict(X_v2))
        mlflow.sklearn.log_model(
            sk_model=pipeline_v2,
            artifact_path="model",
            signature=signature,
            input_example=X_v2.head(5),
            serialization_format="cloudpickle",
        )
        v2_run_id = run.info.run_id

    # --- promotion rule: promote only if v2 MAE is >= 8% lower than v1's ---
    required_mae = mae_v1_on_check * (1 - PROMOTION_IMPROVEMENT)
    promote = mae_v2_on_check <= required_mae

    v2_model_uri = f"runs:/{v2_run_id}/model"
    registered_v2 = client.create_model_version(
        name=registered_model_name, source=v2_model_uri, run_id=v2_run_id
    )

    if promote:
        client.set_registered_model_alias(
            name=registered_model_name, alias="champion", version=registered_v2.version
        )
        decision = "promote"
        champion_version = int(registered_v2.version)
        # Export the new champion the same way Stage 2 did, so the service
        # and anyone cloning the repo can use it without needing the local
        # (gitignored) MLflow artifact store.
        joblib.dump(pipeline_v2, ROOT / "models" / "model.joblib")
    else:
        decision = "keep"
        champion_version = int(client.get_model_version_by_alias(registered_model_name, "champion").version)

    return {
        "mae_v1_on_check": mae_v1_on_check,
        "mae_v2_on_check": mae_v2_on_check,
        "decision": decision,
        "champion_version": champion_version,
        "v2_run_id": v2_run_id,
        "registered_model_name": registered_model_name,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_retrain_and_promote(), indent=2))
