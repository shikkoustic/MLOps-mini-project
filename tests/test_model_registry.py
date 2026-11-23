"""Verify that the configured MLflow model has a champion alias."""
from pathlib import Path

import yaml
from mlflow import MlflowClient


ROOT = Path(__file__).resolve().parents[1]


def test_registered_model_has_resolvable_champion_alias():
    with (ROOT / "params.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_config = config["model"]
    client = MlflowClient(tracking_uri=model_config["tracking_uri"])
    registered = client.get_registered_model(model_config["registered_model_name"])

    assert "champion" in registered.aliases
    champion = client.get_model_version_by_alias(
        model_config["registered_model_name"], "champion"
    )
    assert champion is not None
    assert str(champion.version) == str(registered.aliases["champion"])
