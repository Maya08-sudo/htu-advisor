from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from train_model import sha256_file, train_and_replace


DATASET_PATH = Path("htu_majors_interests.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "best_major_predictor.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.pkl"
METADATA_PATH = MODEL_DIR / "training_metadata.json"


def model_is_stale(dataset_path: Path = DATASET_PATH) -> bool:
    """Return True when the model is missing or the CSV content changed."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return True

    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True

    saved_hash = metadata.get("dataset_sha256")
    return saved_hash != sha256_file(dataset_path)


def ensure_current_model(
    dataset_path: Path = DATASET_PATH,
    auto_retrain: bool = True,
    minimum_accuracy: float = 0.0,
) -> None:
    """
    Retrain only when the CSV content changed.

    For production systems, it is usually better to retrain manually and deploy
    a reviewed model. For this prototype, automatic retraining is supported.
    """
    if model_is_stale(dataset_path):
        if not auto_retrain:
            raise RuntimeError(
                "The dataset changed but the model was not retrained. "
                "Run: python train_model.py"
            )

        train_and_replace(
            dataset_path=dataset_path,
            model_dir=MODEL_DIR,
            minimum_accuracy=minimum_accuracy,
        )


def load_current_model(
    auto_retrain: bool = True,
    minimum_accuracy: float = 0.0,
) -> tuple[Any, Any, list[str]]:
    ensure_current_model(
        auto_retrain=auto_retrain,
        minimum_accuracy=minimum_accuracy,
    )

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, encoder, feature_columns
