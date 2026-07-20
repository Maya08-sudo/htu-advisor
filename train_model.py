from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


DEFAULT_DATASET = Path("htu_majors_interests.csv")
DEFAULT_MODEL_DIR = Path("models")
TARGET_CANDIDATES = (
    "major",
    "recommended_major",
    "target",
    "label",
    "Major",
    "Recommended_Major",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_target_column(df: pd.DataFrame, explicit_target: str | None) -> str:
    if explicit_target:
        if explicit_target not in df.columns:
            raise ValueError(
                f"Target column '{explicit_target}' was not found. "
                f"Available columns: {list(df.columns)}"
            )
        return explicit_target

    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate

    # Safe fallback: use the final column only when it looks categorical.
    last = df.columns[-1]
    if df[last].nunique(dropna=True) <= max(50, int(len(df) * 0.2)):
        return last

    raise ValueError(
        "Could not detect the target column. Run with "
        "--target YOUR_TARGET_COLUMN."
    )


def load_and_validate_dataset(
    csv_path: Path,
    target_column: str | None,
) -> tuple[pd.DataFrame, pd.Series, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("The dataset is empty.")

    target = detect_target_column(df, target_column)

    # Remove fully empty rows and rows with no target.
    df = df.dropna(how="all").copy()
    df = df.dropna(subset=[target]).copy()

    # Remove exact duplicate records so repeated rows do not bias the model.
    df = df.drop_duplicates().reset_index(drop=True)

    if len(df) < 30:
        raise ValueError(
            f"Only {len(df)} valid rows remain. At least 30 rows are required."
        )

    y = df[target].astype(str).str.strip()
    X = df.drop(columns=[target])

    empty_columns = [column for column in X.columns if X[column].isna().all()]
    if empty_columns:
        X = X.drop(columns=empty_columns)

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns were found.")

    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError("The target must contain at least two different majors.")

    rare = class_counts[class_counts < 2]
    if not rare.empty:
        names = ", ".join(f"{name}={count}" for name, count in rare.items())
        raise ValueError(
            "Every major needs at least two examples for a stratified split. "
            f"Insufficient classes: {names}"
        )

    return X, y, target


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [
        column for column in X.columns if column not in numeric_columns
    ]

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def candidate_models(random_state: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
    }


def atomic_joblib_dump(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=destination.name,
        suffix=".tmp",
        delete=False,
    ) as temp:
        temporary_path = Path(temp.name)

    try:
        joblib.dump(value, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def atomic_json_dump(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=destination.name,
        suffix=".tmp",
        delete=False,
    ) as temp:
        json.dump(value, temp, ensure_ascii=False, indent=2)
        temporary_path = Path(temp.name)

    try:
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def backup_existing_models(model_dir: Path) -> Path | None:
    important_files = [
        model_dir / "best_major_predictor.pkl",
        model_dir / "label_encoder.pkl",
        model_dir / "feature_columns.pkl",
        model_dir / "best_major_predictor_meta.pkl",
        model_dir / "training_metadata.json",
    ]

    if not any(path.exists() for path in important_files):
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = model_dir / "backups" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    for path in important_files:
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)

    return backup_dir


def train_and_replace(
    dataset_path: Path = DEFAULT_DATASET,
    model_dir: Path = DEFAULT_MODEL_DIR,
    target_column: str | None = None,
    minimum_accuracy: float = 0.0,
    force: bool = False,
    random_state: int = 42,
) -> dict[str, Any]:
    X, y_text, target = load_and_validate_dataset(dataset_path, target_column)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    min_class_count = int(pd.Series(y).value_counts().min())
    test_size = max(0.20, len(encoder.classes_) / len(y))
    test_size = min(test_size, 0.40)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    folds = min(5, min_class_count)
    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )

    results: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}

    for name, classifier in candidate_models(random_state).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X)),
                ("classifier", classifier),
            ]
        )

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        result = {
            "name": name,
            "cv_f1_macro_mean": float(np.mean(cv_scores)),
            "cv_f1_macro_std": float(np.std(cv_scores)),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_f1_macro": float(
                f1_score(y_test, predictions, average="macro")
            ),
            "classification_report": classification_report(
                y_test,
                predictions,
                target_names=encoder.classes_,
                output_dict=True,
                zero_division=0,
            ),
        }
        results.append(result)
        fitted_models[name] = pipeline

    # Primary selection metric is cross-validated macro F1.
    best_result = max(
        results,
        key=lambda item: (
            item["cv_f1_macro_mean"],
            item["test_f1_macro"],
            item["test_accuracy"],
        ),
    )
    best_model = fitted_models[best_result["name"]]

    # Refit the selected pipeline on all validated rows.
    best_model.fit(X, y)

    current_metadata_path = model_dir / "training_metadata.json"
    old_accuracy = None
    if current_metadata_path.exists():
        try:
            old_metadata = json.loads(
                current_metadata_path.read_text(encoding="utf-8")
            )
            old_accuracy = old_metadata.get("best_model", {}).get("test_accuracy")
        except (json.JSONDecodeError, OSError):
            old_accuracy = None

    passes_threshold = best_result["test_accuracy"] >= minimum_accuracy
    improves_existing = (
        old_accuracy is None
        or best_result["test_accuracy"] >= float(old_accuracy)
    )

    if not force and (not passes_threshold or not improves_existing):
        reason = (
            f"New model was not installed. Accuracy={best_result['test_accuracy']:.4f}, "
            f"required={minimum_accuracy:.4f}, old={old_accuracy}."
        )
        print(reason)
        return {
            "installed": False,
            "reason": reason,
            "best_model": best_result,
            "all_results": results,
        }

    backup_dir = backup_existing_models(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    atomic_joblib_dump(
        best_model,
        model_dir / "best_major_predictor.pkl",
    )
    atomic_joblib_dump(
        encoder,
        model_dir / "label_encoder.pkl",
    )
    atomic_joblib_dump(
        list(X.columns),
        model_dir / "feature_columns.pkl",
    )

    compatibility_metadata = {
        "type": "sklearn_pipeline",
        "model_name": best_result["name"],
        "accuracy": best_result["test_accuracy"],
        "f1_macro": best_result["test_f1_macro"],
    }
    atomic_joblib_dump(
        compatibility_metadata,
        model_dir / "best_major_predictor_meta.pkl",
    )

    metadata = {
        "installed": True,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "rows_used": int(len(X)),
        "feature_count": int(X.shape[1]),
        "feature_columns": list(X.columns),
        "target_column": target,
        "classes": encoder.classes_.tolist(),
        "best_model": best_result,
        "candidate_results": results,
        "backup_directory": str(backup_dir) if backup_dir else None,
    }
    atomic_json_dump(metadata, current_metadata_path)

    print("\nModel update completed.")
    print(f"Selected model: {best_result['name']}")
    print(f"Cross-validated macro F1: {best_result['cv_f1_macro_mean']:.4f}")
    print(f"Test accuracy: {best_result['test_accuracy']:.4f}")
    print(f"Test macro F1: {best_result['test_f1_macro']:.4f}")
    print(f"Rows used: {len(X)}")
    print(f"Saved to: {model_dir.resolve()}")

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, validate and safely replace the HTU major model."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the updated CSV dataset.",
    )
    parser.add_argument(
        "--models",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory in which model files are stored.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target column. Leave blank for automatic detection.",
    )
    parser.add_argument(
        "--minimum-accuracy",
        type=float,
        default=0.0,
        help="Do not install a new model below this test accuracy.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Install the new model even when its accuracy is lower.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_and_replace(
        dataset_path=Path(arguments.dataset),
        model_dir=Path(arguments.models),
        target_column=arguments.target,
        minimum_accuracy=arguments.minimum_accuracy,
        force=arguments.force,
    )
