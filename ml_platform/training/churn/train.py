"""Train, evaluate and persist the telecom churn prediction pipeline.

Run from the repository root:
    uv run --locked python ml_platform/training/churn/train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

DATASET_ID = 563
RANDOM_STATE = 42
TEST_SIZE = 0.2
ARTIFACT_DIR = Path("ml_platform/training/churn/models")


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_ucirepo(id=DATASET_ID)
    features = dataset.data.features
    target = dataset.data.targets.iloc[:, 0]
    return features, target


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def evaluate(pipeline: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict:
    predictions = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)[:, 1]

    report = classification_report(target, predictions, output_dict=True, zero_division=0)
    churn_class = report["1"]

    return {
        "dataset_id": DATASET_ID,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "accuracy": report["accuracy"],
        "churn_precision": churn_class["precision"],
        "churn_recall": churn_class["recall"],
        "churn_f1": churn_class["f1-score"],
        "roc_auc": roc_auc_score(target, probabilities),
    }


def save_artifacts(pipeline: Pipeline, feature_names: list[str], metrics: dict) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, ARTIFACT_DIR / "churn_pipeline.joblib")
    (ARTIFACT_DIR / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))


def main() -> None:
    features, target = load_dataset()
    print(f"Shape: {features.shape}")
    print("Target distribution:")
    print(target.value_counts(normalize=True).round(4))

    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    pipeline = build_pipeline()
    pipeline.fit(features_train, target_train)

    print("\nHeld-out performance:")
    print(classification_report(target_test, pipeline.predict(features_test), zero_division=0))

    metrics = evaluate(pipeline, features_test, target_test)
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    sanity_check = pd.DataFrame(
        {
            "actual": target_test.head(10).to_numpy(),
            "predicted": pipeline.predict(features_test.head(10)),
            "churn_probability": pipeline.predict_proba(features_test.head(10))[:, 1].round(4),
        }
    )
    print("\nSanity check on ten held-out customers:")
    print(sanity_check.to_string(index=False))

    save_artifacts(pipeline, features.columns.tolist(), metrics)
    print(f"\nArtifacts written to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
