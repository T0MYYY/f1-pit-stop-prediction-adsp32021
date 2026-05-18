"""Shared configuration for the F1 pit-stop training pipeline."""

from __future__ import annotations

from pathlib import Path

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_TRAIN_CSV = DATA_DIR / "train.csv"
INTERIM_DIR = DATA_DIR / "interim"
RAW_PARQUET = INTERIM_DIR / "raw.parquet"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_TRAIN = PROCESSED_DIR / "train.parquet"
PROCESSED_TEST = PROCESSED_DIR / "test.parquet"

MLFLOW_DIR = PROJECT_ROOT / "mlflow"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}"
MLFLOW_ARTIFACT_ROOT = MLFLOW_DIR / "artifacts"
EXPERIMENT_NAME = "pit-stop-classifier"
REGISTERED_MODEL_NAME = "f1-pit-stop-classifier"

MODELS_DIR = PROJECT_ROOT / "models"
CHAMPION_EXPORT_DIR = MODELS_DIR / "champion"

TARGET_COL = "PitNextLap"
TRAIN_YEARS = (2022, 2023, 2024)
TEST_YEARS = (2025,)

RAW_COLUMNS = [
    "id", "Driver", "Compound", "Race", "Year", "PitStop", "LapNumber",
    "Stint", "TyreLife", "Position", "LapTime (s)", "LapTime_Delta",
    "Cumulative_Degradation", "RaceProgress", "Position_Change", "PitNextLap",
]

ID_COLUMNS = ["id", "Driver", "Race", "Year"]
CATEGORICAL_COLUMNS = ["Compound", "TyreLife_bucket"]

AUTOML_TIME_BUDGET = 600
AUTOML_METRIC = "macro_f1"
AUTOML_ESTIMATORS = ["lgbm", "xgboost", "xgb_limitdepth", "catboost"]


def ensure_dirs() -> None:
    for d in (INTERIM_DIR, PROCESSED_DIR, MLFLOW_DIR, MLFLOW_ARTIFACT_ROOT, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
