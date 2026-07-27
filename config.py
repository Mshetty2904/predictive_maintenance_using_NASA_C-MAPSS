"""
Project Configuration

Research:
Multi Model Predictive Maintenance using SHAP Explainability
and Automated Drift Triggered Retraining on Cloud
"""
# ==========================================================
# PROJECT PATHS
# ==========================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA_PATH = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "dataset" / "processed"

OUTPUT_PATH = PROJECT_ROOT / "outputs"
MODEL_PATH = PROJECT_ROOT / "models"

DATASETS = [
    "FD001",
    "FD002",
    "FD003",
    "FD004",
]
# ==========================================================
# DATASET
# ==========================================================

DATASET = "FD001"

WINDOW_SIZE = 30

RANDOM_STATE = 42

TEST_SIZE = 0.20

# ==========================================================
# XGBOOST
# ==========================================================

XGB_PARAMS = {

    "n_estimators": 500,

    "learning_rate": 0.05,

    "max_depth": 6,

    "subsample": 0.8,

    "colsample_bytree": 0.8,

    "random_state": RANDOM_STATE

}

# ==========================================================
# LSTM
# ==========================================================

LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64

DENSE_1 = 64
DENSE_2 = 32

DROPOUT = 0.30

LEARNING_RATE = 0.001

BATCH_SIZE = 64

EPOCHS = 75

PATIENCE = 8

# ==========================================================
# CNN-LSTM
# ==========================================================

CNN_LSTM_EPOCHS = 50

CNN_LSTM_BATCH_SIZE = 64

# ==========================================================
# DRIFT
# ==========================================================

PSI_THRESHOLD = 0.20

KS_PVALUE = 0.05

WINDOW_SIZE = 30
STEP_SIZE = 1

VALIDATION_SIZE = 0.20

RANDOM_STATE = 42

SCALER_PATH = PROJECT_ROOT / "scalers"

# ===================================
# Callbacks
# ===================================

VALIDATION_SPLIT = 0.10

LR_FACTOR = 0.5
LR_PATIENCE = 3
MIN_LR = 1e-6

MONITOR = "val_loss"

MAX_RUL = 125