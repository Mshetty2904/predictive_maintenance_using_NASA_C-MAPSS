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

# Run all benchmark models from the same prepared DatasetBundle.
MODEL_NAMES = ("xgboost", "lstm", "cnn_lstm")

TEST_SIZE = 0.20

# ==========================================================
# XGBOOST
# ==========================================================

XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "objective": "reg:squarederror",
}

XGB_EARLY_STOPPING_ROUNDS = 40

# ==========================================================
# LSTM
# ==========================================================

LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64

DENSE_1 = 64
DENSE_2 = 32

LSTM_DROPOUT = 0.20

LSTM_LEARNING_RATE = 5e-4

BATCH_SIZE = 64

EPOCHS = 75

LSTM_PATIENCE = 12

# ==========================================================
# CNN-LSTM
# ==========================================================

CNN_FILTERS = 64
CNN_KERNEL_SIZE = 5
POOL_SIZE = 2
CNN_LSTM_DROPOUT = 0.20
CNN_LSTM_LEARNING_RATE = 2e-4
CNN_LSTM_PATIENCE = 15
#18
CNN_LSTM_BATCH_SIZE = 64
CNN_LSTM_LR_PATIENCE = 6
#5
CNN_LSTM_EPOCHS = 100
# ==========================================================
# DRIFT
# ==========================================================

PSI_THRESHOLD = 0.20

KS_PVALUE = 0.05

WINDOW_SIZE = 30

SCALER_PATH = PROJECT_ROOT / "scalers"

# ===================================
# Callbacks
# ===================================

# Validation is performed by engine group, not by random overlapping windows.
VALIDATION_GROUP_SIZE = 0.10

LR_FACTOR = 0.5
MIN_LR = 1e-6

MONITOR = "val_loss"

USE_REGIME_CLUSTERING = True
REGIME_DATASETS = ("FD002", "FD004")
STEP_SIZE = 1
MAX_RUL = 125
