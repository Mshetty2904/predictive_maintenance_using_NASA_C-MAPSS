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

WINDOW_SIZE = 50

RANDOM_STATE = 42

# Run all benchmark models from the same prepared DatasetBundle.
MODEL_NAMES = ["xgboost","lstm","cnn_lstm"]

# Research-design choice for the final SHAP explanation model.  All three
# metrics (RMSE, MAE and R2) are still reported and ranked before this map is
# applied.  Change a value only when the research question requires a
# different deployable explanation model.
SHAP_MODEL_BY_DATASET = {
    "FD001": "xgboost",    # best overall in the benchmark
    "FD002": "cnn_lstm",   # best deep hybrid model
    "FD003": "lstm",       # best-performing model
    "FD004": "lstm",       # best deep model
}

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
# XGBoost's reg_lambda is its L2 tree penalty.
XGB_PARAMS["reg_lambda"] = 1.0  # change 1.0 to 0.0 to disable L2

# ==========================================================
# LSTM
# ==========================================================

LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 64

DENSE_1 = 64
DENSE_2 = 32

LSTM_DROPOUT = 0.20  # change 0.20 to 0.30 or 0.40

LSTM_LEARNING_RATE = 1e-3  # change 1e-3 to 5e-4 or 1e-4

BATCH_SIZE = 64

EPOCHS = 75

LSTM_PATIENCE = 12
L2_REGULARIZATION = 1e-5  # change 1e-5 to 1e-4 for stronger regularization
HUBER_DELTA = 10.0
WEIGHT_DECAY = 1e-4  # change to 0.0 to disable AdamW weight decay
USE_ADAMW = True
ROBUST_DATASETS = ("FD002", "FD004")
ROBUST_LSTM_UNITS = 32
ROBUST_DENSE_1 = 32
ROBUST_DENSE_2 = 16
ROBUST_DROPOUT = 0.35  # change 0.35 to 0.25 or 0.45
ROBUST_LEARNING_RATE = 3e-4  # change 3e-4 to 1e-4 or 5e-4
ROBUST_L2 = 1e-4  # change 1e-4 to 5e-5 or 5e-4

# ==========================================================
# CNN-LSTM
# ==========================================================

CNN_FILTERS = 64
CNN_KERNEL_SIZE = 5
POOL_SIZE = 2
CNN_LSTM_DROPOUT = 0.20  # change 0.20 to 0.30 or 0.40
CNN_LSTM_LEARNING_RATE = 2e-4  # change 2e-4 to 1e-4 or 5e-4
CNN_LSTM_PATIENCE = 15
#18
CNN_LSTM_BATCH_SIZE = 64
CNN_LSTM_LR_PATIENCE = 6
#5
CNN_LSTM_EPOCHS = 100
ROBUST_CNN_FILTERS = 32
ROBUST_CNN_LSTM_UNITS = 32
ROBUST_CNN_DROPOUT = 0.20  # change 0.25 to 0.20 or 0.35
ROBUST_CNN_LEARNING_RATE = 2e-4  # change 2e-4 to 1e-4 or 3e-4
ROBUST_CNN_L2 = 1e-5  # change 5e-5 to 1e-5 or 1e-4
CNN_USE_COSINE_RESTARTS = False  # change False to True after stabilization
CNN_COSINE_FIRST_DECAY_STEPS = 500
CNN_COSINE_T_MUL = 2.0
CNN_COSINE_M_MUL = 0.9
CNN_COSINE_ALPHA = 1e-6
ACTIVATION = "swish"  # change "swish" to "relu" or "gelu"
CNN_USE_RESIDUAL_BLOCK = True

# Shared CNN-BiLSTM-Attention architecture used for every dataset.
ATTENTION_HEADS = 4
ATTENTION_KEY_DIM = 16
# ==========================================================
# DRIFT
# =================================
# =========================

PSI_THRESHOLD = 0.20

KS_PVALUE = 0.05

# Drift monitoring calibration. Thresholds are learned from normal
# training-engine batches, then bounded by these conservative defaults.
DRIFT_CALIBRATION_QUANTILE = 0.995  # change to 0.99 for more sensitivity
DRIFT_REQUIRE_BOTH_TESTS = True  # change to False to use PSI OR KS
MIN_CONSECUTIVE_DRIFT_BATCHES = 3  # change to 1 for immediate triggering
MIN_CONSECUTIVE_STABLE_BATCHES = 2  # stable batches required to reset a drift episode
MIN_DRIFTED_SENSORS_PER_BATCH = 3  # change to 1 for sensor-level triggering
DRIFT_MIN_BATCH_ROWS = 5  # short engines remain covered; very small batches are flagged

#WINDOW_SIZE = 50

SCALER_PATH = PROJECT_ROOT / "scalers"

# ===================================
# Callbacks
# ===================================

# Validation is performed by engine group, not by random overlapping windows.
VALIDATION_SIZE_BY_DATASET = {
    "FD001": 0.20,
    "FD002": 0.10,
    "FD003": 0.20,
    "FD004": 0.10,
}

LR_FACTOR = 0.5
MIN_LR = 1e-6

MONITOR = "val_loss"

USE_REGIME_CLUSTERING = True
REGIME_DATASETS = ("FD002", "FD004")
# change None to 2 or 6 to test a fixed regime count explicitly
REGIME_K_OVERRIDE = None
REGIME_K_CANDIDATES = (2, 6)
REPORT_PER_REGIME_METRICS = True
ADD_REGIME_ONE_HOT = True
STEP_SIZE = 1
MAX_RUL = 125

# Temporal-feature controls. Lower MAX_TEMPORAL_SENSORS if feature count is
# too high; increase it from 6 to 8 only after validating on held-out engines.
TEMPORAL_FEATURES_PER_REGIME = 3
MAX_TEMPORAL_SENSORS = 6
ADD_SECOND_DERIVATIVE = False  # change False to True after CNN validation
FINAL_WINDOW_CUTOFFS = (0.60, 0.70, 0.80, 0.90)

#NEW ONe
REGIME_ONEHOT_SIZE = 6
USE_REGIME_FILM = True
SAMPLE_WEIGHT_BY_REGIME = True
REGIME_WEIGHT_CAP = 2.5
ROBUST_ATTENTION_HEADS = 6        # was shared ATTENTION_HEADS = 4
ROBUST_ATTENTION_KEY_DIM = 16
ROBUST_ACTIVATION = "swish"
ROBUST_CLIPNORM = None
ROBUST_CNN_USE_COSINE_RESTARTS = True
ROBUST_CNN_COSINE_FIRST_DECAY_STEPS = 300   # shorter than the standard 500 --
                                             # your best robust epochs (13, 34) are early
ROBUST_CNN_COSINE_T_MUL = 2.0
ROBUST_CNN_COSINE_M_MUL = 0.9
ROBUST_CNN_COSINE_ALPHA = 1e-6
