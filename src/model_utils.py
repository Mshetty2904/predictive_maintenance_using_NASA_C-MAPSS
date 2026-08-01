from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.nasa_score import nasa_score


def evaluate_model(y_true, predictions):
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    score = nasa_score(y_true, predictions)
    return pd.DataFrame(
        {
            "RMSE": [round(rmse, 3)],
            "MAE": [round(mae, 3)],
            "R2": [round(r2, 3)],
            "NASA Score": [round(score, 3)],
        }
    )


def save_metrics(metrics, output_path, dataset, model_name):
    metrics_path = Path(output_path) / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path / f"{dataset}_{model_name}_metrics.csv", index=False)


def print_dataset_info(bundle, model_name):
    print("\n" + "=" * 60)
    print(f"Dataset: {bundle.dataset_name} | Model: {model_name}")
    print("=" * 60)
    print(f"Train engines : {bundle.train['Engine_ID'].nunique()}")
    print(f"Test engines  : {bundle.test['Engine_ID'].nunique()}")
    print(f"Train windows : {bundle.X_train.shape[0]}")
    print(f"Test windows  : {bundle.X_test.shape[0]}")
    print(f"Window shape  : {bundle.X_train.shape[1:]}")


def print_cv_fold(fold, rmse, mae, r2, nasa_score):
    print(
        f"Fold {fold}: RMSE={rmse:.3f} MAE={mae:.3f} "
        f"R2={r2:.3f} NASA={nasa_score:.3f}"
    )


def print_cv_summary(rmse_scores, mae_scores, r2_scores, nasa_scores):
    print("\nCross-Validation Average")
    print(f"RMSE       : {np.mean(rmse_scores):.3f}")
    print(f"MAE        : {np.mean(mae_scores):.3f}")
    print(f"R2         : {np.mean(r2_scores):.3f}")
    print(f"NASA Score : {np.mean(nasa_scores):.3f}")


def print_final_metrics(metrics):
    result = metrics.iloc[0]
    print("\nFinal NASA Test Metrics")
    print(f"RMSE       : {result['RMSE']:.3f}")
    print(f"MAE        : {result['MAE']:.3f}")
    print(f"R2         : {result['R2']:.3f}")
    print(f"NASA Score : {result['NASA Score']:.3f}")


def print_training_diagnostics(history, dataset, model_name):
    """Print concise final-fit diagnostics for a neural model."""
    history_data = history.history if hasattr(history, "history") else history
    train_loss = np.asarray(history_data.get("loss", []), dtype=float)
    valid_loss = np.asarray(history_data.get("val_loss", []), dtype=float)

    if train_loss.size == 0 or valid_loss.size == 0:
        print(f"{dataset} {model_name}: validation loss unavailable.")
        return

    best_epoch = int(np.argmin(valid_loss)) + 1
    best_val = float(np.min(valid_loss))
    final_train = float(train_loss[-1])
    final_val = float(valid_loss[-1])
    gap = final_val - final_train

    # This is a diagnostic heuristic, not a statistical test.
    if final_val > best_val * 1.10 and final_train < best_val:
        assessment = "possible overfitting"
    elif best_epoch == len(valid_loss) and final_train > best_val * 1.10:
        assessment = "possible underfitting"
    else:
        assessment = "reasonable fit"

    print(f"\n{dataset} {model_name} final-fit diagnostics")
    print(f"Best epoch             : {best_epoch}")
    print(f"Best validation loss   : {best_val:.5f}")
    print(f"Final training loss    : {final_train:.5f}")
    print(f"Final validation loss  : {final_val:.5f}")
    print(f"Validation gap         : {gap:.5f}")
    print(f"Fit assessment         : {assessment}")

    if "val_mae" in history_data:
        val_mae = np.asarray(history_data["val_mae"], dtype=float)
        print(f"Best validation MAE    : {float(np.min(val_mae)):.5f}")
