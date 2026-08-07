import os
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

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


def save_keras_model_safely(model, target_path, retries=3):
    """Save a Keras model without crashing when Windows locks the old file."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_name(
        f".{target_path.stem}.{uuid4().hex}.tmp.keras"
    )

    model.save(str(temporary_path))

    for attempt in range(retries):
        try:
            os.replace(temporary_path, target_path)
            return target_path
        except OSError as error:
            if attempt == retries - 1:
                fallback_path = target_path.with_name(
                    f"{target_path.stem}_run_{uuid4().hex}.keras"
                )
                try:
                    os.replace(temporary_path, fallback_path)
                except OSError:
                    # The temporary file is still a valid saved model if this
                    # rare second replacement also fails.
                    fallback_path = temporary_path
                print(
                    f"WARNING: Could not replace {target_path.name}: {error}. "
                    f"Saved the model to {fallback_path.name}."
                )
                return fallback_path
            time.sleep(1)


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


def print_per_regime_metrics(bundle, predictions):
    """Report test metrics by operating regime when regime labels exist."""
    if "Regime_ID" not in bundle.test.columns:
        return
    regime_by_engine = (
        bundle.test.sort_values(["Engine_ID", "Cycle"])
        .groupby("Engine_ID", sort=True)["Regime_ID"]
        .last()
        .to_numpy()
    )
    print("\nPer-regime test metrics")
    for regime in sorted(np.unique(regime_by_engine)):
        mask = regime_by_engine == regime
        metrics = evaluate_model(bundle.y_test[mask], predictions[mask]).iloc[0]
        print(
            f"Regime {int(regime)}: RMSE={metrics['RMSE']:.3f} "
            f"MAE={metrics['MAE']:.3f} R2={metrics['R2']:.3f}"
        )


def print_training_diagnostics(
    history,
    dataset,
    model_name,
    model=None,
    X_train=None,
    y_train=None,
    X_valid=None,
    y_valid=None,
):
    """Evaluate and print diagnostics for the restored best checkpoint."""
    history_data = history.history if hasattr(history, "history") else history
    train_loss = np.asarray(history_data.get("loss", []), dtype=float)
    valid_loss = np.asarray(history_data.get("val_loss", []), dtype=float)

    if train_loss.size == 0 or valid_loss.size == 0:
        print(f"{dataset} {model_name}: validation loss unavailable.")
        return

    best_index = int(np.argmin(valid_loss))
    best_epoch = best_index + 1
    best_val = float(valid_loss[best_index])
    final_train = float(train_loss[best_index])
    final_val = best_val

    if model is not None and X_train is not None and X_valid is not None:
        train_result = model.evaluate(
            X_train,
            y_train,
            verbose=0,
            return_dict=True,
        )
        valid_result = model.evaluate(
            X_valid,
            y_valid,
            verbose=0,
            return_dict=True,
        )
        final_train = float(train_result.get("mse", train_result["loss"]))
        final_val = float(valid_result.get("mse", valid_result["loss"]))

    gap = final_val - final_train

    # This is a diagnostic heuristic, not a statistical test.
    if best_epoch < len(valid_loss) and gap > max(abs(final_val), 1e-9) * 0.25:
        assessment = "possible overfitting"
    elif best_epoch == len(valid_loss) and gap < 0:
        assessment = "possible underfitting"
    else:
        assessment = "reasonable fit"

    print(f"\n{dataset} {model_name} final-fit diagnostics")
    print(f"Best epoch             : {best_epoch}")
    print(f"Best validation loss   : {best_val:.5f}")
    print(f"Final training MSE     : {final_train:.5f}")
    print(f"Final validation MSE   : {final_val:.5f}")
    print(f"Validation MSE gap     : {gap:.5f}")
    print(f"Fit assessment         : {assessment}")

    if model is not None and X_valid is not None:
        valid_result = model.evaluate(
            X_valid,
            y_valid,
            verbose=0,
            return_dict=True,
        )
        if "mae" in valid_result:
            print(f"Best validation MAE    : {float(valid_result['mae']):.5f}")
    elif "val_mae" in history_data:
        val_mae = np.asarray(history_data["val_mae"], dtype=float)
        print(f"Best validation MAE    : {float(np.min(val_mae)):.5f}")


def final_window_split(bundle, validation_size, random_state):
    """Hold out engines and validate only on their truncated final windows."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=random_state,
    )
    fit_final, valid_final = next(
        splitter.split(bundle.X_final, bundle.y_final, groups=bundle.final_groups)
    )
    fit_engines = set(bundle.final_groups[fit_final])
    valid_engines = set(bundle.final_groups[valid_final])
    fit_idx = np.asarray(
        [group in fit_engines for group in bundle.train_groups], dtype=bool
    )
    valid_idx = np.asarray(
        [group in valid_engines for group in bundle.final_groups], dtype=bool
    )
    return fit_idx, valid_idx
