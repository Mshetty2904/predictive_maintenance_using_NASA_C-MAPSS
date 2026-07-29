from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.nasa_score import nasa_score


def evaluate_model(y_true, predictions):

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    score = nasa_score(
        y_true,
        predictions,
    )

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
    metrics_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        metrics_path / f"{dataset}_{model_name}_metrics.csv",
        index=False,
    )


def print_dataset_info(bundle, model_name):

    print("\n" + "=" * 60)
    print(f"Dataset: {bundle.dataset_name} | Model: {model_name}")
    print("=" * 60)

    print(
        f"Train engines : "
        f"{bundle.train['Engine_ID'].nunique()}"
    )

    print(
        f"Test engines  : "
        f"{bundle.test['Engine_ID'].nunique()}"
    )

    print(
        f"Train windows : "
        f"{bundle.X_train.shape[0]}"
    )

    print(
        f"Test windows  : "
        f"{bundle.X_test.shape[0]}"
    )

    print(
        f"Window shape  : "
        f"{bundle.X_train.shape[1:]}"
    )


def print_cv_fold(
    fold,
    rmse,
    mae,
    r2,
    nasa_score,
):

    print(
        f"Fold {fold}: "
        f"RMSE={rmse:.3f} "
        f"MAE={mae:.3f} "
        f"R2={r2:.3f} "
        f"NASA={nasa_score:.3f}"
    )


def print_cv_summary(
    rmse_scores,
    mae_scores,
    r2_scores,
    nasa_scores,
):

    print("\nCross-Validation Average")

    print(
        f"RMSE       : {np.mean(rmse_scores):.3f}"
    )

    print(
        f"MAE        : {np.mean(mae_scores):.3f}"
    )

    print(
        f"R2         : {np.mean(r2_scores):.3f}"
    )

    print(
        f"NASA Score : {np.mean(nasa_scores):.3f}"
    )


def print_final_metrics(metrics):

    result = metrics.iloc[0]

    print("\nFinal NASA Test Metrics")

    print(
        f"RMSE       : {result['RMSE']:.3f}"
    )

    print(
        f"MAE        : {result['MAE']:.3f}"
    )

    print(
        f"R2         : {result['R2']:.3f}"
    )

    print(
        f"NASA Score : "
        f"{result['NASA Score']:.3f}"
    )