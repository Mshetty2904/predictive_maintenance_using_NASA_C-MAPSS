from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_model(
    y_true,
    predictions,
):

    mse = mean_squared_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    return pd.DataFrame(
        {
            "RMSE": [round(rmse, 3)],
            "MAE": [round(mae, 3)],
            "R2": [round(r2, 3)],
        }
    )


def save_metrics(
    metrics,
    output_path,
    dataset,
):

    metrics.to_csv(
        Path(output_path)
        / "metrics"
        / f"{dataset}_xgboost_metrics.csv",
        index=False,
    )