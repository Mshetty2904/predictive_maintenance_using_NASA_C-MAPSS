from pathlib import Path

import joblib
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from xgboost import XGBRegressor

from src.window_generator import flatten_windows


class XGBoostTrainer:
    """
    Train an XGBoost model using 5-Fold Cross Validation,
    retrain on the full training data,
    and predict on the NASA official test dataset.
    """

    def __init__(self, model_path):

        self.model_path = Path(model_path)

        self.model_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    def train(self, bundle):

        # -----------------------------------------
        # Prepare Training Data
        # -----------------------------------------

        X_train = flatten_windows(bundle.X_train)
        y_train = bundle.y_train

        # -----------------------------------------
        # 5-Fold Cross Validation
        # -----------------------------------------

        print("\nRunning 5-Fold Cross Validation...")

        kfold = KFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        rmse_scores = []
        mae_scores = []
        r2_scores = []

        fold = 1

        for train_idx, valid_idx in kfold.split(X_train):

            X_tr = X_train[train_idx]
            X_val = X_train[valid_idx]

            y_tr = y_train[train_idx]
            y_val = y_train[valid_idx]

            model = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
            )

            model.fit(
                X_tr,
                y_tr,
            )

            predictions = model.predict(X_val)

            rmse = np.sqrt(
                mean_squared_error(
                    y_val,
                    predictions,
                )
            )

            mae = mean_absolute_error(
                y_val,
                predictions,
            )

            r2 = r2_score(
                y_val,
                predictions,
            )

            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)

            print(
                f"Fold {fold}: "
                f"RMSE={rmse:.3f} "
                f"MAE={mae:.3f} "
                f"R2={r2:.3f}"
            )

            fold += 1

        print("\nAverage Cross Validation Performance")

        print(
            f"RMSE : {np.mean(rmse_scores):.3f}"
        )

        print(
            f"MAE  : {np.mean(mae_scores):.3f}"
        )

        print(
            f"R2   : {np.mean(r2_scores):.3f}"
        )

        # -----------------------------------------
        # Train Final Model
        # -----------------------------------------

        print("\nTraining Final Model...")

        final_model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )

        final_model.fit(
            X_train,
            y_train,
        )

        # -----------------------------------------
        # Test Dataset
        # -----------------------------------------

        X_test = flatten_windows(bundle.X_test)

        predictions = final_model.predict(
            X_test,
        )

        # -----------------------------------------
        # Save Model
        # -----------------------------------------

        model_file = (
            self.model_path
            / f"{bundle.dataset_name}_xgboost.pkl"
        )

        joblib.dump(
            final_model,
            model_file,
        )

        return final_model, predictions