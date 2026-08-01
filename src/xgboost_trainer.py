from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping

from config import MAX_RUL, XGB_EARLY_STOPPING_ROUNDS, XGB_PARAMS
from src.model_utils import print_cv_fold, print_cv_summary
from src.nasa_score import nasa_score
from src.window_generator import flatten_windows


class XGBoostTrainer:

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)

    def build_model(self, n_estimators=None, callbacks=None):
        params = XGB_PARAMS.copy()
        if n_estimators is not None:
            params["n_estimators"] = n_estimators
        return XGBRegressor(
            **params,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="rmse",
            callbacks=callbacks,
        )

    def train(self, bundle):
        X_train = flatten_windows(bundle.X_train)
        X_test = flatten_windows(bundle.X_test)
        y_train = bundle.y_train

        print("\nRunning 5-Fold Engine-wise Cross Validation...")
        rmse_scores, mae_scores, r2_scores = [], [], []
        nasa_scores, best_rounds = [], []
        group_kfold = GroupKFold(n_splits=5)

        for fold, (train_idx, valid_idx) in enumerate(
            group_kfold.split(X_train, y_train, groups=bundle.train_groups),
            start=1,
        ):
            model = self.build_model(
                callbacks=[
                    EarlyStopping(
                        rounds=XGB_EARLY_STOPPING_ROUNDS,
                        save_best=True,
                    )
                ]
            )
            model.fit(
                X_train[train_idx],
                y_train[train_idx],
                eval_set=[(X_train[valid_idx], y_train[valid_idx])],
                verbose=False,
            )
            predictions = np.clip(
                model.predict(X_train[valid_idx]),
                0,
                MAX_RUL,
            )
            rmse = np.sqrt(mean_squared_error(y_train[valid_idx], predictions))
            mae = mean_absolute_error(y_train[valid_idx], predictions)
            r2 = r2_score(y_train[valid_idx], predictions)
            score = nasa_score(y_train[valid_idx], predictions)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            nasa_scores.append(score)
            best_rounds.append(model.best_iteration + 1)
            print_cv_fold(fold, rmse, mae, r2, score)

        print_cv_summary(rmse_scores, mae_scores, r2_scores, nasa_scores)
        final_rounds = int(np.median(best_rounds))
        print("\nTraining Final XGBoost Model...")
        print(f"Selected Trees: {final_rounds}")
        print(
            "XGBoost validation loss is not epoch-based; "
            "the preceding 5-fold engine-wise metrics are the validation results."
        )
        final_model = self.build_model(n_estimators=final_rounds)
        final_model.fit(X_train, y_train, verbose=False)
        predictions = np.clip(final_model.predict(X_test), 0, MAX_RUL)
        joblib.dump(
            final_model,
            self.model_path / f"{bundle.dataset_name}_xgboost.pkl",
        )
        return final_model, predictions
