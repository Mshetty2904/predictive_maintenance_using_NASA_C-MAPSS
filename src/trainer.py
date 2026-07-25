from pathlib import Path

import joblib

from xgboost import XGBRegressor

from src.window_generator import flatten_windows


class XGBoostTrainer:
    """
    Train and save an XGBoost model.
    """

    def __init__(self, model_path):

        self.model_path = Path(model_path)

        self.model_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    def train(self, bundle):

        X_train = flatten_windows(bundle.X_train)
        X_test = flatten_windows(bundle.X_test)

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            bundle.y_train,
        )

        predictions = model.predict(X_test)

        model_file = (
            self.model_path
            / f"{bundle.dataset_name}_xgboost.pkl"
        )

        joblib.dump(
            model,
            model_file,
        )

        return model, predictions