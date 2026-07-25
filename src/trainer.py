from pathlib import Path
from pyexpat import model

import joblib

from xgboost import XGBRegressor

from src.window_generator import flatten_windows

from sklearn.model_selection import train_test_split
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

        X = flatten_windows(bundle.X_train)
        y = bundle.y_train

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )
        validation_predictions = model.predict(
            X_valid
        )

        validation_rmse = (
            (
                (
                    (
                        y_valid
                        - validation_predictions
                    )
                    ** 2
                ).mean()
            ) ** 0.5
        )

        print(
            f"Validation RMSE : {validation_rmse:.3f}"
        )
        X_test = flatten_windows(bundle.X_test)
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