from pathlib import Path

import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """
    Standardize features for deep learning models.
    """

    def __init__(self, scaler_path):

        self.scaler_path = Path(scaler_path)

        self.scaler_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    def scale(self, bundle):

        n_train, time_steps, n_features = bundle.X_train.shape
        n_test = bundle.X_test.shape[0]

        X_train = bundle.X_train.reshape(
            -1,
            n_features,
        )

        X_test = bundle.X_test.reshape(
            -1,
            n_features,
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train,
        )

        X_test = scaler.transform(
            X_test,
        )

        X_train = X_train.reshape(
            n_train,
            time_steps,
            n_features,
        )

        X_test = X_test.reshape(
            n_test,
            time_steps,
            n_features,
        )

        scaler_file = (
            self.scaler_path
            / f"{bundle.dataset_name}_scaler.pkl"
        )

        joblib.dump(
            scaler,
            scaler_file,
        )
        return X_train, X_test