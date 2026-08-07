from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureScaler:

    def __init__(self, scaler_path):
        self.scaler_path = Path(scaler_path)
        self.scaler_path.mkdir(parents=True, exist_ok=True)

    def fit_transform_pair(
        self,
        train_data,
        other_data,
        has_regime=False,
    ):
        train_samples, time_steps, features = train_data.shape
        other_samples, _, _ = other_data.shape

        train_2d = train_data.reshape(-1, features)
        other_2d = other_data.reshape(-1, features)
        feature_start = 1 if has_regime else 0

        scaler = StandardScaler()
        train_scaled = train_2d.copy()
        other_scaled = other_2d.copy()

        train_scaled[:, feature_start:] = scaler.fit_transform(
            train_2d[:, feature_start:]
        )
        other_scaled[:, feature_start:] = scaler.transform(
            other_2d[:, feature_start:]
        )

        return (
            train_scaled.reshape(train_samples, time_steps, features),
            other_scaled.reshape(other_samples, time_steps, features),
            scaler,
        )

    def scale_final_data(self, bundle, train_idx=None, valid_idx=None):
        has_regime = "Regime_ID" in bundle.train.columns
        train_data = (
            bundle.X_train
            if train_idx is None
            else bundle.X_train[train_idx]
        )

        X_train, X_test, scaler = self.fit_transform_pair(
            train_data,
            bundle.X_test,
            has_regime=has_regime,
        )

        joblib.dump(
            scaler,
            self.scaler_path / f"{bundle.dataset_name}_scaler.pkl",
        )

        if valid_idx is None:
            return X_train, X_test, scaler

        _, X_valid, _ = self.fit_transform_pair(
            train_data,
            bundle.X_train[valid_idx],
            has_regime=has_regime,
        )
        return X_train, X_valid, X_test, scaler

    def fit_transform_many(self, train_data, *other_data, has_regime=False):
        """Fit one scaler on training windows and transform other arrays."""
        samples, steps, features = train_data.shape
        train_2d = train_data.reshape(-1, features).astype(np.float64)
        start = 1 if has_regime else 0
        scaler = StandardScaler()
        train_scaled = train_2d.copy()
        train_scaled[:, start:] = scaler.fit_transform(train_2d[:, start:])
        transformed = []
        for data in other_data:
            n = data.shape[0]
            data_2d = data.reshape(-1, features).astype(np.float64)
            data_scaled = data_2d.copy()
            data_scaled[:, start:] = scaler.transform(data_2d[:, start:])
            transformed.append(data_scaled.reshape(n, steps, features))
        return (train_scaled.reshape(samples, steps, features), *transformed, scaler)
