from pathlib import Path

import joblib
from sklearn.preprocessing import StandardScaler


class FeatureScaler:

    def __init__(self, scaler_path):

        self.scaler_path = Path(scaler_path)
        self.scaler_path.mkdir(parents=True, exist_ok=True)

    def fit_transform_pair(self, train_data, other_data):

        train_samples, time_steps, features = train_data.shape
        other_samples, _, _ = other_data.shape

        train_2d = train_data.reshape(-1, features)
        other_2d = other_data.reshape(-1, features)

        scaler = StandardScaler()

        train_scaled = scaler.fit_transform(train_2d)
        other_scaled = scaler.transform(other_2d)

        train_scaled = train_scaled.reshape(
            train_samples,
            time_steps,
            features,
        )

        other_scaled = other_scaled.reshape(
            other_samples,
            time_steps,
            features,
        )

        return train_scaled, other_scaled, scaler

    def scale_final_data(self, bundle):

        X_train, X_test, scaler = self.fit_transform_pair(
            bundle.X_train,
            bundle.X_test,
        )

        scaler_file = (
            self.scaler_path
            / f"{bundle.dataset_name}_scaler.pkl"
        )

        joblib.dump(scaler, scaler_file)

        return X_train, X_test, scaler