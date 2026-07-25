from pathlib import Path
import joblib

from sklearn.preprocessing import StandardScaler


def scale_data(train, test):

    scaler = StandardScaler()

    sensor_columns = [
        col for col in train.columns
        if col.startswith("Sensor_")
    ]

    train[sensor_columns] = scaler.fit_transform(
        train[sensor_columns]
    )

    test[sensor_columns] = scaler.transform(
        test[sensor_columns]
    )

    return train, test, scaler


def save_scaler(scaler, model_path):

    model_path = Path(model_path)

    model_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        scaler,
        model_path / "standard_scaler.pkl"
    )