import numpy as np


def create_windows(data, window_size=30, step_size=1):
    """
    Create sliding windows for one dataset.

    Returns:
        X : (samples, window_size, features)
        y : (samples,)
    """

    sensor_columns = [
        col for col in data.columns
        if col.startswith("Sensor_")
    ]

    X = []
    y = []

    for _, engine_data in data.groupby("Engine_ID"):

        engine_data = engine_data.reset_index(drop=True)

        for start in range(
            0,
            len(engine_data) - window_size + 1,
            step_size,
        ):

            end = start + window_size

            window = engine_data.loc[
                start:end - 1,
                sensor_columns,
            ].values

            target = engine_data.loc[
                end - 1,
                "RUL",
            ]

            X.append(window)
            y.append(target)

    return np.array(X), np.array(y)


def flatten_windows(X):
    """
    Convert 3D windows into 2D features for XGBoost.
    """

    samples = X.shape[0]

    return X.reshape(samples, -1)