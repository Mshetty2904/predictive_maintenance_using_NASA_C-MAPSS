import numpy as np


def _feature_columns(data):
    """Return the original sensor/regime feature order."""
    columns = []

    if "Regime_ID" in data.columns:
        columns.append("Regime_ID")

    columns.extend(
        column
        for column in data.columns
        if column.startswith("Sensor_")
    )
    return columns


def create_train_windows(data, window_size=30, step_size=1):
    feature_columns = _feature_columns(data)
    X = []
    y = []
    groups = []

    for engine_id, engine in data.groupby("Engine_ID"):
        engine = engine.reset_index(drop=True)

        values = engine[feature_columns].to_numpy()
        if len(values) < window_size:
            padded = np.zeros(
                (window_size, len(feature_columns)),
                dtype=values.dtype,
            )
            padded[-len(values):] = values
            X.append(padded)
            y.append(engine.loc[len(engine) - 1, "RUL"])
            groups.append(engine_id)
            continue

        for start in range(
            0,
            len(engine) - window_size + 1,
            step_size,
        ):
            end = start + window_size
            X.append(engine.loc[start:end - 1, feature_columns].values)
            y.append(engine.loc[end - 1, "RUL"])
            groups.append(engine_id)

    return np.array(X), np.array(y), np.array(groups)


def create_test_windows(test, rul, window_size=30):
    feature_columns = _feature_columns(test)
    X = []
    y = []

    engine_ids = sorted(test["Engine_ID"].unique())
    rul_by_engine = dict(zip(engine_ids, rul["RUL"].to_numpy()))

    for engine_id, engine in test.groupby("Engine_ID", sort=True):
        engine = engine.reset_index(drop=True)
        window = engine[feature_columns].values

        if len(window) < window_size:
            padded = np.zeros(
                (window_size, len(feature_columns)),
                dtype=window.dtype,
            )
            padded[-len(window):] = window
            window = padded
        else:
            window = window[-window_size:]

        X.append(window)

        if engine_id not in rul_by_engine:
            raise KeyError(f"No test RUL value found for engine {engine_id}.")
        y.append(rul_by_engine[engine_id])

    return np.array(X), np.array(y)


def flatten_windows(X):
    return X.reshape(X.shape[0], -1)
