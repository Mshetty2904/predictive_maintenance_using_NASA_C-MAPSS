import numpy as np

def create_train_windows(
    data,
    window_size=30,
    step_size=1,
):

    sensor_columns = [
        col for col in data.columns
        if col.startswith("Sensor_")
    ]

    X = []
    y = []
    groups = []

    for engine_id, engine in data.groupby("Engine_ID"):

        engine = engine.reset_index(drop=True)

        for start in range(
            0,
            len(engine) - window_size + 1,
            step_size,
        ):

            end = start + window_size

            X.append(
                engine.loc[
                    start:end - 1,
                    sensor_columns,
                ].values
            )

            y.append(
                engine.loc[
                    end - 1,
                    "RUL",
                ]
            )

            # Store engine id for GroupKFold
            groups.append(engine_id)

    return (
        np.array(X),
        np.array(y),
        np.array(groups),
    )

def create_test_windows(test, rul, window_size=30):

    sensor_columns = [
        col for col in test.columns
        if col.startswith("Sensor_")
    ]

    X = []
    y = []

    for i, (_, engine) in enumerate(test.groupby("Engine_ID")):

        engine = engine.reset_index(drop=True)

        window = engine[sensor_columns].values

        # Pad short sequences
        if len(window) < window_size:

            pad_rows = np.repeat(
                window[[0]],
                window_size - len(window),
                axis=0,
            )

            window = np.vstack((pad_rows, window))

        else:

            window = window[-window_size:]

        X.append(window)

        y.append(rul.iloc[i, 0])

    return np.array(X), np.array(y)
def flatten_windows(X):

    return X.reshape(X.shape[0], -1)