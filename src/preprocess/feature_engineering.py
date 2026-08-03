from pathlib import Path

import joblib
from sklearn.preprocessing import StandardScaler

from config import OUTPUT_PATH, SCALER_PATH


def remove_constant_sensors(
    train,
    test,
    dataset_name,
):
    """
    Remove sensors with zero variance from the training dataset.

    The removed sensors are also removed from the test dataset
    and saved to a text file.
    """

    train = train.copy()
    test = test.copy()

    sensor_columns = [
        col for col in train.columns
        if col.startswith("Sensor_")
    ]

    constant_sensors = []

    for sensor in sensor_columns:
        if train[sensor].nunique() == 1:
            constant_sensors.append(sensor)

    if constant_sensors:
        train.drop(
            columns=constant_sensors,
            inplace=True,
        )

        test.drop(
            columns=constant_sensors,
            inplace=True,
        )

    output_dir = OUTPUT_PATH / "constant_sensors"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir /
        f"{dataset_name}_constant_sensors.txt"
    )

    with open(output_file, "w") as file:

        file.write(f"Dataset : {dataset_name}\n")
        file.write(
            f"Total Constant Sensors : {len(constant_sensors)}\n\n"
        )

        if constant_sensors:

            for sensor in constant_sensors:
                file.write(f"{sensor}\n")

        else:

            file.write("No constant sensors found.\n")

    print(
        f"\nConstant sensor report saved to:\n{output_file}"
    )

    return train, test, constant_sensors


def remove_low_variance_sensors(
    train,
    test,
    dataset_name,
    threshold=0.001,
):
    """
    Remove sensors whose variance is below the specified threshold.
    Variance is computed only on the training dataset.
    """

    train = train.copy()
    test = test.copy()

    sensor_columns = [
        col for col in train.columns
        if col.startswith("Sensor_")
    ]

    low_variance_sensors = []

    for sensor in sensor_columns:

        if train[sensor].var() < threshold:
            low_variance_sensors.append(sensor)

    if low_variance_sensors:

        train.drop(
            columns=low_variance_sensors,
            inplace=True,
        )

        test.drop(
            columns=low_variance_sensors,
            inplace=True,
        )

    output_dir = OUTPUT_PATH / "low_variance_sensors"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir /
        f"{dataset_name}_low_variance_sensors.txt"
    )

    with open(output_file, "w") as file:

        file.write(f"Dataset : {dataset_name}\n")
        file.write(f"Threshold : {threshold}\n")
        file.write(
            f"Total Sensors Removed : {len(low_variance_sensors)}\n\n"
        )

        if low_variance_sensors:

            for sensor in low_variance_sensors:
                file.write(f"{sensor}\n")

        else:

            file.write("No low variance sensors found.\n")

    print(
        f"\nLow variance sensor report saved to:\n{output_file}"
    )
    return train, test, low_variance_sensors


def zscore_sensor_features(train, test, dataset_name):
    """Fit a training-only global Z-score scaler and apply it to both splits."""
    train = train.copy()
    test = test.copy()
    sensor_columns = [
        column for column in train.columns
        if column.startswith("Sensor_")
    ]
    if not sensor_columns:
        raise ValueError("No sensor columns available for Z-score normalization.")

    scaler = StandardScaler()
    train.loc[:, sensor_columns] = scaler.fit_transform(train[sensor_columns])
    test.loc[:, sensor_columns] = scaler.transform(test[sensor_columns])

    SCALER_PATH.mkdir(parents=True, exist_ok=True)
    scaler_path = SCALER_PATH / f"{dataset_name}_global_zscore.pkl"
    joblib.dump(scaler, scaler_path)

    report_dir = OUTPUT_PATH / "normalization"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{dataset_name}_zscore_report.txt"
    train_means = train[sensor_columns].mean().abs().max()
    train_stds = train[sensor_columns].std(ddof=0)
    max_std_error = (train_stds - 1.0).abs().max()
    with report_path.open("w", encoding="utf-8") as report:
        report.write(f"Dataset : {dataset_name}\n")
        report.write("Method  : StandardScaler fitted on training sensors only\n")
        report.write(f"Sensors : {len(sensor_columns)}\n")
        report.write(f"Max absolute train mean : {train_means:.8f}\n")
        report.write(f"Max train std error     : {max_std_error:.8f}\n")
        report.write(f"Scaler                 : {scaler_path}\n")

    print(f"\nZ-score normalization report saved to:\n{report_path}")
    return train, test, scaler


def add_degradation_features(train, test, dataset_name, top_n=8, window=5):
    """Add causal degradation features for the multi-regime datasets."""
    train = train.copy()
    test = test.copy()
    sensor_columns = [
        column for column in train.columns
        if column.startswith("Sensor_")
        and not any(token in column for token in ("_diff1", "_roll_mean", "_roll_std"))
    ]

    correlations = {}
    for sensor in sensor_columns:
        correlation = train[[sensor, "RUL"]].corr().iloc[0, 1]
        correlations[sensor] = 0.0 if correlation != correlation else abs(correlation)

    selected = sorted(
        sensor_columns,
        key=lambda sensor: correlations[sensor],
        reverse=True,
    )[:min(top_n, len(sensor_columns))]

    def transform(dataframe):
        result = dataframe.copy()
        for sensor in selected:
            grouped = result.groupby("Engine_ID")[sensor]
            result[f"{sensor}_diff1"] = grouped.diff().fillna(0.0)
            result[f"{sensor}_roll_mean{window}"] = grouped.transform(
                lambda values: values.rolling(window, min_periods=1).mean()
            )
            result[f"{sensor}_roll_std{window}"] = grouped.transform(
                lambda values: values.rolling(window, min_periods=1).std(ddof=0)
            ).fillna(0.0)
        return result

    train = transform(train)
    test = transform(test)

    output_dir = OUTPUT_PATH / "temporal_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{dataset_name}_selected_sensors.txt"
    with report_path.open("w", encoding="utf-8") as report:
        report.write(f"Dataset : {dataset_name}\n")
        report.write(f"Rolling window : {window}\n")
        report.write("Features per selected sensor: diff1, rolling mean, rolling std\n\n")
        report.write("Selected sensors:\n")
        report.write("\n".join(selected))

    print(f"Temporal degradation features added for: {selected}")
    print(f"Temporal feature report saved to:\n{report_path}")
    return train, test, selected
