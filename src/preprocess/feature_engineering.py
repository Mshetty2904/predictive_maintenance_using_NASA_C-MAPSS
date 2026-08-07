from pathlib import Path

import joblib
import numpy as np
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

    # StandardScaler returns floats; convert the source columns first to
    # prevent pandas integer-to-float assignment warnings.
    train = train.astype({column: "float64" for column in sensor_columns})
    test = test.astype({column: "float64" for column in sensor_columns})

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


def add_regime_one_hot_features(train, test):
    """Expose operating-regime identity as model input features."""
    train = train.copy()
    test = test.copy()
    if "Regime_ID" not in train.columns:
        return train, test, []
    regimes = sorted(train["Regime_ID"].dropna().unique())
    columns = []
    for regime in regimes:
        column = f"Regime_OneHot_{int(regime)}"
        columns.append(column)
        train[column] = (train["Regime_ID"] == regime).astype("float64")
        test[column] = (test["Regime_ID"] == regime).astype("float64")
    return train, test, columns


def add_degradation_features(
    train,
    test,
    dataset_name,
    top_n=6,
    window=5,
    features_per_regime=3,
    add_second_derivative=True,
):
    """Add causal, engine-local degradation and acceleration features."""
    train = train.copy()
    test = test.copy()
    sensor_columns = [
        column for column in train.columns
        if column.startswith("Sensor_")
        and not any(token in column for token in ("_diff1", "_diff2", "_roll_mean", "_roll_std"))
    ]

    correlations = {}
    for sensor in sensor_columns:
        correlation = train[[sensor, "RUL"]].corr().iloc[0, 1]
        correlations[sensor] = 0.0 if correlation != correlation else abs(correlation)

    selected = []
    if "Regime_ID" in train.columns:
        for _, regime_data in train.groupby("Regime_ID", sort=True):
            regime_scores = {
                sensor: abs(regime_data[[sensor, "RUL"]].corr().iloc[0, 1])
                if regime_data[[sensor, "RUL"]].corr().iloc[0, 1] == regime_data[[sensor, "RUL"]].corr().iloc[0, 1]
                else 0.0
                for sensor in sensor_columns
            }
            selected.extend(sorted(regime_scores, key=regime_scores.get, reverse=True)[:features_per_regime])
    if not selected:
        selected = sorted(sensor_columns, key=correlations.get, reverse=True)
    selected = list(dict.fromkeys(selected))[:min(top_n, len(sensor_columns))]

    def transform(dataframe):
        result = dataframe.copy()
        result["_feature_order"] = range(len(result))
        result = result.sort_values(
            ["Engine_ID", "Cycle", "_feature_order"]
        ).reset_index(drop=True)
        for sensor in selected:
            grouped = result.groupby("Engine_ID", sort=False)[sensor]
            result[f"{sensor}_diff1"] = grouped.diff().fillna(0.0)
            result[f"{sensor}_roll_mean{window}"] = grouped.transform(
                lambda values: values.rolling(window, min_periods=1).mean()
            )
            result[f"{sensor}_roll_std{window}"] = grouped.transform(
                lambda values: values.rolling(window, min_periods=1).std(ddof=0)
            ).fillna(0.0)
            if add_second_derivative:
                result[f"{sensor}_diff2"] = grouped.diff().groupby(
                    result["Engine_ID"], sort=False
                ).diff().fillna(0.0)
        return result.sort_values("_feature_order").drop(
            columns="_feature_order"
        ).reset_index(drop=True)

    train = transform(train)
    test = transform(test)

    audit_dir = OUTPUT_PATH / "temporal_features"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"{dataset_name}_causal_audit.txt"
    audit_lines = []
    for split_name, original, engineered in (
        ("train", train.copy(), train),
        ("test", test.copy(), test),
    ):
        # Recompute from sorted source rows and compare every generated value.
        # The comparison catches cross-engine rolling and future-looking code.
        ordered = original.sort_values(["Engine_ID", "Cycle"]).reset_index(drop=True)
        for sensor in selected:
            group = ordered.groupby("Engine_ID", sort=False)[sensor]
            expected_diff1 = group.diff().fillna(0.0).to_numpy()
            actual_diff1 = engineered.sort_values(["Engine_ID", "Cycle"])[f"{sensor}_diff1"].to_numpy()
            if not np.allclose(expected_diff1, actual_diff1):
                raise AssertionError(f"Non-causal diff feature detected: {split_name}/{sensor}")
            expected_mean = group.transform(
                lambda values: values.rolling(window, min_periods=1).mean()
            ).to_numpy()
            expected_std = group.transform(
                lambda values: values.rolling(window, min_periods=1).std(ddof=0)
            ).fillna(0.0).to_numpy()
            ordered_engineered = engineered.sort_values(["Engine_ID", "Cycle"])
            if not np.allclose(
                expected_mean,
                ordered_engineered[f"{sensor}_roll_mean{window}"].to_numpy(),
            ) or not np.allclose(
                expected_std,
                ordered_engineered[f"{sensor}_roll_std{window}"].to_numpy(),
            ):
                raise AssertionError(f"Non-causal rolling feature detected: {split_name}/{sensor}")
            if add_second_derivative:
                expected_diff2 = group.diff().groupby(
                    ordered["Engine_ID"], sort=False
                ).diff().fillna(0.0).to_numpy()
                actual_diff2 = engineered.sort_values(["Engine_ID", "Cycle"])[f"{sensor}_diff2"].to_numpy()
                if not np.allclose(expected_diff2, actual_diff2):
                    raise AssertionError(f"Non-causal second derivative detected: {split_name}/{sensor}")
        audit_lines.append(f"{split_name}: PASSED; engine-local, cycle-ordered, causal")
    audit_path.write_text("\n".join(audit_lines), encoding="utf-8")
    print(f"Temporal leakage audit passed and saved to:\n{audit_path}")

    output_dir = OUTPUT_PATH / "temporal_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{dataset_name}_selected_sensors.txt"
    with report_path.open("w", encoding="utf-8") as report:
        report.write(f"Dataset : {dataset_name}\n")
        report.write(f"Rolling window : {window}\n")
        report.write(
            "Features per selected sensor: diff1, rolling mean, rolling std"
        )
        if add_second_derivative:
            report.write(", diff2")
        report.write("\n\n")
        report.write("Selected sensors:\n")
        report.write("\n".join(selected))

    print(f"Temporal degradation features added for: {selected}")
    print(f"Temporal feature report saved to:\n{report_path}")
    return train, test, selected
