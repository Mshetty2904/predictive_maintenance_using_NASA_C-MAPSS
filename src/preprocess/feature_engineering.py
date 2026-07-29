from pathlib import Path

from config import OUTPUT_PATH


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