from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_table(df, output_path, file_name):
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / file_name, index=False)


def sensor_statistics(train, output_path):

    sensors = train.filter(like="Sensor_")

    stats = sensors.describe().T[
        ["mean", "std", "min", "max"]
    ].round(3)

    stats.reset_index(inplace=True)

    stats.rename(columns={"index": "Sensor"}, inplace=True)

    save_table(
        stats,
        Path(output_path) / "tables",
        "sensor_statistics.csv",
    )


def constant_sensors(train, output_path):

    sensors = train.filter(like="Sensor_")

    constant = (
        sensors.nunique() == 1
    )

    result = pd.DataFrame({
        "Sensor": constant.index,
        "Constant": constant.values
    })

    save_table(
        result,
        Path(output_path) / "tables",
        "constant_sensors.csv",
    )


def low_variance_sensors(train, output_path, threshold=0.01):

    sensors = train.filter(like="Sensor_")

    variance = sensors.var()

    result = pd.DataFrame({
        "Sensor": variance.index,
        "Variance": variance.values
    })

    result["Low Variance"] = result["Variance"] < threshold

    save_table(
        result.round(5),
        Path(output_path) / "tables",
        "low_variance_sensors.csv",
    )


def correlation_heatmap(train, output_path):

    sensors = train.filter(like="Sensor_")

    corr = sensors.corr()

    plt.figure(figsize=(12, 10))

    plt.imshow(corr, cmap="coolwarm", aspect="auto")

    plt.colorbar()

    plt.xticks(range(len(corr)), corr.columns, rotation=90, fontsize=6)

    plt.yticks(range(len(corr)), corr.columns, fontsize=6)

    plt.title("Sensor Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        Path(output_path)
        / "plots"
        / "eda"
        / "correlation_heatmap.png",
        dpi=300,
    )

    plt.close()