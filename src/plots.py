from pathlib import Path

import matplotlib.pyplot as plt


def save_plot(output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_rul_distribution(train, output_folder):
    max_cycle = (
        train.groupby("Engine_ID")["Cycle"]
        .max()
    )

    plt.figure(figsize=(8, 5))

    plt.hist(max_cycle, bins=20)

    plt.title("Engine Life Distribution")
    plt.xlabel("Maximum Cycle")
    plt.ylabel("Number of Engines")

    save_plot(
        Path(output_folder) /
        "plots" /
        "eda" /
        "engine_life_distribution.png"
    )


def plot_sensor(train, sensor, output_folder):

    engine = train[train["Engine_ID"] == 1]

    plt.figure(figsize=(8, 5))

    plt.plot(engine["Cycle"], engine[sensor])

    plt.title(f"{sensor} Trend (Engine 1)")
    plt.xlabel("Cycle")
    plt.ylabel(sensor)

    save_plot(
        Path(output_folder) /
        "plots" /
        "eda" /
        f"{sensor.lower()}_trend.png"
    )