from pathlib import Path

from config import DATASET, OUTPUT_PATH, RAW_DATA_PATH
from src.data_loader import DataLoader
from src.plots import plot_rul_distribution, plot_sensor
from src.validation import validate_dataset
from src.analysis import (
    sensor_statistics,
    constant_sensors,
    low_variance_sensors,
    correlation_heatmap,
)
from src.preprocessing import (
    calculate_rul,
    remove_unused_columns,
    save_processed_data,
)

def main():

    loader = DataLoader(DATASET, RAW_DATA_PATH)

    train, test, rul = loader.load_dataset()

    summary = loader.dataset_summary(train, test)

    validation = validate_dataset(train, test)

    summary.to_csv(
        Path(OUTPUT_PATH) / "tables" / "dataset_summary.csv",
        index=False
    )

    validation.to_csv(
        Path(OUTPUT_PATH) / "tables" / "dataset_validation.csv",
        index=False
    )

    plot_rul_distribution(train, OUTPUT_PATH)

    plot_sensor(train, "Sensor_2", OUTPUT_PATH)

    print(summary)
    print()
    print(validation)

    print("\nOutputs saved successfully.")

    sensor_statistics(train, OUTPUT_PATH)

    constant_sensors(train, OUTPUT_PATH)

    low_variance_sensors(train, OUTPUT_PATH)

    correlation_heatmap(train, OUTPUT_PATH)

    train = calculate_rul(train)

    train = remove_unused_columns(train)
    test = remove_unused_columns(test)

    save_processed_data(
        train,
        test,
        OUTPUT_PATH,
        DATASET,
    )

    print("\nProcessed datasets saved.")
if __name__ == "__main__":
    main()