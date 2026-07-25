from src.data_loader import DataLoader
from src.preprocessing import (
    calculate_rul,
    save_processed_data,
)

from config import (
    DATASETS,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
)


def main():

    print("=" * 60)
    print("NASA C-MAPSS DATA PROCESSING")
    print("=" * 60)

    for dataset in DATASETS:

        print(f"\nProcessing {dataset}...")

        loader = DataLoader(
            dataset,
            RAW_DATA_PATH,
        )

        train, test, rul = loader.load_dataset()

        train = calculate_rul(train)

        save_processed_data(
            train,
            test,
            dataset,
            PROCESSED_DATA_PATH,
        )

        print(f"{dataset} completed.")
        print(f"Train : {train.shape}")
        print(f"Test  : {test.shape}")
        print(f"RUL   : {rul.shape}")

    print("\nAll datasets processed successfully.")


if __name__ == "__main__":
    main()