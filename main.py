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
from src.window_generator import (
    create_windows,
    flatten_windows,
)
from config import (
    DATASETS,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    WINDOW_SIZE,
    STEP_SIZE,
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
        X_train, y_train = create_windows(
            train,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE,
        )

        X_train_xgb = flatten_windows(X_train)

        print(f"Windows : {X_train.shape}")
        print(f"Targets : {y_train.shape}")
        print(f"XGBoost : {X_train_xgb.shape}")

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