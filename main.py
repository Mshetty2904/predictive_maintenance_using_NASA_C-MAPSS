from config import (
    DATASETS,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    WINDOW_SIZE,
    STEP_SIZE,
    OUTPUT_PATH,
    MODEL_PATH,
)

from src.pipeline import TrainingPipeline
from src.trainer import XGBoostTrainer
from src.model_utils import (
    evaluate_model,
    save_metrics,
)
from src.lstm_trainer import LSTMTrainer
from config import SCALER_PATH

def main():

    print("=" * 60)
    print("NASA C-MAPSS DATA PROCESSING")
    print("=" * 60)

    for dataset in DATASETS:

        print(f"\nProcessing {dataset}...")

        pipeline = TrainingPipeline(
            dataset_name=dataset,
            raw_path=RAW_DATA_PATH,
            processed_path=PROCESSED_DATA_PATH,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE,
        )

        bundle = pipeline.run()

        trainer = LSTMTrainer(
            MODEL_PATH,
            SCALER_PATH,
        )

        model, predictions = trainer.train(
            bundle,
        )

        metrics = evaluate_model(
            bundle.y_test,
            predictions,
        )

        save_metrics(
            metrics,
            OUTPUT_PATH,
            bundle.dataset_name,
        )

        print(f"Train Shape      : {bundle.train.shape}")
        print(f"Test Shape       : {bundle.test.shape}")
        print(f"Train Windows    : {bundle.X_train.shape}")
        print(f"Test Windows     : {bundle.X_test.shape}")

        print("\nModel Performance")
        print(metrics)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()