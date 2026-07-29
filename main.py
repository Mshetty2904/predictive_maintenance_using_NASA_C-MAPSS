from config import (
    DATASETS,
    MODEL_PATH,
    OUTPUT_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    SCALER_PATH,
    STEP_SIZE,
    WINDOW_SIZE,
)
import os
from src.cnn_lstm_trainer import CNNLSTMTrainer
from src.model_utils import (
    evaluate_model,
    print_dataset_info,
    print_final_metrics,
    save_metrics,
)
from src.pipeline import TrainingPipeline
from src.plots import ModelPlots

MODEL_NAME = "cnn_lstm"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
def main():
    print("\nNASA C-MAPSS Predictive Maintenance")
    print(f"Model: {MODEL_NAME}")

    for dataset in DATASETS:

        pipeline = TrainingPipeline(
            dataset_name=dataset,
            raw_path=RAW_DATA_PATH,
            processed_path=PROCESSED_DATA_PATH,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE,
        )

        bundle = pipeline.run()

        print_dataset_info(bundle, MODEL_NAME)

        trainer = CNNLSTMTrainer(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
        )

        _, predictions = trainer.train(bundle)

        metrics = evaluate_model(
            bundle.y_test,
            predictions,
        )
        plotter = ModelPlots(
            OUTPUT_PATH / "plots"
        )

        plotter.plot_actual_vs_predicted(
            bundle.y_test,
            predictions,
            bundle.dataset_name,
            MODEL_NAME,
        )

        plotter.plot_residuals(
            bundle.y_test,
            predictions,
            bundle.dataset_name,
            MODEL_NAME,
        )

        plotter.plot_residual_histogram(
            bundle.y_test,
            predictions,
            bundle.dataset_name,
            MODEL_NAME,
        )

        save_metrics(
            metrics,
            OUTPUT_PATH,
            bundle.dataset_name,
            MODEL_NAME,
        )

        print_final_metrics(metrics)

    print("\nAll datasets completed successfully.")


if __name__ == "__main__":
    main()