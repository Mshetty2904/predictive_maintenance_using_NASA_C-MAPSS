"""Train and evaluate all benchmark models on the same prepared bundles."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from config import (
    DATASETS,
    MODEL_NAMES,
    MODEL_PATH,
    OUTPUT_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SCALER_PATH,
    STEP_SIZE,
    WINDOW_SIZE,
)
from src.cnn_lstm_trainer import CNNLSTMTrainer
from src.lstm_trainer import LSTMTrainer
from src.model_utils import evaluate_model, print_dataset_info, print_final_metrics, save_metrics
from src.pipeline import TrainingPipeline
from src.plots import ModelPlots
from src.xgboost_trainer import XGBoostTrainer


class Tee:
    """Write terminal output and log output at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def build_trainer(model_name):
    if model_name == "xgboost":
        return XGBoostTrainer(model_path=MODEL_PATH)
    if model_name == "lstm":
        return LSTMTrainer(model_path=MODEL_PATH, scaler_path=SCALER_PATH)
    if model_name == "cnn_lstm":
        return CNNLSTMTrainer(model_path=MODEL_PATH, scaler_path=SCALER_PATH)
    raise ValueError(f"Unknown model name: {model_name}")


def main():
    seed_everything(RANDOM_STATE)
    plotter = ModelPlots(OUTPUT_PATH / "plots")
    print("\nNASA C-MAPSS Predictive Maintenance")
    print(f"Models: {', '.join(MODEL_NAMES)}")

    for dataset in DATASETS:
        pipeline = TrainingPipeline(
            dataset_name=dataset,
            raw_path=RAW_DATA_PATH,
            processed_path=PROCESSED_DATA_PATH,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE,
        )
        bundle = pipeline.run()

        for model_name in MODEL_NAMES:
            print_dataset_info(bundle, model_name)
            trainer = build_trainer(model_name)
            _, predictions = trainer.train(bundle)
            metrics = evaluate_model(bundle.y_test, predictions)
            plotter.plot_actual_vs_predicted(
                bundle.y_test, predictions, bundle.dataset_name, model_name
            )
            plotter.plot_residuals(
                bundle.y_test, predictions, bundle.dataset_name, model_name
            )
            plotter.plot_residual_histogram(
                bundle.y_test, predictions, bundle.dataset_name, model_name
            )
            save_metrics(metrics, OUTPUT_PATH, bundle.dataset_name, model_name)
            print_final_metrics(metrics)

    print("\nAll datasets and models completed successfully.")


if __name__ == "__main__":
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    start_time = time.perf_counter()

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)

        exit_code = 0
        try:
            main()
        except Exception:
            # Capture the complete traceback in the log before exiting.
            traceback.print_exc()
            exit_code = 1
        finally:
            elapsed = int(time.perf_counter() - start_time)
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)

            print(f"Total execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")
            print(f"Log file saved to: {log_path}")

            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    raise SystemExit(exit_code)
