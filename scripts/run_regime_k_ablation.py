"""Compare K=2 and K=6 using the same XGBoost final-window protocol."""

import pandas as pd

from config import MODEL_PATH, OUTPUT_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH, STEP_SIZE, WINDOW_SIZE
from src.model_utils import evaluate_model
from src.pipeline import TrainingPipeline
from src.xgboost_trainer import XGBoostTrainer


def main():
    rows = []
    for dataset in ("FD002", "FD004"):
        for k in (2, 6):
            print(f"\nRegime ablation: {dataset} | K={k}")
            bundle = TrainingPipeline(
                dataset_name=dataset,
                raw_path=RAW_DATA_PATH,
                processed_path=PROCESSED_DATA_PATH,
                window_size=WINDOW_SIZE,
                step_size=STEP_SIZE,
                regime_k_override=k,
            ).run()
            trainer = XGBoostTrainer(MODEL_PATH)
            _, predictions = trainer.train(bundle)
            metrics = evaluate_model(bundle.y_test, predictions).iloc[0]
            rows.append({"Dataset": dataset, "K": k, **metrics.to_dict()})
    output_dir = OUTPUT_PATH / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "regime_k_ablation.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nRegime K ablation saved to: {path}")


if __name__ == "__main__":
    main()
