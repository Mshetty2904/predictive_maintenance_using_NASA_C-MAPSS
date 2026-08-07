"""Run the local live-stream drift experiment.

Example:
    python drift_monitor.py --dataset FD001 --severity all --noise-std 0.02
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from config import KS_PVALUE, PSI_THRESHOLD, PROCESSED_DATA_PATH, RANDOM_STATE
from src.drift import DriftDetector, DriftGenerator, EngineStream
from src.drift.drift_report import DriftReport
from src.drift.retraining import ConsecutiveDriftGate


def sensor_columns(frame):
    return [column for column in frame.columns if column.startswith("Sensor_") and "_diff" not in column and "_roll_" not in column]


def main():
    parser = argparse.ArgumentParser(description="Replay NASA C-MAPSS telemetry and detect injected drift")
    parser.add_argument("--dataset", choices=["FD001", "FD002", "FD003", "FD004"], required=True)
    parser.add_argument("--severity", choices=["normal", "small", "medium", "severe", "all"], default="all")
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--max-engines", type=int, default=None)
    args = parser.parse_args()

    train_path = PROCESSED_DATA_PATH / f"{args.dataset}_train.csv"
    test_path = PROCESSED_DATA_PATH / f"{args.dataset}_test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sensors = sensor_columns(train)
    output = Path("outputs") / "reports" / "drift" / args.dataset
    detector = DriftDetector(output / "reference_stats.json", PSI_THRESHOLD, KS_PVALUE)
    detector.fit_reference(train, sensors)
    report = DriftReport(output, reset=True)
    trigger_path = output / "retraining_triggers.jsonl"
    if trigger_path.exists():
        trigger_path.unlink()
    skipped_batches = []
    severities = ["normal", "small", "medium", "severe"] if args.severity == "all" else [args.severity]
    for severity in severities:
        gate = ConsecutiveDriftGate()
        generator = DriftGenerator(severity, args.noise_std if severity != "normal" else 0.0, RANDOM_STATE)
        for index, (engine_id, batch) in enumerate(EngineStream(test)):
            if args.max_engines is not None and index >= args.max_engines:
                break
            drifted = generator.transform(batch, sensors)
            batch_id = f"{severity}_{index:04d}"
            rows = detector.detect(drifted, sensors, engine_id, severity, batch_id=batch_id)
            if not rows:
                skipped_batches.append({"dataset": args.dataset, "severity": severity, "engine_id": engine_id, "reason": "batch below minimum sample size"})
            report.append(rows)
            decision = gate.update(rows)
            decision.update({"dataset": args.dataset, "engine_id": engine_id, "severity": severity, "batch_id": batch_id})
            with trigger_path.open("a", encoding="utf-8") as trigger_file:
                trigger_file.write(json.dumps(decision) + "\n")
    report.generate_plots()
    pd.DataFrame(skipped_batches).to_csv(output / "skipped_batches.csv", index=False)
    print(f"Drift reference saved to: {output / 'reference_stats.json'}")
    print(f"Drift report saved to: {output / 'drift_results.csv'}")
    print(f"Drift plots saved to: {output / 'plots'}")
    print(f"Retraining decisions saved to: {trigger_path}")


if __name__ == "__main__":
    main()
