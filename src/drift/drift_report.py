"""Persist batch drift results and generate dissertation-ready plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import MIN_DRIFTED_SENSORS_PER_BATCH


class DriftReport:
    def __init__(self, output_dir, reset=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "drift_results.csv"
        if reset and self.csv_path.exists():
            self.csv_path.unlink()

    def append(self, rows):
        if not rows:
            return
        frame = pd.DataFrame(rows)
        frame.to_csv(self.csv_path, mode="a", header=not self.csv_path.exists(), index=False)

    def generate_plots(self):
        if not self.csv_path.exists():
            return
        frame = pd.read_csv(self.csv_path)
        if frame.empty:
            return
        if "batch_id" in frame and frame["batch_id"].notna().any():
            batch = (
                frame.groupby(["batch_id", "severity"], as_index=False)
                .agg(
                    psi=("psi", "median"),
                    ks_statistic=("ks_statistic", "median"),
                    sensor_checks=("status", "size"),
                    drifted_sensor_checks=("status", lambda values: (values == "drift").sum()),
                    drift_rate=("status", lambda values: (values == "drift").mean()),
                )
            )
            batch["batch"] = pd.factorize(batch["batch_id"])[0]
        else:
            frame["batch"] = frame.groupby(["severity", "engine_id"], dropna=False).ngroup()
            batch = (
                frame.groupby(["batch", "severity"], as_index=False)
                .agg(
                    psi=("psi", "median"),
                    ks_statistic=("ks_statistic", "median"),
                    sensor_checks=("status", "size"),
                    drifted_sensor_checks=("status", lambda values: (values == "drift").sum()),
                    drift_rate=("status", lambda values: (values == "drift").mean()),
                )
            )
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        for metric, filename, threshold in (
            ("psi", "psi_timeline.png", 0.20),
            ("ks_statistic", "ks_statistic_timeline.png", None),
        ):
            plt.figure(figsize=(12, 6))
            sns.lineplot(
                data=batch,
                x="batch",
                y=metric,
                hue="severity",
                legend=(threshold is not None),
            )
            if threshold is not None:
                plt.axhline(threshold, color="red", linestyle="--", label="threshold")
            plt.xlabel("Stream batch")
            plt.ylabel(metric.upper())
            plt.title(f"Drift {metric.upper()} timeline")
            plt.grid(alpha=0.25)
            if threshold is not None:
                plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / filename, dpi=300, bbox_inches="tight")
            plt.close()

        summary = batch.groupby("severity", as_index=False)["drift_rate"].mean()
        sensor_summary = (
            frame.groupby("severity", as_index=False)
            .agg(
                sensor_checks=("status", "size"),
                sensor_drift_checks=("status", lambda values: (values == "drift").sum()),
            )
        )
        sensor_summary["sensor_drift_rate"] = (
            sensor_summary["sensor_drift_checks"] / sensor_summary["sensor_checks"]
        )
        batch_summary = batch.groupby("severity", as_index=False).agg(
            batches=("batch", "nunique"),
            batches_with_drift=("drift_rate", lambda values: (values > 0).sum()),
        )
        batch["meets_sensor_quorum"] = (
            batch["drifted_sensor_checks"] >= MIN_DRIFTED_SENSORS_PER_BATCH
        )
        quorum_summary = batch.groupby("severity", as_index=False)["meets_sensor_quorum"].sum()
        quorum_summary = quorum_summary.rename(
            columns={"meets_sensor_quorum": "batches_meeting_sensor_quorum"}
        )
        batch_summary["batch_drift_rate"] = (
            batch_summary["batches_with_drift"] / batch_summary["batches"]
        )
        sensor_summary.merge(batch_summary, on="severity").merge(quorum_summary, on="severity").to_csv(
            self.output_dir / "detection_summary.csv", index=False
        )
        plt.figure(figsize=(8, 5))
        sns.barplot(data=summary, x="severity", y="drift_rate", order=["normal", "small", "medium", "severe"])
        plt.ylabel("Fraction of sensor checks flagged")
        plt.xlabel("Injected drift severity")
        plt.title("Drift detection sensitivity by severity")
        plt.ylim(0, 1.05)
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_dir / "detection_sensitivity_by_severity.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(12, 5))
        sns.lineplot(data=batch, x="batch", y="drift_rate", hue="severity", marker="o")
        plt.axhline(0.5, color="red", linestyle="--", label="50% sensor checks drifted")
        plt.xlabel("Stream batch")
        plt.ylabel("Drift status rate")
        plt.title("Drift status timeline")
        plt.ylim(-0.05, 1.05)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "drift_status_timeline.png", dpi=300, bbox_inches="tight")
        plt.close()
