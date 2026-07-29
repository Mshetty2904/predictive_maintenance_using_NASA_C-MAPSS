from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class ModelPlots:

    def __init__(self, output_path):

        self.output_path = Path(output_path)
        self.output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    def plot_actual_vs_predicted(
        self,
        y_true,
        y_pred,
        dataset,
        model,
    ):

        plt.figure(figsize=(7, 6))

        plt.scatter(
            y_true,
            y_pred,
            alpha=0.7,
        )

        min_value = min(
            np.min(y_true),
            np.min(y_pred),
        )

        max_value = max(
            np.max(y_true),
            np.max(y_pred),
        )

        plt.plot(
            [min_value, max_value],
            [min_value, max_value],
            "r--",
            linewidth=2,
        )

        plt.xlabel("Actual RUL")
        plt.ylabel("Predicted RUL")

        plt.title(
            f"{dataset} - {model}\nActual vs Predicted"
        )

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            self.output_path
            / f"{dataset}_{model}_actual_vs_predicted.png",
            dpi=300,
        )

        plt.close()

    def plot_residuals(
        self,
        y_true,
        y_pred,
        dataset,
        model,
    ):

        residuals = y_true - y_pred

        plt.figure(figsize=(7, 6))

        plt.scatter(
            y_pred,
            residuals,
            alpha=0.7,
        )

        plt.axhline(
            y=0,
            linestyle="--",
        )

        plt.xlabel("Predicted RUL")
        plt.ylabel("Residual")

        plt.title(
            f"{dataset} - {model}\nResidual Plot"
        )

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            self.output_path
            / f"{dataset}_{model}_residuals.png",
            dpi=300,
        )

        plt.close()

    def plot_residual_histogram(
        self,
        y_true,
        y_pred,
        dataset,
        model,
    ):

        residuals = y_true - y_pred

        plt.figure(figsize=(7, 6))

        plt.hist(
            residuals,
            bins=30,
        )

        plt.xlabel("Residual")

        plt.ylabel("Frequency")

        plt.title(
            f"{dataset} - {model}\nResidual Distribution"
        )

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            self.output_path
            / f"{dataset}_{model}_residual_histogram.png",
            dpi=300,
        )

        plt.close()