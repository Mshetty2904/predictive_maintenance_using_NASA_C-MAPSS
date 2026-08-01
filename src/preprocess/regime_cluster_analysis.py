"""
Operating Regime Cluster Analysis

Finds the optimal number of operating-condition clusters
for the NASA C-MAPSS datasets.

Author: Mayur Shetty
"""
from sklearn.preprocessing import StandardScaler
from pathlib import Path

import json

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


class RegimeClusterAnalysis:
    """
    Analyze operating regimes using KMeans clustering.
    """

    def __init__(
        self,
        output_dir,
        random_state=42,
    ):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

    def analyze(
        self,
        train_df,
        dataset_name,
        min_clusters=2,
        max_clusters=10,
    ):
        """
        Analyze the optimal number of operating-condition clusters.

        Parameters
        ----------
        train_df : pandas.DataFrame
            Training dataframe.

        dataset_name : str
            Dataset name (FD002 / FD004).

        min_clusters : int
            Minimum number of clusters.

        max_clusters : int
            Maximum number of clusters.

        Returns
        -------
        int
            Recommended number of clusters.
        """

        # ----------------------------------
        # Validate input dataset
        # ----------------------------------

        required_columns = [
            "Setting_1",
            "Setting_2",
            "Setting_3",
        ]

        missing = [
            col
            for col in required_columns
            if col not in train_df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing operational setting columns: {missing}"
            )
                # ----------------------------------
        # Check operating conditions
        # ----------------------------------

        if not self._has_multiple_operating_conditions(
            train_df,
        ):

            print(
                "\nOnly one operating condition detected."
            )

            print(
                "Skipping regime clustering."
            )

            return 1
                # ----------------------------------
        # Extract and Standardize Operational Settings
        # ----------------------------------

        X = train_df[required_columns].copy()

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        print("\n========================================")
        print(f"Operating Regime Analysis : {dataset_name}")
        print("========================================")

        print(f"Samples              : {len(X)}")
        print(f"Features             : {len(required_columns)}")
        print(f"Cluster Range        : {min_clusters} - {max_clusters}")

        # ----------------------------------
        # Storage for evaluation metrics
        # ----------------------------------

        cluster_range = range(
            min_clusters,
            max_clusters + 1,
        )

        inertia_scores = []

        silhouette_scores = []

        db_scores = []

        ch_scores = []

        print("\nEvaluating candidate clusters...\n")
                # ----------------------------------
        # Evaluate candidate clusters
        # ----------------------------------

        for k in cluster_range:

            print(f"Evaluating K = {k}")

            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=20,
            )

            labels = model.fit_predict(X_scaled)

            # Inertia (Within Cluster Sum of Squares)
            inertia = model.inertia_

            # Silhouette Score
            silhouette = silhouette_score(
                X_scaled,
                labels,
            )

            # Davies-Bouldin Index
            db_index = davies_bouldin_score(
                X_scaled,
                labels,
            )

            # Calinski-Harabasz Score
            ch_score = calinski_harabasz_score(
                X_scaled,
                labels,
            )

            inertia_scores.append(inertia)
            silhouette_scores.append(silhouette)
            db_scores.append(db_index)
            ch_scores.append(ch_score)

        # ----------------------------------
        # Store results
        # ----------------------------------

        results = pd.DataFrame(
            {
                "Clusters": list(cluster_range),
                "Inertia": inertia_scores,
                "Silhouette": silhouette_scores,
                "Davies_Bouldin": db_scores,
                "Calinski_Harabasz": ch_scores,
            }
        )

        print("\n========================================")
        print("Cluster Evaluation Metrics")
        print("========================================")

        print(results.round(4))
                # ----------------------------------
        # Save Cluster Evaluation Metrics
        # ----------------------------------

        csv_path = (
            self.output_dir /
            f"{dataset_name}_cluster_metrics.csv"
        )

        results.to_csv(
            csv_path,
            index=False,
        )

        print(
            f"\nCluster metrics saved to:\n{csv_path}"
        )

        # ----------------------------------
        # Find Best K using Individual Metrics
        # ----------------------------------

        best_silhouette = results.loc[
            results["Silhouette"].idxmax(),
            "Clusters",
        ]

        best_db = results.loc[
            results["Davies_Bouldin"].idxmin(),
            "Clusters",
        ]

        best_ch = results.loc[
            results["Calinski_Harabasz"].idxmax(),
            "Clusters",
        ]

        print("\n========================================")
        print("Best Cluster Count by Metric")
        print("========================================")

        print(f"Silhouette Score     : {best_silhouette}")
        print(f"Davies-Bouldin Index : {best_db}")
        print(f"Calinski-Harabasz    : {best_ch}")
                # ----------------------------------
        # Estimate Elbow Point
        # ----------------------------------

        inertia_diff = [
            inertia_scores[i] - inertia_scores[i + 1]
            for i in range(len(inertia_scores) - 1)
        ]

        best_elbow = (
            cluster_range[
                inertia_diff.index(max(inertia_diff))
            ]
        )

        print(f"Elbow Method         : {best_elbow}")

        # ----------------------------------
        # Majority Voting
        # ----------------------------------

        votes = {}

        for k in (
            best_silhouette,
            best_db,
            best_ch,
            best_elbow,
        ):
            votes[k] = votes.get(k, 0) + 1

        recommended_k = max(
            votes,
            key=votes.get,
        )

        print("\n========================================")
        print("Final Recommendation")
        print("========================================")

        print(f"Recommended K : {recommended_k}")

        print("\nVote Summary")

        for k in sorted(votes):
            print(f"K = {k} : {votes[k]} vote(s)")
                # ----------------------------------
        # Save Recommendation
        # ----------------------------------

        recommendation = {
            "dataset": dataset_name,
            "recommended_k": int(recommended_k),
            "silhouette": int(best_silhouette),
            "davies_bouldin": int(best_db),
            "calinski_harabasz": int(best_ch),
            "elbow": int(best_elbow),
        }

        json_path = (
            self.output_dir /
            f"{dataset_name}_best_k.json"
        )

        with open(
            json_path,
            "w",
        ) as file:
            json.dump(
                recommendation,
                file,
                indent=4,
            )

        print(
            f"\nRecommendation saved to:\n{json_path}"
        )
                # ----------------------------------
        # Save Evaluation Plots
        # ----------------------------------

        self._save_metric_plot(
            cluster_range,
            inertia_scores,
            "Number of Clusters",
            "Inertia",
            f"{dataset_name} - Elbow Curve",
            f"{dataset_name}_elbow.png",
            recommended_k,
        )

        self._save_metric_plot(
            cluster_range,
            silhouette_scores,
            "Number of Clusters",
            "Silhouette Score",
            f"{dataset_name} - Silhouette Score",
            f"{dataset_name}_silhouette.png",
            recommended_k,
        )

        self._save_metric_plot(
            cluster_range,
            db_scores,
            "Number of Clusters",
            "Davies-Bouldin Index",
            f"{dataset_name} - Davies-Bouldin Index",
            f"{dataset_name}_davies_bouldin.png",
            recommended_k,
        )

        self._save_metric_plot(
            cluster_range,
            ch_scores,
            "Number of Clusters",
            "Calinski-Harabasz Score",
            f"{dataset_name} - Calinski-Harabasz Score",
            f"{dataset_name}_calinski_harabasz.png",
            recommended_k,
        )

        print("\nEvaluation plots saved successfully.")
        return recommended_k
    def _save_pca_plot(
        self,
        X,
        labels,
        dataset_name,
    ):
        """
        Save PCA visualization of the operating regimes.
        """

        pca = PCA(
            n_components=2,
            random_state=self.random_state,
        )

        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(8, 6))

        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=labels,
            s=8,
            alpha=0.7,
        )

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title(f"{dataset_name} Operating Regimes")

        plt.colorbar(
            scatter,
            label="Cluster",
        )

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            self.output_dir /
            f"{dataset_name}_pca_clusters.png",
            dpi=300,
        )

        plt.close()
                # ----------------------------------
        # Fit Final Model with Recommended K
        # ----------------------------------

        final_model = KMeans(
            n_clusters=recommended_k,
            random_state=self.random_state,
            n_init=20,
        )

        final_labels = final_model.fit_predict(X_scaled)

        self._save_pca_plot(
            X_scaled,
            final_labels,
            dataset_name,
        )

        print("PCA visualization saved successfully.")
    def _save_metric_plot(
        self,
        x,
        y,
        xlabel,
        ylabel,
        title,
        filename,
        best_k=None,
    ):
        """
        Save a line plot for a clustering evaluation metric.
        """

        plt.figure(figsize=(8, 5))

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2,
        )

        if best_k is not None:

            index = list(x).index(best_k)

            plt.scatter(
                best_k,
                y[index],
                s=100,
                marker="*",
                label=f"Recommended K = {best_k}",
            )

            plt.legend()

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            self.output_dir / filename,
            dpi=300,
        )

        plt.close()
    def _has_multiple_operating_conditions(
        self,
        train_df,
    ):
        """
        Check whether the dataset contains multiple
        operating conditions.
        """

        settings = train_df[
            [
                "Setting_1",
                "Setting_2",
                "Setting_3",
            ]
        ]

        unique_conditions = (
            settings
            .drop_duplicates()
            .shape[0]
        )

        print(
            f"\nUnique operating conditions: {unique_conditions}"
        )

        return unique_conditions > 1