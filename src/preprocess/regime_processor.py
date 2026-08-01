"""
Operating Regime Processing

Assign operating regimes using KMeans and
perform regime-wise feature scaling.

Author: Mayur Shetty
"""

from pathlib import Path

import joblib
from sklearn.cluster import KMeans
import pandas as pd

from sklearn.preprocessing import StandardScaler

class RegimeProcessor:
    """
    Assign operating regimes and perform
    regime-wise scaling.
    """

    def __init__(
        self,
        output_dir,
        random_state=42,
    ):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.cluster_model = None
        self.settings_scaler = None

        # One scaler per operating regime
        self.scalers = {}

        # Sensor columns to normalize
        self.sensor_columns = None
    def fit_cluster_model(
        self,
        train_df,
        n_clusters,
    ):
        """
        Train the KMeans model.
        """

        settings = train_df[
            [
                "Setting_1",
                "Setting_2",
                "Setting_3",
            ]
        ]

        self.settings_scaler = StandardScaler()
        scaled_settings = self.settings_scaler.fit_transform(settings)

        self.cluster_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=20,
        )

        self.cluster_model.fit(scaled_settings)
        joblib.dump(
            self.cluster_model,
            self.output_dir / "regime_kmeans.pkl",
        )
        joblib.dump(
            self.settings_scaler,
            self.output_dir / "regime_settings_scaler.pkl",
        )

        return self
    def assign_regimes(
        self,
        dataframe,
    ):
        """
        Assign operating regime IDs.
        """
        if self.cluster_model is None:
            raise ValueError(
                "Cluster model has not been fitted."
            )
        settings = dataframe[
            [
                "Setting_1",
                "Setting_2",
                "Setting_3",
            ]
        ]

        dataframe = dataframe.copy()

        if self.settings_scaler is None:
            raise ValueError("Operating-settings scaler has not been fitted.")

        scaled_settings = self.settings_scaler.transform(settings)
        dataframe["Regime_ID"] = self.cluster_model.predict(scaled_settings)

        return dataframe
    def fit_scalers(
        self,
        train_df,
    ):
        """
        Fit one StandardScaler for each operating regime.
        """

        self.sensor_columns = [
            column
            for column in train_df.columns
            if column.startswith("Sensor_")
        ]
        if len(self.sensor_columns) == 0:
            raise ValueError(
                "No sensor columns found."
            )

        self.scalers = {}

        for regime in sorted(train_df["Regime_ID"].unique()):

            scaler = StandardScaler()

            mask = train_df["Regime_ID"] == regime

            scaler.fit(
                train_df.loc[
                    mask,
                    self.sensor_columns,
                ]
            )

            self.scalers[regime] = scaler

        print(
            f"\nFitted {len(self.scalers)} regime scalers."
        )
        joblib.dump(
            self.scalers,
            self.output_dir / "regime_scalers.pkl",
        )
    def transform(
        self,
        dataframe,
    ):
        """
        Apply regime-wise scaling.
        """
        if self.cluster_model is None:
            raise ValueError(
                "Cluster model has not been fitted."
            )

        if len(self.scalers) == 0:
            raise ValueError(
                "Regime scalers have not been fitted."
            )

        dataframe = dataframe.copy()

        unknown_regimes = set(dataframe["Regime_ID"].unique()) - set(
            self.scalers.keys()
        )
        if unknown_regimes:
            raise ValueError(
                f"Unseen operating regimes encountered: {sorted(unknown_regimes)}"
            )
        dataframe[self.sensor_columns] = dataframe[
            self.sensor_columns
        ].astype("float64")

        for regime, scaler in self.scalers.items():

            mask = dataframe["Regime_ID"] == regime

            dataframe.loc[
                mask,
                self.sensor_columns,
            ] = scaler.transform(
                dataframe.loc[
                    mask,
                    self.sensor_columns,
                ]
            )

        return dataframe
    def load_models(self):
        """
        Load trained KMeans model and regime scalers.
        """

        self.cluster_model = joblib.load(
            self.output_dir / "regime_kmeans.pkl"
        )

        self.settings_scaler = joblib.load(
            self.output_dir / "regime_settings_scaler.pkl"
        )

        self.scalers = joblib.load(
            self.output_dir / "regime_scalers.pkl"
        )

        if len(self.scalers) > 0:
            self.sensor_columns = list(
                self.scalers[
                    next(iter(self.scalers))
                ].feature_names_in_
            )

        print(
            "\nLoaded regime clustering models successfully."
        )
