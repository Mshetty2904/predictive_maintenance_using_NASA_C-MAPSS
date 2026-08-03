
from src.preprocess.regime_cluster_analysis import (
    RegimeClusterAnalysis,
)

from src.preprocess.regime_processor import (
    RegimeProcessor,
)
from src.data_loader import DataLoader
from src.preprocessing import (
    calculate_rul,
    save_processed_data,
)
from src.window_generator import (
    create_train_windows,
    create_test_windows,
)
from src.dataset_bundle import DatasetBundle
from src.preprocess.feature_engineering import (
    add_degradation_features,
    remove_constant_sensors,
    remove_low_variance_sensors,
    zscore_sensor_features,
)
from config import (
    RANDOM_STATE,
    REGIME_DATASETS,
    USE_REGIME_CLUSTERING,
)


class TrainingPipeline:
    """
    Runs the complete data preparation pipeline
    for one NASA C-MAPSS dataset.
    """

    def __init__(
        self,
        dataset_name,
        raw_path,
        processed_path,
        window_size,
        step_size,
    ):

        self.dataset_name = dataset_name
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.window_size = window_size
        self.step_size = step_size

    def run(self):

        loader = DataLoader(
            self.dataset_name,
            self.raw_path,
        )

        train, test, rul = loader.load_dataset()

        # Calculate Remaining Useful Life (RUL)
        train = calculate_rul(train)

        # Remove constant sensors
        train, test, removed_sensors = remove_constant_sensors(
            train=train,
            test=test,
            dataset_name=self.dataset_name,
        )

        # Remove low variance sensors
        train, test, removed_low_variance = remove_low_variance_sensors(
            train=train,
            test=test,
            dataset_name=self.dataset_name,
            threshold=0.001,
        )
        # -----------------------------------
        # Analyze operating regimes
        # (FD002 & FD004 only)
        # -----------------------------------

        if USE_REGIME_CLUSTERING and self.dataset_name in REGIME_DATASETS:

        #     # ----------------------------------
        #     # Find optimal number of regimes
        #     # ----------------------------------

            analysis = RegimeClusterAnalysis(
                output_dir=self.processed_path / "regime_analysis" / self.dataset_name,
                random_state=RANDOM_STATE,
            )

            best_k = analysis.analyze(
                train_df=train,
                dataset_name=self.dataset_name,
            )

            print(f"\nOptimal Clusters : {best_k}")

        #     # ----------------------------------
        #     # Train regime processor
        #     # ----------------------------------

            processor = RegimeProcessor(
                output_dir=self.processed_path / "regime_models" / self.dataset_name,
                random_state=RANDOM_STATE,
            )

            processor.fit_cluster_model(
                train_df=train,
                n_clusters=best_k,
            )

            train = processor.assign_regimes(train)
            test = processor.assign_regimes(test)

            processor.fit_scalers(train)
            print(
                "\nOperating regime preprocessing completed."
            )
            train = processor.transform(train)
            test = processor.transform(test)
            print("\nTrain Regime Distribution")
            print(train["Regime_ID"].value_counts().sort_index())

            print("\nTest Regime Distribution")
            print(test["Regime_ID"].value_counts().sort_index())

        # All datasets receive a train-fitted global Z-score pass. For
        # FD002/FD004 this follows the regime-wise sensor normalization.
        train, test, _ = zscore_sensor_features(
            train,
            test,
            self.dataset_name,
        )

        if self.dataset_name in REGIME_DATASETS:
            train, test, _ = add_degradation_features(
                train,
                test,
                self.dataset_name,
            )
        # Save processed datasets
        save_processed_data(
            train,
            test,
            self.dataset_name,
            self.processed_path,
        )

        # Create training windows
        X_train, y_train, train_groups = create_train_windows(
            train,
            self.window_size,
            self.step_size,
        )

        # Create test windows
        X_test, y_test = create_test_windows(
            test,
            rul,
            self.window_size,
        )

        # Return dataset bundle
        return DatasetBundle(
            dataset_name=self.dataset_name,
            train=train,
            test=test,
            rul=rul,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            train_groups=train_groups,
        )
