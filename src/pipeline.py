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

        train = calculate_rul(train)

        save_processed_data(
            train,
            test,
            self.dataset_name,
            self.processed_path,
        )

        X_train, y_train = create_train_windows(
            train,
            self.window_size,
            self.step_size,
        )

        X_test, y_test = create_test_windows(
            test,
            rul,
            self.window_size,
        )

        return DatasetBundle(
            dataset_name=self.dataset_name,

            train=train,
            test=test,
            rul=rul,

            X_train=X_train,
            y_train=y_train,

            X_test=X_test,
            y_test=y_test,
        )