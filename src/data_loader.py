from pathlib import Path

import pandas as pd


class DataLoader:
    def __init__(self, dataset_name, raw_data_path):
        self.dataset_name = dataset_name
        self.dataset_path = Path(raw_data_path) / dataset_name

        self.columns = (
            ["Engine_ID", "Cycle"]
            + [f"Setting_{i}" for i in range(1, 4)]
            + [f"Sensor_{i}" for i in range(1, 22)]
        )

    def _read_file(self, file_name):
        file_path = self.dataset_path / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found.")

        return pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            names=self.columns,
        )

    def load_dataset(self):
        train = self._read_file(f"train_{self.dataset_name}.txt")
        test = self._read_file(f"test_{self.dataset_name}.txt")

        rul = pd.read_csv(
            self.dataset_path / f"RUL_{self.dataset_name}.txt",
            header=None,
            names=["RUL"],
        )

        return train, test, rul

    @staticmethod
    def dataset_summary(train, test):
        summary = pd.DataFrame(
            {
                "Dataset": ["Train", "Test"],
                "Rows": [len(train), len(test)],
                "Columns": [train.shape[1], test.shape[1]],
                "Engines": [
                    train["Engine_ID"].nunique(),
                    test["Engine_ID"].nunique(),
                ],
            }
        )

        return summary