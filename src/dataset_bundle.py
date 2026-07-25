from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    """
    Stores all prepared data for one dataset.
    """

    dataset_name: str

    train: pd.DataFrame
    test: pd.DataFrame
    rul: pd.DataFrame

    X_train: np.ndarray
    y_train: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray