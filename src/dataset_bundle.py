from dataclasses import dataclass

import numpy as np
import pandas as pd
class DatasetBundle:

    def __init__(
        self,
        dataset_name,
        train,
        test,
        rul,
        X_train,
        y_train,
        X_test,
        y_test,
        train_groups,
    ):

        self.dataset_name = dataset_name

        self.train = train
        self.test = test
        self.rul = rul

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.train_groups = train_groups