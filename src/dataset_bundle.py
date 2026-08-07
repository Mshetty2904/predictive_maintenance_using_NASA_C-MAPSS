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
        X_final,
        y_final,
        final_groups,
        final_windows_by_cutoff=None,
        final_targets_by_cutoff=None,
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
        self.X_final = X_final
        self.y_final = y_final
        self.final_groups = final_groups
        self.final_windows_by_cutoff = final_windows_by_cutoff or {}
        self.final_targets_by_cutoff = final_targets_by_cutoff or {}
