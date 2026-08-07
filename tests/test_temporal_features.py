import numpy as np
import pandas as pd

from src.preprocess.feature_engineering import add_degradation_features


def test_temporal_features_are_engine_local_and_causal():
    data = pd.DataFrame(
        {
            "Engine_ID": [2, 1, 2, 1, 2, 1],
            "Cycle": [2, 1, 1, 3, 3, 2],
            "Sensor_1": [20.0, 10.0, 15.0, 30.0, 25.0, 20.0],
            "RUL": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    transformed, _, _ = add_degradation_features(
        data, data, "TEST", top_n=1, window=2, add_second_derivative=True
    )
    ordered = transformed.sort_values(["Engine_ID", "Cycle"])
    first_rows = ordered.groupby("Engine_ID", sort=False).head(1)
    assert np.allclose(first_rows["Sensor_1_diff1"], 0.0)
    assert np.allclose(first_rows["Sensor_1_diff2"], 0.0)
