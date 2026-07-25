from pathlib import Path

import pandas as pd


def calculate_rul(train):
    """Calculate Remaining Useful Life (RUL) for each engine."""

    max_cycle = train.groupby("Engine_ID")["Cycle"].transform("max")
    train["RUL"] = max_cycle - train["Cycle"]

    return train


def remove_unused_columns(df):
    """Remove operating settings."""

    return df.drop(columns=["Setting_1", "Setting_2", "Setting_3"])


def save_processed_data(train, test, output_path, dataset):

    processed_path = (
        Path(output_path).parent
        / "dataset"
        / "processed"
    )

    processed_path.mkdir(parents=True, exist_ok=True)

    train.to_csv(
        processed_path / f"train_{dataset}.csv",
        index=False,
    )

    test.to_csv(
        processed_path / f"test_{dataset}.csv",
        index=False,
    )