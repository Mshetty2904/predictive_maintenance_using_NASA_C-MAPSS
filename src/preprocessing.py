from pathlib import Path
from config import MAX_RUL

def calculate_rul(train_df):
    """
    Calculate Remaining Useful Life (RUL)
    using piecewise linear RUL.
    """

    train = train_df.copy()

    max_cycle = (
        train
        .groupby("Engine_ID")["Cycle"]
        .transform("max")
    )

    train["RUL"] = max_cycle - train["Cycle"]

    # Piecewise RUL Capping
    train["RUL"] = train["RUL"].clip(
        upper=MAX_RUL
    )

    return train


def save_processed_data(
    train,
    test,
    dataset_name,
    processed_path,
):
    """
    Save processed train and test datasets.
    """

    processed_path = Path(processed_path)

    processed_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_csv(
        processed_path / f"{dataset_name}_train.csv",
        index=False,
    )

    test.to_csv(
        processed_path / f"{dataset_name}_test.csv",
        index=False,
    )