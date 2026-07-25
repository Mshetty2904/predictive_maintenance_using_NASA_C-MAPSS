import pandas as pd


def validate_dataset(train, test):
    """Return dataset validation summary."""

    validation = pd.DataFrame({
        "Dataset": ["Train", "Test"],
        "Rows": [len(train), len(test)],
        "Columns": [train.shape[1], test.shape[1]],
        "Missing Values": [
            train.isna().sum().sum(),
            test.isna().sum().sum()
        ],
        "Duplicate Rows": [
            train.duplicated().sum(),
            test.duplicated().sum()
        ],
        "Engines": [
            train["Engine_ID"].nunique(),
            test["Engine_ID"].nunique()
        ]
    })

    return validation