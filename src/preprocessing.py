import os
import time
from pathlib import Path
from uuid import uuid4



def calculate_rul(train_df, max_rul=125):
    """
    Calculate Remaining Useful Life (RUL)
    using piecewise linear RUL.
    """

    train = train_df.copy()

    # Calculate engine-wise RUL
    max_cycle = train.groupby("Engine_ID")["Cycle"].transform("max")
    train["RUL"] = max_cycle - train["Cycle"]

    # Piecewise RUL capping (NASA CMAPSS standard)
    train["RUL"] = train["RUL"].clip(upper=max_rul)

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

    def safe_save(dataframe, target_path):
        """Save without directly truncating a possibly locked Windows file."""
        target_path = Path(target_path)
        temporary_path = target_path.with_name(
            f".{target_path.stem}.{uuid4().hex}.tmp.csv"
        )

        try:
            # Write the complete file first. This avoids leaving a partial CSV
            # if the process is interrupted during serialization.
            dataframe.to_csv(temporary_path, index=False)

            for attempt in range(3):
                try:
                    os.replace(temporary_path, target_path)
                    return target_path
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(1.0)

        except OSError as error:
            # Windows may reject replacement when the previous CSV is open in
            # Excel, a file preview, antivirus, or another Python process.
            # Preserve the newly generated data under a unique filename so
            # model training can continue instead of failing at this optional
            # artifact-writing step.
            fallback_path = target_path.with_name(
                f"{target_path.stem}_run_{uuid4().hex}.csv"
            )

            if temporary_path.exists():
                os.replace(temporary_path, fallback_path)

            print(
                f"WARNING: Could not replace {target_path.name} "
                f"({error}). Saved a fallback copy to "
                f"{fallback_path.name}. Close the old file before the next run."
            )
            return fallback_path

    safe_save(
        train,
        processed_path / f"{dataset_name}_train.csv",
    )

    safe_save(
        test,
        processed_path / f"{dataset_name}_test.csv",
    )
