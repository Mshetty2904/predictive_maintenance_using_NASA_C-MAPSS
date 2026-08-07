"""Replay NASA test data as engine-by-engine live batches."""

from pathlib import Path

import pandas as pd


class EngineStream:
    """Yield one sorted dataframe batch per engine.

    The stream is deterministic and does not mix rows from different engines.
    This mirrors a real deployment where one engine's telemetry arrives as a
    sequence of cycles before the next engine is processed.
    """

    def __init__(self, data, engine_column="Engine_ID", cycle_column="Cycle"):
        self.data = data.copy()
        self.engine_column = engine_column
        self.cycle_column = cycle_column
        if engine_column not in self.data or cycle_column not in self.data:
            raise ValueError("Stream data must contain Engine_ID and Cycle columns")
        self.data = self.data.sort_values([engine_column, cycle_column]).reset_index(drop=True)

    @classmethod
    def from_csv(cls, path, **kwargs):
        return cls(pd.read_csv(Path(path)), **kwargs)

    def __iter__(self):
        for engine_id, batch in self.data.groupby(self.engine_column, sort=True):
            yield int(engine_id), batch.sort_values(self.cycle_column).reset_index(drop=True)

    def __len__(self):
        return int(self.data[self.engine_column].nunique())
