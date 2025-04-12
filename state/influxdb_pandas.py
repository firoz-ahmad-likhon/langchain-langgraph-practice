import pandas as pd
from influxdb_client_3 import InfluxDBClient3


class DataState:
    """Class to hold the state of data processing to avoid tracking DataFrame evaluation issues."""

    def __init__(self, client: InfluxDBClient3, df: pd.DataFrame = None) -> None:
        """Initialize the state."""
        self.client = client
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        """Getters for the dataframe."""
        if self._df is None:
            raise ValueError("DataFrame is not set yet.")
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame) -> None:
        """Setters for the dataframe."""
        self._df = value

    def __bool__(self) -> bool:
        """Override bool evaluation to avoid DataFrame truth value ambiguity."""
        return True
