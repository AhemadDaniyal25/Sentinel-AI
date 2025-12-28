import numpy as np
import pandas as pd


class FailureInjector:
    """
    Injects synthetic industrial failure patterns into sensor time-series data.
    This is used to test anomaly detection robustness under realistic conditions.
    """

    def __init__(self, random_state: int = 42):
        np.random.seed(random_state)

    def inject_overheating(
        self,
        df: pd.DataFrame,
        column: str,
        start_index: int,
        growth_rate: float = 0.02,
    ) -> pd.DataFrame:
        """
        Simulates gradual overheating after a given index.
        """
        df = df.copy()
        for i in range(start_index, len(df)):
            df.loc[i, column] += (i - start_index) * growth_rate
        return df

    def inject_vibration_spike(
        self,
        df: pd.DataFrame,
        column: str,
        spike_index: int,
        magnitude: float = 3.0,
    ) -> pd.DataFrame:
        """
        Simulates a sudden vibration spike (e.g. bearing defect).
        """
        df = df.copy()
        df.loc[spike_index, column] += magnitude
        return df

    def inject_sensor_drift(
        self,
        df: pd.DataFrame,
        column: str,
        total_drift: float = 1.0,
    ) -> pd.DataFrame:
        """
        Simulates slow sensor calibration drift over time.
        """
        df = df.copy()
        drift = np.linspace(0, total_drift, len(df))
        df[column] += drift
        return df
