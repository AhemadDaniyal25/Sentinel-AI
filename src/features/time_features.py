import pandas as pd


def build_rolling_features(
    df: pd.DataFrame,
    column: str,
    window_size: int = 20,
) -> pd.DataFrame:
    """
    Builds rolling statistical features for time-series sensor data.
    These features are robust, interpretable, and industry-standard.
    """
    features = pd.DataFrame(index=df.index)

    features[f"{column}_mean"] = df[column].rolling(window_size).mean()
    features[f"{column}_std"] = df[column].rolling(window_size).std()
    features[f"{column}_min"] = df[column].rolling(window_size).min()
    features[f"{column}_max"] = df[column].rolling(window_size).max()

    return features.dropna()
