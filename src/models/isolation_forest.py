import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestAnomalyDetector:
    """
    Isolation Forest based anomaly detector for multivariate time-series features.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray):
        self.model.fit(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly scores (higher = more anomalous).
        """
        return -self.model.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns binary anomaly labels.
        """
        return self.model.predict(X) == -1
