import numpy as np


class StatisticalBaseline:
    """
    Simple z-score based anomaly detector.
    Serves as an interpretable baseline for industrial systems.
    """

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mean = None
        self.std = None

    def fit(self, values: np.ndarray):
        self.mean = np.mean(values)
        self.std = np.std(values)

    def score(self, values: np.ndarray) -> np.ndarray:
        z_scores = np.abs((values - self.mean) / self.std)
        return z_scores

    def predict(self, values: np.ndarray) -> np.ndarray:
        scores = self.score(values)
        return scores > self.threshold
