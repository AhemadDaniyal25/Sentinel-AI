import numpy as np


class DataDriftDetector:
    """
    Simple statistical drift detector for sensor data.
    Compares reference and current distributions.
    """

    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold

    def detect_mean_shift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> bool:
        """
        Detects relative mean shift beyond threshold.
        """
        ref_mean = np.mean(reference)
        cur_mean = np.mean(current)

        if ref_mean == 0:
            return False

        relative_shift = abs(cur_mean - ref_mean) / abs(ref_mean)
        return relative_shift > self.threshold
