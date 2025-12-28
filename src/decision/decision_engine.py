from typing import List


class DecisionEngine:
    """
    Converts anomaly scores into actionable decisions.
    Suppresses noise and avoids false alarms.
    """

    def __init__(
        self,
        score_threshold: float = 0.7,
        persistence_windows: int = 3,
    ):
        self.score_threshold = score_threshold
        self.persistence_windows = persistence_windows

    def evaluate(self, scores: List[float]) -> List[bool]:
        alerts = []
        consecutive = 0

        for score in scores:
            if score >= self.score_threshold:
                consecutive += 1
            else:
                consecutive = 0

            alerts.append(consecutive >= self.persistence_windows)

        return alerts
