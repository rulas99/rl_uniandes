from __future__ import annotations


class PageHinkleyDetector:
    """Page-Hinkley detector for upward drift on a scalar monitoring signal.

    Reward drops are handled by the caller by passing a negated reward/return signal,
    matching the tabular MORPHIN notebook while keeping this detector one-sided.
    """

    def __init__(
        self,
        delta: float = 0.02,
        threshold: float = 5.0,
        min_instances: int = 80,
    ) -> None:
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.min_instances = int(min_instances)
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.cumulative_sum = 0.0
        self.min_cumulative_sum = 0.0
        self.last_value = 0.0
        self.last_statistic = 0.0

    def update(self, value: float) -> bool:
        value = float(value)
        self.last_value = value
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.cumulative_sum += value - self.mean - self.delta
        self.min_cumulative_sum = min(self.min_cumulative_sum, self.cumulative_sum)
        statistic = self.cumulative_sum - self.min_cumulative_sum
        self.last_statistic = statistic
        return self.count >= self.min_instances and statistic > self.threshold
