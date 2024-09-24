import numpy as np
from numpy.random import SeedSequence, BitGenerator, Generator, default_rng
from numpy.typing import ArrayLike
from collections import namedtuple


ConfidenceInterval = namedtuple("ConfidenceInterval", ["low", "high"])


class StreamingBootstrappedStandardization:
    def __init__(
        self,
        x: float,
        bootstrap_replicates: int = 1000,
        seed: SeedSequence | None = None,
    ) -> None:
        self.x = x
        self.__bootstrap_replicates = bootstrap_replicates
        self.__mean = np.zeros(bootstrap_replicates, dtype=np.float64)
        self.__w_sum = np.zeros(bootstrap_replicates, dtype=np.float64)
        self.__s = np.zeros(bootstrap_replicates, dtype=np.float64)
        self.__prng = default_rng(seed)

    @property
    def standardized_x(self) -> tuple[float, float]:
        """Standardize the input x and return the mean and standard error."""
        bootstrap_estimates = (self.x - self.__mean) / np.sqrt(self.__s / self.__w_sum)
        return bootstrap_estimates.mean(), bootstrap_estimates.std(ddof=1)

    @property
    def standardized_x_confidence(
        self, confidence_level: float = 0.95
    ) -> tuple[float, ConfidenceInterval]:
        bootstrap_estimates = (self.x - self.__mean) / np.sqrt(self.__s / self.__w_sum)

        if confidence_level is not None:
            alpha = (
                1 - confidence_level
            ) / 2  # /2 because we care about two-sided alternative
            confidence = ConfidenceInterval(
                *np.percentile(bootstrap_estimates, [100 * alpha, 100 * (1 - alpha)])
            )
            return bootstrap_estimates.mean(), confidence

    def update(self, sample: float, weight: float | None = None) -> None:
        if weight is None:
            weight = 1.0
        weights = weight * self.__prng.poisson(1, self.__bootstrap_replicates)

        # West's algorithm (1979) (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_incremental_algorithm)
        self.__w_sum += weights
        delta = sample - self.__mean
        self.__mean += delta * weights / self.__w_sum
        self.__s += weights * delta * (sample - self.__mean)

    def update_batch(
        self, samples: np.ndarray, weights: np.ndarray | None = None
    ) -> None:
        """
        Batch update the accumulator with multiple samples and weights.

        Args:
            samples: One-dimensional array of samples.
            weights: One-dimensional array of weights of the same length as samples.
        """
        if weights is None:
            weights = np.ones_like(samples, dtype=np.float64)
        meta_weights = weights * self.__prng.poisson(
            1, (self.__bootstrap_replicates, len(weights))
        )
        meta_samples = np.tile(samples, (self.__bootstrap_replicates, 1))

        w_sum = meta_weights.sum(axis=1)
        mean = np.average(meta_samples, weights=meta_weights, axis=1)
        s = (
            np.average(
                (meta_samples - np.tile(mean, (len(weights), 1)).T) ** 2,
                weights=meta_weights,
                axis=1,
            )
            * w_sum
        )
        self._merge(w_sum, mean, s)

    def merge(self, other) -> None:
        """Merge this accumulator with another one.

        Args:
            other: The other accumulator to merge.
        """
        self._merge(other.__w_sum, other.__mean, other.__s)

    def _merge(self, w_sum: np.ndarray, mean: np.ndarray, s: np.ndarray) -> None:
        tot_weight = self.__w_sum + w_sum
        delta = mean - self.__mean
        mean = (self.__w_sum * self.__mean + w_sum * mean) / tot_weight
        s = self.__s + s + delta**2 * (self.__w_sum * w_sum) / tot_weight

        self.__w_sum = tot_weight
        self.__mean = mean
        self.__s = s
