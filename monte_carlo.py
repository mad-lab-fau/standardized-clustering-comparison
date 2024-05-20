from numpy.random import default_rng, SeedSequence, BitGenerator, Generator
import numpy as np

from mpmath import bell, log
from scipy.sparse import csr_matrix
from scipy.stats import bootstrap
from scipy.stats._common import ConfidenceInterval
from scipy.special import loggamma
from sklearn.metrics.cluster import (
    contingency_matrix,
    rand_score,
)
from utils import AverageMethod, AdjustmentType, RandomModel, stirling2, qlog, tsallis_entropy, generalized_mutual_information
from mutual_information import get_upper_bound
from time import perf_counter


class WalkerRandomSampling(object):
    """Walker's alias method for random objects with different probablities.

    Based on the implementation of Denis Bzowy at the following URL:
    http://code.activestate.com/recipes/576564-walkers-alias-method-for-random-objects-with-diffe/
    licensed under the MIT license.
    """

    def __init__(self, weights, keys=None, seed=None):
        """Builds the Walker tables ``prob`` and ``inx`` for calls to `random()`.
        The weights (a list or tuple or iterable) can be in any order and they
        do not even have to sum to 1.

        Args:
            weights: Weights of the random variates.
            keys: Keys of the random variates.
            seed: Seed for the random number generator.
        """
        n = self.n = len(weights)
        if keys is None:
            self.keys = keys
        else:
            self.keys = np.array(keys)

        self._rng = default_rng(seed)

        if isinstance(weights, (list, tuple)):
            weights = np.array(weights, dtype=float)
        elif isinstance(weights, np.ndarray):
            if weights.dtype != float:
                weights = weights.astype(float)
        else:
            weights = np.array(list(weights), dtype=float)

        if weights.ndim != 1:
            raise ValueError("weights must be a vector")

        weights = weights * n / weights.sum()

        inx = -np.ones(n, dtype=int)
        short = np.where(weights < 1)[0].tolist()
        long = np.where(weights > 1)[0].tolist()
        while short and long:
            j = short.pop()
            k = long[-1]

            inx[j] = k
            weights[k] -= 1 - weights[j]
            if weights[k] < 1:
                short.append(k)
                long.pop()

        self.prob = weights
        self.inx = inx

    def random(self, size=None):
        """Returns a given number of random integers or keys, with probabilities
        being proportional to the weights supplied in the constructor.
        When `size` is ``None``, returns a single integer or key, otherwise
        returns a NumPy array with a length given in `size`.

        Args:
            size: Number of random integers or keys to return.

        Returns:
            Random variates with probabilities being proportional to the weights supplied in the constructor.
        """
        if size is None:
            u = self._rng.random()
            j = self._rng.integers(self.n)
            k = j if u <= self.prob[j] else self.inx[j]
            return self.keys[k] if self.keys is not None else k

        u = self._rng.random(size=size)
        j = self._rng.integers(self.n, size=size)
        k = np.where(u <= self.prob[j], j, self.inx[j])
        return self.keys[k] if self.keys is not None else k


class RandomClusteringGenerator:
    def __init__(
        self,
        labels: np.ndarray,
        random_model: RandomModel,
        seed: None | SeedSequence | BitGenerator | Generator = None,
    ) -> None:
        """
        Initialize the random clustering generator.

        Args:
            labels: The true labels.
            random_model: The random model to use.
            seed: The seed to use.
        """
        self._prng = default_rng(seed)
        self._labels = labels
        self._n = len(self._labels)
        self._k = len(np.unique(self._labels))
        self._random_model = random_model
        if self._random_model == RandomModel.ALL:
            logbelln = float(log(bell(self._n)))
            self._k_max = 2 * self._n
            weights = np.arange(self._k_max + 1)
            weights = self._n * np.log(weights) - loggamma(weights + 1)
            weights = np.exp(weights - logbelln)
            self._knuth_k_sampler = WalkerRandomSampling(
                weights, seed=self._prng)

    def _random_partition_all(self, size: None | int = None) -> np.ndarray:
        """Return a random set partition of a set via Dobiński's formula.

        Args:
            size: number of random partitions to return.

        Returns:
            A 1d array of length n that represents the partition. If size is not None a 2d array with shape (size, n) is returned.
        """
        k = self._knuth_k_sampler.random(size=size)
        if size is None:
            if k == self._k_max:
                pass
            return self._prng.integers(0, k, size=self._n)
        else:
            np.where(k == self._k_max, 0, k)
            return self._prng.integers(np.zeros(size), k, size=(self._n, size)).T

    def _random_partition_num(self, n: int, k: int, min_label: int = 0) -> np.ndarray:
        """Return a random partition of an integer via brute force.

        Args:
            n: size of the set to be partitioned
            k: number of parts
            min_label: minimum label to be used

        Returns:
            An array of length n that represents the partition.
        """
        if k == 1:
            return np.ones(n, dtype=int) * min_label
        # TODO: potentially memoize the stirling numbers
        if self._prng.random() < stirling2(n - 1, k - 1) / stirling2(n, k):
            # n is a singleton in the partition
            return np.append(
                self._random_partition_num(n - 1, k - 1, min_label + 1),
                min_label,
            )
        else:
            # n is in a part with more than one element
            partition = self._random_partition_num(n - 1, k, min_label)
            return np.append(partition, self._prng.integers(min_label, k + min_label))

    def random(self, size: None | int = None) -> np.ndarray:
        """
        Generate a clustering according to the chosen random model.

        Args:
            size: number of random partitions to return.

        Returns:
            np.ndarray: The generated clustering labels shape (n, size).
        """
        match self._random_model:
            case RandomModel.ALL:
                return self._random_partition_all(size=size)
            case RandomModel.NUM:
                if size is None:
                    return self._random_partition_num(self._n, self._k)
                else:
                    return np.array(
                        [
                            self._random_partition_num(self._n, self._k)
                            for _ in range(size)
                        ]
                    )
            case RandomModel.PERM:
                if size is None:
                    return self._prng.permuted(self._labels)
                else:
                    return self._prng.permuted(np.tile(self._labels, (size, 1)), axis=1)
            case RandomModel.PAIR:
                # Pairwise transposition, i.e. two labels are swapped
                if size is None:
                    labels = self._labels.copy()
                    i, j = self._prng.choice(self._n, size=2, replace=False)
                    labels[i], labels[j] = labels[j], labels[i]
                    return labels
                else:
                    labels = np.tile(self._labels, (size, 1))
                    i, j = self._prng.choice(
                        self._n, replace=True, size=(size, 2))
                    labels[np.arrange(size), i], labels[np.arrange(size), j] = (
                        labels[np.arrange(size), j],
                        labels[np.arrange(size), i],
                    )
                    return labels
            case RandomModel.FIXED:
                if size is None:
                    return self._labels
                else:
                    return np.tile(self._labels, (size, 1))


class RandomContingencyGenerator:
    def __init__(
        self,
        labels_true: np.ndarray,
        labels_pred: np.ndarray,
        random_model_true: RandomModel,
        random_model_pred: RandomModel,
        seed: SeedSequence,
    ) -> None:
        """
        Initialize the random contingency generator.

        Parameters:
            labels_true (np.ndarray): The true labels of the data.
            labels_pred (np.ndarray): The predicted labels of the data.
            random_model_true (RandomModel): The random model used for the true clustering.
            random_model_pred (RandomModel): The random model used for the predicted clustering.
            seed (SeedSequence): The seed used for generating contingency tables.
        """
        seed_true, seed_pred = seed.spawn(2)
        self._random_clustering_true = RandomClusteringGenerator(
            labels_true, random_model_true, seed=seed_true
        )
        self._random_clustering_pred = RandomClusteringGenerator(
            labels_pred, random_model_pred, seed=seed_pred
        )

    def random(self) -> csr_matrix:
        """
        Generate a sparse contingency matrix according to the chosen random model.

        Args:
            size: number of random partitions to return.

        Returns:
            csr_matrix: The generated contingency matrix.
        """
        labels_true = self._random_clustering_true.random()
        labels_pred = self._random_clustering_pred.random()
        return contingency_matrix(labels_true, labels_pred, sparse=True)


def generalized_adjusted_mutual_information_mc(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    adjustment: AdjustmentType,
    random_model_true: RandomModel,
    random_model_pred: RandomModel,
    q: float = 1.0,
    average_method: AverageMethod = AverageMethod.ARITHMETIC,
    time_limit: float = 10.0,
    confidence_level: float | None = None,
    seed: None | SeedSequence = None,
) -> float | tuple[float, ConfidenceInterval]:
    """Compute the generalized adjusted mutual information between two clusterings.

    Args:
        labels_true (np.ndarray): The true labels of the data.
        labels_pred (np.ndarray): The predicted labels of the data.
        adjustment (AdjustmentType): The type of adjustment to use (NONE, ADJUSTED, STANDARDIZED).
        random_model_true (RandomModel): The random model used for adjusting the true clustering.
        random_model_pred (RandomModel): The random model used for adjusting the predicted clustering.
        q (float): The non-additivity q of the Tsallis entropy (1 for Shannon entropy, 2 for Rand Index).
        average_method (AverageMethod): The method to use for calculating the upper bound for generalized adjusted mutual information.
        time_limit (float): The time in seconds for how long to generate MC samples.
        confidence_level (float | None): Whether to return the confidence interval at given level.
        seed (None |SeedSequence): The seed used for generating contingency tables.

    Returns:
        The generalized adjusted mutual information and the bootstrap confidence interval if confidence_level is not None.
    """
    if seed == None:
        seed = SeedSequence()
    n = len(labels_true)
    mi = generalized_mutual_information(
        contingency_matrix(labels_true, labels_pred, sparse=True), n, q
    )

    match adjustment:
        case AdjustmentType.NONE:
            if confidence_level is not None:
                return mi, ConfidenceInterval(mi, mi)
            else:
                return mi
        case AdjustmentType.NORMALIZED:
            upper_bound = get_upper_bound(
                labels_true,
                labels_pred,
                q,
                random_model_true,
                random_model_pred,
                average_method=average_method,
            )
            nmi = mi / upper_bound
            if confidence_level is not None:
                return nmi, ConfidenceInterval(nmi, nmi)
            else:
                return nmi
        case AdjustmentType.ADJUSTED:
            upper_bound = get_upper_bound(
                labels_true,
                labels_pred,
                q,
                random_model_true,
                random_model_pred,
                average_method=average_method,
            )

            def statistic(x, axis=None):
                emi = np.mean(x, axis=axis)
                return (mi - emi) / (upper_bound - emi)

        case AdjustmentType.STANDARDIZED:

            def statistic(x, axis=None):
                return (mi - np.mean(x, axis=axis)) / np.std(x, axis=axis, ddof=1)

        case _:
            raise ValueError(f"Unknown adjustment type: {adjustment}")

    random_contingency = RandomContingencyGenerator(
        labels_true, labels_pred, random_model_true, random_model_pred, seed.spawn(1)[
            0]
    )

    start_time = perf_counter()
    mi_samples = []
    while perf_counter() - start_time < time_limit:
        mi_samples.append(
            generalized_mutual_information(random_contingency.random(), n, q)
        )
    mi_samples = np.array(mi_samples)

    result = statistic(mi_samples)

    if confidence_level is not None:
        confidence_interval = bootstrap(
            (mi_samples,),
            statistic=statistic,
            confidence_level=confidence_level,
            vectorized=True,
            batch=1,
            n_resamples=1_000,
            random_state=default_rng(seed.spawn(1)[0]),
        ).confidence_interval
        return result, confidence_interval

    return result


def generalized_adjusted_rand_score_mc(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    random_model_true: RandomModel,
    random_model_pred: RandomModel,
    adjustment: AdjustmentType,
    num_samples: int = 20_000,
    confidence_level: float | None = None,
    seed: None | SeedSequence = None,
) -> float | tuple[float, ConfidenceInterval]:
    """Compute the generalized adjusted rand score between two clusterings.

    Args:
        labels_true (np.ndarray): The true labels of the data.
        labels_pred (np.ndarray): The predicted labels of the data.
        random_model_true (RandomModel): The random model used for adjusting the true clustering.
        random_model_pred (RandomModel): The random model used for adjusting the predicted clustering.
        adjustment (AdjustmentType): The type of adjustment to use (NONE, ADJUSTED, STANDARDIZED).
        num_samples (int): The number of Monte Carlo samples to use.
        confidence_level (float | None): Whether to return the confidence interval at given level.
        seed (None |SeedSequence): The seed used for generating contingency tables.

    Returns:
        The generalized adjusted rand score and the bootstrap confidence interval if confidence_level is not None.
    """
    if seed == None:
        seed = SeedSequence()
    n = len(labels_true)
    ri = rand_score(labels_true, labels_pred)

    match adjustment:
        case AdjustmentType.NONE | AdjustmentType.NORMALIZED:
            if confidence_level is not None:
                return ri, ConfidenceInterval(ri, ri)
            else:
                return ri
        case AdjustmentType.ADJUSTED:

            def statistic(x, axis=None):
                eri = np.mean(x, axis=axis)
                return (ri - eri) / (1.0 - eri)

        case AdjustmentType.STANDARDIZED:

            def statistic(x, axis=None):
                return (ri - np.mean(x, axis=axis)) / np.std(x, axis=axis, ddof=1)

        case _:
            raise ValueError(f"Unknown adjustment type: {adjustment}")

    seed_true, seed_pred = seed.spawn(2)
    random_clustering_true = RandomClusteringGenerator(
        labels_true, random_model_true, seed=seed_true
    )
    random_clustering_pred = RandomClusteringGenerator(
        labels_pred, random_model_pred, seed=seed_pred
    )

    mi_samples = np.array(
        [
            rand_score(random_clustering_true.random(),
                       random_clustering_pred.random())
            for _ in range(num_samples)
        ]
    )

    result = statistic(mi_samples)

    if confidence_level is not None:
        confidence_interval = bootstrap(
            (mi_samples,),
            statistic=statistic,
            confidence_level=confidence_level,
            vectorized=True,
            batch=1,
            n_resamples=1_000,
            random_state=default_rng(seed.spawn(1)[0]),
        ).confidence_interval
        return result, confidence_interval

    return result
