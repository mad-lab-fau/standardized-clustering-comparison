from enum import Enum
from mpmath import stirling2, memoize
import numpy as np
from scipy.sparse import csr_matrix

stirling2 = memoize(stirling2)


class AverageMethod(Enum):
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"
    MIN = "min"
    MAX = "max"


class RandomModel(Enum):
    ALL = 0
    NUM = 1
    PERM = 2


class AdjustmentType(Enum):
    NONE = 0
    NORMALIZED = 1
    ADJUSTED = 2
    STANDARDIZED = 3


def qlog(x: float, q: float = 1.0) -> float:
    """Compute the q-log of a number.

    Args:
        x (float):
            The number.
        q (float):
            The non-additivity q of the q-log.

    Returns:
        The q-log of the number.
    """
    if q == 1.0:
        return np.log(x)
    return (x ** (1 - q) - 1) / (1 - q)


def tsallis_entropy(p: np.ndarray, q: float = 1.0, axis=None) -> float | np.ndarray:
    """Tsallis entropy of a probability distribution.

    Args:
        p (array):
            Probability distribution.
        q (float):
            The non-additivity q of the Tsallis entropy (1 for Shannon entropy, 2 for Rand Index).
        axis (int):
            The axis along which to compute the entropy.

    Returns:
        The Tsallis entropy.
    """
    if q == 1.0:
        return -np.sum(p * np.log(p, where=p > 0), axis=axis)
    return 1 / (q - 1) * (1 - (p**q).sum(axis=axis))


def generalized_mutual_information(contingency: csr_matrix, n: int, q: float):
    """Compute the generalized mutual information between two clusterings.

    Args:
        contingency (csr_matrix):
            The contingency matrix.
        n (int):
            The total number of data points.
        q (float):
            The non-additivity q of the Tsallis entropy (1 for Shannon entropy, 2 for Rand Index).

    Returns:
        The generalized mutual information.
    """
    joint_entropy = tsallis_entropy(contingency.data / n, q=q)
    row_entropy = tsallis_entropy(np.ravel(contingency.sum(axis=1)) / n, q=q)
    col_entropy = tsallis_entropy(np.ravel(contingency.sum(axis=0)) / n, q=q)

    return row_entropy + col_entropy - joint_entropy
