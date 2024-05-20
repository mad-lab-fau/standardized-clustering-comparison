import numpy as np
from mpmath import binomial, bell, memoize
from scipy.stats import hypergeom
from abc import ABC, abstractmethod
from typing import Iterable
from sklearn.metrics.cluster import (
    contingency_matrix
)
from sklearn.metrics.cluster._supervised import _generalized_average
from utils import stirling2, qlog, generalized_mutual_information, tsallis_entropy, AverageMethod, RandomModel, AdjustmentType

bell = memoize(bell)


class SizeProbability(ABC):
    def __init__(self, labels: np.ndarray) -> None:
        self.n = len(labels)
        sizes = np.bincount(labels)
        self.k = len(sizes)
        sizes, counts = np.unique(sizes, return_counts=True)
        self.size_counts = dict(zip(sizes, counts))

    @abstractmethod
    def p(self, a: int) -> float:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def conditional_p_neq(self, a: int, a_given: int) -> float:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def support(self) -> Iterable[int]:
        raise NotImplementedError("Subclasses must implement this property")

    def conditional_p(self, a: int, a_given: int) -> float:
        probability = self.conditional_p_neq(a, a_given)
        if a == a_given:
            probability += a / self.n
        return probability

    def p2(self, a: int, ap: int) -> float:
        return self.p(a) * self.conditional_p(ap, a)


class SizeProbabilityPerm(SizeProbability):
    def __init__(
        self,
        size_counts: dict[int, int] | None,
        n: int | None = None,
        labels: np.ndarray | None = None,
    ) -> None:
        self.random_model = RandomModel.PERM
        if labels is not None:
            super().__init__(labels)
        elif size_counts is None:
            raise ValueError("Either labels or size_counts must be provided.")
        else:
            self.size_counts = size_counts
            if n is None:
                self.n = sum(size * count for size,
                             count in size_counts.items())
            else:
                self.n = n

    def p(self, a: int) -> float:
        return self.size_counts.get(a, 0) * a / self.n

    def conditional_p_neq(self, a: int, a_given: int) -> float:
        return (self.size_counts.get(a, 0) - int(a == a_given)) * a / self.n

    @property
    def support(self) -> Iterable[int]:
        return self.size_counts.keys()


class SizeProbabilityNum(SizeProbability):
    def __init__(
        self, n: int | None, k: int | None, labels: np.ndarray | None = None
    ) -> None:
        self.random_model = RandomModel.NUM
        if labels is not None:
            super().__init__(labels)
        elif n is None or k is None:
            raise ValueError("Either labels or n and k must be provided.")
        else:
            self.n = n
            self.k = k

    def p(self, a: int) -> float:
        return float(
            binomial(self.n, a)
            * (stirling2(self.n - a, self.k - 1) / stirling2(self.n, self.k))
            * a
            / self.n
        )

    def conditional_p_neq(self, a: int, a_given: int) -> float:
        if self.n - a - a_given >= self.k - 2:
            return float(
                binomial(self.n - a_given, a)
                * (
                    stirling2(self.n - a_given - a, self.k - 2)
                    / stirling2(self.n - a_given, self.k - 1)
                )
                * (a / self.n)
            )
        else:
            return 0.0

    @property
    def support(self) -> Iterable[int]:
        return range(1, self.n - self.k + 2)


class SizeProbabilityAll(SizeProbability):
    def __init__(self, n: int | None, labels: np.ndarray | None = None) -> None:
        self.random_model = RandomModel.ALL
        if labels is not None:
            super().__init__(labels)
        elif n is None:
            raise ValueError("Either labels or n must be provided.")
        else:
            self.n = n

    def p(self, a: int) -> float:
        return float(
            binomial(self.n, a) * (bell(self.n - a) /
                                   bell(self.n)) * a / self.n
        )

    def conditional_p_neq(self, a: int, a_given: int) -> float:
        if a + a_given <= self.n:
            return float(
                binomial(self.n - a_given, a)
                * (bell(self.n - a_given - a) / bell(self.n - a_given))
                * (a / self.n)
            )
        else:
            return 0.0

    @property
    def support(self) -> Iterable[int]:
        return range(1, self.n + 1)


def emi(p_true: SizeProbability, p_pred: SizeProbability, q: float = 1.0):
    emi_val = 0
    for a in p_true.support:
        pa = p_true.p(a)
        for b in p_pred.support:
            pb = p_pred.p(b)
            for n in range(max(1, a + b - p_true.n), min(a, b) + 1):
                pn = n * p_true.n / (a * b)
                emi_val += pa * pb * pn * \
                    qlog(pn, q) * hypergeom.pmf(n, p_true.n, a, b)
    return emi_val


def emi2(p_true: SizeProbability, p_pred: SizeProbability, q: float = 1.0):
    emi2_val = 0
    for a in p_true.support:
        pa = p_true.p(a)
        for b in p_pred.support:
            pb = p_pred.p(b)
            for n in range(max(1, a + b - p_true.n), min(a, b) + 1):
                pn = n * p_true.n / (a * b)
                inner_sum_a = 0.0
                for ap in p_true.support:
                    pac = p_true.conditional_p_neq(ap, a)
                    for n_p in range(
                        max(1, a + ap + b - n - p_pred.n), min(ap, b - n) + 1
                    ):
                        inner_sum_a += (
                            pac
                            * (n_p / ap)
                            * qlog(n_p * p_pred.n / (ap * b), q)
                            * hypergeom.pmf(n_p, p_pred.n - a, ap, b - n)
                        )

                inner_sum_b = 0.0
                for bp in p_pred.support:
                    pbc = p_pred.conditional_p_neq(bp, b)
                    for n_p in range(
                        max(0, a + b + bp - n - p_true.n), min(a - n, bp) + 1
                    ):
                        inner_sum_ab = 0.0
                        for ap in p_true.support:
                            pac = p_true.conditional_p_neq(ap, a)
                            for npp in range(
                                max(1, a + ap + bp - n_p - p_true.n),
                                min(ap, bp - n_p) + 1,
                            ):
                                pnpp = npp * p_true.n / (ap * bp)
                                inner_sum_ab += (
                                    pac
                                    * pnpp
                                    * qlog(pnpp, q)
                                    * np.nan_to_num(
                                        hypergeom.pmf(
                                            npp, p_true.n - a, ap, bp - n_p)
                                    )
                                )

                        pnp = n_p * p_true.n / (a * bp)
                        if pnp > 0:
                            term = n_p / bp * qlog(pnp, q)
                        else:
                            term = 0
                        inner_sum_b += (
                            pbc
                            * hypergeom.pmf(n_p, p_true.n - b, a - n, bp)
                            * (term + inner_sum_ab)
                        )

                emi2_val += (
                    pa
                    * pb
                    * pn
                    * qlog(pn, q)
                    * hypergeom.pmf(n, p_true.n, a, b)
                    * ((n / p_true.n) * qlog(pn, q) + inner_sum_a + inner_sum_b)
                )

    return emi2_val


def get_upper_bound(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    q: float,
    random_model_true: RandomModel,
    random_model_pred: RandomModel,
    average_method: AverageMethod = AverageMethod.ARITHMETIC,
) -> float:
    """Compute an upper bound for generalized adjusted mutual information.

    See Gates and Ahn (2017) for more details.

    Args:
        labels_true (np.ndarray):
            The true labels of the data.
        labels_pred (np.ndarray):
            The predicted labels of the data.
        random_model_true (RandomModel):
            The random model used for the true clustering.
        random_model_pred (RandomModel):
            The random model used for the predicted clustering.
        average_method (AverageMethod):
            The method to use for calculating the upper bound for generalized adjusted mutual information.
        q (float):
            The non-additivity q of the Tsallis entropy (1 for Shannon entropy, 2 for Rand Index).

    Returns:
        The upper bound for generalized adjusted mutual information.

    References:
        A. J. Gates and Y.-Y. Ahn, “The impact of random models on clustering similarity,” J. Mach. Learn. Res., vol. 18, p. 87:1-87:28, 2017.
    """
    n = len(labels_true)
    random_model = RandomModel(
        min(random_model_true.value, random_model_pred.value))

    match random_model:
        case RandomModel.ALL:
            # TODO: this is potentially wrong.
            return qlog(n, q)
        case RandomModel.NUM:
            ub_true = qlog(len(np.unique(labels_true)), q)
            ub_pred = qlog(len(np.unique(labels_pred)), q)
            return _generalized_average(ub_true, ub_pred, average_method.value)
        case _:
            _, marginals = np.unique(labels_true, return_counts=True)
            ub_true = tsallis_entropy(marginals / n, q)
            _, marginals = np.unique(labels_pred, return_counts=True)
            ub_pred = tsallis_entropy(marginals / n, q)
            return _generalized_average(ub_true, ub_pred, average_method.value)


def standardized_mutual_information(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    random_model_true: RandomModel,
    random_model_pred: RandomModel,
    q: float = 1.0,
) -> float:
    n_tot = len(labels_true)
    match random_model_true:
        case RandomModel.PERM:
            p_true = SizeProbabilityPerm(None, None, labels_true)
        case RandomModel.NUM:
            p_true = SizeProbabilityNum(n_tot, None, labels_true)
        case RandomModel.ALL:
            p_true = SizeProbabilityAll(n_tot)
        case _:
            raise ValueError("Invalid random model.")

    match random_model_pred:
        case RandomModel.PERM:
            p_pred = SizeProbabilityPerm(None, None, labels_pred)
        case RandomModel.NUM:
            p_pred = SizeProbabilityNum(n_tot, None, labels_pred)
        case RandomModel.ALL:
            p_pred = SizeProbabilityAll(n_tot)
        case _:
            raise ValueError("Invalid random model.")

    mi = generalized_mutual_information(
        contingency_matrix(labels_true, labels_pred, sparse=True), n_tot, q)
    emi_val = emi(p_true, p_pred, q)
    emi2_val = emi2(p_true, p_pred, q)

    if (abs(emi2_val - emi_val**2) < 1e-14) and (abs(mi - emi_val) < 1e-14):
        return 1.0

    return (mi - emi_val) / np.sqrt(emi2_val - emi_val**2)


def adjusted_mutual_information(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    random_model_true: RandomModel,
    random_model_pred: RandomModel,
    q: float = 1.0,
    average_method: AverageMethod = AverageMethod.ARITHMETIC,
):
    n_tot = len(labels_true)
    match random_model_true:
        case RandomModel.PERM:
            p_true = SizeProbabilityPerm(None, None, labels_true)
        case RandomModel.NUM:
            p_true = SizeProbabilityNum(n_tot, None, labels_true)
        case RandomModel.ALL:
            p_true = SizeProbabilityAll(n_tot)
        case _:
            raise ValueError("Invalid random model.")

    match random_model_pred:
        case RandomModel.PERM:
            p_pred = SizeProbabilityPerm(None, None, labels_pred)
        case RandomModel.NUM:
            p_pred = SizeProbabilityNum(n_tot, None, labels_pred)
        case RandomModel.ALL:
            p_pred = SizeProbabilityAll(n_tot)
        case _:
            raise ValueError("Invalid random model.")
    mi = generalized_mutual_information(
        contingency_matrix(labels_true, labels_pred, sparse=True), n_tot, q)
    emi_val = emi(p_true, p_pred, q)
    ub = get_upper_bound(labels_true, labels_pred, q,
                         random_model_true, random_model_pred, average_method)
    return (mi - emi_val) / (ub - emi_val)


def generalized_adjusted_mutual_information(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    adjustment: AdjustmentType,
    random_model_true: RandomModel,
    random_model_pred: RandomModel,
    q: float = 1.0,
    average_method: AverageMethod = AverageMethod.ARITHMETIC,

) -> float:
    match adjustment:
        case AdjustmentType.NONE:
            return generalized_mutual_information(
                contingency_matrix(labels_true, labels_pred, sparse=True),
                len(labels_true),
                q,
            )
        case AdjustmentType.NORMALIZED:
            mi = generalized_mutual_information(
                contingency_matrix(labels_true, labels_pred, sparse=True),
                len(labels_true),
                q,
            )
            ub = get_upper_bound(
                labels_true, labels_pred, q, random_model_true, random_model_pred, average_method
            )
            return mi / ub
        case AdjustmentType.ADJUSTED:
            return adjusted_mutual_information(
                labels_true, labels_pred, random_model_true, random_model_pred, q, average_method
            )
        case AdjustmentType.STANDARDIZED:
            return standardized_mutual_information(
                labels_true, labels_pred, random_model_true, random_model_pred, q
            )
        case _:
            raise ValueError("Invalid adjustment type.")
