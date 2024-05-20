from monte_carlo import generalized_adjusted_mutual_information_mc, generalized_adjusted_rand_score_mc
from rand_index import generalized_adjusted_rand_score
from mutual_information import generalized_adjusted_mutual_information
from utils import RandomModel, AdjustmentType, AverageMethod, MethodFamily, Timeout
from scipy.stats._common import ConfidenceInterval
import warnings

import numpy as np
from numpy.random import SeedSequence


def compare_clusterings(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    method_family: MethodFamily,
    adjustment: AdjustmentType,
    random_model_true: RandomModel = RandomModel.PERM,
    random_model_pred: RandomModel = RandomModel.PERM,
    average_method: AverageMethod = AverageMethod.ARITHMETIC,
    q: float = 1.0,
    mi_time_limit: float | None = 30.0,
    confidence_level: float | None = None,
    seed: SeedSequence | None = None,
) -> float | tuple[float, ConfidenceInterval]:
    match method_family:
        case MethodFamily.RI:
            result = generalized_adjusted_rand_score(
                labels_true, labels_pred, adjustment, random_model_true, random_model_pred
            )
        case MethodFamily.MI:
            half_time_limit = mi_time_limit / 2
            try:
                with Timeout(half_time_limit):
                    result = generalized_adjusted_mutual_information(
                        labels_true, labels_pred, adjustment, random_model_true, random_model_pred, q=q, average_method=average_method
                    )
            except TimeoutError:
                warnings.warn(
                    "Calculation timed out, falling back to Monte Carlo.")
                return generalized_adjusted_mutual_information_mc(
                    labels_true, labels_pred, adjustment, random_model_true, random_model_pred, q=q, average_method=average_method, time_limit=half_time_limit, confidence_level=confidence_level, seed=seed
                )

    if confidence_level is not None:
        return result, ConfidenceInterval(result, result)
    else:
        return result
