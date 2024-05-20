import numpy as np
from mpmath import bell
from sklearn.metrics.cluster import rand_score
from utils import stirling2, RandomModel, AdjustmentType


def p(labels: np.ndarray, random_model, agree: bool):
    # TODO: THIS SHOULD BE REFACTORED. HERE WE CALCULATE THE cluster_sizes OVER AND OVER AGAIN WHICH IS QUITE COSTLY.
    # MAYBE INTRODUCE A CLUSTER CLASS.
    if not agree:
        return 1 - p(labels, random_model, agree=True)
    n = len(labels)
    _, cluster_sizes = np.unique(labels, return_counts=True)
    k = len(cluster_sizes)

    match random_model:
        case RandomModel.ALL:
            return bell(n - 1) / bell(n)
        case RandomModel.NUM:
            return stirling2(n - 1, k) / stirling2(n, k)
        case RandomModel.PERM:
            return (cluster_sizes * (cluster_sizes - 1)).sum() / (n * (n - 1))
        case _:
            raise ValueError(f"Invalid random model {random_model}.")


def p_2(
    labels: np.ndarray,
    random_model: RandomModel,
    agree_alpha: bool,
    agree_beta: bool,
    alpha_eq_beta: bool,
):
    if alpha_eq_beta:
        if agree_alpha is agree_beta:
            return p(labels, random_model, agree=agree_alpha)
        else:
            return 0
    else:
        n = len(labels)
        _, cluster_sizes = np.unique(labels, return_counts=True)
        k = len(cluster_sizes)
        if agree_alpha and agree_beta:
            match random_model:
                case RandomModel.ALL:
                    return bell(n - 2) / bell(n)
                case RandomModel.NUM:
                    return stirling2(n - 2, k) / stirling2(n, k)
                case RandomModel.PERM:
                    q1 = (cluster_sizes * (cluster_sizes - 1)).sum() // 2
                    total = (n * (n - 1)) // 2
                    return (q1 / total) * (q1 - 1) / (total - 1)
                case _:
                    raise ValueError(f"Invalid random model {random_model}.")
        elif agree_alpha or agree_beta:
            return p(labels, random_model, agree=True) - p_2(
                labels, random_model, True, True, False
            )
        else:
            return (
                1
                - p_2(
                    labels,
                    random_model,
                    agree_alpha=True,
                    agree_beta=True,
                    alpha_eq_beta=False,
                )
                - 2
                * p_2(
                    labels,
                    random_model,
                    agree_alpha=True,
                    agree_beta=False,
                    alpha_eq_beta=False,
                )
            )


def e_ri2(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    random_model_true: RandomModel,
    random_model_pred: RandomModel,
):
    n = len(labels_true)
    nc2inv = 2 / (n * (n - 1))
    alpha_eq_beta = sum(
        p_2(labels_true, random_model_true, agree_alpha, agree_beta, True)
        * p_2(labels_pred, random_model_pred, agree_alpha, agree_beta, True)
        for agree_alpha in [True, False]
        for agree_beta in [True, False]
    )
    alpha_neq_beta = sum(
        p_2(labels_true, random_model_true, agree_alpha, agree_beta, False)
        * p_2(labels_pred, random_model_pred, agree_alpha, agree_beta, False)
        for agree_alpha in [True, False]
        for agree_beta in [True, False]
    )
    return nc2inv * alpha_eq_beta + (1 - nc2inv) * alpha_neq_beta


def e_ri(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    random_model_true: RandomModel,
    random_model_pred: RandomModel,
):
    return sum(
        p(labels_true, random_model_true, agree)
        * p(labels_pred, random_model_pred, agree)
        for agree in [True, False]
    )


def standardized_rand_score(
    labels_true, labels_pred, random_model_true, random_model_pred
):
    ri = rand_score(labels_true, labels_pred)
    eri = e_ri(labels_true, labels_pred, random_model_true, random_model_pred)
    eri2 = e_ri2(labels_true, labels_pred,
                 random_model_true, random_model_pred)
    if (abs(eri2 - eri**2) < 1e-14) and (abs(ri - eri) < 1e-14):
        return 1.0
    return (ri - eri) / np.sqrt(eri2 - eri**2)


def adjusted_rand_score(labels_true, labels_pred, random_model_true, random_model_pred):
    ri = rand_score(labels_true, labels_pred)
    eri = e_ri(labels_true, labels_pred, random_model_true, random_model_pred)
    return (ri - eri) / (1.0 - eri)


def generalized_adjusted_rand_score(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    adjustment: AdjustmentType,
    random_model_true: RandomModel,
    random_model_pred: RandomModel,
) -> float:
    match adjustment:
        case AdjustmentType.NONE | AdjustmentType.NORMALIZED:
            return rand_score(labels_true, labels_pred)
        case AdjustmentType.ADJUSTED:
            return adjusted_rand_score(
                labels_true, labels_pred, random_model_true, random_model_pred
            )
        case AdjustmentType.STANDARDIZED:
            return standardized_rand_score(
                labels_true, labels_pred, random_model_true, random_model_pred
            )
        case _:
            raise ValueError(f"Invalid adjustment type {adjustment}.")
