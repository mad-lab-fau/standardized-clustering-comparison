from monte_carlo import (
    standardized_mutual_information_mc,
    RandomClusteringGenerator,
)
from utils import RandomModel, AdjustmentType, Timeout
from mutual_information import generalized_adjusted_mutual_information
from rand_index import generalized_adjusted_rand_score
from numpy import logspace
from typing import Iterable, Any, Callable
import pandas as pd
from numpy.random import SeedSequence, BitGenerator, Generator
from tqdm import tqdm, trange
from time import perf_counter
from multiprocessing import Pool
import sys
from functools import partial
from random import shuffle

sys.setrecursionlimit(10**6)

standardized_mutual_information = partial(
    generalized_adjusted_mutual_information, adjustment=AdjustmentType.STANDARDIZED
)
standardized_rand_index = partial(
    generalized_adjusted_rand_score, adjustment=AdjustmentType.STANDARDIZED
)


def time_clustering_comparison_method(
    method, method_name, u, v, random_model, trial, n, ku, kv
):
    kwargs = {
        "random_model_true": random_model,
        "random_model_pred": random_model,
    }

    try:
        with Timeout(seconds=100):
            start_time = perf_counter()
            value = method(
                u,
                v,
                **kwargs,
            )
            time_difference = perf_counter() - start_time
    except TimeoutError:
        time_difference = None
        value = None

    std_err = 0.0
    if (method_name == "SMI_MC") and (value is not None):
        value, std_err = value
    return {
        "n": n,
        "ku": ku,
        "kv": kv,
        "trial": trial,
        "method": method_name,
        "random_model": random_model.name,
        "runtime_seconds": time_difference,
        "value": value,
        "standard_error": std_err,
    }


def efficiency_experiment(
    n_values: Iterable[int] = None,
    ku: int = 5,
    kv: int = 5,
    num_trials: int = 100,
    seed: None | SeedSequence = None,
) -> list[dict[str, Any]]:
    if n_values is None:
        n_values = logspace(1, 4, num=7, dtype=int, base=10)

    if seed is None:
        seed = SeedSequence(0)

    results = []
    with Pool(16) as pool:
        for n in n_values:
            if ku > n or kv > n:
                raise ValueError(
                    "The number of clusters ku, kv must be less than or equal to the number of data points n."
                )

            u_fix = list(range(ku)) + [0] * (n - ku)
            v_fix = list(range(kv)) + [0] * (n - kv)

            u_seed, v_seed = seed.spawn(2)

            u_generator = RandomClusteringGenerator(u_fix, RandomModel.NUM, seed=u_seed)
            v_generator = RandomClusteringGenerator(v_fix, RandomModel.NUM, seed=v_seed)

            for trial in range(num_trials):
                u = u_generator.random()
                v = v_generator.random()

                for random_model in RandomModel:
                    for method_name, method in [
                        ("SMI_MC", standardized_mutual_information_mc),
                        ("SMI", standardized_mutual_information),
                        ("SRI", standardized_rand_index),
                    ]:
                        results.append(
                            pool.apply_async(
                                time_clustering_comparison_method,
                                args=(
                                    method,
                                    method_name,
                                    u,
                                    v,
                                    random_model,
                                    trial,
                                    n,
                                    ku,
                                    kv,
                                ),
                            )
                        )

        results = [result.get() for result in tqdm(results)]

    return results


if __name__ == "__main__":
    n_values = logspace(1, 4, num=7, dtype=int, base=10).tolist() + [
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
    ]
    shuffle(n_values)
    results = efficiency_experiment(n_values=n_values)
    df = pd.DataFrame(results)
    df.to_csv("./results/efficiency_experiment.csv", index=False)
