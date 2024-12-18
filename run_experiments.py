import argparse
from dataset import RealExperimentMCPdeSouto, ClusteringAlgorithms
from utils import RandomModel, AdjustmentType, MethodFamily
from numpy.random import SeedSequence
from abc import ABC, abstractmethod
from psutil import cpu_count
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from compare_clusterings import compare_clusterings
from scipy.stats._common import ConfidenceInterval
from typing import Any, Iterable
import pandas as pd
from monte_carlo import RandomClusteringGenerator
from pathlib import Path
from itertools import islice, repeat

ComparisonMethod = tuple[MethodFamily,
                         AdjustmentType, RandomModel, RandomModel]


def _compare_clusterings(params: tuple[tuple[dict[str, Any], np.ndarray, np.ndarray, ComparisonMethod], SeedSequence, float | None]) -> tuple[float, ConfidenceInterval, dict[str, Any]]:
    (sample_params, labels_true, labels_pred,
     comparison_method), seed, time_limit = params
    sample_params["method_family"] = comparison_method[0].name
    sample_params["adjustment"] = comparison_method[1].name
    sample_params["random_model_true"] = comparison_method[2].name
    sample_params["random_model_pred"] = comparison_method[3].name
    value, confidence = compare_clusterings(
        labels_true, labels_pred, *comparison_method, mi_time_limit=time_limit, seed=seed, confidence_level=0.95)
    return value, confidence, sample_params


class Experiment(ABC):
    def __init__(self, comparison_methods: list[ComparisonMethod], seed: SeedSequence | None = None):
        """
        Initialize the RunExperiments class.

        Args:
            comparison_methods (list[ComparisonMethod]): A list of comparison methods to be used in the experiments.
            seed (SeedSequence): The seed sequence for random number generation.
        """
        self.comparison_methods = comparison_methods
        if seed is None:
            seed = SeedSequence()
        self.seed = seed

    @abstractmethod
    def __len__(self):
        raise NotImplementedError(
            "This method must be implemented in a subclass.")

    @abstractmethod
    def __iter__(self) -> Iterable[tuple[dict[str, Any], np.ndarray, np.ndarray, ComparisonMethod]]:
        raise NotImplementedError(
            "This method must be implemented in a subclass.")

    def run(self, total_time_limit: float = 86400.0, n_jobs: int | None = None, chunk_number: int = 0, total_chunks: int = 1) -> pd.DataFrame:
        if n_jobs is None:
            n_jobs = cpu_count(logical=False)
        chunk_size = (len(self) // total_chunks) + \
            (1 if ((len(self) % total_chunks) != 0) else 0)
        time_per_trial = total_time_limit / (chunk_size * n_jobs)
        self.seed.spawn(chunk_number * chunk_size)
        params = zip(islice(iter(self), chunk_number * chunk_size,
                            min((chunk_number + 1) * chunk_size, len(self))), self.seed.spawn(chunk_size), repeat(time_per_trial))
        total = min((chunk_number + 1) * chunk_size, len(self)) - \
            chunk_number * chunk_size

        with Pool(n_jobs) as pool:
            results = list({"value": value, "confidence_low": confidence[0], "confidence_high": confidence[1], **sample_params} for value, confidence, sample_params in
                           tqdm(
                pool.imap_unordered(_compare_clusterings, params),
                total=total,
            )
            )
        return pd.DataFrame(results)


class SyntheticExperiment(Experiment):
    def __init__(
        self,
        comparison_methods: list[ComparisonMethod],
        number_datapoints: int = 500,
        reference_number_clusters: int = 10,
        compare_number_clusters: list[int] = [2, 6, 10, 14, 18, 22],
        trials: int = 5_000,
        seed: SeedSequence | None = None,
    ):
        """Run the experiment performed in S. Romano et al. "Standardized Mutual Information
        for Clustering Comparisons: One Step Further in Adjustment for Chance" (2014).

        We generate a reference clustering with reference_number_clusters evenly sized clusters.
        Then we generate a number of clusterings with compare_number_clusters clusters and
        for every comparison metric we record the selected cluster's size. After trials trials
        we return the selection probability for each cluster size for each comparison metric.

        Args:
            comparison_methods: List of ComparisonMethods to test
            number_datapoints: Number of datapoints N
            reference_number_clusters: Number of clusters K
            compare_number_clusters: List of number of clusters to compare to the reference clustering
            trials: Number of trials to run
            seed: random seed sequence to use for reproducibility
        """
        super().__init__(comparison_methods, seed)
        if number_datapoints % reference_number_clusters != 0:
            raise ValueError(
                "The number of datapoints must be divisible by the number of reference clusters."
            )
        if max(compare_number_clusters) > number_datapoints:
            raise ValueError(
                "The number of datapoints must be greater than the number of clusters."
            )

        self.k_reference = reference_number_clusters
        self.reference_clustering = np.array(
            list(range(reference_number_clusters))
            * (number_datapoints // reference_number_clusters)
        )

        self.trials = trials

        self.cluster_generators = [
            RandomClusteringGenerator(list(range(k)) + [0]*(number_datapoints - k), RandomModel.NUM, gen_seed) for k, gen_seed in zip(compare_number_clusters, seed.spawn(len(compare_number_clusters)))
        ]

    def __len__(self):
        return len(self.comparison_methods) * len(self.cluster_generators) * self.trials

    def __iter__(self) -> Iterable[tuple[dict[str, Any], np.ndarray, np.ndarray, ComparisonMethod]]:
        # Weird ordering for smoother progress bar
        for trial_number in range(self.trials):
            for cluster_generator in self.cluster_generators:
                for comparison_method in self.comparison_methods:
                    labels_pred = cluster_generator.random()
                    yield {"n": cluster_generator._n, "k_reference": self.k_reference, "k_compare": cluster_generator._k, "trial": trial_number}, self.reference_clustering, labels_pred, comparison_method


class RealExperiment(Experiment):
    def __init__(self, comparison_methods: list[tuple[MethodFamily, AdjustmentType, RandomModel, RandomModel]], data_dir: Path = Path("./data/MCPdeSouto"), seed: SeedSequence | None = None):
        super().__init__(comparison_methods, seed)
        data_dir.mkdir(exist_ok=True, parents=True)

        self.real_experiment = RealExperimentMCPdeSouto(data_dir)
        self.real_experiment.download_data()

    def __len__(self):
        return len(self.comparison_methods) * len(ClusteringAlgorithms) * len(self.real_experiment.datasets.keys())

    def __iter__(self) -> Iterable[tuple[dict[str, Any], np.ndarray, np.ndarray, ComparisonMethod]]:
        for dataset_name in self.real_experiment.datasets.keys():
            labels_true = self.real_experiment.get_dataset(dataset_name)[1]
            for algorithm, clustering_seed in zip(ClusteringAlgorithms, self.seed.spawn(len(ClusteringAlgorithms))):
                labels_pred = self.real_experiment.get_clustering(
                    dataset_name, algorithm, clustering_seed)
                for comparison_method in self.comparison_methods:
                    yield {"dataset": dataset_name, "algorithm": algorithm.name}, labels_true, labels_pred, comparison_method


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("--output_dir", type=Path,
                        help="Output directory for the results", default=Path("./results"))
    parser.add_argument(
        "--experiment", nargs="+", choices=["synthetic", "real"], help="Type of experiment to run", default=["synthetic", "real"])

    # Comparison method arguments
    parser.add_argument("--method-families", nargs="+", choices=[
                        "RI", "MI"], help="Method families to compare", default=["RI", "MI"])
    parser.add_argument("--adjustments", nargs="+", choices=["NONE", "NORMALIZED", "ADJUSTED",
                        "STANDARDIZED"], help="Adjustments to use", default=["NORMALIZED", "ADJUSTED", "STANDARDIZED"])
    parser.add_argument("--random-models", nargs="+", choices=[
                        "PERM", "NUM", "ALL"], help="Random models to use", default=["PERM", "NUM", "ALL"])

    # Real dataset arguments
    parser.add_argument("--data_dir", type=Path,
                        help="Path to the data directory for the real experiment", default=Path("./data/MCPdeSouto"))

    # Synthetic dataset arguments
    parser.add_argument("--number_datapoints", type=int,
                        help="Number of datapoints for the synthetic experiment", default=500)
    parser.add_argument("--reference_number_clusters", type=int,
                        help="Number of clusters for the synthetic experiment", default=10)
    parser.add_argument("--compare_number_clusters", type=int, nargs="+",
                        help="Number of clusters to compare to the reference clustering", default=[2, 6, 10, 14, 18, 22])
    parser.add_argument(
        "--trials", type=int, help="Number of trials for the synthetic experiment", default=1000)

    # Optional arguments
    parser.add_argument("--time_per_experiment", type=float,
                        help="Total time limit for the synthetic/real experiment in seconds", default=86400.0)
    parser.add_argument("--n_jobs", type=int,
                        help="Number of parallel jobs to run", default=None)
    parser.add_argument("--seed", type=int,
                        help="Seed for random number generation", default=42)
    parser.add_argument("--chunk_number", type=int,
                        help="Chunk number for distributed computing", default=0)
    parser.add_argument("--total_chunks", type=int,
                        help="Total number of chunks for distributed computing", default=1)
    args = parser.parse_args()

    seed = SeedSequence(args.seed)
    comparison_methods = []
    for random_model_pred in args.random_models:
        for random_model_true in {RandomModel.PERM, RandomModel[random_model_pred]}:
            for family in args.method_families:
                for adjustment in args.adjustments:
                    comparison_methods.append(
                        (MethodFamily[family], AdjustmentType[adjustment], random_model_true, RandomModel[random_model_pred]))

    # Make sure the output directory exists
    args.output_dir.mkdir(exist_ok=True, parents=True)

    output_name_suffix = ""
    if args.total_chunks > 1:
        output_name_suffix = f"_chunk{args.chunk_number}_of{args.total_chunks}"

    if "synthetic" in args.experiment:
        print("Running synthetic experiment")
        experiment = SyntheticExperiment(
            comparison_methods,
            args.number_datapoints,
            args.reference_number_clusters,
            args.compare_number_clusters,
            args.trials,
            seed
        )
        results = experiment.run(
            args.time_per_experiment, args.n_jobs, args.chunk_number, args.total_chunks)
        results.to_csv(args.output_dir /
                       f"synthetic{output_name_suffix}.csv", index=False)
    if "real" in args.experiment:
        print("Running real experiment")
        experiment = RealExperiment(
            comparison_methods,
            args.data_dir,
            seed
        )
        results = experiment.run(
            args.time_per_experiment, args.n_jobs, args.chunk_number, args.total_chunks)
        results.to_csv(args.output_dir /
                       f"real{output_name_suffix}.csv", index=False)
