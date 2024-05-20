import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import (
    MiniBatchKMeans,
    KMeans,
    AffinityPropagation,
    MeanShift,
    SpectralClustering,
    AgglomerativeClustering,
    DBSCAN,
    HDBSCAN,
    OPTICS,
    Birch,
)
from sklearn.mixture import GaussianMixture
from enum import Enum
from numpy.random import SeedSequence, MT19937, RandomState
import pickle
import warnings


class ClusteringAlgorithms(Enum):
    MINI_BATCH_KMEANS = "mini_batch_kmeans"
    KMEANS = "kmeans"
    AFFINITY_PROPAGATION = "affinity_propagation"
    MEAN_SHIFT = "mean_shift"
    SPECTRAL_CLUSTERING = "spectral_clustering"
    WARD = "ward"
    AGGLOMERATIVE_CLUSTERING = "agglomerative_clustering"
    DBSCAN = "dbscan"
    HDBSCAN = "hdbscan"
    OPTICS = "optics"
    BIRCH = "birch"
    GAUSSIAN_MIXTURE = "gaussian_mixture"


def clustering(
    X: np.ndarray,
    algorithm: ClusteringAlgorithms,
    n_clusters: int,
    random_state: SeedSequence | None = None,
) -> np.ndarray:
    """Applies a clustering algorithm to the data.

    Args:
        X: The data to cluster.
        algorithm: The algorithm to use.
        n_clusters: The number of clusters.
        random_state: The random state to use.

    Returns:
        The labels of the clusters.
    """
    if random_state is None:
        random_state = SeedSequence()

    legacy_random_state = RandomState(MT19937(random_state))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        match algorithm:
            case ClusteringAlgorithms.MINI_BATCH_KMEANS:
                return MiniBatchKMeans(
                    n_clusters=n_clusters, random_state=legacy_random_state
                ).fit_predict(X)
            case ClusteringAlgorithms.KMEANS:
                return KMeans(
                    n_clusters=n_clusters, random_state=legacy_random_state
                ).fit_predict(X)
            case ClusteringAlgorithms.AFFINITY_PROPAGATION:
                return AffinityPropagation(
                    random_state=legacy_random_state
                ).fit_predict(X)
            case ClusteringAlgorithms.MEAN_SHIFT:
                return MeanShift().fit_predict(X)
            case ClusteringAlgorithms.SPECTRAL_CLUSTERING:
                return SpectralClustering(
                    n_clusters=n_clusters, random_state=legacy_random_state
                ).fit_predict(X)
            case ClusteringAlgorithms.WARD:
                return AgglomerativeClustering(
                    n_clusters=n_clusters, linkage="ward"
                ).fit_predict(X)
            case ClusteringAlgorithms.AGGLOMERATIVE_CLUSTERING:
                return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)
            case ClusteringAlgorithms.DBSCAN:
                return DBSCAN().fit_predict(X)
            case ClusteringAlgorithms.HDBSCAN:
                return HDBSCAN().fit_predict(X)
            case ClusteringAlgorithms.OPTICS:
                return OPTICS().fit_predict(X)
            case ClusteringAlgorithms.BIRCH:
                return Birch(n_clusters=n_clusters).fit_predict(X)
            case ClusteringAlgorithms.GAUSSIAN_MIXTURE:
                return GaussianMixture(
                    n_components=n_clusters, random_state=legacy_random_state
                ).fit_predict(X)


class SyntheticExperimentGatesAndAhn:
    def __init__(self, clusters: dict[str, np.ndarray] | None = None):
        if clusters is None:
            self.clusters = {
                "W": np.array(
                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
                ),
                "X": np.array(
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
                ),
                "Y": np.array(
                    [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
                ),
                "Z": np.array(
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
                ),
            }
        else:
            self.clusters = clusters


class RealExperimentMCPdeSouto:
    datasets = {
        "Armstrong-2002-v1": "Affymetrix",
        "Armstrong-2002-v2": "Affymetrix",
        "Bhattacharjee-2001": "Affymetrix",
        "Chowdary-2006": "Affymetrix",
        "Dyrskjot-2003": "Affymetrix",
        "Golub-1999-v1": "Affymetrix",
        "Golub-1999-v2": "Affymetrix",
        "Gordon-2002": "Affymetrix",
        "Laiho-2007": "Affymetrix",
        "Nutt-2003-v1": "Affymetrix",
        "Nutt-2003-v2": "Affymetrix",
        "Nutt-2003-v3": "Affymetrix",
        "Pomeroy-2002-v1": "Affymetrix",
        "Pomeroy-2002-v2": "Affymetrix",
        "Ramaswamy-2001": "Affymetrix",
        "Shipp-2002-v1": "Affymetrix",
        "Singh-2002": "Affymetrix",
        "Su-2001": "Affymetrix",
        "West-2001": "Affymetrix",
        "Yeoh-2002-v1": "Affymetrix",
        "Yeoh-2002-v2": "Affymetrix",
        "Alizadeh-2000-v1": "CDNA",
        "Alizadeh-2000-v2": "CDNA",
        "Alizadeh-2000-v3": "CDNA",
        "Bittner-2000": "CDNA",
        "Bredel-2005": "CDNA",
        "Chen-2002": "CDNA",
        "Garber-2001": "CDNA",
        "Khan-2001": "CDNA",
        "Lapointe-2004-v1": "CDNA",
        "Lapointe-2004-v2": "CDNA",
        "Liang-2005": "CDNA",
        "Risinger-2003": "CDNA",
        "Tomlins-2006": "CDNA",
        "Tomlins-2006-v2": "CDNA",
    }
    base_url = "https://schlieplab.org/Static/Supplements/CompCancer/"

    def __init__(self, data_dir: Path) -> None:
        if not data_dir.is_dir() or not data_dir.exists():
            raise ValueError(f"{data_dir} is not a valid directory.")

        self.data_dir = data_dir

    def download_data(
        self, include_description: bool = False, silent: bool = True
    ) -> None:
        """Download the datasets.

        Args:
            include_description (bool): Whether to download the description.
                Defaults to False.
            silent (bool): Whether to print progress. Defaults to False.
        """
        for dataset_name, dataset_type in tqdm(
            self.datasets.items(),
            desc="Downloading datasets",
            leave=None,
            disable=silent,
        ):
            dataset_dir = self.data_dir / dataset_name.lower()
            dataset_dir.mkdir(exist_ok=True)
            dataset_filename = dataset_dir / f"database.txt"

            if not dataset_filename.exists():
                dataset_base_url = (
                    f"{self.base_url}{dataset_type}/{dataset_name.lower()}/"
                )
                # Download in chunks as these files can be large:
                with requests.get(
                    f"{dataset_base_url}{dataset_name.lower()}_database.txt",
                    stream=True,
                ) as r:
                    r.raise_for_status()
                    with open(dataset_filename, "wb") as f:
                        for chunk in tqdm(
                            r.iter_content(chunk_size=8192),
                            leave=None,
                            desc=f"Downloading {dataset_name}",
                            disable=silent,
                        ):
                            f.write(chunk)
            elif not silent:
                tqdm.write(
                    f"{dataset_name} data already downloaded. Skipping download."
                )

            if include_description:
                description_filename = dataset_dir / f"description.htm"
                if not description_filename.exists():
                    author = dataset_name.split("-")[0].lower()
                    # Download normally:
                    with requests.get(
                        f"{dataset_base_url}{author}_description.htm"
                    ) as r:
                        with open(description_filename, "wb") as f:
                            f.write(r.content)
                elif not silent:
                    tqdm.write(
                        f"{dataset_name} description already downloaded. Skipping download."
                    )

    def get_dataset(self, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read the dataset and return the data and the labels.

        Args:
            dataset_name (str): The name of the dataset.

        Returns:
            tuple[np.ndarray, np.ndarray]: The data X and the labels y.
                The data X is an m x n matrix, where m is the number of
                samples and n is the number of genes. The labels y is an
                array of length m.
        """
        filename = self.data_dir / dataset_name.lower() / "database.txt"

        if (not filename.exists()) or (not filename.is_file()):
            raise ValueError(
                f"{filename} does not exist. Perhaps you need to download it?"
            )

        # Get the first row of the dataset:
        with open(filename, "r") as f:
            original_sample_labels = [
                label.strip('"') for label in f.readline().split()
            ]

        # Read everything else:
        df = pd.read_csv(
            self.data_dir / dataset_name.lower() / "database.txt",
            sep="\t",
            header=1,
            index_col=0,
        )
        # df.index.name = "genes"
        df = df.transpose()

        if df.values.dtype == "object":
            # Mean imputation if necessary.
            df = df.apply(pd.to_numeric, errors="coerce")
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mean())

        # Get the labels:
        labels = np.array([sample_label.split(".")[0]
                          for sample_label in df.index])

        # Make labels numeric:
        labels = LabelEncoder().fit_transform(labels)

        return df.values, labels

    def get_clustering(
        self,
        dataset_name: str,
        algorithm: ClusteringAlgorithms,
        seed: SeedSequence | None = None,
    ) -> np.ndarray:
        """Get the clustering of the dataset.

        Args:
            dataset_name (str): The name of the dataset.
            algorithm (ClusteringAlgorithms): The algorithm to use.
            seed (SeedSequence | None, optional): The seed to use. Defaults to None.

        Returns:
            np.ndarray: The clustering.
        """
        cache_files = list(
            (self.data_dir / dataset_name.lower()
             ).glob(f"{algorithm.value}*.pkl")
        )

        if seed is None:
            if len(cache_files) > 0:
                with open(cache_files[0], "rb") as f:
                    return pickle.load(f)
            seed = SeedSequence()

        result_filename = (
            self.data_dir
            / dataset_name.lower()
            / f"{algorithm.value}_{hash(seed.entropy):x}.pkl"
        )

        if result_filename.exists() and result_filename.is_file():
            with open(result_filename, "rb") as f:
                return pickle.load(f)

        X, y_true = self.get_dataset(dataset_name)
        n_clusters = len(np.unique(y_true))
        y_pred = clustering(X, algorithm, n_clusters, random_state=seed)

        # Save the clustering:
        with open(result_filename, "wb") as f:
            pickle.dump(y_pred, f)

        # Delete the old cache files:
        if len(cache_files) > 0:
            for cache_file in cache_files:
                cache_file.unlink()
        return y_pred
