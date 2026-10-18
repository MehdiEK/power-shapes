import numpy as np
import pandas as pd
import gudhi as gd

from itertools import combinations
from joblib import Parallel, delayed
from gudhi.hera import wasserstein_distance as _wd_hera 

from power_shapes.loader import load_rte_data
from power_shapes.utils import scale_columns_by_daily_mean


def persistence_diagram(
        net_load: np.ndarray, 
        filtration:str, 
        clip:bool=False
) -> np.array:
    """
    """
    assert (
            filtration in ["sublevel", "suplevel"], 
            "filtration must be either 'sublevel' or 'suplevel'."
        )
    
    x = np.asarray(net_load, dtype=float)

    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("net_load must be a 1D array of at least two finite numbers.")

    # Sublevel : creux ; superlevel : pointes
    f = x if filtration == "sublevel" else -x

    # Construire le graphe-chemin
    st = gd.SimplexTree()

    for i, value in enumerate(f):
        st.insert([i], filtration=float(value))

    for i in range(len(f) - 1):
        st.insert([i, i + 1],
                  filtration=float(max(f[i], f[i + 1])))

    # Calculer la persistance
    st.compute_persistence()

    # Extraire les intervalles H0
    intervals = st.persistence_intervals_in_dimension(0)

    # Replace infinite death times with the maximum value of f
    if clip:
        max_value = np.max(f)
        intervals[intervals[:, 1] == np.inf, 1] = max_value

    return intervals


def _pair_distance(d1, d2, order, internal_p, delta):
    """
    Compute the Wasserstein distance between two persistence diagrams.
    """
    return _wd_hera(d1, d2, order=order, internal_p=internal_p, delta=delta)


def _chunk(pairs, diagrams, order, internal_p, delta):
    return [
        _pair_distance(
            diagrams[i], diagrams[j], order, internal_p, delta
        )
        for i, j in pairs
    ]


def wasserstein_distance_matrix(
        diagrams, 
        order=1, 
        internal_p=2, 
        delta=0.01, 
        n_jobs=1, 
        chunk_size=2000
    ):
    """
    Symmetric matrix of pairwise Wasserstein distances between persistence 
    diagrams.

    diagrams : list of (n_i, 2) arrays (finite points only, see to_h0_array)
    order    : Wasserstein order p (1 or 2 are the usual choices)
    internal_p : ground metric between pts (2 = Euclidean, np.inf = sup norm)
    """
    n = len(diagrams)
    pairs = list(combinations(range(n), 2))  # only the upper triangle
    chunks = [pairs[k:k + chunk_size] for k in range(0, len(pairs), chunk_size)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_chunk)(c, diagrams, order, internal_p, delta) for c in chunks
    )

    D = np.zeros((n, n))
    flat = np.concatenate(results) if results else np.array([])
    iu = np.triu_indices(n, k=1)
    D[iu] = flat
    D = D + D.T  # symmetrize
    return D


def run_compute_wasserstein_distance( 
        date_min:str,
        date_max:str,
        order:int, 
        internal_p:int, 
        delta:float, 
        n_jobs:int
) -> np.array:
    """
    Compute the Wasserstein distance between all pairs of persistence diagrams.
    """
    # load data
    df = load_rte_data()

    # filter data by date
    df = df[
        (df['datetime'].dt.date >= pd.to_datetime(date_min).date()) 
        & (df['datetime'].dt.date <= pd.to_datetime(date_max).date())
    ]

    # scale data
    scaled_df = scale_columns_by_daily_mean(
        df,
        columns_by_mean={
            "consumption_mw": ["solar_mw", "wind_mw", "net_demand_solar_mw", "net_demand_mw"]
        }
    )

    # compute all the diagrams
    diagrams = {}

    for date, day_ts in scaled_df.groupby(scaled_df['datetime'].dt.date):
        net_load = day_ts['net_demand_solar_mw'].values
        diagram = persistence_diagram(net_load, filtration="sublevel", clip=True)
        diagrams[date] = diagram

    # compute the distance matrix
    distance_matrix = wasserstein_distance_matrix(
        diagrams=list(diagrams.values()),
        order=order,
        internal_p=internal_p,
        delta=delta,
        n_jobs=n_jobs
    )

    # save the results in a dataframe
    idx = pd.DatetimeIndex(list(diagrams.keys()), name="date")

    return pd.DataFrame(
        distance_matrix,
        index=idx,
        columns=idx
    )