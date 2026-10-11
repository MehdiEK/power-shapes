import numpy as np
import gudhi as gd


def persistence_diagram(
        net_load: np.ndarray, 
        filtration="sublevel"
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

    return intervals
