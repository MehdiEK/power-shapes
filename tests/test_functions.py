import numpy as np
from power_shapes.functions import persistence_diagram


def to_h0_array(dgm, inf_policy="clip", clip_value=None):
    """
    Convert one persistence diagram to an (n, 2) float array of H0 points.

    inf_policy:
      - "drop": remove the essential class (death = inf)
      - "clip": replace inf by `clip_value` (e.g. the day's max value),
                so the day's amplitude (max - min) is kept in the diagram
    """
    # adapter for gudhi's [(dim, (b, d)), ...] format
    if len(dgm) and isinstance(dgm[0], tuple) and len(dgm[0]) == 2 and not np.isscalar(dgm[0][1]):
        dgm = [pt for dim, pt in dgm if dim == 0]

    dgm = np.asarray(dgm, dtype=float).reshape(-1, 2)

    inf_mask = np.isinf(dgm[:, 1])
    if inf_policy == "drop":
        dgm = dgm[~inf_mask]
    elif inf_policy == "clip":
        if clip_value is None:
            raise ValueError("clip_value is required when inf_policy='clip'")
        dgm[inf_mask, 1] = clip_value
    else:
        raise ValueError("inf_policy must be 'drop' or 'clip'")

    # remove zero-persistence points (they don't matter for the distance)
    return dgm[dgm[:, 1] > dgm[:, 0]]


def test_persistence_diagram():
    # Test with a simple 1D array
    net_load = np.array([1, 3, 2, 5, 4])
    dgm = persistence_diagram(net_load, filtration="sublevel", clip=True)
    h0_array = to_h0_array(dgm, inf_policy="clip", clip_value=np.max(net_load))
    np.testing.assert_array_equal(h0_array, dgm)