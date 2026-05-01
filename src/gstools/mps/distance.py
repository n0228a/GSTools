"""Pure distance functions for MPS pattern comparison.

No class state — takes arrays and scalars, returns floats.
``TrainingImage.distance()`` uses these internally; other algorithms
can import them directly.
"""

import numpy as np

__all__ = [
    "compute_node_weights",
    "categorical_dist",
    "l1_dist",
    "l2_dist",
    "lp_dist",
    "variation_dist",
]


def compute_node_weights(
    n, lag_norms, distance_power, cond_mask=None, cond_weight=1.0
):
    """Compute normalized spatial-decay weights for a data event.

    Combines spatial decay (Mariethoz2010 Eq. 5) with conditioning data
    multipliers (Mariethoz2010 §3 ¶26).

    Parameters
    ----------
    n : int
        Number of neighbours in the data event.
    lag_norms : array-like or None, shape (n,)
        Euclidean norms ``‖h_i‖`` of each lag vector. ``None`` or
        ``distance_power == 0`` → uniform spatial weights.
    distance_power : float
        Exponent δ. ``0.0`` → uniform.
    cond_mask : array-like of bool, optional
        ``True`` where the neighbour is a conditioning datum.
    cond_weight : float, optional
        Bonus weight multiplier for conditioning nodes.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Node weights normalized to sum to 1.
    """
    if lag_norms is not None and distance_power != 0.0:
        norms = np.asarray(lag_norms, dtype=np.float64)
        norms = np.where(norms == 0.0, 1e-10, norms)
        raw_w = norms ** (-distance_power)
    else:
        raw_w = np.ones(n, dtype=np.float64)

    if cond_mask is not None:
        raw_w = raw_w.copy()
        raw_w[np.asarray(cond_mask, dtype=bool)] *= cond_weight

    return raw_w / raw_w.sum()


def categorical_dist(data_event_sim, data_event_ti, node_weights):
    """Weighted categorical distance (Mariethoz2010 Eq. 3).

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    data_event_ti : numpy.ndarray, shape (n,)
    node_weights : numpy.ndarray, shape (n,)
        Normalized spatial and conditioning weights.

    Returns
    -------
    float
        Distance in [0, 1].
    """
    return float(
        np.dot(
            node_weights,
            (data_event_sim != data_event_ti).astype(np.float64),
        )
    )


def l1_dist(data_event_sim, data_event_ti, node_weights, d_max):
    """Weighted L1 distance / Manhattan (Mariethoz2010 Eq. 6).

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    data_event_ti : numpy.ndarray, shape (n,)
    node_weights : numpy.ndarray, shape (n,)
        Normalized spatial and conditioning weights.
    d_max : float
        Data range for normalization.

    Returns
    -------
    float
        Distance in [0, 1].
    """
    return float(
        np.dot(node_weights, np.abs(data_event_sim - data_event_ti) / d_max)
    )


def l2_dist(data_event_sim, data_event_ti, node_weights, d_max):
    """Weighted L2 / RMS distance (Mariethoz2010 Eq. 4–5).

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    data_event_ti : numpy.ndarray, shape (n,)
    node_weights : numpy.ndarray, shape (n,)
        Normalized spatial and conditioning weights.
    d_max : float
        Data range for normalization.

    Returns
    -------
    float
        Distance in [0, 1].
    """
    return float(
        np.sqrt(
            np.dot(
                node_weights,
                ((data_event_sim - data_event_ti) / d_max) ** 2,
            )
        )
    )


def lp_dist(data_event_sim, data_event_ti, node_weights, d_max, p):
    """Weighted Lp (Minkowski) distance.

    Warning: Computationally heavier than l1_dist or l2_dist due to
    the generic C-level pow() evaluation. Use only when p != 1.0 or 2.0.

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    data_event_ti : numpy.ndarray, shape (n,)
    node_weights : numpy.ndarray, shape (n,)
        Normalized spatial and conditioning weights.
    d_max : float
        Data range for normalization.
    p : float
        The Minkowski exponent (e.g., 1.5, 3.0, 5.0).

    Returns
    -------
    float
        Distance in [0, 1].
    """
    diffs = np.abs(data_event_sim - data_event_ti) / d_max
    return float(np.sum(node_weights * (diffs**p)) ** (1.0 / p))


def variation_dist(data_event_sim, data_event_ti, node_weights, d_max):
    """Weighted variation distance (Mariethoz2010 Eq. 9, de-meaned).

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    data_event_ti : numpy.ndarray, shape (n,)
    node_weights : numpy.ndarray, shape (n,)
        Normalized spatial and conditioning weights.
    d_max : float
        Data range for normalization.

    Returns
    -------
    float
        Distance in [0, 1].
    """
    diffs = (data_event_sim - data_event_sim.mean()) - (
        data_event_ti - data_event_ti.mean()
    )
    return float(np.sqrt(np.dot(node_weights, (diffs / d_max) ** 2)))
