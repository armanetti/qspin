"""
Gauge-fixing utilities for generalized Ising spin variables.

The package works with zero-sum (gauge-fixed) spin states.  Raw integer-coded
data {1, 2, ..., Q} is transformed into

    {-(Q-1)/2, -(Q-1)/2 + 1, ...,-1, 0, 1, ..., (Q-1)/2}     if Q is odd   (integers)
    {-(Q-1)/2, -(Q-1)/2 + 1, ..., -1/2, 1/2, ..., (Q-1)/2}    if Q is even (half-integers)

matching the paper convention v_q = -(R-1)/2 + (q-1).  For Q even the states
are half-integers and are stored as float64.  All inference and MCMC routines
in :mod:`isingq` expect this representation.
"""

import numpy as np


def bins(Q):
    """Return histogram bin edges adapted to the gauge-fixed Q-state alphabet.

    Parameters
    ----------
    Q : int
        Number of distinct spin states.

    Returns
    -------
    ndarray
        Bin edges placed at unit-spacing positions between gauge-fixed states.
    """
    edges = np.arange(1, Q + 2, dtype=float)
    edges = edges - np.mean(edges)
    return edges


def possible_states(Q):
    """Return the array of gauge-fixed spin values for a Q-state model.

    Parameters
    ----------
    Q : int
        Number of distinct spin states.

    Returns
    -------
    ndarray of float
        Centred spin states: integers for Q odd, half-integers for Q even.
    """
    states = np.arange(1, Q + 1, dtype=float)
    states = states - np.mean(states)
    return states


def gaugefixing_data(Y, Q):
    """Apply gauge-fixing to a dataset of raw spin codes.

    Parameters
    ----------
    Y : ndarray, shape (N, M)
        Raw data, with integer entries drawn from a Q-state alphabet.
    Q : int
        Number of spin states.

    Returns
    -------
    ndarray, shape (N, M), dtype float
        Gauge-fixed data with zero-sum spin states (half-integers for Q even).
    """
    states = np.array(list(set(Y.flatten())), dtype=float)
    return Y.astype(float) - np.mean(states)


def gaugefixing_data_float(Y, Q):
    """Alias of :func:`gaugefixing_data` kept for backwards compatibility."""
    states = np.array(list(set(Y.flatten())), dtype=float)
    return Y.astype(float) - np.mean(states)
