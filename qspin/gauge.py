"""
Gauge-fixing utilities for generalized Ising spin variables.

The package works with zero-sum (gauge-fixed) spin states.  Raw integer-coded
data {1, 2, ..., Q} is transformed into

    {-(Q-1)/2, ..., 0, ..., (Q-1)/2}     if Q is odd
    {-(Q-1), -(Q-1)+2, ..., -1, 1, ..., Q-1}    if Q is even

so that the spin values are symmetric about zero.  All inference and MCMC
routines in :mod:`isingq` expect this representation.
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
        Bin edges placed at half-integer positions between gauge-fixed states.
    """
    if Q % 2 == 1:
        edges = np.arange(1, Q + 2)
        edges = edges - np.mean(edges)
    else:
        edges = np.arange(1, Q + 2)
        edges = 2 * (edges - np.mean(edges))
    return edges


def possible_states(Q):
    """Return the array of gauge-fixed spin values for a Q-state model.

    Parameters
    ----------
    Q : int
        Number of distinct spin states.

    Returns
    -------
    ndarray
        Centred, integer-valued spin states.
    """
    if Q % 2 == 1:
        states = np.arange(1, Q + 1)
        states = states - np.mean(states)
    else:
        states = np.arange(1, Q + 1)
        states = 2 * (states - np.mean(states))
    return states


def gaugefixing_data(Y, Q):
    """Apply integer gauge-fixing to a dataset of raw spin codes.

    Parameters
    ----------
    Y : ndarray, shape (N, M)
        Raw data, with integer entries drawn from a Q-state alphabet.
    Q : int
        Number of spin states.

    Returns
    -------
    ndarray, shape (N, M), dtype int
        Gauge-fixed data with zero-sum spin states.
    """
    states = np.array(list(set(Y.flatten())), dtype=int)
    if Q % 2 == 1:
        Y = Y - int(np.mean(states))
    else:
        Y = np.array(2 * (Y - np.mean(states)), dtype=int)
    return Y


def gaugefixing_data_float(Y, Q):
    """Float-valued version of :func:`gaugefixing_data`.

    Useful when the gauge correction yields a non-integer mean.
    """
    states = np.array(list(set(Y.flatten())), dtype=int)
    if Q % 2 == 1:
        Y = Y - np.mean(states)
    else:
        Y = np.array(2 * (Y - np.mean(states)))
    return Y
