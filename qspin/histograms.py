"""
Confidence intervals for histogram bin counts.

Two analytic methods (Wilson, Agresti-Coull) and one bootstrap method are
provided.  All return either the (lower, upper) interval endpoints or the
half-widths (suitable as ``yerr`` for matplotlib errorbar plots).
"""

import numpy as np
from sklearn.utils import resample
from scipy.stats import norm as normaldist


def wilson_score_interval(p, n, mypvalue, return_interval_lengths=False):
    r"""Wilson score confidence interval for a binomial proportion.

    Given a fraction :math:`p` estimated from :math:`n` samples, returns the
    two-sided confidence interval at level :math:`1 - {\rm mypvalue}`.

    Parameters
    ----------
    p : float or ndarray
        Observed proportion(s).
    n : int
        Sample size.
    mypvalue : float
        Two-sided tail probability (e.g. 0.05).
    return_interval_lengths : bool
        If True, return ``[p - p_minus, p_plus - p]`` (half-widths).
    """
    z = normaldist.ppf(1 - mypvalue/2, loc=0, scale=1)
    p_minus = (p + z ** 2 / (2. * n) - z * np.sqrt(4 * n * p * (1 - p) + z ** 2) / (2 * n)) \
              * (1 + z ** 2 / n) ** (-1)
    p_plus = (p + z ** 2 / (2. * n) + z * np.sqrt(4 * n * p * (1 - p) + z ** 2) / (2 * n)) \
             * (1 + z ** 2 / n) ** (-1)
    if return_interval_lengths:
        return np.array([p - p_minus, p_plus - p])
    return np.array([p_minus, p_plus])


def agresti_coull_interval(p, n, mypvalue, return_interval_lengths=False):
    """Agresti-Coull confidence interval for a binomial proportion."""
    z = normaldist.ppf(1 - mypvalue/2, loc=0, scale=1)
    ntilde = n + z ** 2
    ptilde = (n * p + z ** 2 / 2) / ntilde
    p_minus = ptilde - z * np.sqrt(ptilde * (1 - ptilde) / ntilde)
    p_plus = ptilde + z * np.sqrt(ptilde * (1 - ptilde) / ntilde)
    if return_interval_lengths:
        return np.array([p - p_minus, p_plus - p])
    return np.array([p_minus, p_plus])


def bootstrap_histogram(Y, bins, n_bootstraps=1000, mypvalue=0.05,
                        return_interval_lengths=False):
    """Bootstrap confidence band for a histogram of ``Y``.

    Parameters
    ----------
    Y : ndarray, shape (N,)
        Data to histogram.
    bins : array-like
        Histogram bin edges.
    n_bootstraps : int
        Number of bootstrap replicates.
    mypvalue : float
        Tail probability used for the percentile interval.
    return_interval_lengths : bool
        If True, return half-widths around the mean instead of (low, high).
    """
    N = len(Y)
    mybins_bootstrap = np.copy(bins)
    hists_bootstrapped = np.zeros((n_bootstraps, len(bins) - 1))
    for n in range(n_bootstraps):
        subject_indices = resample(range(N), replace=True, n_samples=N)
        HB_aux = np.histogram(Y[subject_indices], bins=mybins_bootstrap, density=True)
        hists_bootstrapped[n] = HB_aux[0]
    H_mean = np.mean(hists_bootstrapped, axis=0)
    conf_intervals = np.quantile(hists_bootstrapped, [mypvalue, 1 - mypvalue], axis=0)
    if return_interval_lengths:
        return np.array([H_mean - conf_intervals[0, :], conf_intervals[1, :] - H_mean])
    return conf_intervals
