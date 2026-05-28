"""
Jackknife error estimation with block resampling.

Used to estimate the integrated autocorrelation time tau_int of a Markov-chain
observable: for a time series of length N, the block-Jackknife variance
divided by the naive (i.i.d.) variance estimates 2 tau_int as the block size
grows past the chain correlation time.
"""

import numpy as np
from scipy.stats import chi2


def JKerror(x, bslist, alpha=0.925):
    r"""Block-Jackknife error and integrated autocorrelation time of a time series.

    For each block size :math:`b_s \in {\rm bslist}` we form
    :math:`n_b = \lfloor N/b_s \rfloor - 1` non-overlapping block means and compute
    their sample variance.  The ratio between this and the naive i.i.d. variance
    estimator is :math:`2\,\tau_{\rm int}`; we report :math:`\tau_{\rm int}` directly.

    The relative confidence interval :math:`(\tau_-, \tau_+)` at level ``alpha``
    is obtained from the chi-squared distribution with :math:`n_b - 1` degrees
    of freedom.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Time series of observable values.
    bslist : array-like of int
        Block sizes to evaluate.
    alpha : float
        Confidence level for the tau error bars (default 0.925).

    Returns
    -------
    Oerrors : ndarray
        Block-Jackknife standard errors.
    taus : ndarray
        Integrated autocorrelation times, one per block size.
    tauerrors : ndarray, shape (len(bslist), 2)
        (upper, lower) limits of the alpha-confidence interval for each tau.
    nbs : ndarray
        Number of blocks for each block size.
    """
    N = len(x)
    naiveerror2 = np.var(x) / N

    n_blocks = len(bslist)
    Oerrors = np.zeros((n_blocks))
    taus = np.zeros((n_blocks))
    tauerrors = np.zeros((n_blocks, 2))
    nbs = np.array(N / bslist - 1, dtype=int)

    for ib, bs in enumerate(bslist):
        nb = int(N / bs) - 1
        blocks = [np.mean(x[i * bs:(i + 1) * bs]) for i in range(nb)]

        myerror2 = np.var(blocks, ddof=1) / nb
        Oerror = np.sqrt(myerror2)

        l = chi2.ppf((1. - alpha) * .5, nb - 1.)
        u = chi2.ppf((1. + alpha) * .5, nb - 1.)
        erroru = myerror2 * (nb - 1.) / (u * naiveerror2)
        errorl = myerror2 * (nb - 1.) / (l * naiveerror2)

        Oerrors[ib] = Oerror
        taus[ib] = myerror2 / naiveerror2 * 0.5
        tauerrors[ib] = [erroru, errorl]

    return Oerrors, taus, tauerrors, nbs
