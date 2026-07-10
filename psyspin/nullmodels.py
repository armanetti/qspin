"""
Null models and similarity-matrix helpers.

This module collects three families of helpers used to benchmark the
generalized-Ising fits against simpler baselines:

* **Simple data-driven generators** (categorical-independent, Gaussian,
  Gaussian-with-discretization, Gaussian-copula).  They produce synthetic
  datasets that share some marginal or correlation structure with the data,
  but not the higher-order interactions captured by the Ising-type models.
* **Spectral / similarity null models** (Newman modularity, GMM-style
  spectral cleaning) for sanity-checking inferred coupling structure.
* **Analytical distance distributions** (mixture-of-chi-squared
  characteristic function) used in distance-histogram diagnostics.
"""

import numpy as np
from random import sample, choice
from sklearn.decomposition import FactorAnalysis
from scipy.sparse.linalg import eigs
from scipy.stats import spearmanr, norm
from scipy.integrate import quad
from scipy.special import gamma, gammaincinv, gammainc

from .gauge import bins as _gauge_bins


# ---------------------------------------------------------------------------
# Simple count-based null model
# ---------------------------------------------------------------------------

def countFreq(arr):
    """Return [[value, relative_frequency], ...] for the entries of ``arr``."""
    mp = {}
    ans = []
    L = len(arr)
    for num in arr:
        mp[num] = mp.get(num, 0) + 1
    for num, freq in mp.items():
        ans.append([num, freq / L])
    return ans


def catind_model(Y, R, N=1000):
    """Categorical-independent null model: each column drawn from its own marginal.

    Parameters
    ----------
    Y : ndarray, shape (N_data, M)
        Gauge-fixed dataset.
    R : int
        Number of distinct spin states.
    N : int
        Sample size to generate.
    """
    _, M = Y.shape
    constant = 1.

    frequencies = np.zeros((M, R))
    mybins = _gauge_bins(R)
    for j in range(M):
        frequencies[j], _ = np.histogram(Y[:, j], bins=mybins, density=True)
        frequencies[j] *= constant

    states = np.sort(np.array(list(set(Y.flatten())), dtype=int))
    X_catind = np.zeros((N, M), dtype=int)
    for j in range(M):
        X_catind[:, j] = np.random.choice(states, p=frequencies[j], size=(N))
    return X_catind


# ---------------------------------------------------------------------------
# Gaussian-copula and Gaussian-with-discretization null models
# ---------------------------------------------------------------------------

def null_gaussian_copula(X_data, N, M, R, nfa):
    """Gaussian-copula null model with Spearman-based dependence structure.

    Steps:
        1. Compute Spearman correlation rho_S(X_data).
        2. Map to the latent Gaussian correlation:
              Sigma = 2 sin(pi rho_S / 6).
        3. Project Sigma to the closest PSD matrix (eigenvalue clipping).
        4. Sample from N(0, Sigma), pass through the normal CDF
           and through each item's empirical inverse CDF.
    """
    rho_spearman, _ = spearmanr(X_data)
    Sigma_copula = 2 * np.sin(np.pi / 6 * rho_spearman)
    np.fill_diagonal(Sigma_copula, 1.0)

    eigvals, eigvecs = np.linalg.eigh(Sigma_copula)
    if np.any(eigvals < 0):
        eigvals[eigvals < 0] = 1e-8
        Sigma_copula = eigvecs @ np.diag(eigvals) @ eigvecs.T
        np.fill_diagonal(Sigma_copula, 1.0)

    mu = np.zeros(M)
    Z_samples = np.random.multivariate_normal(mu, Sigma_copula, size=N)
    U_samples = norm.cdf(Z_samples)

    X_disc = np.empty((N, M), dtype=X_data.dtype)
    for j in range(M):
        counts = np.bincount(X_data[:, j], minlength=R + 1)
        cum_probs = np.cumsum(counts) / np.shape(X_data)[0]
        cutoffs = cum_probs[:-1]
        X_disc[:, j] = np.searchsorted(cutoffs, U_samples[:, j])

    return X_disc


def model_gaussdisc(Y, R, N):
    """Gaussian-then-discretize null model.

    Draws ``N`` rows from N(mean(Y), cov(Y)) and discretizes each column to
    R levels via empirical quantile cutoffs derived from the *continuous*
    sample (keeping each item's marginal close to the empirical one).
    """
    _, M = Y.shape
    frequencies = np.zeros((M, R))
    mybins = _gauge_bins(R)
    X_disc = np.empty((N, M), dtype=Y.dtype)

    constant = 1.

    mu = np.mean(Y, axis=0)
    Sigma = np.cov(Y.T)
    X_cont = np.random.multivariate_normal(mu, Sigma, size=N)

    for j in range(M):
        frequencies[j], _ = np.histogram(Y[:, j], bins=mybins, density=True)
        frequencies[j] *= constant
        quantile_boundaries = np.cumsum(100. * frequencies[j])[:-1]
        cuts = np.percentile(X_cont[:, j], quantile_boundaries)
        X_disc[:, j] = np.searchsorted(cuts, X_cont[:, j])
    return X_disc


def _generate_null_for_spectral(X, N, M, R, nfa, null_model):
    """Generate a single null realization for spectral filtering.

    ``null_model`` is one of:

    * ``'subj_randomisation'``: shuffle each column independently across subjects;
    * ``'fa_generative'``: factor-analysis generative model, then discretize;
    * ``'gauss'``: multivariate Gaussian with empirical mean and covariance,
      then discretize.
    """
    X = np.asarray(X)
    if null_model == 'subj_randomisation':
        ssi = np.random.choice(range(N), (N, M))
        return np.array([X[ssi[:, j], j] for j in range(M)]).T

    if null_model == 'fa_generative':
        fa = FactorAnalysis(n_components=nfa, random_state=None)
        fa.fit(X)
        mu = fa.mean_
        Sigma = reconstruct_covariance_fa(fa)
        X_cont = np.random.multivariate_normal(mu, Sigma, size=N)
        quantile_boundaries = np.linspace(0, 100, R + 1)[1:-1]
        X_disc = np.empty((N, M), dtype=X.dtype)
        for j in range(M):
            cuts = np.percentile(X[:, j], quantile_boundaries)
            cuts = np.unique(cuts)
            X_disc[:, j] = np.searchsorted(cuts, X_cont[:, j])
        return X_disc

    if null_model == 'gauss':
        mu = np.mean(X, axis=0)
        Sigma = np.cov(X.T)
        X_cont = np.random.multivariate_normal(mu, Sigma, size=N)
        quantile_boundaries = np.linspace(0, 100, R + 1)[1:-1]
        X_disc = np.empty((N, M), dtype=X.dtype)
        for j in range(M):
            cuts = np.percentile(X[:, j], quantile_boundaries)
            cuts = np.unique(cuts)
            X_disc[:, j] = np.searchsorted(cuts, X_cont[:, j])
        return X_disc

    raise ValueError(f"Unknown null_model: '{null_model}'. "
                     "Use 'subj_randomisation', 'fa_generative' or 'gauss'.")


def reconstruct_covariance_fa(fa):
    """Rebuild the FA-implied covariance Sigma = W W^T + diag(noise)."""
    W = fa.components_.T
    return W @ W.T + np.diag(fa.noise_variance_)


# ---------------------------------------------------------------------------
# Spectral / similarity null models
# ---------------------------------------------------------------------------

def newmanmodularity(S):
    """Newman modularity matrix B = S - ell ell^T / L (with diagonal of S removed).

    ``S`` is a similarity matrix; the returned matrix subtracts the
    configuration-model background.
    """
    N = len(S)
    S0 = S - np.eye(N)
    ell = np.sum(S0, axis=1)
    L = np.sum(ell)
    return (S0 - np.outer(ell, ell) / L)


def noiseclean_general(list_eigvecs, w, V):
    """Reconstruct a low-rank approximation from selected eigen-pairs."""
    N, _ = np.shape(V)
    Osignal = np.zeros((N, N))
    for m in list_eigvecs:
        Osignal += np.real(w[m] * np.outer(V.T[m], V.T[m]))
    return Osignal


def modularity_general(S, nb_to_keep, remove1stmode=True):
    """Keep the top-``nb_to_keep`` eigenmodes of ``S``, optionally removing the leading one."""
    w, V = eigs(S, nb_to_keep + 1)
    ordine = np.argsort(w)[::-1]
    w = w[ordine]
    V = V[:, ordine]
    rango = range(1, nb_to_keep + 1) if remove1stmode else range(0, nb_to_keep)
    return noiseclean_general(rango, w, V)


def resize_spectrum(Lambda, m, m0=0):
    """Rescale an eigenvalue subset to preserve total variance under truncation."""
    L = len(Lambda)
    total_var = np.sum(Lambda)
    partial_var = np.sum(Lambda[m0:-m])
    return (total_var / partial_var) * Lambda[m0:-m] * ((L - m - m0) / L)


def modularity_GMM(X, my_similarity, remove1stmode=True, many_returns=False):
    """GMM-style spectral cleaning of an empirical similarity matrix.

    Compares the empirical eigenvalue spectrum to the spectrum of a
    subject-shuffled null and iteratively removes noise eigenvalues until
    the remaining bulk falls inside the null support.

    Parameters
    ----------
    X : ndarray, shape (N, M)
        Empirical dataset.
    my_similarity : callable
        Function (X -> M x M) that builds a similarity matrix.
    remove1stmode : bool
        If True, ignore the leading eigenvalue ("market mode") when comparing
        to the null spectrum.
    many_returns : bool
        If True, also return the null and signal spectra.
    """
    Y = np.copy(X)
    N, M = Y.shape

    ssi = np.random.choice(range(N), (N, M))
    Yshu = np.array([Y[ssi[:, i], i] for i in range(M)]).T

    S_Y = my_similarity(Y)
    w_Y, V_Y = eigs(S_Y, M)
    indices = np.argsort(w_Y)
    w_Y = w_Y[indices]
    V_Y = V_Y[:, indices]

    S_Yshu = my_similarity(Yshu)
    w_Yshu, _ = eigs(S_Yshu, M)
    w_Yshu = np.sort(w_Yshu)

    if remove1stmode:
        lambdaminus, lambdaplus = w_Yshu[0], w_Yshu[-2]
    else:
        lambdaminus, lambdaplus = w_Yshu[0], w_Yshu[-1]

    if remove1stmode:
        m = 2
        Lambda_noise = np.copy(w_Y[:-1])
        while Lambda_noise[-1] > lambdaplus or Lambda_noise[0] < lambdaminus:
            Lambda_noise = resize_spectrum(w_Y[:-1], m)
            m += 1
    else:
        m = 1
        Lambda_noise = np.copy(w_Y)
        while Lambda_noise[-1] > lambdaplus or Lambda_noise[0] < lambdaminus:
            Lambda_noise = resize_spectrum(w_Y, m)
            m += 1

    nb_to_keep = M - m
    if remove1stmode:
        rango = range(M - nb_to_keep - 1, M - 1, 1)
    else:
        rango = range(M - nb_to_keep, M, 1)
    Ssignal = noiseclean_general(rango, w_Y, V_Y)

    if many_returns:
        return Ssignal, w_Yshu, w_Y, Lambda_noise
    return Ssignal


# ---------------------------------------------------------------------------
# Analytical distance distributions
# ---------------------------------------------------------------------------

def energydensity_gauss(xs, d):
    r"""Chi-squared(d) probability density (parametrized in ``xs``)."""
    return xs ** (d / 2 - 1) * np.exp(-xs) / gamma(d / 2)


def energy(x, C, mu):
    r"""Mahalanobis-style quadratic form 1/2 (x-mu)^T C^{-1} (x-mu)."""
    return 0.5 * (x - mu).T @ np.linalg.inv(C) @ (x - mu)


def fourier_transform_integral(func, nu, T):
    r"""Numerically evaluate the Fourier transform of ``func`` at frequency ``nu``
    over the interval [-T/2, T/2]."""
    integrand_real = lambda t: np.real(func(t) * np.exp(-1j * nu * t))
    integrand_imag = lambda t: np.imag(func(t) * np.exp(-1j * nu * t))
    real_part, _ = quad(integrand_real, -T / 2, T / 2)
    imag_part, _ = quad(integrand_imag, -T / 2, T / 2)
    return complex(real_part, imag_part) / (2 * np.pi)


def mixture_chisquared_characteristicfunction(z, lambdas):
    r"""Characteristic function of a mixture of chi-squared distributions.

    For lambdas = (lambda_1, ..., lambda_n), returns
        phi(z) = prod_k (1 - 2 i z lambda_k)^{-1/2}.
    """
    myarray = np.array(1 - 2. * z * 1j * lambdas) ** 0.5
    return np.prod(myarray) ** -1
