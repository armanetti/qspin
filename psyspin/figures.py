"""
Shared plotting helpers for the figures shown in the accompanying paper.

Each function corresponds to one figure type and saves the PDF directly to
``savepath``.  All functions follow the same signature convention:

    plot_<figure_type>(data_args, ..., dataset, savepath, colors, cfg)

where
  - dataset  : str, questionnaire name (used in titles / filenames)
  - savepath : str, directory where PDFs are saved
  - colors   : dict, colour palette
               keys: 'emp','ising','bc','beg','gauss','gaussdisc','nullcat','copula'
  - cfg      : dict, global configuration (nbins, mypvalue, alpha_emp, ...)

Key cfg entries for distance figures:
  - nbins_emp_dist : int, empirical bins for E2d / Mahalanobis figures
                     (falls back to nbins_emp if not set; default 20)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import norm as normaldist
from scipy.signal import find_peaks
from sklearn.decomposition import FactorAnalysis
import matplotlib as mpl


from .histograms import wilson_score_interval
from .jackknife import JKerror
from .nullmodels import (fourier_transform_integral, 
                         energydensity_gauss, 
                         reconstruct_covariance_fa,
                         mixture_chisquared_characteristicfunction,)


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

def compute_pseudolikelihood_moments(X, states, J, h, K=None):
    r"""Compute the "sleep"-phase conditional moments of the pseudo-likelihood.

    These are exactly the second term of each gradient in
    Eq.~(\ref{eq:PLgradients}) (Appendix \ref{sec:PseudoLik}): the model's
    single-site conditional expectation :math:`\langle \cdot \rangle_{P(x_j |
    {\bf x}_{s,\setminus j},\theta)}`, evaluated at every training
    configuration :math:`s` and averaged over the dataset.  Gradient ascent
    on the pseudo-likelihood drives these quantities towards their empirical
    ("wake") counterparts, so at convergence they must match.

    Handles Ising (``K=None``, diag(J)=0), BC (``K=None``) and BEG
    (``K`` given) alike -- ``varphi_i = 1/2 J_ii + sum_j K_ij x_j^2``.

    Parameters
    ----------
    X : ndarray, shape (N, M)
    states : ndarray, shape (R,)
    J, h : ndarray
        Fitted couplings / fields.
    K : ndarray or None
        Fitted quartic couplings (BEG only). ``None`` reduces to the BC form.

    Returns
    -------
    expected_x : ndarray, shape (M,)
    expected_x2 : ndarray, shape (M,)
    expected_covariance : ndarray, shape (M, M)
        Off-diagonal: (1/N) sum_s x_si <x_j>_{...} - <x_i><x_j> (Jgradpselik sleep term, centred).
        Diagonal: expected_x2 - expected_x**2.
    expected_x2x2 : ndarray, shape (M, M) or None
        (1/N) sum_s x_si^2 <x_j^2>_{...} (Kgradpselik sleep term). ``None`` if ``K`` is ``None``.
    """
    n_samples, n_nodes = np.shape(X)
    phi = X @ J + h - X * np.diag(J)
    varphi = 0.5 * np.diag(J)[np.newaxis, :] * np.ones((n_samples, 1))
    if K is not None:
        varphi = varphi + (X ** 2) @ K
    exponent = phi[:, :, np.newaxis] * states[np.newaxis, np.newaxis, :]
    exponent = exponent + varphi[:, :, np.newaxis] * states[np.newaxis, np.newaxis, :] ** 2
    max_exp = np.max(exponent, axis=2, keepdims=True)
    probs = np.exp(
        exponent - max_exp
        - np.log(np.sum(np.exp(exponent - max_exp), axis=2, keepdims=True))
    )
    expected_x = np.mean(np.sum(probs * states, axis=2), axis=0)
    expected_x2 = np.mean(np.sum(probs * states ** 2, axis=2), axis=0)
    expected_xixj = X.T @ np.sum(probs * states, axis=2) / n_samples
    expected_covariance = expected_xixj - np.outer(expected_x, expected_x)
    expected_covariance[np.diag_indices(n_nodes)] = expected_x2 - expected_x ** 2

    expected_x2x2 = None
    if K is not None:
        expected_x2x2 = (X ** 2).T @ np.sum(probs * states ** 2, axis=2) / n_samples

    return expected_x, expected_x2, expected_covariance, expected_x2x2


def compute_pseudolikelihood_moments_BC(X, states, J, h):
    r"""Compute the BC pseudo-likelihood conditional moments.

    Kept for backward compatibility -- thin wrapper around
    :func:`compute_pseudolikelihood_moments` with ``K=None``, dropping the
    (always ``None``) fourth return value.

    Returns
    -------
    expected_x : ndarray, shape (M,)
    expected_x2 : ndarray, shape (M,)
    expected_covariance : ndarray, shape (M, M)
        Conditional first/second/covariance moments under the pseudo-likelihood
        density.  These are the quantities required to vanish at a stationary
        point of the pseudo-likelihood objective.
    """
    expected_x, expected_x2, expected_covariance, _ = compute_pseudolikelihood_moments(
        X, states, J, h, K=None)
    return expected_x, expected_x2, expected_covariance


def condition(el):
    """Default filter for simulation dictionaries.

    Selects PCD runs (not pseudo-likelihood), with an empirical or random
    reset and non-trivial ``n_intersample_sweeps``.
    """
    return (
        (el['reset'] == 'rand' or el['reset'] == 'emp')
        and el['n_intersample_sweeps'] > 1
        and not el['pseudolikelihood']
    )


def model_label(el, model_name):
    r"""Return a TeX-formatted legend label, e.g. ``'Ising ($\tau_c=1000$)'``."""
    tau = el['n_intersample_sweeps']
    reset = el['reset']
    pse = el['pseudolikelihood']
    suffix = r'$\tau_c=' + str(tau) + r'$'
    if pse:
        suffix += r', pse-lik'
    if reset == 'rand':
        suffix += r', rand'
    return f'{model_name} ({suffix})'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _remove_diag(M):
    """Return the off-diagonal entries of a square matrix, flattened."""
    n = M.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return M[mask]


def _wilson_error(counts, n, pvalue, bin_lengths):
    """Wilson score CI half-widths, scaled by bin_length, for errorbar plots."""
    Hmass = counts * bin_lengths
    return wilson_score_interval(p=Hmass, n=n, mypvalue=pvalue,
                                 return_interval_lengths=True) / bin_lengths


def _emp_hist_with_errors(ax, values, nbins_emp, N, mypvalue, alpha_emp, color,
                          label='empirical'):
    _ms = MODEL_STYLES['emp']
    HB = ax.hist(values, histtype='step', bins=nbins_emp,
                 label=label, density=True, lw=_ms['lw'], color='black', alpha=_ms['alphaline'], 
                 zorder=_ms['zorder'])
    ax.hist(values, alpha=alpha_emp, bins=nbins_emp, density=True, lw=0.,
            color=color, zorder=_ms['zorder'])
    bl = HB[1][1:] - HB[1][:-1]
    yerr = _wilson_error(HB[0], N, mypvalue, bl)
    ax.errorbar(0.5 * (HB[1][:-1] + HB[1][1:]), HB[0],
                yerr=yerr, color=color, capsize=4.,
                fmt='.', ls='', lw=0.65, ms=2.5, capthick=0.65,
                zorder=_ms['zorder'] + 0.5)
    return HB


def _savefig(savepath, fname):
    os.makedirs(savepath, exist_ok=True)
    plt.savefig(os.path.join(savepath, fname), bbox_inches='tight')
    plt.close()


def _model_linestyle(el):
    return ':' if el['reset'] == 'rand' else '-'


def _gauss_sampled_hist(ax, data, bins, col, label=None, zorder=1):
    """Thin continuous step + very thin hatching for sampled Gaussian histograms.

    The step outline is thin (lw=0.4) and the hatch pattern uses the same
    thin linewidth, giving a lightweight texture without filling.
    Returns the histogram output (n, bin_edges, patches).
    """
    HB = ax.hist(data, bins=bins, density=True, histtype='stepfilled',
                 fc='none', edgecolor=col, linewidth=0.4, hatch='//',
                 zorder=zorder, label=label)
    return HB


def _gauss_own_domain_bins(values, nbins):
    """Bin edges spanning the *own* range of ``values`` (not some externally
    fixed / empirical-domain array), so that a sampled Gauss/FA histogram --
    and any analytic curve overlaid on it -- are drawn over their own support
    rather than being clipped to another curve's domain."""
    # cut the max value so that the super rare tail of the sampled Gauss/FA 
    # doesn't get its own bin and make the domain too wide (and the histogram too sparse) for the analytic curve to be visible
    values = values[values < np.percentile(values, 99.8)]
    vmin, vmax = np.min(values), np.max(values)
    pad = 0.02 * (vmax - vmin) if vmax > vmin else 1.0
    return np.linspace(vmin, vmax + pad, nbins + 1)


# ---------------------------------------------------------------------------
# Simple null-model fitting: full Gaussian vs Factor-Analyzers (FA)
# ---------------------------------------------------------------------------

def fit_gauss_or_fa_model(X, method='FA', n_components=None, random_state=None):
    r"""Fit either a full multivariate Gaussian or a Factor-Analyzers (FA)
    model on ``X``, to be used as the "Gauss"/"FA" null baseline throughout
    the figures below.

    * ``method='gauss'``: full multivariate Gaussian with empirical mean and
      covariance, :math:`\Sigma = \mathrm{Cov}(X)`.
    * ``method='FA'`` (default): low-rank + diagonal-noise covariance,
      :math:`\Sigma = WW^\dag + \mathrm{diag}(\psi)`, fitted via
      :class:`sklearn.decomposition.FactorAnalysis`.

    Both are Gaussian generative models (mean + covariance); sampling from
    either one is done with :func:`sample_gauss_or_fa_model`.

    Parameters
    ----------
    X : ndarray, shape (N, M)
    method : {'FA', 'gauss'}
    n_components : int
        Number of latent factors. Required if ``method='FA'``.
    random_state : int or None

    Returns
    -------
    dict with keys:
        'mean'    : ndarray (M,)
        'cov'     : ndarray (M, M), reconstructed covariance
        'eigvals' : ndarray (M,), eigenvalues of 'cov', sorted descending
        'label'   : 'FA' or 'Gauss' -- use in legends (point 3 of the spec)
        'fa'      : fitted FactorAnalysis instance, or None for 'gauss'
        'method'  : 'FA' or 'gauss'
    """
    method_key = 'FA' if str(method).lower() in ('fa', 'factoranalysis', 'factor_analysis') else 'gauss'
    if method_key == 'FA':
        if n_components is None:
            raise ValueError("n_components must be provided when method='FA'")
        fa = FactorAnalysis(n_components=n_components, random_state=random_state)
        fa.fit(X)
        mean = fa.mean_
        cov = reconstruct_covariance_fa(fa)
        label = 'FA'
    else:
        fa = None
        mean = np.mean(X, axis=0)
        cov = np.cov(X.T)
        label = 'Gauss'
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    return dict(mean=mean, cov=cov, eigvals=eigvals, label=label, fa=fa, method=method_key)


def sample_gauss_or_fa_model(model, N, random_state=None):
    r"""Draw ``N`` i.i.d. samples from a model fitted by
    :func:`fit_gauss_or_fa_model` (multivariate normal with the model's mean
    and reconstructed covariance -- for FA this is exactly its generative
    distribution)."""
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        return rng.multivariate_normal(model['mean'], model['cov'], size=N)
    return np.random.multivariate_normal(model['mean'], model['cov'], size=N)


def _add_peak_arrows(ax, values, bins, n_peaks=3, color='black'):
    r"""Draw straight black arrows, tilted at 45 degrees, pointing at the
    ``n_peaks`` tallest local maxima of the histogram of ``values``.

    Each arrow comes from the upper-right (i.e. points top-to-bottom,
    right-to-left) directly onto the peak. The offset is expressed in
    display points (not data units) so the 45-degree tilt is exact on
    screen regardless of the axes' data scaling (e.g. a log y-axis).

    Used for the GCBS first-factor histogram, whose empirical distribution
    is markedly multi-modal.

    ``scipy.signal.find_peaks`` only ever flags *interior* points -- by
    construction it never considers the first or last bin, since a peak is
    defined by having lower neighbours on both sides and an edge bin only
    has one. A mode sitting right at the boundary of the histogram's range
    (as can happen once bins are set from the data's own min/max) would
    otherwise silently go unmarked, so the two edge bins are checked by hand
    and folded into the candidate list before picking the tallest ones.
    """
    counts, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    peak_idx, _ = find_peaks(counts)
    peak_idx = list(peak_idx)
    if len(counts) > 1:
        if counts[0] > counts[1]:
            peak_idx.append(0)
        if counts[-1] > counts[-2]:
            peak_idx.append(len(counts) - 1)
    peak_idx = np.array(sorted(set(peak_idx)))
    if len(peak_idx) == 0:
        return
    n_keep = min(n_peaks, len(peak_idx))
    top = peak_idx[np.argsort(counts[peak_idx])[::-1][:n_keep]]
    top = np.sort(top)

    for idx in top:
        x0, y0 = centers[idx], counts[idx]
        ax.annotate('', xy=(x0, y0), xycoords='data',
                    xytext=(32, 32), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.4))


# ---------------------------------------------------------------------------
# Per-model histogram style sheet  (edit here — no need to touch plot funcs)
# ---------------------------------------------------------------------------
#   ls    : linestyle for the step-border
#   lw    : linewidth for the step-border
#   alpha : fill transparency — set to 0.0 for all models (no shadow),
#           empirical fill is handled directly in _emp_hist_with_errors via cfg
#   zorder: drawing order — higher = drawn on top
#           gauss(1) < ising(2) < bc(3) < beg(4) < emp(5)
#
# 'gauss'         → analytic Gauss/FA curves: dashed (--), thicker lw
#                   (shared style regardless of whether the underlying null
#                   model is a full Gaussian or a Factor-Analyzers fit --
#                   only the legend text, driven by the `gauss_label`
#                   argument of each plot function, changes: 'Gauss' vs 'FA')
# 'gauss_sampled' → MC-sampled Gauss/FA histograms: thin step + thin hatching

mpl.rcParams['hatch.linewidth'] = 0.4

MODEL_STYLES = {
    'ising':         dict(ls='-',   lw=1.8, alpha=0.0, alphaline=0.85, zorder=2),
    'bc':            dict(ls='-',   lw=1.8, alpha=0.0, alphaline=0.85, zorder=3),
    'beg':           dict(ls='-',   lw=1.8, alpha=0.0, alphaline=0.85, zorder=4),
    'gauss':         dict(ls='--',  lw=1.8, alpha=0.0, alphaline=0.85, zorder=1),   # analytic Gauss
    'gauss_sampled': dict(ls='-',   lw=0.4, alpha=0.0, alphaline=0.85, zorder=1),   # sampled Gauss
    'gaussdisc':     dict(ls='-.',  lw=1.8, alpha=0.0, alphaline=0.85, zorder=1),
    'copula':        dict(ls='--',  lw=1.8, alpha=0.0, alphaline=0.85, zorder=1),
    'nullcat':       dict(ls='-',   lw=1.8, alpha=0.0, alphaline=0.85, zorder=1),
    'emp':           dict(ls='-',   lw=1.2, alpha=0.15, alphaline=1, zorder=5),  # alpha handled by cfg
}

# ---------------------------------------------------------------------------
# Figure: training losses
# ---------------------------------------------------------------------------

def plot_losses(losses_ising, losses_bc, losses_beg, dataset, savepath, colors):
    r"""Loss curves $L_{\bf h}$, $L_J$, $L_K$ vs PCD iteration."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].plot(losses_ising[:, 1], '.-', label='Ising', color=colors['ising'], lw=0.5, rasterized=True)
    axs[0].plot(losses_bc[:, 1], '.-', label='BC', color=colors['bc'], lw=0.5, rasterized=True)
    axs[0].plot(losses_beg[:, 1], '.-', label='BEG', color=colors['beg'], lw=0.5, rasterized=True)
    axs[0].set_yscale('log')
    axs[0].set_title(r'Loss $L_{\bf h}$, ' + dataset)
    axs[0].set_xlabel('PCD iteration')
    axs[0].legend()
    axs[0].grid(True, which='both', ls='--', alpha=0.4)

    axs[1].plot(losses_ising[:, 2], '.-', label='Ising', color=colors['ising'], lw=0.5, rasterized=True)
    axs[1].plot(losses_bc[:, 2], '.-', label='BC', color=colors['bc'], lw=0.5, rasterized=True)
    axs[1].plot(losses_beg[:, 2], '.-', label='BEG', color=colors['beg'], lw=0.5, rasterized=True)
    axs[1].set_yscale('log')
    axs[1].set_title(r'Loss $L_J$, ' + dataset)
    axs[1].set_xlabel('PCD iteration')
    axs[1].legend()
    axs[1].grid(True, which='both', ls='--', alpha=0.4)

    axs[2].plot(losses_beg[:, 3], '.-', label='BEG', color=colors['beg'], lw=0.5, rasterized=True)
    axs[2].set_yscale('log')
    axs[2].set_title(r'Loss $L_K$, ' + dataset)
    axs[2].set_xlabel('PCD iteration')
    axs[2].legend()
    axs[2].grid(True, which='both', ls='--', alpha=0.4)

    fig.suptitle(f'Training losses -- {dataset}', size=14)
    fig.tight_layout()
    _savefig(savepath, f'losses_data{dataset}.pdf')


# ---------------------------------------------------------------------------
# Figure: moment matching (learning phase)
# ---------------------------------------------------------------------------

def plot_moment_matching_learning(inverseisingPCD, inversebcPCD, inversebegPCD,
                                  dataset, savepath, colors, cfg):
    r"""Empirical vs theoretical sufficient statistics after PCD convergence."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fl = axs.flatten()
    fs = cfg.get('xlabel_fsize', 14)

    emp_mean = inversebegPCD.x_mean.flatten()
    emp_x2 = np.diag(inversebegPCD.xxdag_mean)
    emp_cov = _remove_diag(
        inversebegPCD.xxdag_mean
        - np.outer(inversebegPCD.x_mean, inversebegPCD.x_mean)
    ).flatten()
    emp_x2x2 = inversebegPCD.x2x2dag_mean.flatten()

    for ax, emp, theo_ising, theo_bc, theo_beg, lbl in [
        (fl[0], emp_mean,
         inverseisingPCD.theoretical_mean.flatten(),
         inversebcPCD.theoretical_mean.flatten(),
         inversebegPCD.theoretical_mean.flatten(),
         r'$\langle x_i \rangle$'),
        (fl[1], emp_x2,
         np.diag(inverseisingPCD.theoretical_xxdag),
         np.diag(inversebcPCD.theoretical_xxdag),
         np.diag(inversebegPCD.theoretical_xxdag),
         r'$\langle x_i^2 \rangle$'),
        (fl[2], emp_cov,
         _remove_diag(inverseisingPCD.theoretical_xxdag
                      - np.outer(inverseisingPCD.theoretical_mean,
                                 inverseisingPCD.theoretical_mean)).flatten(),
         _remove_diag(inversebcPCD.theoretical_xxdag
                      - np.outer(inversebcPCD.theoretical_mean,
                                 inversebcPCD.theoretical_mean)).flatten(),
         _remove_diag(inversebegPCD.theoretical_xxdag
                      - np.outer(inversebegPCD.theoretical_mean,
                                 inversebegPCD.theoretical_mean)).flatten(),
         r'$\langle x_ix_j\rangle - \langle x_i\rangle\langle x_j\rangle$'),
    ]:
        ax.errorbar(emp, emp, fmt='-', rasterized=True)
        ax.errorbar(emp, theo_ising, fmt='+', color=colors['ising'], rasterized=True, label='Ising')
        ax.errorbar(emp, theo_bc, fmt='x', color=colors['bc'], rasterized=True, label='BC')
        ax.errorbar(emp, theo_beg, fmt='o', mfc='None', color=colors['beg'], rasterized=True, label='BEG')
        ax.set_xlabel(lbl + ' - empirical', size=fs)
        ax.set_ylabel(lbl + ' - model', size=fs)
        ax.legend()

    fl[3].errorbar(emp_x2x2, emp_x2x2, fmt='-', rasterized=True)
    fl[3].errorbar(emp_x2x2, inversebegPCD.theoretical_x2x2dag.flatten(),
                   fmt='o', mfc='None', color=colors['beg'], rasterized=True, label='BEG')
    fl[3].set_xlabel(r'$\langle x_i^2 x_j^2 \rangle$ - empirical', size=fs)
    fl[3].set_ylabel(r'$\langle x_i^2 x_j^2 \rangle$ - model', size=fs)
    fl[3].legend()

    fig.suptitle(f'Moment matching -- learning phase -- {dataset}',
                 size=cfg.get('suptitle_fsize', 16))
    fig.tight_layout()
    _savefig(savepath, f'moment_matching_learning_data={dataset}.pdf')


# ---------------------------------------------------------------------------
# Figure: moment matching (sampling phase)
# ---------------------------------------------------------------------------

def plot_moment_matching_sampling(X, sim_dict_list_ising, sim_dict_list_bc,
                                  sim_dict_list_beg, dataset, savepath, colors, cfg,
                                  inverseisingPCD=None, inversebcPCD=None,
                                  inversebegPCD=None, pseudolikelihood=False):
    r"""Empirical vs MCMC-sampled sufficient statistics.

    When the models were fit via pseudo-likelihood (``el['pseudolikelihood']
    is True`` for at least one simulated element) *and* the fitted BEG
    object is supplied, this also overlays grey "+" crosses showing the
    "sleep" part of the pseudo-likelihood gradient, Eq.~(\ref{eq:PLgradients})
    in Appendix \ref{sec:PseudoLik} -- the model's single-site conditional
    expectations evaluated on the training data, which the pseudo-likelihood
    optimisation requires to match the empirical ("wake") averages at
    convergence. The BEG object is used because it is the most general model
    (Ising/BC are the special cases K=0 and K=0, diag(J)=0), so its
    conditional moments are shown for all four panels.
    """
    N = X.shape[0]
    emp_mean = np.mean(X, axis=0)
    emp_x2 = np.mean(X ** 2, axis=0)
    emp_cov = np.cov(X.T)
    emp_cov_flat = emp_cov[~np.eye(emp_cov.shape[0], dtype=bool)]
    emp_x2x2 = ((X ** 2).T @ (X ** 2) / N).flatten()
    fs = cfg.get('xlabel_fsize', 14)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fl = axs.flatten()

    fl[0].plot(emp_mean, emp_mean, '-')
    fl[1].plot(emp_x2, emp_x2, '-')
    fl[2].errorbar(emp_cov_flat, emp_cov_flat, capsize=5, fmt='-')
    fl[3].errorbar(emp_x2x2, emp_x2x2, capsize=5, fmt='-')

    for simlist, col, name in [
        (sim_dict_list_ising, colors['ising'], 'Ising'),
        (sim_dict_list_bc, colors['bc'], 'BC'),
        (sim_dict_list_beg, colors['beg'], 'BEG'),
    ]:
        for el in simlist:
            if not condition(el):
                continue
            lbl = model_label(el, name)
            N_mc = len(el['configurations'])
            mc_mean = el['dict_means']['x']
            mc_x2 = np.diag(el['dict_means']['xxdag'])
            mc_cov = el['dict_means']['xxdag'] - np.outer(el['dict_means']['x'],
                                                          el['dict_means']['x'])
            mc_cov_flat = mc_cov[~np.eye(mc_cov.shape[0], dtype=bool)]
            mc_x2x2 = el['dict_means']['x2x2dag'].flatten()

            fl[0].errorbar(emp_mean, mc_mean,
                           xerr=np.std(X, axis=0) / np.sqrt(N),
                           yerr=el['dict_stds']['x'] / np.sqrt(N),
                           capsize=5, fmt='.', label=lbl, color=col, rasterized=True)
            fl[1].errorbar(emp_x2, mc_x2,
                           xerr=np.std(X ** 2, axis=0) / np.sqrt(N),
                           yerr=np.diag(el['dict_stds']['xxdag']) / np.sqrt(N),
                           capsize=5, fmt='.', label=lbl, color=col, rasterized=True)
            offdiag = ~np.eye(mc_cov.shape[0], dtype=bool)
            fl[2].errorbar(emp_cov_flat, mc_cov_flat,
                           yerr=el['dict_stds']['xxdag'][offdiag] / np.sqrt(N_mc),
                           capsize=5, fmt='.', label=lbl, color=col, rasterized=True)
            fl[3].errorbar(emp_x2x2, mc_x2x2,
                           yerr=el['dict_stds']['x2x2dag'].flatten() / np.sqrt(N_mc),
                           capsize=5, fmt='.', label=lbl, color=col, rasterized=True)

    # --- Pseudo-likelihood "sleep" moments, Eq. (\ref{eq:PLgradients}) ------
    # Grey crosses: the model's single-site conditional expectations on the
    # training data (BEG parametrisation, most general), which the
    # pseudo-likelihood gradient requires to match the empirical averages.
  
    if pseudolikelihood and inversebegPCD is not None and hasattr(inversebegPCD, 'J_fit'):
        states = inversebegPCD.states
        J, h, K = inversebegPCD.J_fit, inversebegPCD.h_fit, inversebegPCD.K_fit
        sleep_x, sleep_x2, sleep_cov, sleep_x2x2 = compute_pseudolikelihood_moments(
            X, states, J, h, K=K)
        _pl_lbl = 'pseudo-lik. sleep moments (BEG)'
        _pl_kw = dict(fmt='+', color='grey', ms=8, mew=1.3,
                      rasterized=True, zorder=6, label=_pl_lbl)
        fl[0].errorbar(emp_mean, sleep_x, **_pl_kw)
        fl[1].errorbar(emp_x2, sleep_x2, **_pl_kw)
        fl[2].errorbar(emp_cov_flat, _remove_diag(sleep_cov).flatten(), **_pl_kw)
        fl[3].errorbar(emp_x2x2, sleep_x2x2.flatten(), **_pl_kw)

    fl[0].set_xlabel(r'$\langle x_i \rangle$ - empirical', size=fs)
    fl[0].set_ylabel(r'$\langle x_i \rangle$ - MCMC', size=fs)
    fl[1].set_xlabel(r'$\langle x_i^2 \rangle$ - empirical', size=fs)
    fl[1].set_ylabel(r'$\langle x_i^2 \rangle$ - MCMC', size=fs)
    fl[2].set_xlabel(r'$\langle x_ix_j\rangle - \langle x_i\rangle\langle x_j\rangle$ - empirical', size=fs)
    fl[2].set_ylabel(r'$\langle x_ix_j\rangle - \langle x_i\rangle\langle x_j\rangle$ - MCMC', size=fs)
    fl[3].set_xlabel(r'$\langle x_i^2 x_j^2 \rangle$ - empirical', size=fs)
    fl[3].set_ylabel(r'$\langle x_i^2 x_j^2 \rangle$ - MCMC', size=fs)
    for ax in fl:
        ax.legend(fontsize=7)

    fig.suptitle(f'Moment matching -- sampling phase -- {dataset}',
                 size=cfg.get('suptitle_fsize', 16))
    fig.tight_layout()
    _savefig(savepath, f'moment_matching_sampling_data={dataset}.pdf')


# ---------------------------------------------------------------------------
# Figure: item histograms (maxent models)
# ---------------------------------------------------------------------------

def plot_item_histogram(X, sim_dict_list_ising, sim_dict_list_bc, sim_dict_list_beg,
                        item_bins, dataset, savepath, colors, cfg):
    r"""Histograms ${\sf h}_{x_i}$ for each item: empirical vs Ising / BC / BEG.

    Style/logic harmonised with the other maxent comparison figures
    (:func:`plot_pc_histogram_maxent`, :func:`plot_factor_histogram_maxent`,
    the E2d / Mahalanobis histograms): shared empirical-with-Wilson-CI
    helper, per-model style sheet (``MODEL_STYLES``, incl. ``alphaline``),
    and legend labels built via ``model_label`` (so e.g. ``tau_c``/reset
    info shows up exactly as in every other figure) instead of a bare
    model name.
    """
    N = X.shape[0]
    M = X.shape[1]
    nb_items = min(M, cfg.get('nb_items', 12))
    ncolumns = cfg.get('item_ncols', 4)
    nrows = int(nb_items / ncolumns + 0.99)
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    fig, axs = plt.subplots(nrows, ncolumns,
                            figsize=(8, int(8 * nrows / 3.)),
                            sharex=True, sharey=True)
    fl = axs.flatten()

    for idx in range(nb_items):
        ax = fl[idx]
        _emp_hist_with_errors(ax, X[:, idx], item_bins, N, pvalue, alpha_emp,
                              colors['emp'])
        ax.set_title(f'item {idx + 1}', size=8)

        for simlist, col, name, mkey in [
            (sim_dict_list_ising, colors['ising'], 'Ising', 'ising'),
            (sim_dict_list_bc,    colors['bc'],    'BC',    'bc'),
            (sim_dict_list_beg,   colors['beg'],   'BEG',   'beg'),
        ]:
            for el in simlist:
                if condition(el):
                    lbl = model_label(el, name) if idx == nb_items - 1 else None
                    _ms = MODEL_STYLES[mkey]
                    ax.hist(el['configurations'][:, idx], bins=item_bins,
                            histtype='step', density=True, lw=_ms['lw'],
                            color=col, alpha=_ms['alphaline'], ls=_ms['ls'],
                            label=lbl, zorder=_ms['zorder'])
                    break

    fl[nb_items - 1].legend(fontsize=7)
    fig.suptitle(f'Item histograms -- {dataset}', size=cfg.get('suptitle_fsize', 16))
    fig.tight_layout()
    _savefig(savepath, f'itemhistogram_comparison_data={dataset}.pdf')


# ---------------------------------------------------------------------------
# Figure: Euclidean distance histogram (spin models only)
# ---------------------------------------------------------------------------
def function_to_fouriertransform(z, lambdas):
    myarray = np.array(1 - 2. * z * 1j * lambdas) ** 0.5
    return np.prod(myarray) ** -1


def plot_E2d_histogram_maxent(X, sim_dict_list_ising, sim_dict_list_bc,
                              sim_dict_list_beg, dataset, savepath, colors, cfg,
                              lambdas=None, X_gauss=None):
    r"""Histogram ${\sf h}_{d_{\bf x}}$ (Euclidean distance): Ising / BC / BEG vs empirical.

    Gauss arguments are accepted for backward compatibility but not plotted here;
    use plot_E2d_histogram_beg_gauss_catind for the second figure type.

    Saved as: E2d_histogram_logscale={True|False}data={dataset}.pdf
    """
    N = X.shape[0]
    nbins = cfg.get('nbins_E2d', 80)
    nbins_emp = cfg.get('nbins_emp_dist', cfg.get('nbins_emp', 20))
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    E2d_X = np.sum((X - np.mean(X, axis=0)) ** 2, axis=1)
    E2dmax = np.max(E2d_X) * 1.1
    bins_E2d = np.arange(1e-2, E2dmax, E2dmax / nbins)

    for logscale in [True, False]:
        fig, ax = plt.subplots(figsize=(7, 5))
        HB_e2d_X = _emp_hist_with_errors(ax, E2d_X, nbins_emp, N, pvalue, alpha_emp,
                              colors['emp'])
        # spin models
        for simlist, col, name, mkey in [
            (sim_dict_list_ising, colors['ising'], 'Ising', 'ising'),
            (sim_dict_list_bc,    colors['bc'],    'BC',    'bc'),
            (sim_dict_list_beg,   colors['beg'],   'BEG',   'beg'),
        ]:
            for el in simlist:
                if condition(el):
                    _ms = MODEL_STYLES[mkey]
                    ax.hist(el['distances_maxent'], histtype='step',
                            bins=bins_E2d, label=model_label(el, name),
                            density=True, lw=_ms['lw'], color=col, alpha=_ms['alphaline'],
                            ls=_ms['ls'], zorder=_ms['zorder'])
                    break

        y_min = np.sort(HB_e2d_X[0][np.where(HB_e2d_X[0] > 1.0E-12)[0]])[0]
        ylim = ax.get_ylim()
        ax.set_ylim([np.max([ylim[0], y_min]), ylim[1]])

        if logscale:
            ax.set_yscale('log')
            ax.set_ylim([np.max([ylim[0], y_min]), ylim[1]*1.3])
        ax.set_xlabel(r'$d_{\bf x}^2 = \|{\bf x} - \boldsymbol{\mu}\|_2^2$',
                      size=cfg.get('xlabel_fsize', 12))
        ax.set_ylabel(r'${\sf h}_{d_{\bf x}}$', size=cfg.get('xlabel_fsize', 12)+1, rotation=0, labelpad=10)
        ax.set_title(f'Euclidean distance histogram — {dataset}', fontsize=cfg.get('suptitle_fsize', 12))
        #ax.legend(fontsize=8, bbox_to_anchor=(0.01, 1.1), loc='upper left', ncol=4)
        ax.legend(fontsize=9,  loc='upper right', ncol=1)
        _savefig(savepath, f'E2d_histogram_logscale={logscale}data={dataset}.pdf')


def plot_E2d_histogram_beg_gauss_catind(X, sim_dict_list_beg, X_catind, X_gauss,
                                         dataset, savepath, colors, cfg,
                                         lambdas=None, gauss_label='FA'):
    r"""Histogram ${\sf h}_{d_{\bf x}}$ (Euclidean distance): BEG + Gauss/FA + Cat-ind vs empirical.

    Gauss/FA sampled histogram: thin step + thin hatching, drawn over its
    *own* domain (range of ``X_gauss``'s distances), not clipped to the
    empirical/BEG bins -- the analytic curve (if ``lambdas`` provided) is
    drawn over that same domain so the two always agree.

    Parameters
    ----------
    gauss_label : str
        Legend label for the null baseline: ``'FA'`` (default, low-rank +
        noise covariance) or ``'Gauss'`` (full empirical covariance) --
        see :func:`fit_gauss_or_fa_model`.

    Saved as: E2d_histogram_beg_gauss_catind_logscale={True|False}data={dataset}.pdf
    """
    N = X.shape[0]
    nbins = cfg.get('nbins_E2d', 80)
    nbins_emp = cfg.get('nbins_emp_dist', cfg.get('nbins_emp', 20))
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    E2d_X = np.sum((X - np.mean(X, axis=0)) ** 2, axis=1)
    E2dmax = np.max(E2d_X) * 1.1
    bins_E2d = np.arange(1e-2, E2dmax, E2dmax / nbins)

    for logscale in [True, False]:
        fig, ax = plt.subplots(figsize=(7, 5))
        HB_e2d_X = _emp_hist_with_errors(ax, E2d_X, nbins_emp, N, pvalue, alpha_emp,
                                          colors['emp'])
        # BEG
        for el in sim_dict_list_beg:
            if condition(el):
                _ms = MODEL_STYLES['beg']
                ax.hist(el['distances_maxent'], histtype='step',
                        bins=bins_E2d, label=model_label(el, 'BEG'),
                        density=True, lw=_ms['lw'], color=colors['beg'], alpha=_ms['alphaline'],
                        ls=_ms['ls'], zorder=_ms['zorder'])
                break

        # Gauss/FA sampled (thin step + thin hatching) + analytic curve,
        # both drawn over the sampled model's own domain (point 1 of spec)
        if X_gauss is not None:
            E2d_g = np.sum((X_gauss - np.mean(X_gauss, axis=0)) ** 2, axis=1)
            bins_gauss = _gauss_own_domain_bins(E2d_g, nbins)
            #print(bins_gauss[-1])
            _ms_g = MODEL_STYLES['gauss']
            HB_e2d_g = _gauss_sampled_hist(ax, E2d_g, bins_gauss, colors['gauss'],
                                            label=f'{gauss_label} (sampled)',
                                            zorder=_ms_g['zorder'])
            if lambdas is not None:
                integrand = lambda z: function_to_fouriertransform(z, lambdas)
                maxz = 10.
                bx2_gauss = HB_e2d_g[1]
                hx2_gauss = np.array([fourier_transform_integral(integrand, x, maxz)
                                      for x in bx2_gauss])
                ax.plot(bx2_gauss, hx2_gauss, ls='--', lw=_ms_g['lw'],
                        color=colors['gauss'], label=f'{gauss_label} (analytic)', alpha=_ms_g['alphaline'],
                        zorder=_ms_g['zorder'])

        # Cat-ind
        if X_catind is not None:
            E2d_catind = np.sum((X_catind - np.mean(X_catind, axis=0)) ** 2, axis=1)
            _ms_c = MODEL_STYLES['nullcat']
            ax.hist(E2d_catind, histtype='step', bins=bins_E2d,
                    label='Cat-ind', density=True, lw=_ms_c['lw'], alpha=_ms_c['alphaline'],
                    color=colors['nullcat'], ls=_ms_c['ls'], zorder=_ms_c['zorder'])

        y_min = np.sort(HB_e2d_X[0][np.where(HB_e2d_X[0] > 1.0E-12)[0]])[0]
        ylim = ax.get_ylim()
        ax.set_ylim([np.max([ylim[0], y_min]), ylim[1]])

        if logscale:
            ax.set_yscale('log')
            ax.set_ylim([np.max([ylim[0], y_min]), ylim[1]*1.3])
        ax.set_xlabel(r'$d_{\bf x}^2 = \|{\bf x} - \boldsymbol{\mu}\|_2^2$',
                      size=cfg.get('xlabel_fsize', 14))
        ax.set_ylabel(r'${\sf h}_{d_{\bf x}}$', size=cfg.get('xlabel_fsize', 14)+1, rotation=0, labelpad=10)
        ax.set_title(f'Euclidean distance — BEG / Gauss / Cat-ind — {dataset}', fontsize=cfg.get('suptitle_fsize', 14))
        ax.legend(fontsize=9, loc='upper right', ncol=1)
        _savefig(savepath,
                 f'E2d_histogram_beg_gauss_catind_logscale={logscale}data={dataset}.pdf')


def plot_E2d_histogram_simple(X, X_catind, X_gauss, X_gaussdisc, X_copula,
                              dataset, savepath, colors, cfg, gauss_label='FA'):
    r"""${\sf h}_{d_{\bf x}}$: simple models vs empirical (legacy function)."""
    N = X.shape[0]
    nbins = cfg.get('nbins_E2d', 80)
    nbins_emp = cfg.get('nbins_emp_dist', cfg.get('nbins_emp', 20))
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    E2d_X = np.sum((X - np.mean(X, axis=0)) ** 2, axis=1)
    E2dmax = np.max(E2d_X) * 1.1
    bins_E2d = np.arange(1e-2, E2dmax, E2dmax / nbins)

    simple = [
        (X_gauss,     colors['gauss'],     gauss_label,  'gauss'),
        (X_gaussdisc, colors['gaussdisc'], 'Gauss-disc', 'gaussdisc'),
        (X_copula,    colors['copula'],    'Copula',     'copula'),
        (X_catind,    colors['nullcat'],   'Cat-ind',    'nullcat'),
    ]

    for logscale in [True, False]:
        fig, ax = plt.subplots(figsize=(7, 5))
        _emp_hist_with_errors(ax, E2d_X, nbins_emp, N, pvalue, alpha_emp, colors['emp'])
        for Xm, col, lbl, mkey in simple:
            if Xm is not None:
                E2d_m = np.sum((Xm - np.mean(Xm, axis=0)) ** 2, axis=1)
                if mkey == 'gauss':
                    bins_gauss = _gauss_own_domain_bins(E2d_m, nbins)
                    _gauss_sampled_hist(ax, E2d_m, bins_gauss, col, label=lbl,
                                        zorder=MODEL_STYLES[mkey]['zorder'])
                else:
                    _ms = MODEL_STYLES[mkey]
                    ax.hist(E2d_m, histtype='step', bins=bins_E2d,
                            label=lbl, density=True, lw=_ms['lw'], color=col,
                            ls=_ms['ls'], zorder=_ms['zorder'])
        if logscale:
            ax.set_yscale('log')
        ax.set_xlabel(r'$d_{\bf x}^2 = \|{\bf x} - \boldsymbol{\mu}\|_2^2$',
                      size=cfg.get('xlabel_fsize', 14))
        ax.set_ylabel(r'${\sf h}_{d_{\bf x}}$', size=cfg.get('xlabel_fsize', 14))
        ax.set_title(f'Euclidean distance -- simple models -- {dataset}')
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        _savefig(savepath, f'E2d_histogram_simplemodels_data={dataset}logscale={logscale}.pdf')


def plot_E2d_histogram_begvscopula(X, sim_dict_list_beg, X_copula,
                                   dataset, savepath, colors, cfg):
    r"""${\sf h}_{d_{\bf x}}$: BEG vs copula vs empirical.

    Saved as: E2d_histogram_begvscopula_data={dataset}_logscale=_{True|False}.pdf
    """
    N = X.shape[0]
    nbins = cfg.get('nbins_E2d', 60)
    nbins_emp = cfg.get('nbins_emp_dist', cfg.get('nbins_emp', 20))
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    E2d_X = np.sum((X - np.mean(X, axis=0)) ** 2, axis=1)
    E2dmax = np.max(E2d_X) * 1.1
    bins_E2d = np.arange(1e-2, E2dmax, E2dmax / nbins)

    E2d_cop = np.sum((X_copula - np.mean(X_copula, axis=0)) ** 2, axis=1)

    for logscale in [True, False]:
        fig, ax = plt.subplots(figsize=(7, 5))
        _emp_hist_with_errors(ax, E2d_X, nbins_emp, N, pvalue, alpha_emp,
                              colors['emp'])
        _ms_cop = MODEL_STYLES['copula']
        ax.hist(E2d_cop, histtype='step', bins=bins_E2d, label='Copula',
                density=True, lw=_ms_cop['lw'], color=colors['copula'],
                ls=_ms_cop['ls'], zorder=_ms_cop['zorder'])
        for el in sim_dict_list_beg:
            if condition(el):
                _ms_beg = MODEL_STYLES['beg']
                ax.hist(el['distances_maxent'], histtype='step', bins=bins_E2d,
                        label=model_label(el, 'BEG'), density=True,
                        lw=_ms_beg['lw'], color=colors['beg'],
                        ls=_ms_beg['ls'], zorder=_ms_beg['zorder'])
                break
        if logscale:
            ax.set_yscale('log')
        ax.set_xlabel(r'$d_{\bf x}^2 = \|{\bf x} - \boldsymbol{\mu}\|_2^2$',
                      size=cfg.get('xlabel_fsize', 14))
        ax.set_ylabel(r'${\sf h}_{d_{\bf x}}$', size=cfg.get('ylabel_fsize', 14))
        ax.set_title(f'Euclidean distance — BEG vs Copula — {dataset}')
        ax.legend(fontsize=8)
        _savefig(savepath,
                 f'E2d_histogram_begvscopula_data={dataset}_logscale=_{logscale}.pdf')


# ---------------------------------------------------------------------------
# Figure: Mahalanobis distance histograms
# ---------------------------------------------------------------------------

def plot_mahalanobis_commonC_maxent(energies_X, sim_dict_list_ising,
                                    sim_dict_list_bc, sim_dict_list_beg,
                                    N, dataset, savepath, colors, cfg,
                                    energies_Xgauss=None):
    r"""Histogram ${\sf h}_{d_{\bf x}^{({\rm M})}}$ (Mahalanobis, common C):
    Ising / BC / BEG vs empirical.

    Gauss argument accepted for backward compatibility but not plotted here;
    use plot_mahalanobis_commonC_beg_gauss_catind for the second figure type.

    Saved as: energy_histogram_commonC_{dataset}_log={True|False}.pdf
    """
    nbins = cfg.get('nbins_energy', 80)
    nbins_emp = cfg.get('nbins_emp_dist', cfg.get('nbins_emp', 20))
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    energymax = np.max(energies_X) * 1.1
    bins_energy = np.arange(1e-2, energymax, energymax / nbins)

    for logscale in [True, False]:
        fig, ax = plt.subplots(figsize=(7, 5))
        _emp_hist_with_errors(ax, energies_X, nbins_emp, N, pvalue, alpha_emp,
                              colors['emp'])
        HB_maxent = None
        for simlist, col, name, mkey in [
            (sim_dict_list_ising, colors['ising'], 'Ising', 'ising'),
            (sim_dict_list_bc,    colors['bc'],    'BC',    'bc'),
            (sim_dict_list_beg,   colors['beg'],   'BEG',   'beg'),
        ]:
            for el in simlist:
                if condition(el):
                    _ms = MODEL_STYLES[mkey]
                    HB_maxent = ax.hist(el['energies_maxent_commonC'], histtype='step',
                            bins=bins_energy, label=model_label(el, name),
                            density=True, lw=_ms['lw'], color=col, alpha=_ms['alphaline'],
                            ls=_ms['ls'], zorder=_ms['zorder'])
                    break

        assert HB_maxent is not None, "No model element satisfies condition — cannot compute y_min"
        y_min = np.sort(HB_maxent[0][np.where(HB_maxent[0] > 1.0E-12)[0]])[0]
        ymax = np.max(HB_maxent[0]) * 0.9
        ylim = ax.get_ylim()
        ax.set_ylim([np.max([ylim[0], y_min]), np.max([ylim[1], ymax])])

        if logscale:
            ax.set_yscale('log')
            ax.set_ylim([np.max([ylim[0], y_min]), np.max([ylim[1], ymax])*1.3])
        ax.set_xlabel(
            r'$d_{\bf x}^{({\rm M})} = \frac{1}{2}\left({\bf x} - \boldsymbol{\mu}\right)^\dag C^{-1}\left({\bf x} - \boldsymbol{\mu}\right)$',
            size=cfg.get('xlabel_fsize', 14))
        ax.set_ylabel(r'${\sf h}_{d_{\bf x}^{({\rm M})}}$', size=cfg.get('ylabel_fsize', 14)+1, rotation=0, labelpad=10)
        ax.set_title(f'Mahalanobis distance (common $C$) — {dataset}', fontsize=cfg.get('suptitle_fsize', 14))
        ax.legend(fontsize=9, loc='upper right', ncol=1)
        _savefig(savepath,
                 f'energy_histogram_commonC_{dataset}_log={logscale}.pdf')


def plot_mahalanobis_commonC_beg_gauss_catind(energies_X, sim_dict_list_beg,
                                              energies_Xcatind, energies_Xgauss,
                                              bins_energy, N, M,
                                              dataset, savepath, colors, cfg,
                                              gauss_label='FA',
                                              lambdas=None):
    r"""Histogram ${\sf h}_{d_{\bf x}^{({\rm M})}}$ (Mahalanobis, common C):
    BEG + Gauss/FA + Cat-ind vs empirical.

    Gauss/FA sampled histogram: thin step + thin hatching, drawn over its own
    domain (point 1 of spec). Analytic curve: dashed, thick, same domain --
    always the plain chi-squared(M) density (i.e. the classic Gaussian
    theoretical curve), deliberately *not* corrected for the FA null model's
    own (low-rank) covariance, since that correction requires a generalized
    eigendecomposition that turned out to be numerically unstable and isn't
    the point of this comparison.

    Parameters
    ----------
    gauss_label : str
        Legend label for the null baseline: ``'FA'`` (default) or ``'Gauss'``.

    Saved as: energy_histogram_commonC_beg_gauss_catind_{dataset}_log={True|False}.pdf
    """
    nbins_emp = cfg.get('nbins_emp_dist', cfg.get('nbins_emp', 20))
    nbins = cfg.get('nbins_energy', 80)
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    for logscale in [True, False]:
        fig, ax = plt.subplots(figsize=(7, 5))
        _emp_hist_with_errors(ax, energies_X, nbins_emp, N, pvalue, alpha_emp,
                              colors['emp'])
        # BEG
        for el in sim_dict_list_beg:
            if condition(el):
                _ms = MODEL_STYLES['beg']
                ax.hist(el['energies_maxent_commonC'], histtype='step',
                        bins=bins_energy, label=model_label(el, 'BEG'),
                        density=True, lw=_ms['lw'], color=colors['beg'], alpha=_ms['alphaline'],
                        ls=_ms['ls'], zorder=_ms['zorder'])
                break

        # Gauss/FA sampled (thin + hatch) + analytic curve, own domain
        if energies_Xgauss is not None:
            bins_gauss = _gauss_own_domain_bins(energies_Xgauss, nbins)
            _ms_g = MODEL_STYLES['gauss']
            HB_Xgauss = _gauss_sampled_hist(ax, energies_Xgauss, bins_gauss,
                                             colors['gauss'],
                                             label=f'{gauss_label} (sampled)',
                                             zorder=_ms_g['zorder'])
            xs_analytic = HB_Xgauss[1]
            if gauss_label == 'FA':
                integrand = lambda z: function_to_fouriertransform(z, lambdas/2)
                ys_analytic = np.array([fourier_transform_integral(integrand, x, 10.)
                                        for x in xs_analytic])
                ax.plot(xs_analytic, ys_analytic, ls='--', lw=_ms_g['lw'],
                        color=colors['gauss'], alpha=_ms_g['alphaline'],
                        label=f'{gauss_label} (analytic)', zorder=_ms_g['zorder'])
            else:
                ax.plot(xs_analytic, energydensity_gauss(xs_analytic, M),
                        ls='--', lw=_ms_g['lw'], color=colors['gauss'], alpha=_ms_g['alphaline'],
                        label=f'{gauss_label} (analytic)', zorder=_ms_g['zorder'])

        # Cat-ind
        if energies_Xcatind is not None:
            _ms_c = MODEL_STYLES['nullcat']
            ax.hist(energies_Xcatind, histtype='step', bins=bins_energy,
                    label='Cat-ind', density=True, lw=_ms_c['lw'], alpha=_ms_c['alphaline'],
                    color=colors['nullcat'], ls=_ms_c['ls'], zorder=_ms_c['zorder'])

        ymin = np.sort(HB_Xgauss[0][np.where(HB_Xgauss[0] > 1.0E-12)[0]])[0]
        ylim = ax.get_ylim()
        ax.set_ylim([np.max([ylim[0], ymin]), ylim[1]])
        if logscale:
            ax.set_yscale('log')
            ax.set_ylim([np.max([ylim[0], ymin]), ylim[1]*1.3])
        ax.set_xlabel(
            r'$d_{\bf x}^{({\rm M})} = \frac{1}{2}\left({\bf x} - \boldsymbol{\mu}\right)^\dag C^{-1}\left({\bf x} - \boldsymbol{\mu}\right)$',
            size=cfg.get('xlabel_fsize', 14))
        ax.set_ylabel(r'${\sf h}_{d_{\bf x}^{({\rm M})}}$', size=cfg.get('ylabel_fsize', 14)+1, rotation=0, labelpad=10)
        ax.set_title(f'Mahalanobis (common $C$) — BEG / Gauss / Cat-ind — {dataset}', fontsize=cfg.get('suptitle_fsize', 14))
        ax.legend(fontsize=9, loc='upper right', ncol=1)
        _savefig(savepath,
                 f'energy_histogram_commonC_beg_gauss_catind_{dataset}_log={logscale}.pdf')


def plot_mahalanobis_modelcov(energies_X, sim_dict_list_ising,
                              sim_dict_list_bc, sim_dict_list_beg,
                              N, M, bins_energy, dataset, savepath, colors, cfg):
    r"""Mahalanobis distance (model $\Sigma$) histogram."""
    from scipy.special import gamma as _gamma

    def _gauss_density(xs, d):
        return xs ** (d / 2 - 1) * np.exp(-xs) / _gamma(d / 2)

    nbins_emp = cfg.get('nbins_emp_dist', cfg.get('nbins_emp', 20))
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    for logscale in [True, False]:
        fig, ax = plt.subplots(figsize=(7, 5))
        HB =  _emp_hist_with_errors(ax, energies_X, nbins_emp, N, pvalue, alpha_emp, colors['emp'])
        for simlist, col, name, mkey in [
            (sim_dict_list_ising, colors['ising'], 'Ising', 'ising'),
            (sim_dict_list_bc,    colors['bc'],    'BC',    'bc'),
            (sim_dict_list_beg,   colors['beg'],   'BEG',   'beg'),
        ]:
            for el in simlist:
                if condition(el):
                    _ms = MODEL_STYLES[mkey]
                    ax.hist(el['energies_maxent'], histtype='step',
                            bins=bins_energy, label=model_label(el, name),
                            density=True, lw=_ms['lw'], color=col, alpha=_ms['alphaline'],
                            ls=_ms['ls'], zorder=_ms['zorder'])
                    break
       
        if logscale:
            ymin = np.sort(HB[0][np.where(HB[0] > 1.0E-12)[0]])[0]
            ylim = ax.get_ylim()
            ax.set_ylim([np.max([ylim[0], ymin]), ylim[1]*1.3])
            ax.set_yscale('log')
        ax.set_xlabel(r'$d_{\bf x}^{({\rm M})} = \frac{1}{2}\left({\bf x} - \boldsymbol{\mu}\right)^\dag \Sigma^{-1}\left({\bf x} - \boldsymbol{\mu}\right)$',
                      size=cfg.get('xlabel_fsize', 14))
        ax.set_ylabel(r'${\sf h}_{d_{\bf x}^{({\rm M})}}$', size=cfg.get('ylabel_fsize', 14)+1, rotation=0, labelpad=10)
        ax.set_title(f'Mahalanobis distance (model $\\Sigma$) -- {dataset}', fontsize=cfg.get('suptitle_fsize', 14))
        ax.legend(fontsize=9, loc='upper right', ncol=1)
        _savefig(savepath, f'energy_histogram_{dataset}_log={logscale}.pdf')


def plot_mahalanobis_commonC_simple(energies_X, energies_Xcatind,
                                    energies_Xgauss, energies_Xgaussdisc,
                                    energies_Xcopula, bins_energy,
                                    N, M, dataset, savepath, colors, cfg,
                                    gauss_label='FA'):
    r"""${\sf h}_{d_{\bf x}^{({\rm M})}}$ (common $C$): simple models vs empirical (legacy).

    Saved as: energy_histogram_commonC_simplemodels_{dataset}_log={True|False}.pdf
    """
    nbins = cfg.get('nbins_energy', 80)
    nbins_emp = cfg.get('nbins_emp_dist', cfg.get('nbins_emp', 20))
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    simple = [
        (energies_Xgauss,     colors['gauss'],     gauss_label,  'gauss'),
        (energies_Xgaussdisc, colors['gaussdisc'], 'Gauss-disc', 'gaussdisc'),
        (energies_Xcopula,    colors['copula'],    'Copula',     'copula'),
        (energies_Xcatind,    colors['nullcat'],   'Cat-ind',    'nullcat'),
    ]

    for logscale in [True, False]:
        fig, ax = plt.subplots(figsize=(7, 5))
        _emp_hist_with_errors(ax, energies_X, nbins_emp, N, pvalue, alpha_emp,
                              colors['emp'])
        for ens, col, lbl, mkey in simple:
            if ens is not None:
                if mkey == 'gauss':
                    bins_gauss = _gauss_own_domain_bins(ens, nbins)
                    _gauss_sampled_hist(ax, ens, bins_gauss, col, label=lbl,
                                        zorder=MODEL_STYLES[mkey]['zorder'])
                else:
                    _ms = MODEL_STYLES[mkey]
                    ax.hist(ens, histtype='step', bins=bins_energy,
                            label=lbl, density=True, lw=_ms['lw'],
                            color=col, ls=_ms['ls'], zorder=_ms['zorder'])
        if logscale:
            ax.set_yscale('log')
        ax.set_xlabel(
            r'$d_{\bf x}^{({\rm M})} = \frac{1}{2}\left({\bf x} - \boldsymbol{\mu}\right)^\dag C^{-1}\left({\bf x} - \boldsymbol{\mu}\right)$',
            size=cfg.get('xlabel_fsize', 14))
        ax.set_ylabel(r'${\sf h}_{d_{\bf x}^{({\rm M})}}$', size=cfg.get('ylabel_fsize', 14)+1, rotation=0, labelpad=10)
        ax.set_title(f'Mahalanobis distance — simple models — {dataset}', fontsize=cfg.get('suptitle_fsize', 14))
        ax.legend(fontsize=9, loc='upper right', ncol=1)
        _savefig(savepath,
                 f'energy_histogram_commonC_simplemodels_{dataset}_log={logscale}.pdf')


def plot_mahalanobis_commonC_begvscopula(energies_X, sim_dict_list_beg,
                                         energies_Xcopula, bins_energy,
                                         N, dataset, savepath, colors, cfg):
    r"""${\sf h}_{d_{\bf x}^{({\rm M})}}$ (common $C$): BEG vs copula vs empirical.

    Saved as: energy_histogram_commonC_begvscopula_{dataset}_log={True|False}.pdf
    """
    nbins_emp = cfg.get('nbins_emp_dist', cfg.get('nbins_emp', 20))
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    for logscale in [True, False]:
        fig, ax = plt.subplots(figsize=(7, 5))
        _emp_hist_with_errors(ax, energies_X, nbins_emp, N, pvalue, alpha_emp,
                              colors['emp'])
        _ms_cop = MODEL_STYLES['copula']
        ax.hist(energies_Xcopula, histtype='step', bins=bins_energy,
                label='Copula', density=True, lw=_ms_cop['lw'],
                color=colors['copula'], ls=_ms_cop['ls'], zorder=_ms_cop['zorder'])
        for el in sim_dict_list_beg:
            if condition(el):
                _ms_beg = MODEL_STYLES['beg']
                ax.hist(el['energies_maxent_commonC'], histtype='step',
                        bins=bins_energy, label=model_label(el, 'BEG'),
                        density=True, lw=_ms_beg['lw'], color=colors['beg'],
                        ls=_ms_beg['ls'], zorder=_ms_beg['zorder'])
                break
        if logscale:
            ax.set_yscale('log')
        ax.set_xlabel(
            r'$d_{\bf x}^{({\rm M})} = \frac{1}{2}\left({\bf x} - \boldsymbol{\mu}\right)^\dag C^{-1}\left({\bf x} - \boldsymbol{\mu}\right)$',
            size=cfg.get('xlabel_fsize', 14))
        ax.set_ylabel(r'${\sf h}_{d_{\bf x}^{({\rm M})}}$', size=cfg.get('ylabel_fsize', 14))
        ax.set_title(f'Mahalanobis distance — BEG vs Copula — {dataset}')
        ax.legend(fontsize=8)
        _savefig(savepath,
                 f'energy_histogram_commonC_begvscopula_{dataset}_log={logscale}.pdf')


# ---------------------------------------------------------------------------
# Figure: principal-component histograms (spin models only)
# ---------------------------------------------------------------------------

def plot_pc_histogram_maxent(X, sim_dict_list_ising, sim_dict_list_bc,
                             sim_dict_list_beg, U, mu_X, n_pcs,
                             dataset, savepath, colors, cfg,
                             lambdas_gauss=None, Xprime_gauss=None,
                             zoom=False, pcs_toplot=None):
    r"""Histograms ${\sf h}_{x'_k}$ of principal components: Ising / BC / BEG vs empirical.

    Gauss arguments accepted for backward compatibility but not plotted here;
    use plot_pc_histogram_beg_gauss_catind for the second figure type.

    Saved as: PCshistogram_comparison_data={dataset}.pdf  (or _zoom.pdf)
    """
    N = X.shape[0]
    nbins = cfg.get('nbins_pc', 60)
    nbins_emp = cfg.get('nbins_emp', 20)
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    if pcs_toplot is None:
        pcs_toplot = list(range(n_pcs))
    n_plot = len(pcs_toplot)
    ncolumns = min(n_plot, 3)
    nrows = int(n_plot / ncolumns + 0.99)

    Xprime = (X - mu_X) @ U.T

    fig, axs = plt.subplots(nrows, ncolumns,
                            figsize=(12, int(12 * nrows / 3.)),
                            sharey=True)
    fl = np.array(axs).flatten() if n_plot > 1 else [axs]

    for i, comp in enumerate(pcs_toplot):
        ax = fl[i]
        HB = ax.hist(Xprime[:, comp], bins=nbins_emp, histtype='step',
                     density=True, color=colors['emp'], lw=1.)
        ax.hist(Xprime[:, comp], bins=nbins_emp, alpha=alpha_emp,
                density=True, color=colors['emp'], lw=0.)
        bl = HB[1][1:] - HB[1][:-1]
        yerr = _wilson_error(HB[0], N, pvalue, bl)
        ax.errorbar(0.5 * (HB[1][:-1] + HB[1][1:]), HB[0], yerr=yerr,
                    color=colors['emp'], capsize=4., fmt='.', ls='',
                    lw=0.85, ms=2.5, capthick=0.85)

        mybins = HB[1] if zoom else nbins
        for simlist, col, name, mkey in [
            (sim_dict_list_ising, colors['ising'], 'Ising', 'ising'),
            (sim_dict_list_bc,    colors['bc'],    'BC',    'bc'),
            (sim_dict_list_beg,   colors['beg'],   'BEG',   'beg'),
        ]:
            for el in simlist:
                if condition(el):
                    lbl = model_label(el, name) if i == n_plot - 1 else None
                    _ms = MODEL_STYLES[mkey]
                    ax.hist(el['PCs'][:, comp], bins=mybins, histtype='step',
                            density=True, color=col, lw=_ms['lw'],
                            ls=_ms['ls'], label=lbl, zorder=_ms['zorder'])
                    break

        ax.set_xlabel(r'$x^{\prime}_{' + str(comp + 1) + r'}$', size=11)
        ax.set_yscale('log')

    fl[n_plot - 1].legend(fontsize=7)
    fig.suptitle(f'PC histograms — {dataset}', size=cfg.get('suptitle_fsize', 16))
    fig.supylabel(r"$\mathsf{h}_{x'_k}$", size=12)
    fig.tight_layout()

    fname = (f'PCshistogram_comparison_data={dataset}_zoom.pdf'
             if (zoom or pcs_toplot != list(range(n_pcs)))
             else f'PCshistogram_comparison_data={dataset}.pdf')
    _savefig(savepath, fname)


def plot_pc_histogram_beg_gauss_catind(X, sim_dict_list_beg, X_catind, X_gauss,
                                        U, mu_X, n_pcs,
                                        dataset, savepath, colors, cfg,
                                        lambdas_gauss=None, gauss_label='FA'):
    r"""Histograms ${\sf h}_{x'_k}$: BEG + Gauss/FA + Cat-ind vs empirical.

    Gauss/FA sampled histogram: thin step + thin hatching (domain is the
    sample's own range, already consistent with the analytic curve below).
    Analytic curve (if ``lambdas_gauss`` provided): dashed, thick -- the
    plain Gaussian theoretical curve N(0, lambdas_gauss[comp]), i.e.
    ``lambdas_gauss[comp]`` should simply be the empirical covariance
    eigenvalue along PCA direction ``U[comp]``. Deliberately kept as the
    classic Gaussian assumption rather than derived from the FA null model's
    own covariance, which is unnecessary here and can be numerically
    unstable.

    Parameters
    ----------
    gauss_label : str
        Legend label for the null baseline: ``'FA'`` (default) or ``'Gauss'``.

    Saved as: PCshistogram_beg_gauss_catind_data={dataset}.pdf
    """
    N = X.shape[0]
    nbins = cfg.get('nbins_pc', 60)
    nbins_emp = cfg.get('nbins_emp', 20)
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    ncolumns = min(n_pcs, 3)
    nrows = int(n_pcs / ncolumns + 0.99)

    Xprime = (X - mu_X) @ U.T

    fig, axs = plt.subplots(nrows, ncolumns,
                            figsize=(12, int(12 * nrows / 3.)),
                            sharey=True)
    fl = np.array(axs).flatten() if n_pcs > 1 else [axs]

    for i, comp in enumerate(range(n_pcs)):
        ax = fl[i]
        HB = ax.hist(Xprime[:, comp], bins=nbins_emp, histtype='step',
                     density=True, color=colors['emp'], lw=1.)
        ax.hist(Xprime[:, comp], bins=nbins_emp, alpha=alpha_emp,
                density=True, color=colors['emp'], lw=0.)
        bl = HB[1][1:] - HB[1][:-1]
        yerr = _wilson_error(HB[0], N, pvalue, bl)
        ax.errorbar(0.5 * (HB[1][:-1] + HB[1][1:]), HB[0], yerr=yerr,
                    color=colors['emp'], capsize=4., fmt='.', ls='',
                    lw=0.85, ms=2.5, capthick=0.85)

        # BEG
        for el in sim_dict_list_beg:
            if condition(el):
                lbl_beg = model_label(el, 'BEG') if i == n_pcs - 1 else None
                _ms = MODEL_STYLES['beg']
                ax.hist(el['PCs'][:, comp], bins=nbins, histtype='step',
                        density=True, color=colors['beg'], lw=_ms['lw'],
                        ls=_ms['ls'], label=lbl_beg, zorder=_ms['zorder'])
                break

        # Gauss/FA sampled (thin + hatch) + analytic normal
        if X_gauss is not None:
            Xgprime = (X_gauss - mu_X) @ U.T
            lbl_gs = f'{gauss_label} (sampled)' if i == n_pcs - 1 else None
            _ms_g = MODEL_STYLES['gauss']
            HB_pcgauss = _gauss_sampled_hist(ax, Xgprime[:, comp], nbins,
                                              colors['gauss'], label=lbl_gs,
                                              zorder=_ms_g['zorder'])
            if lambdas_gauss is not None:
                lbl_ga = f'{gauss_label} (analytic)' if i == n_pcs - 1 else None
                ax.plot(HB_pcgauss[1],
                        normaldist.pdf(HB_pcgauss[1], loc=0,
                                       scale=lambdas_gauss[comp] ** 0.5),
                        color=colors['gauss'], lw=_ms_g['lw'],
                        label=lbl_ga, ls=_ms_g['ls'],
                        zorder=_ms_g['zorder'])

        # Cat-ind
        if X_catind is not None:
            Xcprime = (X_catind - mu_X) @ U.T
            lbl_c = 'Cat-ind' if i == n_pcs - 1 else None
            _ms_c = MODEL_STYLES['nullcat']
            ax.hist(Xcprime[:, comp], bins=nbins, histtype='step',
                    density=True, color=colors['nullcat'], lw=_ms_c['lw'],
                    ls=_ms_c['ls'], label=lbl_c, zorder=_ms_c['zorder'])

        ax.set_xlabel(r'$x^{\prime}_{' + str(comp + 1) + r'}$', size=11)
        ax.set_yscale('log')

    fl[n_pcs - 1].legend(fontsize=7)
    fig.suptitle(f'PC histograms — BEG / {gauss_label} / Cat-ind — {dataset}',
                 size=cfg.get('suptitle_fsize', 16))
    fig.supylabel(r"$\mathsf{h}_{x'_k}$", size=12)
    fig.tight_layout()
    _savefig(savepath, f'PCshistogram_beg_gauss_catind_data={dataset}.pdf')


def plot_pc_histogram_simple(X, X_catind, X_gauss, X_gaussdisc, X_copula,
                             U, mu_X, n_pcs, dataset, savepath, colors, cfg,
                             gauss_label='FA'):
    r"""Histograms ${\sf h}_{x'_k}$: simple models vs empirical (legacy).

    Saved as: PCshistogram_comparison_simplemodels_data={dataset}.pdf
    """
    N = X.shape[0]
    nbins = cfg.get('nbins_pc', 60)
    nbins_emp = cfg.get('nbins_emp', 20)
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    ncolumns = 3
    nrows = int(n_pcs / ncolumns + 0.99)

    Xprime = (X - mu_X) @ U.T

    simple = [
        (X_gauss,     colors['gauss'],     gauss_label,  'gauss'),
        (X_gaussdisc, colors['gaussdisc'], 'Gauss-disc', 'gaussdisc'),
        (X_copula,    colors['copula'],    'Copula',     'copula'),
        (X_catind,    colors['nullcat'],   'Cat-ind',    'nullcat'),
    ]

    fig, axs = plt.subplots(nrows, ncolumns,
                            figsize=(12, int(12 * nrows / 3.)),
                            sharey=True)
    fl = axs.flatten()

    for comp in range(n_pcs):
        ax = fl[comp]
        HB = ax.hist(Xprime[:, comp], bins=nbins_emp, histtype='step',
                     density=True, color=colors['emp'], lw=1.)
        ax.hist(Xprime[:, comp], bins=nbins_emp, alpha=alpha_emp,
                density=True, color=colors['emp'], lw=0.)
        bl = HB[1][1:] - HB[1][:-1]
        yerr = _wilson_error(HB[0], N, pvalue, bl)
        ax.errorbar(0.5 * (HB[1][:-1] + HB[1][1:]), HB[0], yerr=yerr,
                    color=colors['emp'], capsize=4., fmt='.', ls='',
                    lw=0.85, ms=2.5, capthick=0.85)

        for Xm, col, lbl, mkey in simple:
            if Xm is not None:
                Xm_pc = (Xm - mu_X) @ U.T
                show_lbl = lbl if comp == n_pcs - 1 else None
                if mkey == 'gauss':
                    _gauss_sampled_hist(ax, Xm_pc[:, comp], nbins, col,
                                        label=show_lbl,
                                        zorder=MODEL_STYLES[mkey]['zorder'])
                else:
                    _ms = MODEL_STYLES[mkey]
                    ax.hist(Xm_pc[:, comp], bins=nbins, histtype='step',
                            density=True, color=col, lw=_ms['lw'],
                            ls=_ms['ls'], label=show_lbl, zorder=_ms['zorder'])

        ax.set_xlabel(r'$x^{\prime}_{' + str(comp + 1) + r'}$', size=11)
        ax.set_yscale('log')

    fl[n_pcs - 1].legend(fontsize=7)
    fig.suptitle(f'PC histograms — simple models — {dataset}',
                 size=cfg.get('suptitle_fsize', 16))
    fig.supylabel(r"$\mathsf{h}_{x'_k}$", size=12)
    fig.tight_layout()
    _savefig(savepath,
             f'PCshistogram_comparison_simplemodels_data={dataset}.pdf')


def plot_factor_histogram_beg_gauss_catind(Xf, sim_dict_list_beg,
                                            Xf_catind, Xf_gauss,
                                            n_components,
                                            dataset, savepath, colors, cfg,
                                            gauss_label='FA'):
    r"""Histograms ${\sf h}_{f_k}$ of factor scores: BEG + Gauss/FA + Cat-ind vs empirical.

    Parameters
    ----------
    Xf         : ndarray (N, n_components), empirical factor scores
    Xf_catind  : ndarray or None, factor scores of cat-ind samples
    Xf_gauss   : ndarray or None, factor scores of Gauss/FA null-model samples
    All factor scores should be pre-computed via fa.transform() before calling.
    gauss_label : str
        Legend label for the null baseline: ``'FA'`` (default) or ``'Gauss'``.

    Gauss/FA sampled: thin step + thin hatching, own domain.
    Analytic curve: plain normal Gaussian fit to the *empirical* factor
    scores (mean/std of ``Xf``) -- kept deliberately simple/stable rather
    than derived from the FA null model itself (which can be numerically
    unstable); domain spans the sampled-model histogram's own range so the
    two are still visually consistent.

    Saved as: factorhistogram_beg_gauss_catind_data={dataset}.pdf
    """
    from scipy.stats import norm as _norm

    N = Xf.shape[0]
    nbins = cfg.get('nbins_pc', 60)
    nbins_emp = nbins // 2
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    ncolumns = min(n_components, 3)
    nrows = int(n_components / ncolumns + 0.99)

    fig, axs = plt.subplots(nrows, ncolumns,
                            figsize=(12, int(12 * nrows / 3.)),
                            sharey=True)
    fl = np.array(axs).flatten() if n_components > 1 else [axs]

    for i in range(n_components):
        ax = fl[i]
        HB = ax.hist(Xf[:, i], bins=nbins_emp, histtype='step',
                     density=True, color='black', lw=1.2)
        ax.hist(Xf[:, i], bins=nbins_emp, alpha=alpha_emp,
                density=True, color=colors['emp'], lw=0.)
        bl = HB[1][1:] - HB[1][:-1]
        yerr = _wilson_error(HB[0], N, pvalue, bl)
        ax.errorbar(0.5 * (HB[1][:-1] + HB[1][1:]), HB[0], yerr=yerr,
                    color=colors['emp'], capsize=4., fmt='.', ls='',
                    lw=0.85, ms=2.5, capthick=0.85)

        # BEG
        for el in sim_dict_list_beg:
            if condition(el) and 'Yf' in el:
                lbl_beg = model_label(el, 'BEG') if i == n_components - 1 else None
                _ms = MODEL_STYLES['beg']
                ax.hist(el['Yf'][:, i], bins=nbins, histtype='step',
                        density=True, color=colors['beg'], lw=_ms['lw'], alpha=_ms['alphaline'],
                        ls=_ms['ls'], label=lbl_beg, zorder=_ms['zorder'])
                break

        # Gauss/FA sampled (thin + hatch) + analytic normal, same domain
        if Xf_gauss is not None:
            lbl_gs = f'{gauss_label} (sampled)' if i == n_components - 1 else None
            _ms_g = MODEL_STYLES['gauss']
            bins_gauss = _gauss_own_domain_bins(Xf_gauss[:, i], nbins)
            HB_gauss = _gauss_sampled_hist(ax, Xf_gauss[:, i], bins_gauss, colors['gauss'],
                                           label=lbl_gs, zorder=_ms_g['zorder'])
            xs_th = np.linspace(HB_gauss[1][0], HB_gauss[1][-1], 300)
            lbl_ga = f'{gauss_label} (analytic)' if i == n_components - 1 else None
            ax.plot(xs_th, _norm.pdf(xs_th, loc=Xf[:, i].mean(), scale=Xf[:, i].std()),
                    ls=_ms_g['ls'], alpha=_ms_g['alphaline'],
                    color=colors['gauss'], lw=_ms_g['lw'], label=lbl_ga,
                    zorder=_ms_g['zorder'])

        # Cat-ind
        if Xf_catind is not None:
            lbl_c = 'Cat-ind' if i == n_components - 1 else None
            _ms_c = MODEL_STYLES['nullcat']
            ax.hist(Xf_catind[:, i], bins=nbins, histtype='step', alpha=_ms_c['alphaline'],
                    density=True, color=colors['nullcat'], lw=_ms_c['lw'],
                    ls=_ms_c['ls'], label=lbl_c, zorder=_ms_c['zorder'])

        ax.set_xlabel(r'$f_{' + str(i + 1) + r'}$', size=cfg.get('xlabel_fsize', 11))
        ax.set_yscale('log')

    handles, labels = fl[n_components - 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
           bbox_to_anchor=(0.5, 0.965), ncol=len(handles), fontsize=9)

    #fl[0].legend(fontsize=9, bbox_to_anchor=(0.05, 1.1), loc='upper left', ncol=4)
    #fl[n_components - 1].legend(fontsize=9, bbox_to_anchor=(1.05, 1.), loc='upper left', ncol=1)
    fig.suptitle(f'Factor histograms — {dataset}',
                 size=cfg.get('suptitle_fsize', 16), y=1.0)
    fig.supylabel(r"$\mathsf{h}_{f_k}$", size=cfg.get('xlabel_fsize', 12)+1, rotation=0)
    fig.tight_layout()
    _savefig(savepath, f'factorhistogram_beg_gauss_catind_data={dataset}.pdf')


def plot_factor_histogram_maxent(Xf, sim_dict_list_ising, sim_dict_list_bc,
                                 sim_dict_list_beg, n_components,
                                 dataset, savepath, colors, cfg, zoom=False,
                                 factors_to_plot=None):
    r"""Histograms ${\sf h}_{f_k}$ of factor scores: maxent vs empirical.

    Requires that each sim_dict already has `el['Yf']` populated (via FactorAnalysis).

    Saved as: factorhistogram_comparison_data={dataset}.pdf  (or _zoom.pdf)
    """
    N = Xf.shape[0]
    nbins = cfg.get('nbins_pc', 60)
    nbins_emp = nbins // 2
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    #print(f"plot_factor_histogram_maxent: N={N}, nbins={nbins}, nbins_emp={nbins_emp}, pvalue={pvalue}, alpha_emp={alpha_emp}")

    if factors_to_plot is None:
        factors_to_plot = list(range(n_components))
    n_plot = len(factors_to_plot)
    if zoom:
        ncolumns = 1
    else:
        ncolumns = min(n_plot, 3)
    nrows = int(n_plot / ncolumns + 0.99)

    fig, axs = plt.subplots(nrows, ncolumns,
                            figsize=(12, int(12 * nrows / 3.)),
                            sharey=True)
    fl = np.array(axs).flatten() if n_plot > 1 else [axs]

    for i, comp in enumerate(factors_to_plot):
        ax = fl[i]
        HB = ax.hist(Xf[:, comp], bins=nbins_emp, histtype='step',
                     density=True, color='black', lw=1.2)
        ax.hist(Xf[:, comp], bins=nbins_emp, alpha=alpha_emp,
                density=True, color=colors['emp'], lw=0.)
        bl = HB[1][1:] - HB[1][:-1]
        yerr = _wilson_error(HB[0], N, pvalue, bl)
        ax.errorbar(0.5 * (HB[1][:-1] + HB[1][1:]), HB[0], yerr=yerr,
                    color=colors['emp'], capsize=4., fmt='.', ls='',
                    lw=0.85, ms=2.5, capthick=0.85)

        mybins = HB[1] if zoom else nbins
        for simlist, col, name, mkey in [
            (sim_dict_list_ising, colors['ising'], 'Ising', 'ising'),
            (sim_dict_list_bc,    colors['bc'],    'BC',    'bc'),
            (sim_dict_list_beg,   colors['beg'],   'BEG',   'beg'),
        ]:
            for el in simlist:
                if condition(el) and 'Yf' in el:
                    lbl = model_label(el, name) if i == n_plot - 1 else None
                    _ms = MODEL_STYLES[mkey]
                    ax.hist(el['Yf'][:, comp], bins=mybins, histtype='step',
                            density=True, color=col, lw=_ms['lw'], alpha=_ms['alphaline'],
                            ls=_ms['ls'], label=lbl, zorder=_ms['zorder'])
                    break

        
        ax.set_xlabel(r'$f_{' + str(comp + 1) + r'}$', size=cfg.get('xlabel_fsize', 11))
        ax.set_yscale('log')
        if zoom:
            ylim = ax.get_ylim()
            ax.set_ylim([3e-2, ylim[1]])
    handles, labels = fl[n_plot - 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
           bbox_to_anchor=(0.5, 0.965), ncol=len(handles), fontsize=9)

    #fl[n_plot - 1].legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.suptitle(f'Factor histograms — {dataset}', size=cfg.get('suptitle_fsize', 16), y=1.)
    fig.supylabel(r"$\mathsf{h}_{f_k}$", size=cfg.get('xlabel_fsize', 12)+1, rotation=0)
    fig.tight_layout()

    fname = (f'factorhistogram_comparison_data={dataset}_zoom.pdf'
             if zoom else f'factorhistogram_comparison_data={dataset}.pdf')
    _savefig(savepath, fname)


# ---------------------------------------------------------------------------
# Figure: zoomed first-factor histogram, all models together
# ---------------------------------------------------------------------------

def plot_factor0_histogram_zoom(Xf, sim_dict_list_ising, sim_dict_list_bc,
                                sim_dict_list_beg, Xf_catind, Xf_gauss,
                                dataset, savepath, colors, cfg,
                                gauss_label='FA', n_maxima=3,
                                y_floor_frac=0.15):
    r"""Single-panel, zoomed-in histogram of the *first* factor only,
    overlaying every model at once: empirical, Ising / BC / BEG (maxent) and
    Gauss/FA / Cat-ind (null models) -- i.e. all the curves shown across
    :func:`plot_factor_histogram_maxent` and
    :func:`plot_factor_histogram_beg_gauss_catind`, combined for factor 1
    only.

    Unlike those multi-panel figures, the y-axis floor is raised (to
    ``y_floor_frac`` of the empirical peak height, rather than down to the
    Wilson-CI noise floor) so the *relative* heights of the local maxima
    -- e.g. the three modes of the GCBS first factor -- are easy to compare
    instead of being compressed by several decades of near-empty tail.

    Black arrows, tilted at 45 degrees and coming from the upper right
    straight onto each peak (top-to-bottom, right-to-left), mark the
    ``n_maxima`` tallest local maxima of the empirical histogram.

    Parameters
    ----------
    Xf_catind, Xf_gauss : ndarray or None
        Factor scores of the null-model samples (as in
        :func:`plot_factor_histogram_beg_gauss_catind`); may be None.
    gauss_label : str
        Legend label for the null baseline: ``'FA'`` (default) or ``'Gauss'``.
    n_maxima : int
        Number of empirical local maxima to mark with an arrow.
    y_floor_frac : float
        Lower y-limit as a fraction of the empirical histogram's peak
        density.

    Saved as: factorhistogram_zoom_factor0_data={dataset}.pdf
    """
    from scipy.stats import norm as _norm

    N = Xf.shape[0]
    nbins = cfg.get('nbins_pc', 60)
    nbins_emp = nbins // 2
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    HB = ax.hist(Xf[:, 0], bins=nbins_emp, histtype='step',
                 density=True, color='black', lw=1.2, label='empirical')
    ax.hist(Xf[:, 0], bins=nbins_emp, alpha=alpha_emp,
            density=True, color=colors['emp'], lw=0.)
    bl = HB[1][1:] - HB[1][:-1]
    yerr = _wilson_error(HB[0], N, pvalue, bl)
    ax.errorbar(0.5 * (HB[1][:-1] + HB[1][1:]), HB[0], yerr=yerr,
                color=colors['emp'], capsize=4., fmt='.', ls='',
                lw=0.85, ms=2.5, capthick=0.85)

    # maxent models (Ising / BC / BEG)
    for simlist, col, name, mkey in [
        #(sim_dict_list_ising, colors['ising'], 'Ising', 'ising'),
        #(sim_dict_list_bc,    colors['bc'],    'BC',    'bc'),
        (sim_dict_list_beg,   colors['beg'],   'BEG',   'beg'),
    ]:
        for el in simlist:
            if condition(el) and 'Yf' in el:
                _ms = MODEL_STYLES[mkey]
                ax.hist(el['Yf'][:, 0], bins=nbins, histtype='step',
                        density=True, color=col, lw=_ms['lw'], alpha=_ms['alphaline'],
                        ls=_ms['ls'], label=model_label(el, name), zorder=_ms['zorder'])
                break

    # Gauss/FA sampled + analytic (plain normal fit to the empirical scores --
    # deliberately not FA-derived, see plot_factor_histogram_beg_gauss_catind)
    if Xf_gauss is not None:
        _ms_g = MODEL_STYLES['gauss']
        bins_gauss = _gauss_own_domain_bins(Xf_gauss[:, 0], nbins)
        HB_gauss = _gauss_sampled_hist(ax, Xf_gauss[:, 0], bins_gauss, colors['gauss'],
                                       label=f'{gauss_label} (sampled)', zorder=_ms_g['zorder'])
        xs_th = np.linspace(HB_gauss[1][0], HB_gauss[1][-1], 300)
        ax.plot(xs_th, _norm.pdf(xs_th, loc=Xf[:, 0].mean(), scale=Xf[:, 0].std()),
                ls=_ms_g['ls'], alpha=_ms_g['alphaline'], color=colors['gauss'],
                lw=_ms_g['lw'], label=f'{gauss_label} (analytic)', zorder=_ms_g['zorder'])

    # Cat-ind
    #if Xf_catind is not None:
    #    _ms_c = MODEL_STYLES['nullcat']
    #    ax.hist(Xf_catind[:, 0], bins=nbins, histtype='step', alpha=_ms_c['alphaline'],
    #            density=True, color=colors['nullcat'], lw=_ms_c['lw'],
    #            ls=_ms_c['ls'], label='Cat-ind', zorder=_ms_c['zorder'])

    ax.set_yscale('log')
    ymax = np.max(HB[0])
    ylim = ax.get_ylim()
    ax.set_ylim(max(y_floor_frac * ymax, ylim[0]), ylim[1])
    ax.set_xlim(HB[1][0]-0.5, HB[1][-1]+0.5)

    _add_peak_arrows(ax, Xf[:, 0], nbins_emp, n_peaks=n_maxima, color='black')

    ax.set_xlabel(r'$f_{1}$', size=cfg.get('xlabel_fsize', 13))
    ax.set_ylabel(r"$\mathsf{h}_{f_1}$", size=cfg.get('xlabel_fsize', 13) + 1,
                  rotation=0, labelpad=12)
    ax.set_title(f'First-factor histogram (zoom) — {dataset}',
                 fontsize=cfg.get('suptitle_fsize', 13))
    ax.legend(fontsize=8, loc='upper right', ncol=1)
    fig.tight_layout()
    _savefig(savepath, f'factorhistogram_zoom_factor0_data={dataset}.pdf')


# ---------------------------------------------------------------------------
# Figure: correlation-time analysis (JK tau + ACF)
# ---------------------------------------------------------------------------

def plot_correlation_time_analysis(sim_dict_list_ising, sim_dict_list_bc,
                                   sim_dict_list_beg, U, n_pcs_acf,
                                   dataset, savepath, colors,
                                   nlags=100, n_bs=10):
    r"""Three diagnostic plots for the integrated autocorrelation time:

    1. Jack-Knife tau vs block size for ``x'_1``;
    2. ACF of ``x'_1`` for Ising / BC / BEG with shaded JK-tau region;
    3. ACF of all PCs for the BEG model.
    """
    from statsmodels.tsa.stattools import acf as _acf

    def _first_valid(sdl):
        for el in sdl:
            if condition(el):
                return el
        return sdl[0]

    el_ising = _first_valid(sim_dict_list_ising)
    el_bc = _first_valid(sim_dict_list_bc)
    el_beg = _first_valid(sim_dict_list_beg)

    Ypc_ising = el_ising['configurations'] @ U[:n_pcs_acf].T
    Ypc_bc = el_bc['configurations'] @ U[:n_pcs_acf].T
    Ypc_beg = el_beg['configurations'] @ U[:n_pcs_acf].T

    bslist = np.array(np.logspace(1., 3., n_bs), dtype=int)

    taus_ising = np.zeros((n_pcs_acf, n_bs))
    tauerrs_ising = np.zeros((n_pcs_acf, n_bs, 2))
    taus_bc = np.zeros((n_pcs_acf, n_bs))
    tauerrs_bc = np.zeros((n_pcs_acf, n_bs, 2))
    taus_beg = np.zeros((n_pcs_acf, n_bs))
    tauerrs_beg = np.zeros((n_pcs_acf, n_bs, 2))

    for k in range(n_pcs_acf):
        _, taus_ising[k], tauerrs_ising[k], _ = JKerror(Ypc_ising[:, k], bslist, alpha=0.95)
        _, taus_bc[k], tauerrs_bc[k], _ = JKerror(Ypc_bc[:, k], bslist, alpha=0.95)
        _, taus_beg[k], tauerrs_beg[k], _ = JKerror(Ypc_beg[:, k], bslist, alpha=0.95)

    acf_ising = np.zeros((nlags + 1, n_pcs_acf))
    acf_bc = np.zeros((nlags + 1, n_pcs_acf))
    acf_beg = np.zeros((nlags + 1, n_pcs_acf))

    for k in range(n_pcs_acf):
        acf_ising[:, k] = _acf(Ypc_ising[:, k], adjusted=False, nlags=nlags,
                                qstat=False, fft=True, alpha=None,
                                bartlett_confint=True, missing='none')
        acf_bc[:, k] = _acf(Ypc_bc[:, k], adjusted=False, nlags=nlags,
                             qstat=False, fft=True, alpha=None,
                             bartlett_confint=True, missing='none')
        acf_beg[:, k] = _acf(Ypc_beg[:, k], adjusted=False, nlags=nlags,
                              qstat=False, fft=True, alpha=None,
                              bartlett_confint=True, missing='none')

    tempi = list(range(nlags + 1))
    ymin, ymax = -0.02, 1.02

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(bslist, taus_ising[0], yerr=tauerrs_ising[0].T,
                capsize=2, fmt='^-', label='Ising', color=colors['ising'], lw=0.5)
    ax.errorbar(bslist, taus_bc[0], yerr=tauerrs_bc[0].T,
                capsize=4, fmt='o-', label='BC', color=colors['bc'], lw=0.5)
    ax.errorbar(bslist, taus_beg[0], yerr=tauerrs_beg[0].T,
                capsize=6, fmt='s-', label='BEG', color=colors['beg'], lw=0.5)
    ax.axvspan(tauerrs_ising[0, -1, 0], tauerrs_ising[0, -1, 1], color=colors['ising'], alpha=0.2, zorder=-1)
    ax.axvspan(tauerrs_bc[0, -1, 0], tauerrs_bc[0, -1, 1], color=colors['bc'], alpha=0.2, zorder=-1)
    ax.axvspan(tauerrs_beg[0, -1, 0], tauerrs_beg[0, -1, 1], color=colors['beg'], alpha=0.2, zorder=-1)

    ax.set_xlabel('Jack-Knife block size', size=14)
    ax.set_ylabel(r"correlation time of $x'_1$", size=14)
    ax.set_title('Jack-Knife correlation time -- ' + dataset, size=10)
    ax.legend()
    _savefig(savepath, f'correlationtime_data={dataset}.pdf')

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(acf_ising[:, 0], '.-', color=colors['ising'], label='Ising', lw=0.5)
    ax.plot(acf_bc[:, 0], '+-', color=colors['bc'], label='BC', lw=0.5)
    ax.plot(acf_beg[:, 0], 'x-', color=colors['beg'], label='BEG', lw=0.5)

    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$t$", size=14)
    ax.set_ylabel(r"$g_{x'_1}(t)$", size=14)
    ax.set_title(r"$x'_1$ auto-correlation function -- " + dataset, size=10)
    ax.legend(fontsize=7)
    _savefig(savepath, f'acf_models={dataset}.pdf')

    _cycle_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                     '#bcbd22', '#17becf']
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, col in zip(range(n_pcs_acf), _cycle_colors):
        ax.plot(acf_beg[:, k], '.-', label=r'$j=' + str(k) + r'$',
                color=col, lw=0.5)
        ax.axvline(np.sum(acf_beg[:, k]), color=col, ls='--', lw=0.8)
        ax.axvspan(tauerrs_beg[k, -1, 0], tauerrs_beg[k, -1, 1], color=col, alpha=0.2, zorder=-1)
    

    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$t$", size=14)
    ax.set_ylabel(r"$g_{x'_j}(t)$", size=14)
    ax.set_title(r"$x'_j$ BEG auto-correlation function -- " + dataset, size=10)
    ax.legend(fontsize=7)
    _savefig(savepath, f'acf_beg_PCs={dataset}.pdf')


# ---------------------------------------------------------------------------
# Default colour palette
# ---------------------------------------------------------------------------

DEFAULT_COLORS = {
    'emp': '#222222',
    'ising': '#1f77b4',
    'bc': '#ff7f0e',
    'beg': '#2ca02c',
    'gauss': '#9467bd',
    'gaussdisc': '#8c564b',
    'nullcat': '#e377c2',
    'copula': '#17becf',
}

DEFAULT_CFG = {
    'nbins_E2d': 80,
    'nbins_energy': 80,
    'nbins_pc': 60,
    'nbins_emp': 20,
    'nbins_emp_dist': 20,   # empirical bins for distance (E2d / Mahalanobis) figures
    'mypvalue': 0.05,
    'alpha_emp': 0.1,
    'nb_items': 12,
    'item_ncols': 4,
    'xlabel_fsize': 14,
    'ylabel_fsize': 14,
    'suptitle_fsize': 16,
    # Default "simple" null-model baseline (points 2/3 of the figures spec):
    # 'FA' = Factor-Analyzers (low-rank + diagonal-noise covariance);
    # 'gauss' = full multivariate Gaussian (empirical mean/covariance).
    # See fit_gauss_or_fa_model() / sample_gauss_or_fa_model().
    'gauss_model_method': 'FA',
    'n_components_gauss_model': 5,
}
