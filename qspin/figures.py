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
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import norm as normaldist 

from .histograms import wilson_score_interval
from .jackknife import JKerror
from .nullmodels import fourier_transform_integral, energydensity_gauss


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

def compute_pseudolikelihood_moments_BC(X, states, J, h):
    r"""Compute the BC pseudo-likelihood conditional moments.

    Returns
    -------
    expected_x : ndarray, shape (M,)
    expected_x2 : ndarray, shape (M,)
    expected_covariance : ndarray, shape (M, M)
        Conditional first/second/covariance moments under the pseudo-likelihood
        density.  These are the quantities required to vanish at a stationary
        point of the pseudo-likelihood objective.
    """
    n_samples, n_nodes = np.shape(X)
    phi = X @ J + h - X * np.diag(J)
    exponent = phi[:, :, np.newaxis] * states[np.newaxis, np.newaxis, :]
    exponent += 0.5 * np.diag(J)[np.newaxis, :, np.newaxis] * states[np.newaxis, np.newaxis, :] ** 2
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

def _wilson_error(counts, n, pvalue, bin_lengths):
    """Wilson score CI half-widths, scaled by bin_length, for errorbar plots."""
    Hmass = counts * bin_lengths
    return wilson_score_interval(p=Hmass, n=n, mypvalue=pvalue,
                                 return_interval_lengths=True) / bin_lengths


def _emp_hist_with_errors(ax, values, nbins_emp, N, mypvalue, alpha_emp, color,
                          label='empirical'):
    HB = ax.hist(values, histtype='step', bins=nbins_emp,
                 label=label, density=True, lw=1., color=color)
    ax.hist(values, alpha=alpha_emp, bins=nbins_emp, density=True, lw=0., color=color)
    bl = HB[1][1:] - HB[1][:-1]
    yerr = _wilson_error(HB[0], N, mypvalue, bl)
    ax.errorbar(0.5 * (HB[1][:-1] + HB[1][1:]), HB[0],
                yerr=yerr, color=color, capsize=4.,
                fmt='.', ls='', lw=0.65, ms=2.5, capthick=0.65)
    return HB


def _savefig(savepath, fname):
    os.makedirs(savepath, exist_ok=True)
    plt.savefig(os.path.join(savepath, fname), bbox_inches='tight')
    plt.close()


def _model_linestyle(el):
    return ':' if el['reset'] == 'rand' else '-'


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
    def _remove_diag(M):
        n = M.shape[0]
        mask = ~np.eye(n, dtype=bool)
        return M[mask]

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
                                  sim_dict_list_beg, dataset, savepath, colors, cfg):
    r"""Empirical vs MCMC-sampled sufficient statistics."""
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
    r"""Histograms ${\sf h}_{x_i}$ for each item: empirical vs Ising / BC / BEG."""
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
        ax.hist(X[:, idx], bins=item_bins, histtype='step', density=True,
                color=colors['emp'], lw=1.)
        HB = ax.hist(X[:, idx], bins=item_bins, alpha=alpha_emp, density=True,
                     lw=0., color=colors['emp'])
        bl = HB[1][1:] - HB[1][:-1]
        yerr = _wilson_error(HB[0], N, pvalue, bl)
        ax.errorbar(0.5 * (HB[1][:-1] + HB[1][1:]), HB[0], yerr=yerr,
                    color=colors['emp'], capsize=4., fmt='.', ls='',
                    lw=0.65, ms=2.5, capthick=0.65)
        ax.set_title(f'item {idx + 1}', size=8)

        for simlist, col, name in [
            (sim_dict_list_ising, colors['ising'], 'Ising'),
            (sim_dict_list_bc, colors['bc'], 'BC'),
            (sim_dict_list_beg, colors['beg'], 'BEG'),
        ]:
            for el in simlist:
                if condition(el):
                    lbl = name if idx == nb_items - 1 else None
                    ax.hist(el['configurations'][:, idx], bins=item_bins,
                            histtype='step', density=True, color=col,
                            lw=1.75, label=lbl)
                    break

    fl[nb_items - 1].legend(fontsize=7)
    fig.suptitle(f'Item histograms -- {dataset}', size=cfg.get('suptitle_fsize', 16))
    fig.tight_layout()
    _savefig(savepath, f'itemhistogram_comparison_data={dataset}.pdf')


# ---------------------------------------------------------------------------
# Figure: Euclidean distance histogram (maxent + Gaussian)
# ---------------------------------------------------------------------------
def function_to_fouriertransform(z,lambdas):
    myarray=np.array(1-2.*z*1j*lambdas)**0.5
    return np.prod(myarray)**-1


def plot_E2d_histogram_maxent(X, sim_dict_list_ising, sim_dict_list_bc,
                              sim_dict_list_beg, dataset, savepath, colors, cfg,
                              lambdas=None,X_gauss=None):
    r"""Histogram ${\sf h}_{d_{\bf x}}$ (Euclidean distance to mean): maxent vs empirical.

    Saved as: E2d_histogram_logscale={True|False}data={dataset}.pdf
    """
    N = X.shape[0]
    nbins = cfg.get('nbins_E2d', 80)
    nbins_emp = cfg.get('nbins_emp', 20)
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    E2d_X = np.sum((X - np.mean(X, axis=0)) ** 2, axis=1)
    E2dmax = np.max(E2d_X) * 1.1
    bins_E2d = np.arange(1e-2, E2dmax, E2dmax / nbins)

    for logscale in [True, False]:
        fig, ax = plt.subplots(figsize=(7, 5))
        HB_e2d_X = _emp_hist_with_errors(ax, E2d_X, nbins_emp, N, pvalue, alpha_emp,
                              colors['emp'])
        # maxent models
        for simlist, col, name in [
            (sim_dict_list_ising, colors['ising'], 'Ising'),
            (sim_dict_list_bc,    colors['bc'],    'BC'),
            (sim_dict_list_beg,   colors['beg'],   'BEG'),
        ]:
            for el in simlist:
                if condition(el):
                    ax.hist(el['distances_maxent'], histtype='step',
                            bins=bins_E2d, label=model_label(el, name),
                            density=True, lw=1.75, color=col,
                            ls=_model_linestyle(el))
                    break
        # Gaussian
        ylim = ax.get_ylim()
        if X_gauss is not None:
            E2d_g = np.sum((X_gauss - np.mean(X_gauss, axis=0)) ** 2, axis=1)
            HB_e2d_g=ax.hist(E2d_g, histtype='step', bins=bins_E2d,label='Gauss',density=True, lw=0.75, color=colors['gauss'],hatch='\\',ls='-')
            if lambdas is not None:
                integrand = lambda z: function_to_fouriertransform(z,lambdas)
                maxz=10.
                bx2_gauss = HB_e2d_g[1]
                hx2_gauss = np.array([fourier_transform_integral(integrand,x,maxz) for x in bx2_gauss])
                ax.plot(bx2_gauss,hx2_gauss,ls='--',lw=2.,color=colors['gauss'],label='sum-chi')
        
        y_min = np.sort(HB_e2d_X[0][np.where(HB_e2d_X[0]>1.0E-12)[0]])[0]
        ax.set_ylim([np.max([ylim[0],y_min]),ylim[1]])

        if logscale:
            ax.set_yscale('log')
        ax.set_xlabel(r'$d_{\bf x}^2 = \|{\bf x} - \boldsymbol{\mu}\|_2^2$',
                      size=cfg.get('xlabel_fsize', 14))
        ax.set_ylabel(r'${\sf h}_{d_{\bf x}}$', size=cfg.get('xlabel_fsize', 14))
        ax.set_title(f'Euclidean distance histogram — {dataset}')
        ax.legend(fontsize=8)
        _savefig(savepath, f'E2d_histogram_logscale={logscale}data={dataset}.pdf')


def plot_E2d_histogram_simple(X, X_catind, X_gauss, X_gaussdisc, X_copula,
                              dataset, savepath, colors, cfg):
    r"""${\sf h}_{d_{\bf x}}$: simple models vs empirical."""
    N = X.shape[0]
    nbins = cfg.get('nbins_E2d', 80)
    nbins_emp = cfg.get('nbins_emp', 20)
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    E2d_X = np.sum((X - np.mean(X, axis=0)) ** 2, axis=1)
    E2dmax = np.max(E2d_X) * 1.1
    bins_E2d = np.arange(1e-2, E2dmax, E2dmax / nbins)

    simple = [
        (X_gauss, colors['gauss'], 'Gauss', '--'),
        (X_gaussdisc, colors['gaussdisc'], 'Gauss-disc', '-.'),
        (X_copula, colors['copula'], 'Copula', '-'),
        (X_catind, colors['nullcat'], 'Cat-ind', '-'),
    ]

    for logscale in [True, False]:
        fig, ax = plt.subplots(figsize=(7, 5))
        _emp_hist_with_errors(ax, E2d_X, nbins_emp, N, pvalue, alpha_emp, colors['emp'])
        for Xm, col, lbl, ls in simple:
            if Xm is not None:
                E2d_m = np.sum((Xm - np.mean(Xm, axis=0)) ** 2, axis=1)
                ax.hist(E2d_m, histtype='step', bins=bins_E2d,
                        label=lbl, density=True, lw=1.75, color=col, ls=ls)
        if logscale:
            ax.set_yscale('log')
        ax.set_xlabel(r'$d_{\bf x}^2 = \|{\bf x} - \boldsymbol{\mu}\|_2^2$',
                      size=cfg.get('xlabel_fsize', 14))
        ax.set_ylabel(r'${\sf h}_{d_{\bf x}}$', size=cfg.get('xlabel_fsize', 14))
        ax.set_title(f'Euclidean distance -- simple models -- {dataset}')
        ax.legend(fontsize=8)
        _savefig(savepath, f'E2d_histogram_simplemodels_data={dataset}logscale={logscale}.pdf')


def plot_E2d_histogram_begvscopula(X, sim_dict_list_beg, X_copula,
                                   dataset, savepath, colors, cfg):
    r"""${\sf h}_{d_{\bf x}}$: BEG vs copula vs empirical.

    Saved as: E2d_histogram_begvscopula_data={dataset}_logscale=_{True|False}.pdf
    """
    N = X.shape[0]
    nbins = cfg.get('nbins_E2d', 60)
    nbins_emp = cfg.get('nbins_emp', 20)
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
        ax.hist(E2d_cop, histtype='step', bins=bins_E2d, label='Copula',
                density=True, lw=1.75, color=colors['copula'])
        for el in sim_dict_list_beg:
            if condition(el):
                ax.hist(el['distances_maxent'], histtype='step', bins=bins_E2d,
                        label=model_label(el, 'BEG'), density=True,
                        lw=1.75, color=colors['beg'], ls=_model_linestyle(el))
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
    r"""Histogram ${\sf h}_{d_{\bf x}^{({\rm M})}}$ (Mahalanobis, common C): maxent vs empirical.

    Saved as: energy_histogram_commonC_{dataset}_log={True|False}.pdf
    """
    nbins = cfg.get('nbins_energy', 80)
    nbins_emp = cfg.get('nbins_emp', 20)
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    energymax = np.max(energies_X) * 1.1
    bins_energy = np.arange(1e-2, energymax, energymax / nbins)

    for logscale in [True, False]:
        fig, ax = plt.subplots(figsize=(7, 5))
        _emp_hist_with_errors(ax, energies_X, nbins_emp, N, pvalue, alpha_emp,
                              colors['emp'])
        HB_maxent = None
        for simlist, col, name in [
            (sim_dict_list_ising, colors['ising'], 'Ising'),
            (sim_dict_list_bc,    colors['bc'],    'BC'),
            (sim_dict_list_beg,   colors['beg'],   'BEG'),
        ]:
            for el in simlist:
                if condition(el):
                    HB_maxent = ax.hist(el['energies_maxent_commonC'], histtype='step',
                            bins=bins_energy, label=model_label(el, name),
                            density=True, lw=1.75, color=col,
                            ls=_model_linestyle(el))
                    break
        assert HB_maxent is not None, "No model element satisfies condition — cannot compute y_min"
        y_min = np.sort(HB_maxent[0][np.where(HB_maxent[0]>1.0E-12)[0]])[0]
        ylim = ax.get_ylim()

        if energies_Xgauss is not None:
            #ax.hist(energies_Xgauss, histtype='step', bins=bins_energy,
            #        label='Gauss', density=True, lw=1.75,
            #        color=colors['gauss'], ls='--')
            
            #####################
            # gauss
            HB_Xgauss = ax.hist(energies_Xgauss,hatch='//',histtype=u'step',bins=bins_energy,label='Gauss',density=True,lw=0.5,color=colors['gauss'],ls='-')
            #HB_Xgauss = np.histogram(energies_Xgauss,bins=bins_energy,density=True)

            M=sim_dict_list_ising[0]['configurations'].shape[1]
            ax.plot(HB_Xgauss[1],energydensity_gauss(HB_Xgauss[1],M),ls='--',color=colors['gauss'],lw=2.,label='sum-chi')
            #####################

        
        ax.set_ylim([np.max([ylim[0],y_min]),ylim[1]])

        if logscale:
            ax.set_yscale('log')
        ax.set_xlabel(
            r'$d_{\bf x}^{({\rm M})} = \frac{1}{2}{\bf x}^\dag C^{-1}{\bf x}$',
            size=cfg.get('xlabel_fsize', 14))
        ax.set_ylabel(r'${\sf h}_{d_{\bf x}^{({\rm M})}}$', size=cfg.get('ylabel_fsize', 14))
        ax.set_title(f'Mahalanobis distance (common $C$) — {dataset}')
        ax.legend(fontsize=8)
        _savefig(savepath,
                 f'energy_histogram_commonC_{dataset}_log={logscale}.pdf')


def plot_mahalanobis_modelcov(energies_X, sim_dict_list_ising,
                              sim_dict_list_bc, sim_dict_list_beg,
                              N, M, bins_energy, dataset, savepath, colors, cfg):
    r"""Mahalanobis distance (model $\Sigma$) histogram."""
    from scipy.special import gamma as _gamma

    def _gauss_density(xs, d):
        return xs ** (d / 2 - 1) * np.exp(-xs) / _gamma(d / 2)

    nbins_emp = cfg.get('nbins_emp', 20)
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    for logscale in [True, False]:
        fig, ax = plt.subplots(figsize=(7, 5))
        _emp_hist_with_errors(ax, energies_X, nbins_emp, N, pvalue, alpha_emp, colors['emp'])
        for simlist, col, name in [
            (sim_dict_list_ising, colors['ising'], 'Ising'),
            (sim_dict_list_bc, colors['bc'], 'BC'),
            (sim_dict_list_beg, colors['beg'], 'BEG'),
        ]:
            for el in simlist:
                if condition(el):
                    ax.hist(el['energies_maxent'], histtype='step',
                            bins=bins_energy, label=model_label(el, name),
                            density=True, lw=1.75, color=col,
                            ls=_model_linestyle(el))
                    break
        xs_th = np.linspace(max(bins_energy[0], 1e-6), bins_energy[-1], 300)
        ax.plot(xs_th, _gauss_density(xs_th, M), ls='--',
                color=colors['gauss'], lw=2., label='$\\chi^2$-theory')
        if logscale:
            ax.set_yscale('log')
        ax.set_xlabel(r'$d_{\bf x}^{({\rm M})} = \frac{1}{2}{\bf x}^\dag \Sigma^{-1}{\bf x}$',
                      size=cfg.get('xlabel_fsize', 14))
        ax.set_ylabel(r'${\sf h}_{d_{\bf x}^{({\rm M})}}$', size=cfg.get('ylabel_fsize', 14))
        ax.set_title(f'Mahalanobis distance (model $\\Sigma$) -- {dataset}')
        ax.legend(fontsize=8)
        _savefig(savepath, f'energy_histogram_{dataset}_log={logscale}.pdf')


def plot_mahalanobis_commonC_simple(energies_X, energies_Xcatind,
                                    energies_Xgauss, energies_Xgaussdisc,
                                    energies_Xcopula, bins_energy,
                                    N, M, dataset, savepath, colors, cfg):
    r"""${\sf h}_{d_{\bf x}^{({\rm M})}}$ (common $C$): simple models vs empirical.

    Saved as: energy_histogram_commonC_simplemodels_{dataset}_log={True|False}.pdf
    """
    from scipy.special import gamma as _gamma

    def _gauss_density(xs, d):
        return xs ** (d / 2 - 1) * np.exp(-xs) / _gamma(d / 2)

    nbins_emp = cfg.get('nbins_emp', 20)
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    simple = [
        (energies_Xgauss,    colors['gauss'],     'Gauss',      '--'),
        (energies_Xgaussdisc,colors['gaussdisc'], 'Gauss-disc', '-.'),
        (energies_Xcopula,   colors['copula'],    'Copula',     '-'),
        (energies_Xcatind,   colors['nullcat'],   'Cat-ind',    '-'),
    ]

    for logscale in [True, False]:
        fig, ax = plt.subplots(figsize=(7, 5))
        _emp_hist_with_errors(ax, energies_X, nbins_emp, N, pvalue, alpha_emp,
                              colors['emp'])
        for ens, col, lbl, ls in simple:
            if ens is not None:
                ax.hist(ens, histtype='step', bins=bins_energy,
                        label=lbl, density=True, lw=1.75, color=col, ls=ls)
        if logscale:
            ax.set_yscale('log')
        ax.set_xlabel(
            r'$d_{\bf x}^{({\rm M})} = \frac{1}{2}{\bf x}^\dag C^{-1}{\bf x}$',
            size=cfg.get('xlabel_fsize', 14))
        ax.set_ylabel(r'${\sf h}_{d_{\bf x}^{({\rm M})}}$', size=cfg.get('ylabel_fsize', 14))
        ax.set_title(f'Mahalanobis distance — simple models — {dataset}')
        ax.legend(fontsize=8)
        _savefig(savepath,
                 f'energy_histogram_commonC_simplemodels_{dataset}_log={logscale}.pdf')


def plot_mahalanobis_commonC_begvscopula(energies_X, sim_dict_list_beg,
                                         energies_Xcopula, bins_energy,
                                         N, dataset, savepath, colors, cfg):
    r"""${\sf h}_{d_{\bf x}^{({\rm M})}}$ (common $C$): BEG vs copula vs empirical.

    Saved as: energy_histogram_commonC_begvscopula_{dataset}_log={True|False}.pdf
    """
    nbins_emp = cfg.get('nbins_emp', 20)
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    for logscale in [True, False]:
        fig, ax = plt.subplots(figsize=(7, 5))
        _emp_hist_with_errors(ax, energies_X, nbins_emp, N, pvalue, alpha_emp,
                              colors['emp'])
        ax.hist(energies_Xcopula, histtype='step', bins=bins_energy,
                label='Copula', density=True, lw=1.75, color=colors['copula'])
        for el in sim_dict_list_beg:
            if condition(el):
                ax.hist(el['energies_maxent_commonC'], histtype='step',
                        bins=bins_energy, label=model_label(el, 'BEG'),
                        density=True, lw=1.75, color=colors['beg'],
                        ls=_model_linestyle(el))
                break
        if logscale:
            ax.set_yscale('log')
        ax.set_xlabel(
            r'$d_{\bf x}^{({\rm M})} = \frac{1}{2}{\bf x}^\dag C^{-1}{\bf x}$',
            size=cfg.get('xlabel_fsize', 14))
        ax.set_ylabel(r'${\sf h}_{d_{\bf x}^{({\rm M})}}$', size=cfg.get('ylabel_fsize', 14))
        ax.set_title(f'Mahalanobis distance — BEG vs Copula — {dataset}')
        ax.legend(fontsize=8)
        _savefig(savepath,
                 f'energy_histogram_commonC_begvscopula_{dataset}_log={logscale}.pdf')


# ---------------------------------------------------------------------------
# Figure: principal-component histograms (maxent + Gaussian)
# ---------------------------------------------------------------------------

def plot_pc_histogram_maxent(X, sim_dict_list_ising, sim_dict_list_bc,
                             sim_dict_list_beg, U, mu_X, n_pcs,
                             dataset, savepath, colors, cfg,lambdas_gauss,
                             Xprime_gauss=None, zoom=False, pcs_toplot=None):
    r"""Histograms ${\sf h}_{x'_k}$ of principal components: maxent vs empirical.

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
    mylw = 2.5 if zoom else 1.75
    figsize = 8 if zoom else 12

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
        for simlist, col, name in [
            (sim_dict_list_ising, colors['ising'], 'Ising'),
            (sim_dict_list_bc,    colors['bc'],    'BC'),
            (sim_dict_list_beg,   colors['beg'],   'BEG'),
        ]:
            for el in simlist:
                if condition(el):
                    lbl = model_label(el, name) if i == n_plot - 1 else None
                    ax.hist(el['PCs'][:, comp], bins=mybins, histtype='step',
                            density=True, color=col, lw=mylw, label=lbl)
                    break

        if Xprime_gauss is not None:
            lbl_g = 'Gauss' if i == n_plot - 1 else None
            HB_pcgauss=ax.hist(Xprime_gauss[:, comp], bins=mybins, histtype='step',
                    density=True, color=colors['gauss'], lw=0.7,hatch='\\',
                    ls='-', label=lbl_g)

            ax.plot(HB_pcgauss[1],normaldist.pdf(HB_pcgauss[1],loc=0,scale=lambdas_gauss[comp]**0.5),color=colors['gauss'],lw=2.,label='normal dist.',ls='--')



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

def plot_pc_histogram_simple(X, X_catind, X_gauss, X_gaussdisc, X_copula,
                             U, mu_X, n_pcs, dataset, savepath, colors, cfg):
    r"""Histograms ${\sf h}_{x'_k}$: simple models vs empirical.

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
        (X_gauss,    colors['gauss'],    'Gauss',     '--'),
        (X_gaussdisc,colors['gaussdisc'],'Gauss-disc', '-.'),
        (X_copula,   colors['copula'],   'Copula',    '-'),
        (X_catind,   colors['nullcat'],  'Cat-ind',   '-'),
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

        for Xm, col, lbl, ls in simple:
            if Xm is not None:
                Xm_pc = (Xm - mu_X) @ U.T
                show_lbl = lbl if comp == n_pcs - 1 else None
                ax.hist(Xm_pc[:, comp], bins=nbins, histtype='step',
                        density=True, color=col, lw=1.75, ls=ls, label=show_lbl)

        ax.set_xlabel(r'$x^{\prime}_{' + str(comp + 1) + r'}$', size=11)

        ax.set_yscale('log')
    fl[n_pcs - 1].legend(fontsize=7)
    fig.suptitle(f'PC histograms — simple models — {dataset}',
                 size=cfg.get('suptitle_fsize', 16))
    fig.supylabel(r"$\mathsf{h}_{x'_k}$", size=12)
    fig.tight_layout()
    _savefig(savepath,
             f'PCshistogram_comparison_simplemodels_data={dataset}.pdf')

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
    nbins_emp = cfg.get('nbins_emp', 20)
    pvalue = cfg.get('mypvalue', 0.05)
    alpha_emp = cfg.get('alpha_emp', 0.1)

    if factors_to_plot is None:
        factors_to_plot = list(range(n_components))
    n_plot = len(factors_to_plot)
    ncolumns = min(n_plot, 3)
    nrows = int(n_plot / ncolumns + 0.99)
    mylw = 2.5 if zoom else 1.75

    fig, axs = plt.subplots(nrows, ncolumns,
                            figsize=(12, int(12 * nrows / 3.)),
                            sharey=True)
    fl = np.array(axs).flatten() if n_plot > 1 else [axs]

    for i, comp in enumerate(factors_to_plot):
        ax = fl[i]
        HB = ax.hist(Xf[:, comp], bins=nbins_emp, histtype='step',
                     density=True, color=colors['emp'], lw=1.)
        ax.hist(Xf[:, comp], bins=nbins_emp, alpha=alpha_emp,
                density=True, color=colors['emp'], lw=0.)
        bl = HB[1][1:] - HB[1][:-1]
        yerr = _wilson_error(HB[0], N, pvalue, bl)
        ax.errorbar(0.5 * (HB[1][:-1] + HB[1][1:]), HB[0], yerr=yerr,
                    color=colors['emp'], capsize=4., fmt='.', ls='',
                    lw=0.85, ms=2.5, capthick=0.85)

        mybins = HB[1] if zoom else nbins
        for simlist, col, name in [
            (sim_dict_list_ising, colors['ising'], 'Ising'),
            (sim_dict_list_bc,    colors['bc'],    'BC'),
            (sim_dict_list_beg,   colors['beg'],   'BEG'),
        ]:
            for el in simlist:
                if condition(el) and 'Yf' in el:
                    lbl = model_label(el, name) if i == n_plot - 1 else None
                    ax.hist(el['Yf'][:, comp], bins=mybins, histtype='step',
                            density=True, color=col, lw=mylw, label=lbl)
                    break

        from scipy.stats import norm as _norm
        xs_th = np.linspace(HB[1][0], HB[1][-1], 300)
        lbl_norm = 'Normal' if i == n_plot - 1 else None
        ax.plot(xs_th, _norm.pdf(xs_th), ls='--',
                color=colors['gauss'], lw=2., label=lbl_norm)

        ax.set_xlabel(r'$f_{' + str(comp + 1) + r'}$', size=11)

        ax.set_yscale('log')
        if zoom:
            ylim = ax.get_ylim()
            ax.set_ylim([3e-2, ylim[1]])
    fl[n_plot - 1].legend(fontsize=7)
    fig.suptitle(f'Factor histograms — {dataset}', size=cfg.get('suptitle_fsize', 16))
    fig.supylabel(r"$\mathsf{h}_{f_k}$", size=12)
    fig.tight_layout()

    fname = (f'factorhistogram_comparison_data={dataset}_zoom.pdf'
             if zoom else f'factorhistogram_comparison_data={dataset}.pdf')
    _savefig(savepath, fname)


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
    'mypvalue': 0.05,
    'alpha_emp': 0.1,
    'nb_items': 12,
    'item_ncols': 4,
    'xlabel_fsize': 14,
    'ylabel_fsize': 14,
    'suptitle_fsize': 16,
}
