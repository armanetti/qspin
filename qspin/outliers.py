"""
Outlier-analysis helpers.

Compare an outlier subset of a dataset to the bulk (or to the full dataset)
in terms of per-feature means and per-feature histograms.  Plots are written
as PDF files to the current working directory using a naming convention based
on ``method`` and ``dataset``.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem


def outlier_analysis_factormeans(Xf, indices_outliers_X, method, dataset, show=True):
    """Compare factor-score means of an outlier subset vs the full dataset.

    Produces three PDFs (matshow, errorbar, histogram-grid) in the cwd.
    """
    N, n_components = Xf.shape

    means_outliers_Xf = np.mean(Xf[indices_outliers_X], axis=0)
    stds_outliers_Xf = sem(Xf[indices_outliers_X], axis=0)
    means_Xf = np.mean(Xf, axis=0)
    stds_Xf = sem(Xf, axis=0)

    plt.matshow(np.array([means_Xf, means_outliers_Xf]))
    plt.colorbar()
    plt.savefig(f'factormeans_comparison_matshow_{method}_{dataset}.pdf')
    plt.show() if show else plt.close()

    plt.errorbar(np.arange(0, len(means_Xf)), y=means_Xf, yerr=stds_Xf,
                 fmt='s--', capsize=10)
    plt.errorbar(np.arange(0, len(means_outliers_Xf)), y=means_outliers_Xf,
                 yerr=stds_outliers_Xf, fmt='x--', capsize=10, label='Xf outliers')
    plt.legend()
    plt.savefig(f'factormean_comparison_{method}_{dataset}.pdf')
    plt.show() if show else plt.close()

    ncolumns = 3
    nrows = int(n_components / 3) + 1 * (n_components % 3)
    fig, axs = plt.subplots(nrows, ncolumns, layout='constrained',
                            figsize=(8, int(8 * nrows / 3.)), sharex=True, sharey=False)
    axs_flattened = axs.flatten()
    for k in range(n_components):
        axs_flattened[k].hist(Xf[:, k], bins=40, alpha=0.5, density=True)
        axs_flattened[k].hist(Xf[indices_outliers_X, k], bins=40, alpha=0.5, density=True)
    plt.yscale('log')
    plt.savefig(f'factorhistogram_comparison_{method}_{dataset}.pdf')
    plt.show() if show else plt.close()


def outlier_analysis_means(X, indices_outliers_X, method, dataset, show=True):
    """Compare per-item means of an outlier subset vs the bulk."""
    N, M = X.shape
    bulk_indices = list(set(range(N)) - set(indices_outliers_X))

    means_outliers_X = np.mean(X[indices_outliers_X], axis=0)
    stds_outliers_X = sem(X[indices_outliers_X], axis=0)
    means_Xbulk = np.mean(X[bulk_indices], axis=0)
    stds_Xbulk = sem(X[bulk_indices], axis=0)
    means_X = np.mean(X, axis=0)
    stds_X = sem(X, axis=0)

    plt.errorbar(np.arange(0, len(means_X)), y=means_X, yerr=stds_X,
                 fmt='s--', capsize=10)
    plt.errorbar(np.arange(0, len(means_outliers_X)), y=means_outliers_X,
                 yerr=stds_outliers_X, fmt='x--', capsize=10, label='X outliers')
    plt.errorbar(np.arange(0, len(means_Xbulk)), y=means_Xbulk,
                 yerr=stds_Xbulk, fmt='s--', capsize=10, label='X bulk')
    plt.legend()
    plt.savefig(f'mean_{method}_{dataset}.pdf')
    plt.show() if show else plt.close()

    plt.matshow(np.array([means_X, means_outliers_X]))
    plt.colorbar()
    plt.savefig(f'means_comparison_matshow_{method}_{dataset}.pdf')
    plt.show() if show else plt.close()
