"""
isingq — Generalized Ising / Blume--Capel / Blume--Emery--Griffiths models
for ordinal questionnaire data.

Quick reference
---------------
The package is organized in the following submodules:

* :mod:`isingq.gauge`           gauge-fixing utilities (``bins``, ``possible_states``,
                                ``gaugefixing_data``).
* :mod:`isingq.inference`       pseudo-likelihood classes for Ising / BC / BEG.
* :mod:`isingq.inference_pcd`   persistent Contrastive-Divergence classes
                                + Adam optimizer.
* :mod:`isingq.mcmc`            Gibbs samplers and MCMC wrapper classes
                                (``mcmc_ising``, ``mcmc_beg``) + observables.
* :mod:`isingq.sampling`        strategies to draw configurations from a
                                thermalized chain.
* :mod:`isingq.jackknife`       block-Jackknife integrated autocorrelation time.
* :mod:`isingq.histograms`      Wilson / Agresti-Coull / bootstrap CIs on histograms.
* :mod:`isingq.outliers`        outlier-subset analysis plots.
* :mod:`isingq.nullmodels`      categorical-independent, Gaussian, copula, and
                                spectral null models.
* :mod:`isingq.data`            questionnaire data loaders and simple data
                                utilities (``load_data``, ``discretize``,
                                ``reshuffle``).
* :mod:`isingq.figures`         plotting helpers for the figures of the paper.

The most common entry points are re-exported at the top level for convenience.
"""

from . import gauge
from . import inference
from . import inference_pcd
from . import mcmc
from . import sampling
from . import jackknife
from . import histograms
from . import outliers
from . import nullmodels
from . import data
from . import figures

# top-level convenience re-exports
from .gauge import (
    bins,
    possible_states,
    gaugefixing_data,
    gaugefixing_data_float,
)
from .inference import (
    generalizedIsing_inference,
    generalizedBC_inference,
    generalizedBEG_inference,
)
from .inference_pcd import (
    Adam,
    generalizedIsing_inferencePCD,
    generalizedBC_inferencePCD,
    generalizedBEG_inferencePCD,
)
from .mcmc import (
    gibbssampling_ising,
    gibbssampling_BC,
    gibbssampling_BC_pseudolikelihood,
    gibbssampling_beg,
    gibbssampling_beg_pseudolikelihood,
    mcmc_ising,
    mcmc_beg,
    energy_ising,
    energy_beg,
    magnetization,
    color_fraction,
)
from .sampling import (
    sample_configurations,
    sample_configurations_likelearning,
)
from .jackknife import JKerror
from .figures import (
    condition,
    model_label,
    plot_losses,
    plot_moment_matching_learning,
    plot_moment_matching_sampling,
    plot_item_histogram,
    plot_E2d_histogram_maxent,
    plot_E2d_histogram_simple,
    plot_E2d_histogram_beg_gauss_catind,
    plot_E2d_histogram_begvscopula,
    plot_mahalanobis_commonC_maxent,
    plot_mahalanobis_modelcov,
    plot_mahalanobis_commonC_simple,
    plot_mahalanobis_commonC_beg_gauss_catind,
    plot_mahalanobis_commonC_begvscopula,
    plot_pc_histogram_maxent,
    plot_pc_histogram_simple,
    plot_pc_histogram_beg_gauss_catind,
    plot_factor_histogram_beg_gauss_catind,
    plot_factor_histogram_maxent,
    plot_correlation_time_analysis,
)

__version__ = "1.1.1"

__all__ = [
    # submodules
    "gauge", "inference", "inference_pcd", "mcmc", "sampling",
    "jackknife", "histograms", "outliers", "nullmodels", "data", "figures",
    # gauge
    "bins", "possible_states", "gaugefixing_data", "gaugefixing_data_float",
    # inference (pseudo-likelihood)
    "generalizedIsing_inference", "generalizedBC_inference", "generalizedBEG_inference",
    # inference (PCD)
    "Adam",
    "generalizedIsing_inferencePCD", "generalizedBC_inferencePCD", "generalizedBEG_inferencePCD",
    # MCMC
    "gibbssampling_ising", "gibbssampling_BC", "gibbssampling_BC_pseudolikelihood",
    "gibbssampling_beg", "gibbssampling_beg_pseudolikelihood",
    "mcmc_ising", "mcmc_beg",
    "energy_ising", "energy_beg", "magnetization", "color_fraction",
    # sampling
    "sample_configurations", "sample_configurations_likelearning",
    # jackknife
    "JKerror",
    # figures
    "condition", "model_label",
    "plot_losses", "plot_moment_matching_learning", "plot_moment_matching_sampling",
    "plot_item_histogram",
    "plot_E2d_histogram_maxent", "plot_E2d_histogram_simple",
    "plot_E2d_histogram_beg_gauss_catind", "plot_E2d_histogram_begvscopula",
    "plot_mahalanobis_commonC_maxent", "plot_mahalanobis_modelcov",
    "plot_mahalanobis_commonC_simple", "plot_mahalanobis_commonC_beg_gauss_catind",
    "plot_mahalanobis_commonC_begvscopula",
    "plot_pc_histogram_maxent", "plot_pc_histogram_simple",
    "plot_pc_histogram_beg_gauss_catind",
    "plot_factor_histogram_beg_gauss_catind",
    "plot_factor_histogram_maxent", "plot_correlation_time_analysis",
    # version
    "__version__",
]
