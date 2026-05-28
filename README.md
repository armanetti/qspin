# qspin

**Generalized Ising / Blume–Capel / Blume–Emery–Griffiths inference and
sampling for ordinal questionnaire data.**

`qspin` fits energy-based models with discrete spin variables
`x ∈ {-(Q-1)/2, ..., (Q-1)/2}` (Q odd) or `{-(Q-1), -(Q-1)+2, ..., Q-1}` (Q even)
to ordinal datasets — for instance Likert-scale questionnaire responses. The
package provides three model families of increasing expressive power:

| model | Hamiltonian | comment |
|------|-------------|---------|
| **Ising** | `H(x) = -h·x - ½ xᵀ J x`, `diag(J)=0` | minimal pairwise model |
| **Blume–Capel (BC)** | `H(x) = -h·x - ½ xᵀ J x`, `diag(J)` free | adds a per-site anisotropy on `x_i²` |
| **Blume–Emery–Griffiths (BEG)** | `H(x) = -h·x - ½ xᵀ J x - ½ ∑_{i≠j} K_{ij} x_i² x_j²` | adds a biquadratic coupling, captures variance correlations |

For each model the package offers two inference strategies:

- **Pseudo-likelihood** (PL): closed-form gradient, L-BFGS-B optimization
  (`generalizedIsing_inference`, `generalizedBC_inference`,
  `generalizedBEG_inference`).
- **Persistent Contrastive Divergence** (PCD): persistent Gibbs chains, Adam
  gradient updates (`generalizedIsing_inferencePCD`,
  `generalizedBC_inferencePCD`, `generalizedBEG_inferencePCD`).

It also includes optimized random-sequential Gibbs samplers (`mcmc_ising`,
`mcmc_beg`), null models for comparison (Gaussian, Gaussian copula,
categorical-independent, GMM spectral cleaning), and plotting helpers used to
build the figures of the accompanying paper.

---

## Installation

The package targets Python ≥ 3.10 (uses structural `match`/`case`).

```bash
git clone https://github.com/armanetti/qspin.git
cd qspin
pip install -e .
```

The optional dev extras add Jupyter and openpyxl (the latter is needed only by
the DASS data loader):

```bash
pip install -e .[dev]
```

### Dependencies

`numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `tqdm`,
`statsmodels`.

---

## Quickstart

```python
import numpy as np
from qspin import (
    gaugefixing_data,
    generalizedBEG_inference,
    generalizedBEG_inferencePCD,
    mcmc_beg,
    sample_configurations_likelearning,
)

# X is your raw integer-coded Likert-scale data, shape (N_subjects, M_items),
# with values in {1, ..., Q}.
Q = 5
X_gf = gaugefixing_data(X, Q)            # centre the spin alphabet to zero sum

# 1) Fast warm-start with pseudo-likelihood
pl = generalizedBEG_inference(Q=Q, l2_lambda=1e-2)
J_pl, h_pl, K_pl = pl.fit(X_gf)

# 2) Refine with persistent Contrastive Divergence
pcd = generalizedBEG_inferencePCD(Q=Q, l2_lambda=0.)
J, h, K = pcd.naif_fit_euler(
    X_gf,
    niterations=120, learning_rate=1e-4,
    ncopies=100, tau_PCD=100, tau_therm=100,
    iicc='given', J0=J_pl, h0=h_pl, K0=K_pl,
)

# 3) Sample from the inferred model
sampler = mcmc_beg(-J, -h, -K, Q=Q)        # note the sign convention
sampler.thermalize(betai=0., betaf=1., nsweeps=1000, iicc='random', nb_chunks=100)
sim = sample_configurations_likelearning(
    sampler, X=X_gf, states=pcd.states,
    N_configurations=10_000, n_intersample_sweeps=1000, reset='emp',
)
configs = sim['configurations']           # synthetic dataset, shape (10_000, M)
```

`tutorials/tutorial.ipynb` walks through the full workflow on a single
dataset, with diagnostic plots.

---

## Module map

| submodule | what it provides |
|-----------|-----------------|
| `qspin.gauge` | `bins`, `possible_states`, `gaugefixing_data`, `gaugefixing_data_float` |
| `qspin.inference` | `generalizedIsing_inference`, `generalizedBC_inference`, `generalizedBEG_inference` (pseudo-likelihood) |
| `qspin.inference_pcd` | `generalizedIsing_inferencePCD`, `generalizedBC_inferencePCD`, `generalizedBEG_inferencePCD`, `Adam` |
| `qspin.mcmc` | `gibbssampling_ising`, `gibbssampling_BC`, `gibbssampling_BC_pseudolikelihood`, `gibbssampling_beg`, `gibbssampling_beg_pseudolikelihood`, `mcmc_ising`, `mcmc_beg`, energy / magnetization helpers |
| `qspin.sampling` | `sample_configurations`, `sample_configurations_likelearning` |
| `qspin.jackknife` | `JKerror` — block-Jackknife integrated autocorrelation time |
| `qspin.histograms` | `wilson_score_interval`, `agresti_coull_interval`, `bootstrap_histogram` |
| `qspin.outliers` | `outlier_analysis_means`, `outlier_analysis_factormeans` |
| `qspin.nullmodels` | `catind_model`, `null_gaussian_copula`, `model_gaussdisc`, `newmanmodularity`, `modularity_general`, `modularity_GMM`, `mixture_chisquared_characteristicfunction`, ... |
| `qspin.data` | `load_data`, `discretize`, `reshuffle`, `compare2matrices`, `plot2matrices`, `remove_diagonal` |
| `qspin.figures` | `plot_losses`, `plot_moment_matching_*`, `plot_item_histogram`, `plot_E2d_histogram_*`, `plot_mahalanobis_*`, `plot_pc_histogram_maxent`, `plot_correlation_time_analysis`, default `DEFAULT_COLORS` / `DEFAULT_CFG` |

The most commonly used names are re-exported at the package root for convenience
(e.g. `from qspin import generalizedBEG_inferencePCD, mcmc_beg`).

---

## Sign conventions

A quick reminder, since it bites everyone the first time:

- The **inference** classes parametrize the energy with `-h·x` and `-½ xᵀ J x`
  (and `-½ xᵀ K x²` for BEG). The `J_fit`, `h_fit`, `K_fit` returned by `fit()`
  refer to that energy.
- The **MCMC** wrappers `mcmc_ising` and `mcmc_beg` take `J`, `h`, (`K`) as
  *positive-energy* coefficients: `E = ½ σᵀ J σ + h·σ`. So when you sample
  from a fitted model, pass `-J_fit`, `-h_fit`, `-K_fit` to the MCMC
  constructor (see the quickstart above).

---

## Reproducing the paper

The original repository ships the per-dataset scripts that produced every
figure in the paper. They are kept as reference under
`inverse_spinmodel/learning_cd_3rdattempt/`. The tutorial notebook is a
compact substitute for the most common workflow; it does not run every
combination of hyperparameters/resets/models that the paper explored.

---

## Citation

If you use `qspin` in academic work, please cite **both** the paper and the
software:

```bibtex
@unpublished{Armanetti2026_qspin_paper,
  author    = {Armanetti, Arianna and Cecchetti, Luca and Sarti, Paolo and Garlaschelli, Diego and Ibañez-Berganzam Miguel},
  title     = {Generalized Ising models for ordinal questionnaire data},
  note      = {In preparation},
  year      = {2026},
}

@software{Armanetti2026_qspin_pkg,
  author    = {Ibañez-Berganza, Miguel and Armanetti, Arianna},
  title     = {{qspin}: Generalized Ising / Blume--Capel /
                Blume--Emery--Griffiths inference for ordinal questionnaire data},
  year      = {2026},
  version   = {0.1.0},
  url       = {https://github.com/armanetti/qspin},
}
```

Citation details for the paper will be updated once it is published.

---

## License & contact

Released under the [MIT License](LICENSE), with an additional **citation
request** and a **friendly note**: please drop me a line at
<arianna.armanetti@imtlucca.it> if you use this package — I would be glad to hear about your application, answer questions, or discuss collaborations.

The citation request is not legally binding under MIT, but it matters to me
that the work is credited.
