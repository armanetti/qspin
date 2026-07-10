# psyspin

**Generalized Ising / Blume–Capel / Blume–Emery–Griffiths inference and
sampling for ordinal questionnaire data.**

`psyspin` fits energy-based models with discrete spin variables
`x ∈ {-(Q-1)/2, ..., (Q-1)/2}` (Q odd) or `{-(Q-1), -(Q-1)+2, ..., Q-1}` (Q even)
to ordinal datasets — for instance Likert-scale questionnaire responses. The
package provides three model families of increasing expressive power:

| model | Hamiltonian | comment |
|------|-------------|---------|
| **Ising** | $\displaystyle H(x) = -h\cdot x - \frac{1}{2} x^\top J x$<br>$\mathrm{diag}(J)=0$ | minimal pairwise model |
| **Blume–Capel (BC)** | $\displaystyle H(x) = -h\cdot x - \frac{1}{2} x^\top J x$<br>$\mathrm{diag}(J)$ free | adds a per-site anisotropy on $x_i^2$ |
| **Blume–Emery–Griffiths (BEG)** | $\displaystyle H(x) = -h\cdot x - \frac{1}{2} x^\top J x - \frac{1}{2}\sum_{i\neq j} K_{ij}\, x_i^2 x_j^2$ | adds a biquadratic coupling, captures intensity-intensity relations |

Each model can be fit along two alternative paths, not two independent
strategies:

- **Pseudo-likelihood alone** (PL): closed-form gradient, L-BFGS-B
  optimization (`generalizedIsing_inference`, `generalizedBC_inference`,
  `generalizedBEG_inference`). Fast and easy, but only a rough estimate —
  moment matching between the fitted model and the data is not guaranteed.
- **Pseudo-likelihood + Persistent Contrastive Divergence** (PL→PCD): PL
  provides a fast warm-start, then persistent Gibbs chains refined with Adam
  gradient updates (`generalizedIsing_inferencePCD`,
  `generalizedBC_inferencePCD`, `generalizedBEG_inferencePCD`) take over.
  Computationally heavier, but it guarantees moment matching — this is the
  path used for the results in the paper.

It also includes optimized random-sequential Gibbs samplers (`mcmc_ising`,
`mcmc_beg`), null models for comparison (Gaussian, Gaussian copula,
categorical-independent, GMM spectral cleaning), and plotting helpers used to
build the figures of the accompanying paper, [*Inverse generalised spin
models of answers to questionnaires*](https://arxiv.org/abs/2605.29739)
(arXiv:2605.29739).

---

## Installation

The package targets Python ≥ 3.10 (uses structural `match`/`case`).

From PyPI:

```bash
pip install psyspin
```

For development (editable install from source):

```bash
git clone https://github.com/armanetti/psyspin.git
cd psyspin
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
from psyspin import (
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
sampler = mcmc_beg(J, h, K, Q=Q)           # same sign convention as fit()
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
| `psyspin.gauge` | `bins`, `possible_states`, `gaugefixing_data`, `gaugefixing_data_float` |
| `psyspin.inference` | `generalizedIsing_inference`, `generalizedBC_inference`, `generalizedBEG_inference` (pseudo-likelihood) |
| `psyspin.inference_pcd` | `generalizedIsing_inferencePCD`, `generalizedBC_inferencePCD`, `generalizedBEG_inferencePCD`, `Adam` |
| `psyspin.mcmc` | `gibbssampling_ising`, `gibbssampling_BC`, `gibbssampling_BC_pseudolikelihood`, `gibbssampling_beg`, `gibbssampling_beg_pseudolikelihood`, `mcmc_ising`, `mcmc_beg`, energy / magnetization helpers |
| `psyspin.sampling` | `sample_configurations`, `sample_configurations_likelearning` |
| `psyspin.jackknife` | `JKerror` — block-Jackknife integrated autocorrelation time |
| `psyspin.histograms` | `wilson_score_interval`, `agresti_coull_interval`, `bootstrap_histogram` |
| `psyspin.outliers` | `outlier_analysis_means`, `outlier_analysis_factormeans` |
| `psyspin.nullmodels` | `catind_model`, `null_gaussian_copula`, `model_gaussdisc`, `newmanmodularity`, `modularity_general`, `modularity_GMM`, `mixture_chisquared_characteristicfunction`, ... |
| `psyspin.data` | `load_data`, `discretize`, `reshuffle`, `compare2matrices`, `plot2matrices`, `remove_diagonal` |
| `psyspin.figures` | `plot_losses`, `plot_moment_matching_*`, `plot_item_histogram`, `plot_E2d_histogram_*`, `plot_mahalanobis_*`, `plot_pc_histogram_maxent`, `plot_correlation_time_analysis`, default `DEFAULT_COLORS` / `DEFAULT_CFG` |

The most commonly used names are re-exported at the package root for convenience
(e.g. `from psyspin import generalizedBEG_inferencePCD, mcmc_beg`).

---

## Sign conventions

Inference and sampling now share a single, consistent convention: the energy
is `H(x) = -h·x - ½ xᵀ J x` (and `-½ xᵀ K x²` for BEG). The `J_fit`, `h_fit`,
`K_fit` returned by `fit()` can be passed **directly, unnegated**, into
`mcmc_ising` / `mcmc_beg` — there is no sign flip to remember (see the
quickstart above). This also matches what the PCD training loop does
internally, since it samples from the persistent chains using the very same
parameters it is fitting.

---

## Reproducing the paper

Everything needed to reproduce the paper's results lives under `tutorials/`:

- `inverse_BEG-CD_all.py` runs the full pipeline for one or more datasets —
  PL fit, PL→PCD refinement, and MCMC sampling for the Ising, BC, and BEG
  models — and saves the fitted parameters and sampled configurations.
  `inverse_BEG-pselik_all.py` is the PL-only variant.
- `figures_all_datasets.ipynb` turns those saved results into the figures
  used in the paper.
- `tutorial.ipynb` is a lighter, single-dataset walkthrough for getting
  familiar with the workflow interactively.

As a rough timing reference: with `N_WORKERS=10` (parallel PCD chains /
sampling), a full run of `inverse_BEG-CD_all.py` on one dataset (~8,000
training subjects, all three models, PL→PCD fit + 1,000,000-configuration
sampling + observables) took about 21 minutes for a 12-item questionnaire
and about 37 minutes for a 22-item one in previous runs on this machine;
expect up to ~1h for larger questionnaires (30+ items), since the cost grows
with the number of items `M`.

---

## Citation

If you use `psyspin` in academic work, please cite **both** the paper and the
software:

```bibtex
@article{Armanetti2026_psyspin_paper,
  author        = {Armanetti, Arianna and Cecchetti, Luca and Sarti, Paolo and Garlaschelli, Diego and Ibañez-Berganza, Miguel},
  title         = {Inverse generalised spin models of answers to questionnaires},
  journal       = {arXiv preprint arXiv:2605.29739},
  eprint        = {2605.29739},
  archivePrefix = {arXiv},
  year          = {2026},
  url           = {https://arxiv.org/abs/2605.29739},
}

@software{Armanetti2026_psyspin_pkg,
  author    = {Ibañez-Berganza, Miguel and Armanetti, Arianna},
  title     = {{psyspin}: Generalized Ising / Blume--Capel /
                Blume--Emery--Griffiths inference for ordinal questionnaire data},
  year      = {2026},
  version   = {1.1.2},
  url       = {https://github.com/armanetti/psyspin},
  doi       = {10.5281/zenodo.XXXXXXX},
}
```

The paper is available on arXiv: [arXiv:2605.29739](https://arxiv.org/abs/2605.29739).
The software DOI above will be filled in automatically by Zenodo the first
time a GitHub release is archived — replace `XXXXXXX` once you have the real
record number, or better, point people to the "concept DOI" that always
resolves to the latest version.

---

## License & contact

Released under the [MIT License](LICENSE), with an additional **citation
request** and a **friendly note**: please drop me a line at
<arianna.armanetti@imtlucca.it> if you use this package — I would be glad to hear about your application, answer questions, or discuss collaborations.

The citation request is not legally binding under MIT, but it matters to me
that the work is credited.
