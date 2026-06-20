#!/usr/bin/env python
# coding: utf-8
"""
Unified pipeline — all datasets.

process_dataset(dataset) runs the full pipeline for one dataset:
  1. load & gauge-fix data
  2. pseudo-likelihood (PL) fitting for Ising, BC, BEG
  3. PCD fitting (warm-start chained blocks) for Ising, BC, BEG
  4. MCMC sampling from the inferred models
  5. observables and distances from sampled configurations
  6. save all results as pickle files

At the bottom, joblib runs process_dataset() in parallel over all datasets.
Each worker prints timestamped progress lines prefixed with [dataset].

CONFIGURATION
-------------
The only paths you need to set are PATHDATA and SAVEPATH below.
qspin is loaded from the qspin-package subfolder of this repo; if you have
installed it via `pip install -e ./qspin-package` you can remove the sys.path block.
"""

import os
import os.path as op
import pickle
import sys
import time

import numpy as np
import qspin
from qspin.data import load_data
from qspin import (
    gaugefixing_data,
    # pseudo-likelihood inference
    generalizedIsing_inference,
    generalizedBC_inference,
    generalizedBEG_inference,
    # persistent contrastive divergence inference
    generalizedIsing_inferencePCD,
    generalizedBC_inferencePCD,
    generalizedBEG_inferencePCD,
    # MCMC wrappers
    mcmc_ising,
    mcmc_beg,
    # sampling
    sample_configurations_likelearning,
)


#######################################################
# GENERAL SETTINGS AND PARAMETERS
# -------------------------------------------------------
DATASETS_LIST = ['big5','cfcs', 'dass', 'ei',  
                 'gcbs', 'hsns', 'iri', 'mach',
                 'pwe', 'rwas', 'sd3'] # 'acme','hexaco','msscq',
#DATASETS_LIST = ['gcbs', 'rwas']  # for quick testing
NMAX = 10000

# *** PATHS TO CONFIGURE ***
PATHDATA = "/Users/ariannaarmanetti/Desktop/CODES/datasets/"
SAVEPATH = "/Users/ariannaarmanetti/Desktop/CODES/inverse-spin/learning-naif/"
# **************************

os.makedirs(SAVEPATH, exist_ok=True)

TRAIN_FRAC = 0.8
# -------------------------------------------------------
# PSEUDO-LIKELIHOOD LEARNING
L2_LAMBDA          = 1.0E-2
NITERATIONS_PSELIK = 1000
L_RATES_PSELIK     = 1.0E-7
# -------------------------------------------------------
# FULL-LIKELIHOOD LEARNING — Ising and BC share the same schedule
OPTIMIZER = 'sgd'  # 'adam' or 'sgd'
NITERATIONS_ISING = [120, 120, 120]
L_RATES_ISING     = [1.0E-3, 1.0E-3, 1.0E-4]
TAU_PCD_ISING     = 100
TAU_THERM_ISING   = 10
NCOPIES_ISING     = [100, 1000, 1000]

NITERATIONS_BEG = [20, 120, 120, 120]
L_RATES_BEG     = [1.0E-4, 1.0E-4, 1.0E-4, 1.0E-5]
TAU_PCD_BEG     = 100
TAU_THERM_BEG   = 100
NCOPIES_BEG     = [10, 100, 1000, 1000]
N_WORKERS       = 10 # number of parallel workers for PCD learning 
# (set to 1 to disable parallelism in PCD) - Note: does not support -1 
# -------------------------------------------------------
# SAMPLING
NCONFIGURATIONS = 1_000_000
# [pseudolikelihood, reset, n_intersample_sweeps]
TRIADS_OF_SAMPLING_PARAMETERS = [
    [False, 'emp', 1000],
]
BETA_IN   = 0.
BETA_F    = 1.
NSWEEPS   = 1000
IICC      = 'random'
NB_CHUNKS = 100
# -------------------------------------------------------
# PARALLELIZATION
# -1 = all available cores; -2 = all but one
N_JOBS = 1


########################################################
# VECTORISED HELPERS
# -------------------------------------------------------
def _compute_observables_vec(configs):
    """Vectorised mean and std of the three sufficient statistics.

    Returns dict_means and dict_stds for keys 'x', 'xxdag', 'x2x2dag':
        E[\\sigma]           — shape (M,)
        E[\\sigma\times sigma]     — shape (M, M)
        E[\\sigma^2 \times sigma^2]   — shape (M, M)

    Identities:
        Var(\\sigma_i \\sigma_j)     = E[\\sigma_i^2\\sigma_j^2] - E[\\sigma_i\\sigma_j]^2
        Var(\\sigma_i^2\\sigma_j^2)  = E[\\sigma_i^4\\sigma_j^4] - E[\\sigma_i^2\\sigma_j^2]^2
    """
    N = configs.shape[0]
    cs = configs ** 2

    mean_x     = configs.mean(axis=0)
    std_x      = configs.std(axis=0)

    mean_xxdag = configs.T @ configs / N
    mean_x2x2  = cs.T @ cs / N
    std_xxdag  = np.sqrt(np.maximum(mean_x2x2 - mean_xxdag**2, 0.0))

    cs2 = configs ** 4
    mean_x4x4    = cs2.T @ cs2 / N
    std_x2x2dag  = np.sqrt(np.maximum(mean_x4x4 - mean_x2x2**2, 0.0))

    dict_means = {'x': mean_x,  'xxdag': mean_xxdag,  'x2x2dag': mean_x2x2}
    dict_stds  = {'x': std_x,   'xxdag': std_xxdag,   'x2x2dag': std_x2x2dag}
    return dict_means, dict_stds


def _mahalanobis_vec(configs, C, mu):
    """Vectorised Mahalanobis: 0.5*(x-\\mu)^T C^{-1} (x-\\mu) for each row of configs."""
    inv_C = np.linalg.inv(C)
    delta = configs - mu
    return 0.5 * np.sum((delta @ inv_C) * delta, axis=1)


########################################################
def process_dataset(dataset, learning='lbfgs'):
    """
    Full pipeline for one dataset.
    All heavy loops are sequential inside this function;
    parallelism is obtained by running multiple datasets in parallel.
    """
    t0 = time.time()

    def log(msg):
        print(f'[{dataset}] {time.time() - t0:7.1f}s | {msg}', flush=True)

    log('START')

    # -------------------------------------------------------
    # OUTPUT FOLDER
    dataset_folder = op.join(SAVEPATH, dataset + '-cd', '')
    os.makedirs(dataset_folder, exist_ok=True)

    # -------------------------------------------------------
    # LOAD AND SPLIT DATA
    X = load_data(dataset, PATHDATA)
    R = len(set(X.flatten().tolist()))
    N, M = X.shape
    Neffective = min(N, NMAX)
    idx = np.arange(N)
    np.random.shuffle(idx)
    idx = idx[:Neffective]
    Ntrain = int(TRAIN_FRAC * Neffective)
    Ntest  = Neffective - Ntrain
    Xtrain = X[idx[:Ntrain], :]
    Xtest  = X[idx[Ntrain:Ntrain + Ntest], :]

    Xtrain = gaugefixing_data(Xtrain, R)
    Xtest  = gaugefixing_data(Xtest,  R)
    np.savez(dataset_folder + 'data.npz', Xtrain=Xtrain, Xtest=Xtest, dataset=dataset)
    log(f'data loaded: N_train={Ntrain}, N_test={Ntest}, M={M}, R={R}')

    # =======================================================
    # PSEUDO-LIKELIHOOD (PL) LEARNING
    # =======================================================

    log('PL | Ising ...')
    inverseising = generalizedIsing_inference(Q=R, l2_lambda=L2_LAMBDA)
    match learning:
        case 'lbfgs':
            JstarIsing, hstarIsing = inverseising.fit(Xtrain)
        case 'naif':
            JstarIsing, hstarIsing = inverseising.naif_fit_euler(
                Xtrain, niterations=NITERATIONS_PSELIK, learning_rate=L_RATES_PSELIK)
        case 'both':
            JstarIsing, hstarIsing = inverseising.fit(Xtrain)
            JstarIsing, hstarIsing = inverseising.naif_fit_euler(
                Xtrain, niterations=NITERATIONS_PSELIK, learning_rate=L_RATES_PSELIK)
        case _:
            raise ValueError(f'Invalid learning method: {learning}')

    log('PL | BC ...')
    inversebc = generalizedBC_inference(Q=R, l2_lambda=L2_LAMBDA)
    match learning:
        case 'lbfgs':
            JstarBC, hstarBC = inversebc.fit(Xtrain, iicc='meanfield')
        case 'naif':
            JstarBC, hstarBC = inversebc.naif_fit_euler(
                Xtrain, niterations=NITERATIONS_PSELIK, learning_rate=L_RATES_PSELIK)
        case 'both':
            JstarBC, hstarBC = inversebc.fit(Xtrain, iicc='meanfield')
            JstarBC, hstarBC = inversebc.naif_fit_euler(
                Xtrain, niterations=NITERATIONS_PSELIK, learning_rate=L_RATES_PSELIK)
        case _:
            raise ValueError(f'Invalid learning method: {learning}')

    log('PL | BEG ...')
    inversebeg = generalizedBEG_inference(Q=R, l2_lambda=L2_LAMBDA)
    match learning:
        case 'lbfgs':
            JstarBEG, hstarBEG, KstarBEG = inversebeg.fit(Xtrain)
        case 'naif':
            JstarBEG, hstarBEG, KstarBEG = inversebeg.naif_fit_euler(
                Xtrain, niterations=NITERATIONS_PSELIK, learning_rate=L_RATES_PSELIK)
        case 'both':
            JstarBEG, hstarBEG, KstarBEG = inversebeg.fit(Xtrain)
            JstarBEG, hstarBEG, KstarBEG = inversebeg.naif_fit_euler(
                Xtrain, niterations=NITERATIONS_PSELIK, learning_rate=L_RATES_PSELIK)
        case _:
            raise ValueError(f'Invalid learning method: {learning}')

    log('PL | done')

    # =======================================================
    # PERSISTENT CONTRASTIVE DIVERGENCE (PCD) LEARNING
    # =======================================================
    # NOTE: blocks are chained — each block warm-starts from the
    #       parameters learned by the previous block.
    # losses are concatenated across all blocks.

    # --- Ising PCD ---
    log('PCD Ising | starting')
    inverseisingPCD = generalizedIsing_inferencePCD(Q=R, l2_lambda=0.)
    J0, h0 = JstarIsing, hstarIsing          # warm start from PL for block 0
    losses_isingcd_blocks = []
    n_blocks = len(NITERATIONS_ISING)
    for i in range(n_blocks):
        log(f'PCD Ising | block {i+1}/{n_blocks}  '
            f'ncopies={NCOPIES_ISING[i]}  niter={NITERATIONS_ISING[i]}  lr={L_RATES_ISING[i]}')
        J0, h0 = inverseisingPCD.fit(
            np.copy(Xtrain),
            optimizer=OPTIMIZER,
            niterations=NITERATIONS_ISING[i],
            learning_rate=L_RATES_ISING[i],
            tau_PCD=TAU_PCD_ISING,
            tau_therm=TAU_THERM_ISING,
            ncopies=NCOPIES_ISING[i],
            iicc='given',
            J0=J0,
            h0=h0,
            verbose=False,
            n_workers=N_WORKERS)  
        losses_isingcd_blocks.append(np.copy(inverseisingPCD.losses))
    losses_isingcd = np.concatenate(losses_isingcd_blocks)
    log('PCD Ising | done')

    # --- BC PCD ---
    log('PCD BC | starting')
    inversebcPCD = generalizedBC_inferencePCD(Q=R, l2_lambda=0.)
    J0, h0 = JstarBC, hstarBC
    losses_bccd_blocks = []
    for i in range(n_blocks):
        log(f'PCD BC | block {i+1}/{n_blocks}  '
            f'ncopies={NCOPIES_ISING[i]}  niter={NITERATIONS_ISING[i]}  lr={L_RATES_ISING[i]}')
        J0, h0 = inversebcPCD.fit(
            np.copy(Xtrain),
            optimizer=OPTIMIZER,
            niterations=NITERATIONS_ISING[i],
            learning_rate=L_RATES_ISING[i],
            tau_PCD=TAU_PCD_ISING,
            tau_therm=TAU_THERM_ISING,
            ncopies=NCOPIES_ISING[i],
            iicc='given',
            J0=J0,
            h0=h0,
            verbose=False,
            n_workers=N_WORKERS)
        losses_bccd_blocks.append(np.copy(inversebcPCD.losses))
    losses_bccd = np.concatenate(losses_bccd_blocks)
    log('PCD BC | done')

    # --- BEG PCD ---
    log('PCD BEG | starting')
    inversebegPCD = generalizedBEG_inferencePCD(Q=R, l2_lambda=0.)
    J0, h0, K0 = JstarBEG, hstarBEG, KstarBEG
    losses_begcd_blocks = []
    n_blocks_beg = len(NITERATIONS_BEG)
    for i in range(n_blocks_beg):
        log(f'PCD BEG | block {i+1}/{n_blocks_beg}  '
            f'ncopies={NCOPIES_BEG[i]}  niter={NITERATIONS_BEG[i]}  lr={L_RATES_BEG[i]}')
        J0, h0, K0 = inversebegPCD.fit(
            np.copy(Xtrain),
            optimizer=OPTIMIZER,
            niterations=NITERATIONS_BEG[i],
            learning_rate=L_RATES_BEG[i],
            tau_PCD=TAU_PCD_BEG,
            tau_therm=TAU_THERM_BEG,
            ncopies=NCOPIES_BEG[i],
            iicc='given',
            J0=J0,
            h0=h0,
            K0=K0,
            verbose=False,
            n_workers=N_WORKERS)
        losses_begcd_blocks.append(np.copy(inversebegPCD.losses))
    losses_begcd = np.concatenate(losses_begcd_blocks)
    log('PCD BEG | done')

    # --- save losses and inferred models ---
    for fname, obj in [('losses_isingcd', losses_isingcd),
                       ('losses_bccd',    losses_bccd),
                       ('losses_begcd',   losses_begcd)]:
        with open(dataset_folder + fname + '.pickle', 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    for fname, obj in [('inverseising',    inverseising),
                       ('inversebc',       inversebc),
                       ('inversebeg',      inversebeg),
                       ('inverseising_cd', inverseisingPCD),
                       ('inversebc_cd',    inversebcPCD),
                       ('inversebeg_cd',   inversebegPCD)]:
        with open(dataset_folder + fname + '.pickle', 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    log('models and losses saved')

    # =======================================================
    # SAMPLING FROM THE INFERRED MODELS
    # =======================================================
    # Sign convention: qspin inference and mcmc use the same convention,
    # so J_fit and h_fit are passed directly — no negation needed.

    states = inverseisingPCD.states
    JstarIsingCD, hstarIsingCD             = inverseisingPCD.J_fit, inverseisingPCD.h_fit
    JstarBCCD,    hstarBCCD                = inversebcPCD.J_fit,    inversebcPCD.h_fit
    JstarBEGCD,   hstarBEGCD, KstarBEGCD   = inversebegPCD.J_fit,   inversebegPCD.h_fit, inversebegPCD.K_fit

    # thermalize all three chains
    log('sampling | thermalizing Ising ...')
    mcmc_ising_pcd = mcmc_ising(JstarIsingCD, hstarIsingCD, R)
    mcmc_ising_pcd.thermalize(betai=BETA_IN, betaf=BETA_F, nsweeps=NSWEEPS,
                               iicc=IICC, nb_chunks=NB_CHUNKS, verbose=True)

    log('sampling | thermalizing BC ...')
    mcmc_bc_pcd = mcmc_ising(JstarBCCD, hstarBCCD, R, anisotropy=True)
    mcmc_bc_pcd.thermalize(betai=BETA_IN, betaf=BETA_F, nsweeps=NSWEEPS,
                            iicc=IICC, nb_chunks=NB_CHUNKS, verbose=True)

    log('sampling | thermalizing BEG ...')
    mcmc_beg_pcd = mcmc_beg(JstarBEGCD, hstarBEGCD, KstarBEGCD, R)
    mcmc_beg_pcd.thermalize(betai=BETA_IN, betaf=BETA_F, nsweeps=NSWEEPS,
                             iicc=IICC, nb_chunks=NB_CHUNKS, verbose=True)

    def _sample(mcmc_instance, label):
        log(f'sampling | {label}  (N={NCONFIGURATIONS}, tau={TRIADS_OF_SAMPLING_PARAMETERS[0][2]}) ...')
        results = []
        for pseudolikelihood, reset, n_intersample_sweeps in TRIADS_OF_SAMPLING_PARAMETERS:
            sim = sample_configurations_likelearning(
                mcmc_instance,
                X=Xtrain,
                states=states,
                N_configurations=NCONFIGURATIONS,
                n_intersample_sweeps=n_intersample_sweeps,
                reset=reset,
                pseudolikelihood=pseudolikelihood,
                verbose=True,
                n_jobs=N_WORKERS)
            results.append(sim)
        log(f'sampling | {label} done')
        return results

    sim_dict_list_ising_cd = _sample(mcmc_ising_pcd, 'Ising')
    sim_dict_list_bc_cd    = _sample(mcmc_bc_pcd,    'BC')
    sim_dict_list_beg_cd   = _sample(mcmc_beg_pcd,   'BEG')

    # =======================================================
    # OBSERVABLES AND DISTANCES
    # =======================================================

    log('observables | starting ...')

    # data covariance (used for common-covariance Mahalanobis distances)
    CovX  = np.cov(Xtrain.T)
    meanX = np.mean(Xtrain, axis=0)

    observables_names = ['xxdag', 'x', 'x2x2dag']

    for label, sim_dict_list in [('Ising', sim_dict_list_ising_cd),
                                  ('BC',    sim_dict_list_bc_cd),
                                  ('BEG',   sim_dict_list_beg_cd)]:
        for el in sim_dict_list:
            configs = el['configurations']

            # sufficient statistics — vectorised
            dict_means, dict_stds = _compute_observables_vec(configs)
            el['dict_means']        = dict_means
            el['dict_stds']         = dict_stds
            el['observables_names'] = observables_names

            # Euclidean distances from the sample mean
            el['distances_maxent'] = np.sum(
                (configs - configs.mean(axis=0))**2, axis=1)

            # Mahalanobis distances:
            #   _commonC  uses data covariance (common reference across models)
            #   (no suffix) uses the model's own sample covariance
            CovX_model  = np.cov(configs.T)
            meanX_model = configs.mean(axis=0)

            el['energies_maxent_commonC'] = _mahalanobis_vec(configs, CovX,       meanX)
            el['energies_maxent']         = _mahalanobis_vec(configs, CovX_model, meanX_model)

        log(f'observables | {label} done')

    # --- save simulation results ---
    for fname, obj in [('sim_dict_list_ising_cd', sim_dict_list_ising_cd),
                       ('sim_dict_list_bc_cd',    sim_dict_list_bc_cd),
                       ('sim_dict_list_beg_cd',   sim_dict_list_beg_cd)]:
        with open(dataset_folder + fname + '.pickle', 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - t0
    log(f'DONE  total={elapsed/3600:.2f}h ({elapsed:.0f}s)')
    return dataset


########################################################
if __name__ == '__main__':

    from joblib import Parallel, delayed
    from tqdm import tqdm

    class ProgressParallel(Parallel):
        """joblib.Parallel with a tqdm job-completion bar."""
        def __init__(self, total=None, desc=None, **kwargs):
            super().__init__(**kwargs)
            self._total = total
            self._desc  = desc

        def __call__(self, *args, **kwargs):
            with tqdm(total=self._total, desc=self._desc,
                      unit='dataset', dynamic_ncols=True) as self._pbar:
                return super().__call__(*args, **kwargs)

        def print_progress(self):
            self._pbar.n = self.n_completed_tasks
            self._pbar.refresh()

    n_datasets = len(DATASETS_LIST)

    if n_datasets == 1:
        # Single dataset: run directly (avoids joblib stdout buffering
        # that breaks tqdm in-place updates)
        completed = [process_dataset(DATASETS_LIST[0])]
    else:
        print(f'\nRunning {n_datasets} datasets in parallel  (n_jobs={N_JOBS})')
        print('Per-job progress is printed as  [dataset]  elapsed | step\n')
        completed = ProgressParallel(
            total=n_datasets,
            desc='datasets done',
            n_jobs=N_JOBS,
            backend='loky',
            verbose=0,
        )(delayed(process_dataset)(d) for d in DATASETS_LIST)

    print(f'\nAll done: {completed}')
