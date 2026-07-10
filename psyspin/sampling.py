"""
Sampling strategies for drawing configurations from a thermalized MCMC instance.

Both helpers take a thermalized :class:`psyspin.mcmc.mcmc_ising` or
:class:`psyspin.mcmc.mcmc_beg` and produce a dictionary containing the sampled
configurations, the energy trace, and the sampling parameters.

Two reset strategies are supported between consecutive samples:
    * ``reset='None'`` keeps the chain running;
    * ``reset='emp'``  resets to a random empirical configuration;
    * ``reset='rand'`` resets to a random configuration drawn uniformly.
"""

import copy

import numpy as np
from tqdm import tqdm


def sample_configurations(model_mcmc_instance, X, states,
                          N_configurations=1000,
                          n_intersample_sweeps=20,
                          reset='None',
                          pseudolikelihood=False,
                          verbose=False):
    """Draw ``N_configurations`` independent samples (one per chain restart).

    Parameters
    ----------
    model_mcmc_instance : mcmc_ising or mcmc_beg
        Already thermalized MCMC instance.
    X : ndarray, shape (N, M)
        Empirical dataset (only used when ``reset='emp'``).
    states : ndarray
        Gauge-fixed spin states (only used when ``reset='rand'``).
    N_configurations : int
        Number of samples to draw.
    n_intersample_sweeps : int
        Number of Gibbs sweeps between two samples.
    reset : {'None', 'emp', 'rand'}
        Chain-reset strategy between samples.
    pseudolikelihood : bool
        Use the pseudo-likelihood sampler instead of the standard Gibbs sampler.

    Returns
    -------
    dict
        With keys ``configurations``, ``energies``, ``n_intersample_sweeps``,
        ``reset``, ``pseudolikelihood``.
    """
    N, M = X.shape
    configurations = np.zeros((N_configurations, M))
    energies_nullmodel = []

    if verbose:
        print('sampling with N_configurations = %d, pseudolikelihood = %s, reset = %s, n_intersample_sweeps = %d' %
              (N_configurations, str(pseudolikelihood), reset, n_intersample_sweeps))

    for n in range(N_configurations):
        if verbose and n % max(1, (N_configurations // 10)) == 0:
            print(str(n) + ', ', end='')
        _, energies = model_mcmc_instance.sample(
            beta=1, nsweeps_sampling=n_intersample_sweeps,
            observables_list=[], pseudolikelihood=pseudolikelihood)
        configurations[n] = model_mcmc_instance.sigmas
        energies_nullmodel = np.concatenate([energies_nullmodel, energies])

        if reset == 'emp':
            i = np.random.choice(range(N))
            model_mcmc_instance.sigmas = np.copy(X[i])
        if reset == 'rand':
            model_mcmc_instance.sigmas = np.random.choice(states, size=(M))

    if verbose:
        print()

    return {
        'configurations': configurations,
        'energies': energies_nullmodel,
        'n_intersample_sweeps': n_intersample_sweeps,
        'reset': reset,
        'pseudolikelihood': pseudolikelihood,
    }


def sample_configurations_likelearning(model_mcmc_instance, X, states,
                                       N_configurations=1000,
                                       n_intersample_sweeps=20,
                                       reset='None',
                                       mymean=lambda sigmas: sigmas.copy(),
                                       pseudolikelihood=False,
                                       verbose=False,
                                       n_jobs=1):
    """Draw configurations the same way the PCD learner does.

    Differs from :func:`sample_configurations` in that ``n_intersample_sweeps``
    consecutive configurations are registered before resetting, so the chain
    is restarted every ``N_configurations / n_intersample_sweeps`` samples.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers (passed to ``joblib.Parallel``).
        ``n_jobs=1`` (default) runs the original sequential loop unchanged.
        ``n_jobs=-1`` uses all available cores.

        .. note::
            With ``n_jobs > 1`` each of the ``n_copies`` iterations is run as
            an **independent** chain starting from ``model_mcmc_instance.sigmas``,
            regardless of the ``reset`` argument.  For ``reset='None'`` this
            differs from the sequential behaviour (a single long chain): the
            parallel version produces statistically equivalent samples for
            ergodic systems but without inter-copy correlations.
    """
    N, M = X.shape
    n_copies = int(N_configurations / n_intersample_sweeps)

    if verbose:
        print('sampling with N_configurations = %d, pseudolikelihood = %s, reset = %s, '
              'n_intersample_sweeps = %d, n_jobs = %d' %
              (N_configurations, str(pseudolikelihood), reset, n_intersample_sweeps, n_jobs))

    # ------------------------------------------------------------------ #
    # Sequential path                                                      #
    # ------------------------------------------------------------------ #
    if n_jobs == 1:
        configurations = np.zeros((N_configurations, M))
        energies_nullmodel = []

        for n in tqdm(range(n_copies), desc='sampling', unit='copy'):
            dict_obs, energies = model_mcmc_instance.sample(
                beta=1, nsweeps_sampling=n_intersample_sweeps,
                observables_list=[mymean], observables_names=['x'],
                pseudolikelihood=pseudolikelihood, verbose=False)
            configurations[n * n_intersample_sweeps:(n + 1) * n_intersample_sweeps] = dict_obs['x']
            energies_nullmodel = np.concatenate([energies_nullmodel, energies])

            if reset == 'emp':
                i = np.random.choice(range(N))
                model_mcmc_instance.sigmas = np.copy(X[i])
            if reset == 'rand':
                model_mcmc_instance.sigmas = np.random.choice(states, size=(M))

        return {
            'configurations': configurations,
            'energies': energies_nullmodel,
            'n_intersample_sweeps': n_intersample_sweeps,
            'reset': reset,
            'pseudolikelihood': pseudolikelihood,
        }

    # ------------------------------------------------------------------ #
    # Parallel path                                                        #
    # ------------------------------------------------------------------ #
    from joblib import Parallel, delayed  # noqa: PLC0415

    # One independent integer seed per copy; derived from a SeedSequence so
    # they are guaranteed to be statistically independent even on fork-based
    # systems where child processes would otherwise share the parent RNG state.
    int_seeds = [
        int(s.generate_state(1)[0])
        for s in np.random.SeedSequence().spawn(n_copies)
    ]

    def _run_copy(n, seed):
        np.random.seed(seed)
        inst = copy.deepcopy(model_mcmc_instance)
        if reset == 'emp':
            inst.sigmas = np.copy(X[np.random.randint(N)])
        elif reset == 'rand':
            inst.sigmas = np.random.choice(states, size=(M,))
        dict_obs, energies = inst.sample(
            beta=1, nsweeps_sampling=n_intersample_sweeps,
            observables_list=[mymean], observables_names=['x'],
            pseudolikelihood=pseudolikelihood, verbose=False)
        return dict_obs['x'], energies

    # return_as='generator' lets tqdm update as each job completes
    results = list(tqdm(
        Parallel(n_jobs=n_jobs, return_as='generator')(
            delayed(_run_copy)(n, int_seeds[n]) for n in range(n_copies)
        ),
        total=n_copies, desc='sampling', unit='copy',
    ))

    configs_list, energies_list = zip(*results)
    configurations = np.concatenate(configs_list, axis=0)
    energies_nullmodel = np.concatenate(energies_list)

    return {
        'configurations': configurations,
        'energies': energies_nullmodel,
        'n_intersample_sweeps': n_intersample_sweeps,
        'reset': reset,
        'pseudolikelihood': pseudolikelihood,
    }
