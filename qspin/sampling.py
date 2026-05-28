"""
Sampling strategies for drawing configurations from a thermalized MCMC instance.

Both helpers take a thermalized :class:`isingq.mcmc.mcmc_ising` or
:class:`isingq.mcmc.mcmc_beg` and produce a dictionary containing the sampled
configurations, the energy trace, and the sampling parameters.

Two reset strategies are supported between consecutive samples:
    * ``reset='None'`` keeps the chain running;
    * ``reset='emp'``  resets to a random empirical configuration;
    * ``reset='rand'`` resets to a random configuration drawn uniformly.
"""

import numpy as np


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
                                       verbose=False):
    """Draw configurations the same way the PCD learner does.

    Differs from :func:`sample_configurations` in that ``n_intersample_sweeps``
    consecutive configurations are registered before resetting, so the chain
    is restarted every ``N_configurations / n_intersample_sweeps`` samples.
    """
    N, M = X.shape
    configurations = np.zeros((N_configurations, M))
    energies_nullmodel = []

    if verbose:
        print('sampling with N_configurations = %d, pseudolikelihood = %s, reset = %s, n_intersample_sweeps = %d' %
              (N_configurations, str(pseudolikelihood), reset, n_intersample_sweeps))

    n_copies = int(N_configurations / n_intersample_sweeps)
    for n in range(n_copies):
        if verbose and n % max(1, (n_copies // 10)) == 0:
            print(str(n) + ', ', end='')
        dict_obs, energies = model_mcmc_instance.sample(
            beta=1, nsweeps_sampling=n_intersample_sweeps,
            observables_list=[mymean], observables_names=['x'],
            pseudolikelihood=pseudolikelihood)
        configurations[n * n_intersample_sweeps:(n + 1) * n_intersample_sweeps] = dict_obs['x']
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
