"""
Gibbs MCMC samplers for generalized Ising models.

Two flavours are provided for each model:

* a *standard* sampler that updates one spin at a time using the conditional
  Boltzmann distribution;
* a *pseudo-likelihood* sampler that targets the pseudo-likelihood density
  via a Metropolis acceptance step.

Cached effective-field updates make each spin flip O(N) (instead of O(N^2) for
a naive implementation).  The samplers are wrapped by two convenience classes,
:class:`mcmc_ising` (which also handles Blume--Capel when ``anisotropy=True``)
and :class:`mcmc_beg`, both of which expose a ``thermalize`` /
``sample`` / ``expected_values`` API.
"""

import numpy as np
from tqdm import tqdm

from .gauge import possible_states


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _setup_spin_values(Q):
    """Return (gauge-fixed spin values, mean of {1..Q}) for Gibbs updates.

    Delegates the spin-value computation to :func:`gauge.possible_states` so
    that the convention is defined in a single place.  ``mv`` (the mean of the
    raw alphabet {1, ..., Q}) is needed for the O(1) spin-to-index mapping
    ``index = round(sigma + mv - 1)``.
    """
    sv = possible_states(Q)
    mv = (Q + 1) / 2.0   # mean of {1, ..., Q}
    return sv, mv


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

def color_fraction(sigmas, Q):
    """Fraction of sites in each of the Q colours."""
    N_test = len(sigmas)
    return np.array([len(np.where(sigmas == q)[0]) / N_test for q in range(Q)])


def magnetization_potts(sigmas, Q):
    """Potts magnetization derived from :func:`color_fraction`."""
    return np.mean(np.abs(color_fraction(sigmas, Q) - Q ** -1))


def magnetization(sigmas, Q):
    """Per-site mean of the spin configuration."""
    return np.mean(sigmas, axis=0)


def effective_nb_colors(sigmas, Q):
    """Effective number of colours: Q (1 - Q m / 2)."""
    return Q * (1. - Q * magnetization(sigmas, Q) / 2.)


def effnbcol(mag, Q):
    """Effective number of colours, taking a precomputed magnetization."""
    return Q * (1 - Q * mag / 2.)


def energy_ising(sigmas, J, h):
    r"""Ising energy E(sigma) = 1/2 sigma^T J sigma + h^T sigma."""
    return 0.5 * ((sigmas.T @ J) @ sigmas) + h.T @ sigmas


def energy_beg(sigmas, J, h, K):
    r"""BEG energy E(sigma) = 1/2 sigma^T J sigma + h^T sigma + 1/2 (sigma^2)^T K (sigma^2)."""
    return 0.5 * ((sigmas.T @ J) @ sigmas) + h.T @ sigmas \
        + 0.5 * (((sigmas ** 2).T @ K) @ (sigmas ** 2))


# ---------------------------------------------------------------------------
# Sampling functions
# ---------------------------------------------------------------------------

def gibbssampling_ising(Q, sigmas, J, h, beta, nsweeps,
                        energies_test=False, observables_list=[],
                        observables_names=[], verbose=False):
    """Random-sequential Gibbs sampler for the generalized Ising model (diag(J)=0).

    Optimizations vs a naive loop-based implementation (same Markov chain,
    same stationary distribution):
      - cached effective field  phi = J @ sigma + h  (updated O(N) per spin flip);
      - random site indices / uniform samples pre-generated once per sweep;
      - CDF inversion replaces np.random.choice(range(Q), p=probs).
    """
    sv, mv = _setup_spin_values(Q)
    N = len(sigmas)
    dict_observables: dict = {nome: [] for nome in observables_names}

    phi = J @ sigmas + h
    tot_energy = energy_ising(sigmas, J, h)
    tot_energies = np.zeros(nsweeps)
    energies_test_array = np.zeros(nsweeps)

    sweep_iter = tqdm(range(nsweeps), desc='Gibbs sweeps', leave=False) if verbose else range(nsweeps)
    for sweep in sweep_iter:
        i_seq = np.random.randint(0, N, size=N)
        u_seq = np.random.uniform(size=N)
        for t in range(N):
            i = i_seq[t]
            energy_site = sv * phi[i]
            log_p = beta * energy_site
            log_p -= log_p.max()
            prob = np.exp(log_p); prob /= prob.sum()
            new_index = min(int(np.searchsorted(np.cumsum(prob), u_seq[t])), Q - 1)
            new_spin = sv[new_index]

            # convert current spin to index: sv[k] = k - (Q-1)/2, so k = sv[k] + mv - 1
            index = int(round(sigmas[i] + mv - 1))
            tot_energy += energy_site[new_index] - energy_site[index]

            delta = new_spin - sigmas[i]
            if delta:
                phi += J[:, i] * delta
            sigmas[i] = new_spin

        tot_energies[sweep] = tot_energy
        if energies_test:
            energies_test_array[sweep] = energy_ising(sigmas, J, h)
        for obs, name in zip(observables_list, observables_names):
            dict_observables[name].append(obs(sigmas[:]))

    for obs, name in zip(observables_list, observables_names):
        dict_observables[name] = np.array(dict_observables[name])

    if energies_test:
        return sigmas, tot_energies, dict_observables, energies_test_array
    return sigmas, tot_energies, dict_observables


def gibbssampling_BC(Q, sigmas, J, h, beta, nsweeps,
                     energies_test=False, observables_list=[],
                     observables_names=[], verbose=False):
    """Random-sequential Gibbs sampler for the Blume--Capel model.

    Conditional energy at site i:
        E_i(q) = q * (phi_i - J_{ii} sigma_i) + (1/2) q^2 J_{ii}.
    Setting ``diag(J)=0`` recovers the plain Ising sampler.
    """
    sv, mv = _setup_spin_values(Q)
    sv2 = sv ** 2
    N = len(sigmas)
    dict_observables: dict = {nome: [] for nome in observables_names}

    diag_J = np.diag(J).copy()
    phi = J @ sigmas + h
    tot_energy = energy_ising(sigmas, J, h)
    tot_energies = np.zeros(nsweeps)
    energies_test_array = np.zeros(nsweeps)

    sweep_iter = tqdm(range(nsweeps), desc='Gibbs sweeps', leave=False) if verbose else range(nsweeps)
    for sweep in sweep_iter:
        i_seq = np.random.randint(0, N, size=N)
        u_seq = np.random.uniform(size=N)
        for t in range(N):
            i = i_seq[t]
            eff = phi[i] - diag_J[i] * sigmas[i]
            energy_site = sv * eff + 0.5 * sv2 * diag_J[i]
            log_p = beta * energy_site
            log_p -= log_p.max()
            prob = np.exp(log_p); prob /= prob.sum()
            new_index = min(int(np.searchsorted(np.cumsum(prob), u_seq[t])), Q - 1)
            new_spin = sv[new_index]

            index = int(round(sigmas[i] + mv - 1))
            tot_energy += energy_site[new_index] - energy_site[index]
            delta = new_spin - sigmas[i]
            if delta:
                phi += J[:, i] * delta
            sigmas[i] = new_spin

        tot_energies[sweep] = tot_energy
        if energies_test:
            energies_test_array[sweep] = energy_ising(sigmas, J, h)
        for obs, name in zip(observables_list, observables_names):
            dict_observables[name].append(obs(sigmas[:]))

    for obs, name in zip(observables_list, observables_names):
        dict_observables[name] = np.array(dict_observables[name])

    if energies_test:
        return sigmas, tot_energies, dict_observables, energies_test_array
    return sigmas, tot_energies, dict_observables


def gibbssampling_BC_pseudolikelihood(Q, sigmas, J, h, beta, nsweeps,
                                      energies_test=False, observables_list=[],
                                      observables_names=[], verbose=False):
    """Pseudo-likelihood Metropolis sampler for the Blume--Capel model.

    Acceptance ratio includes the change in the sum of local log-partition
    functions, so the stationary distribution is the pseudo-likelihood density.
    """
    sv, mv = _setup_spin_values(Q)

    dict_observables: dict = {nome: [] for nome in observables_names}
    N = len(sigmas)

    tot_energy = energy_ising(sigmas, J, h)
    tot_energies = np.zeros(nsweeps)
    energies_test_array = np.zeros(nsweeps)

    diag_J = np.diag(J)
    phi = J @ sigmas + h
    eff_all = phi - diag_J * sigmas

    sv2 = sv ** 2
    energy_all = sv[:, np.newaxis] * eff_all[np.newaxis, :] + \
                 0.5 * sv2[:, np.newaxis] * diag_J[np.newaxis, :]

    def _logsumexp_cols(E):
        m = E.max(axis=0)
        return np.log(np.sum(np.exp(E - m), axis=0)) + m

    sweep_iter = tqdm(range(nsweeps), desc='Gibbs sweeps', leave=False) if verbose else range(nsweeps)
    for sweep in sweep_iter:
        for t in range(N):
            i = np.random.randint(N)
            new_index = np.random.randint(Q)
            new_spin = sv[new_index]

            index = int(round(sigmas[i] + mv - 1))
            delta_energy_metropolis = energy_all[new_index, i] - energy_all[index, i]

            delta = new_spin - sigmas[i]
            if delta == 0:
                continue

            d_eff = J[:, i] * delta
            d_eff[i] = 0.0

            energy_attempt = energy_all + sv[:, np.newaxis] * d_eff[np.newaxis, :]

            log_Z_current = _logsumexp_cols(beta * energy_all)
            log_Z_attempt = _logsumexp_cols(beta * energy_attempt)
            deltaZ = np.sum(log_Z_attempt - log_Z_current)

            attempt_prob = np.exp(beta * delta_energy_metropolis - deltaZ)

            xi = np.random.random()
            if xi < attempt_prob:
                tot_energy += energy_all[new_index, i] - energy_all[index, i]
                sigmas[i] = new_spin
                phi += J[:, i] * delta
                eff_all += d_eff
                energy_all = energy_attempt

        tot_energies[sweep] = tot_energy
        if energies_test:
            energies_test_array[sweep] = energy_ising(sigmas, J, h)
        for obs, name in zip(observables_list, observables_names):
            dict_observables[name].append(obs(sigmas[:]))

    for obs, name in zip(observables_list, observables_names):
        dict_observables[name] = np.array(dict_observables[name])

    if energies_test:
        return sigmas, tot_energies, dict_observables, energies_test_array
    return sigmas, tot_energies, dict_observables


def gibbssampling_beg(Q, sigmas, J, h, K, beta, nsweeps,
                      energies_test=False, observables_list=[],
                      observables_names=[], verbose=False):
    """Random-sequential Gibbs sampler for the Blume--Emery--Griffiths model.

    Two cached fields are maintained:
        phi = J @ sigma + h        (updates when sigma_i changes)
        psi = K @ sigma^2          (updates when sigma_i^2 changes)
    Conditional energy at site i:
        E_i(q) = q (phi_i - J_{ii} sigma_i) + (1/2) q^2 J_{ii} + q^2 psi_i.
    K has zero diagonal, so psi has no self term.
    """
    sv, mv = _setup_spin_values(Q)
    sv2 = sv ** 2
    N = len(sigmas)
    dict_observables: dict = {nome: [] for nome in observables_names}

    diag_J = np.diag(J).copy()
    phi = J @ sigmas + h
    psi = K @ (sigmas ** 2).astype(float)
    tot_energy = energy_beg(sigmas, J, h, K)
    tot_energies = np.zeros(nsweeps)
    energies_test_array = np.zeros(nsweeps)

    sweep_iter = tqdm(range(nsweeps), desc='Gibbs sweeps', leave=False) if verbose else range(nsweeps)
    for sweep in sweep_iter:
        i_seq = np.random.randint(0, N, size=N)
        u_seq = np.random.uniform(size=N)
        for t in range(N):
            i = i_seq[t]
            eff = phi[i] - diag_J[i] * sigmas[i]
            energy_site = sv * eff + 0.5 * sv2 * diag_J[i] + sv2 * psi[i]
            log_p = beta * energy_site
            log_p -= log_p.max()
            prob = np.exp(log_p); prob /= prob.sum()
            new_index = min(int(np.searchsorted(np.cumsum(prob), u_seq[t])), Q - 1)
            new_spin = sv[new_index]

            index = int(round(sigmas[i] + mv - 1))
            tot_energy += energy_site[new_index] - energy_site[index]

            d_sigma = new_spin - sigmas[i]
            d_sigma2 = new_spin ** 2 - sigmas[i] ** 2
            if d_sigma:
                phi += J[:, i] * d_sigma
            if d_sigma2:
                psi += K[:, i] * d_sigma2
            sigmas[i] = new_spin

        tot_energies[sweep] = tot_energy
        if energies_test:
            energies_test_array[sweep] = energy_beg(sigmas, J, h, K)
        for obs, name in zip(observables_list, observables_names):
            dict_observables[name].append(obs(sigmas[:]))

    for obs, name in zip(observables_list, observables_names):
        dict_observables[name] = np.array(dict_observables[name])

    if energies_test:
        return sigmas, tot_energies, dict_observables, energies_test_array
    return sigmas, tot_energies, dict_observables


def gibbssampling_beg_pseudolikelihood(Q, sigmas, J, h, K, beta, nsweeps,
                                       energies_test=False, observables_list=[],
                                       observables_names=[], verbose=False):
    """Pseudo-likelihood Metropolis sampler for the BEG model."""
    sv, mv = _setup_spin_values(Q)

    dict_observables: dict = {nome: [] for nome in observables_names}
    N = len(sigmas)

    tot_energy = energy_beg(sigmas, J, h, K)
    tot_energies = np.zeros(nsweeps)
    energies_test_array = np.zeros(nsweeps)

    diag_J = np.diag(J)
    phi = J @ sigmas + h
    psi = K @ (sigmas ** 2).astype(float)
    eff_all = phi - diag_J * sigmas

    sv2 = sv ** 2
    energy_all = sv[:, np.newaxis] * eff_all[np.newaxis, :] + \
                 0.5 * sv2[:, np.newaxis] * diag_J[np.newaxis, :] + \
                 sv2[:, np.newaxis] * psi[np.newaxis, :]

    def _logsumexp_cols(E):
        m = E.max(axis=0)
        return np.log(np.sum(np.exp(E - m), axis=0)) + m

    sweep_iter = tqdm(range(nsweeps), desc='Gibbs sweeps', leave=False) if verbose else range(nsweeps)
    for sweep in sweep_iter:
        for t in range(N):
            i = np.random.randint(N)
            new_index = np.random.randint(Q)
            new_spin = sv[new_index]

            index = int(round(sigmas[i] + mv - 1))
            delta_energy_metropolis = energy_all[new_index, i] - energy_all[index, i]

            delta = new_spin - sigmas[i]
            if delta == 0:
                continue

            delta2 = new_spin ** 2 - sigmas[i] ** 2

            d_eff = J[:, i] * delta
            d_eff[i] = 0.0
            d_psi = K[:, i] * delta2

            energy_attempt = energy_all \
                + sv[:, np.newaxis] * d_eff[np.newaxis, :] \
                + sv2[:, np.newaxis] * d_psi[np.newaxis, :]

            log_Z_current = _logsumexp_cols(beta * energy_all)
            log_Z_attempt = _logsumexp_cols(beta * energy_attempt)
            deltaZ = np.sum(log_Z_attempt - log_Z_current)

            attempt_prob = np.exp(beta * delta_energy_metropolis - deltaZ)

            xi = np.random.random()
            if xi < attempt_prob:
                tot_energy += energy_all[new_index, i] - energy_all[index, i]
                sigmas[i] = new_spin
                phi += J[:, i] * delta
                psi += K[:, i] * delta2
                eff_all += d_eff
                energy_all = energy_attempt

        tot_energies[sweep] = tot_energy
        if energies_test is not False:
            energies_test_array[sweep] = energy_beg(sigmas, J, h, K)
        for obs, name in zip(observables_list, observables_names):
            dict_observables[name].append(obs(sigmas[:]))

    for obs, name in zip(observables_list, observables_names):
        dict_observables[name] = np.array(dict_observables[name])

    if energies_test:
        return sigmas, tot_energies, dict_observables, energies_test_array
    return sigmas, tot_energies, dict_observables


# ---------------------------------------------------------------------------
# MCMC wrapper classes
# ---------------------------------------------------------------------------

class mcmc_ising:
    """High-level MCMC interface for Ising / Blume--Capel models.

    Parameters
    ----------
    J : ndarray, shape (M, M)
        Coupling matrix.
    h : ndarray, shape (M,)
        External field.
    Q : int
        Number of spin states.
    anisotropy : bool
        If True, use the BC sampler (non-zero diag(J)); otherwise use the
        plain Ising sampler.
    """

    def __init__(self, J, h, Q, anisotropy=False):
        N, _ = J.shape
        self.J = np.copy(J)
        self.h = h
        self.Q = Q
        self.N = N
        self.anisotropy = anisotropy

    def thermalize(self, betai, betaf, nsweeps, iicc='ordered', algorithm='G',
                   nb_chunks=20, pseudolikelihood=False, sigmas0=None, verbose=False):
        """Thermalize via (optionally) simulated annealing from betai to betaf.

        Setting ``betai = betaf`` performs a plain quench at fixed inverse
        temperature.
        """
        spin_values, _ = _setup_spin_values(self.Q)

        if pseudolikelihood:
            mysampler = gibbssampling_BC_pseudolikelihood
        else:
            mysampler = gibbssampling_BC if self.anisotropy else gibbssampling_ising

        sigmas = np.zeros(self.N, dtype=float)
        match iicc:
            case 'random':
                sigmas = np.random.choice(spin_values, self.N)
            case 'ordered':
                sigmas = np.zeros(self.N, dtype=float)
            case 'given':
                sigmas = sigmas0

        mag = lambda sigmas: magnetization(sigmas, self.Q)

        deltabeta = betaf - betai
        energies_series = []
        mag_series = []
        chunk_iter = tqdm(range(nb_chunks + 1), desc='Thermalize chunks', leave=False) \
            if verbose else range(nb_chunks + 1)
        for chunk in chunk_iter:
            beta = betai + deltabeta * chunk / nb_chunks
            sigmas, energies, dict_obs, *_ = mysampler(
                Q=self.Q, sigmas=sigmas, J=self.J, h=self.h, beta=beta,
                nsweeps=int(nsweeps / nb_chunks),
                observables_list=[mag], observables_names=['m'], verbose=verbose)
            energies_series = np.concatenate((energies_series, energies))
            mag_series = np.concatenate((mag_series, dict_obs['m']))

        assert sigmas is not None
        self.sigmas = np.copy(sigmas)
        self.betaf = betaf
        self.nb_chunks = nb_chunks
        return sigmas, energies_series, mag_series

    def sample(self, beta, nsweeps_sampling, observables_list=[],
               observables_names=[], pseudolikelihood=False, verbose=False):
        """Draw a Markov-chain trajectory of ``nsweeps_sampling`` sweeps at ``beta``."""
        if pseudolikelihood:
            mysampler = gibbssampling_BC_pseudolikelihood
        else:
            mysampler = gibbssampling_BC if self.anisotropy else gibbssampling_ising

        sigmas, energies, obs_dict, *_ = mysampler(
            Q=self.Q, sigmas=self.sigmas, J=self.J, h=self.h, beta=beta,
            nsweeps=nsweeps_sampling,
            observables_list=observables_list,
            observables_names=observables_names, verbose=verbose)

        self.sigmas = np.copy(sigmas)
        return obs_dict, energies

    def expected_values(self, beta, n_samples, n_sweeps_between_samples,
                        observables_list, observables_names, pseudolikelihood=False):
        """Compute (mean, std) of each observable across ``n_samples`` snapshots."""
        dict_means = {}
        dict_stds = {}
        sigmas = np.copy(self.sigmas)

        if pseudolikelihood:
            mysampler = gibbssampling_BC_pseudolikelihood
        else:
            mysampler = gibbssampling_BC if self.anisotropy else gibbssampling_ising

        for obs, name in zip(observables_list, observables_names):
            value = obs(np.copy(sigmas))
            dict_means[name] = value
            dict_stds[name] = value ** 2

        for n in range(n_samples - 1):
            sigmas, *_ = mysampler(
                Q=self.Q, sigmas=sigmas, J=self.J, h=self.h, beta=beta,
                nsweeps=n_sweeps_between_samples,
                observables_list=observables_list, observables_names=observables_names)

            for obs, name in zip(observables_list, observables_names):
                value = obs(np.copy(sigmas))
                dict_means[name] += value
                dict_stds[name] += value ** 2

        self.sigmas = sigmas

        for name in observables_names:
            dict_means[name] = dict_means[name] / (1. * n_samples)
            dict_stds[name] = (dict_stds[name] / (1. * n_samples) - dict_means[name] ** 2) ** 0.5

        return dict_means, dict_stds


class mcmc_beg:
    """High-level MCMC interface for the Blume--Emery--Griffiths model.

    Parameters
    ----------
    J, K : ndarrays, shape (M, M)
        Quadratic and biquadratic coupling matrices.  K has zero diagonal.
    h : ndarray, shape (M,)
        External field.
    Q : int
        Number of spin states.
    """

    def __init__(self, J, h, K, Q):
        N, _ = J.shape
        self.J = np.copy(J)
        self.h = np.copy(h)
        self.K = np.copy(K)
        self.Q = Q
        self.N = N

    def thermalize(self, betai, betaf, nsweeps, iicc='ordered', algorithm='G',
                   nb_chunks=20, pseudolikelihood=False, sigmas0=None, verbose=False):
        spin_values, _ = _setup_spin_values(self.Q)

        sigmas = np.zeros(self.N, dtype=float)
        match iicc:
            case 'random':
                sigmas = np.random.choice(spin_values, self.N)
            case 'ordered':
                sigmas = np.zeros(self.N, dtype=float)
            case 'given':
                sigmas = sigmas0

        mag = lambda sigmas: magnetization(sigmas, self.Q)

        mysampler = gibbssampling_beg_pseudolikelihood if pseudolikelihood else gibbssampling_beg

        deltabeta = betaf - betai
        energies_series = []
        mag_series = []
        chunk_iter = tqdm(range(nb_chunks + 1), desc='Thermalize chunks', leave=False) \
            if verbose else range(nb_chunks + 1)
        for chunk in chunk_iter:
            beta = betai + deltabeta * chunk / nb_chunks
            sigmas, energies, dict_obs, *_ = mysampler(
                Q=self.Q, sigmas=sigmas, J=self.J, h=self.h, K=self.K, beta=beta,
                nsweeps=int(nsweeps / nb_chunks),
                observables_list=[mag], observables_names=['m'], verbose=verbose)
            energies_series = np.concatenate((energies_series, energies))
            mag_series = np.concatenate((mag_series, dict_obs['m']))

        assert sigmas is not None
        self.sigmas = np.copy(sigmas)
        self.betaf = betaf
        self.nb_chunks = nb_chunks
        return sigmas, energies_series, mag_series

    def sample(self, beta, nsweeps_sampling, observables_list=[],
               observables_names=[], pseudolikelihood=False, verbose=False):
        mysampler = gibbssampling_beg_pseudolikelihood if pseudolikelihood else gibbssampling_beg

        sigmas, energies, dict_obs, *_ = mysampler(
            Q=self.Q, sigmas=self.sigmas, J=self.J, h=self.h, K=self.K, beta=beta,
            nsweeps=nsweeps_sampling,
            observables_list=observables_list,
            observables_names=observables_names, verbose=verbose)

        self.sigmas = np.copy(sigmas)
        return dict_obs, energies

    def expected_values(self, beta, n_samples, n_sweeps_between_samples,
                        observables_list, observables_names, pseudolikelihood=False):
        dict_means = {}
        dict_stds = {}
        sigmas = np.copy(self.sigmas)

        mysampler = gibbssampling_beg_pseudolikelihood if pseudolikelihood else gibbssampling_beg

        for obs, name in zip(observables_list, observables_names):
            value = obs(np.copy(sigmas))
            dict_means[name] = value
            dict_stds[name] = value ** 2

        for n in range(n_samples - 1):
            sigmas, *_ = mysampler(
                Q=self.Q, sigmas=sigmas, J=self.J, h=self.h, K=self.K, beta=beta,
                nsweeps=n_sweeps_between_samples,
                observables_list=observables_list, observables_names=observables_names)

            for obs, name in zip(observables_list, observables_names):
                value = obs(np.copy(sigmas))
                dict_means[name] += value
                dict_stds[name] += value ** 2

        self.sigmas = sigmas

        for name in observables_names:
            dict_means[name] = dict_means[name] / (1. * n_samples)
            dict_stds[name] = (dict_stds[name] / (1. * n_samples) - dict_means[name] ** 2) ** 0.5

        return dict_means, dict_stds
