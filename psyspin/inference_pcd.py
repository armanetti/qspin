"""
Persistent Contrastive Divergence (PCD) inference for generalized Ising models.

Each PCD class maintains a pool of persistent Markov chains (``ncopies`` chains)
that are run for ``tau_PCD`` Gibbs sweeps between gradient updates.  Their
sufficient statistics provide an unbiased estimate of the model expectations
needed for the maximum-likelihood gradient:

    grad_h  ~ <x>_data - <x>_model
    grad_J  ~ <x x^T>_data - <x x^T>_model
    grad_K  ~ <x^2 x^2^T>_data - <x^2 x^2^T>_model    (BEG only)

Optimization uses the Adam optimizer (:class:`Adam`).  Chains can be
periodically reset to random empirical configurations (Algorithm 2,
parameter ``tau_c``) to combat slow mixing.

Parallelisation
---------------
When ``n_workers > 1`` a :class:`~concurrent.futures.ProcessPoolExecutor` is used
to run multiple chains simultaneously across real OS processes, bypassing the GIL.
The pool is created once per ``naif_fit_euler`` call and reused across all gradient
iterations to avoid subprocess-spawn overhead.

On Linux and macOS the ``fork`` start method is used automatically, so no special
guard is needed in the calling script or notebook.  On Windows, ``spawn`` is the
only available method; scripts (not notebooks) must protect the entry point with
``if __name__ == '__main__':``.
"""

import multiprocessing
import sys
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from .gauge import possible_states
from .mcmc import gibbssampling_ising, gibbssampling_BC, gibbssampling_beg

# Use 'fork' on Unix (Linux + macOS) so worker processes inherit memory without
# re-importing the calling script — no 'if __name__ == "__main__"' guard needed.
# On Windows 'fork' is unavailable; fall back to 'spawn'.
_MP_CONTEXT = multiprocessing.get_context(
    'fork' if sys.platform != 'win32' else 'spawn'
)


# ---------------------------------------------------------------------------
# Module-level observable helpers 
# ---------------------------------------------------------------------------

def _obs_x(x):
    return x.copy()


def _obs_xxdag(x):
    return np.outer(x, x)


def _obs_x2x2dag(x):
    return np.outer(x ** 2, x ** 2)


# ---------------------------------------------------------------------------
# Module-level chain runners  (must be top-level for pickle)
# Each function takes a single tuple of arguments so executor.map works.
# ---------------------------------------------------------------------------

def _run_chain_ising(args):
    """Run one persistent chain for the Ising model."""
    Q, sigmas_n, J, h, tau_PCD = args
    sigmas, _, dict_obs, *_ = gibbssampling_ising(
        Q, sigmas_n, J, h, 1., tau_PCD,
        energies_test=False,
        observables_list=[_obs_x, _obs_xxdag],
        observables_names=['x', 'xxdag'],
    )
    return sigmas, dict_obs


def _thermalize_chain_ising(args):
    """Thermalize one chain for the Ising model."""
    Q, sigmas0, J0, h0, tau_therm = args
    sigmas_n, *_ = gibbssampling_ising(Q, sigmas0, J0, h0, 1., tau_therm)
    return sigmas_n


def _run_chain_BC(args):
    """Run one persistent chain for the Blume-Capel model."""
    Q, sigmas_n, J, h, tau_PCD = args
    sigmas, _, dict_obs, *_ = gibbssampling_BC(
        Q, sigmas_n, J, h, 1., tau_PCD,
        energies_test=False,
        observables_list=[_obs_x, _obs_xxdag],
        observables_names=['x', 'xxdag'],
    )
    return sigmas, dict_obs


def _thermalize_chain_BC(args):
    """Thermalize one chain for the Blume-Capel model."""
    Q, sigmas0, J0, h0, tau_therm = args
    sigmas_n, *_ = gibbssampling_BC(Q, sigmas0, J0, h0, 1., tau_therm)
    return sigmas_n


def _run_chain_BEG(args):
    """Run one persistent chain for the BEG model."""
    Q, sigmas_n, J, h, K, tau_PCD = args
    sigmas, _, dict_obs, *_ = gibbssampling_beg(
        Q, sigmas_n, J, h, K, 1., tau_PCD,
        energies_test=False,
        observables_list=[_obs_x, _obs_xxdag, _obs_x2x2dag],
        observables_names=['x', 'xxdag', 'x2x2dag'],
    )
    return sigmas, dict_obs


def _thermalize_chain_BEG(args):
    """Thermalize one chain for the BEG model."""
    Q, sigmas0, J0, h0, K0, tau_therm = args
    sigmas_n, *_ = gibbssampling_beg(Q, sigmas0, J0, h0, K0, 1., tau_therm)
    return sigmas_n


# ---------------------------------------------------------------------------
# Adam optimizer
# ---------------------------------------------------------------------------

class Adam:
    """Minimal Adam optimizer (Kingma & Ba, 2014).

    Update rule:
        m_t = beta1 m_{t-1} + (1 - beta1) g_t
        v_t = beta2 v_{t-1} + (1 - beta2) g_t^2
        m_hat = m_t / (1 - beta1^t),  v_hat = v_t / (1 - beta2^t)
        params -= lr m_hat / (sqrt(v_hat) + eps).
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        assert self.v is not None
        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params


# ---------------------------------------------------------------------------
# PCD inference classes
# ---------------------------------------------------------------------------

class generalizedIsing_inferencePCD:
    """PCD inference for the gauge-fixed generalized Ising model (diag(J)=0)."""

    def __init__(self, Q=3, l2_lambda=0.01):
        self.Q = Q
        self.l2_lambda = l2_lambda
        self.states = possible_states(Q)

    def _unpack_params(self, theta, n_nodes):
        h = theta[:n_nodes]
        J_flat = theta[n_nodes:]
        J = np.zeros((n_nodes, n_nodes))
        tri_indices = np.triu_indices(n_nodes, k=1)
        J[tri_indices] = J_flat
        J = J + J.T
        return J, h

    def _objective(self, theta, X):
        n_samples, n_nodes = X.shape
        J, h = self._unpack_params(theta, n_nodes)

        theoretical_mean = np.zeros(n_nodes)
        theoretical_xxdag = np.zeros((n_nodes, n_nodes))

        args_list = [
            (self.Q, np.copy(self.sigmas[n]), J, h, self.tau_PCD)
            for n in range(self.ncopies)
        ]
        if self._pool is not None:
            results = list(self._pool.map(_run_chain_ising, args_list))
        else:
            results = [_run_chain_ising(a) for a in args_list]

        for n, (sigmas, dict_obs) in enumerate(results):
            self.sigmas[n] = sigmas
            theoretical_mean += np.mean(dict_obs['x'], axis=0)
            theoretical_xxdag += np.mean(dict_obs['xxdag'], axis=0)

        theoretical_mean /= self.ncopies
        theoretical_xxdag /= self.ncopies
        grad_h = -(self.x_mean - theoretical_mean - self.l2_lambda * h)
        diff_xxdag = self.xxdag_mean - theoretical_xxdag
        grad_J = -(diff_xxdag - self.l2_lambda * J)

        tri_indices = np.triu_indices(n_nodes, k=1)
        grad_J_flat = grad_J[tri_indices]

        self.lossh = np.sum(grad_h ** 2)
        self.lossJ = np.sum((grad_J - np.eye(n_nodes) * np.diag(grad_J)) ** 2)
        loss = (self.lossh + self.lossJ) \
               + 0.5 * self.l2_lambda * (np.sum(J ** 2) / 2 + np.sum(h ** 2))

        self.theoretical_xxdag = theoretical_xxdag
        self.theoretical_mean = theoretical_mean
        return loss, np.concatenate([grad_h, grad_J_flat])

    def fit(self, X, niterations=1000, learning_rate=1.0E-3, ncopies=20,
            tau_PCD=20, tau_therm=100, tau_c=None,
            iicc='meanfield', iicc_configurations='emp',
            J0=None, h0=None, optimizer='adam', verbose=False, n_workers=1):
        """Fit (J, h) via PCD.

        Parameters
        ----------
        X : ndarray, shape (N, M)
            Gauge-fixed training data.
        niterations : int
            Number of gradient updates.
        learning_rate : float
            Learning rate (for both Adam and SGD).
        ncopies : int
            Number of persistent Markov chains.
        tau_PCD : int
            Gibbs sweeps between consecutive gradient updates.
        tau_therm : int
            Gibbs sweeps for the initial thermalization of each chain.
        tau_c : int or None
            If set, every ``tau_c`` iterations every chain is reset to a random
            empirical configuration (Algorithm 2).
        iicc : {'meanfield', 'given'}
            Initial parameter values.
        iicc_configurations : {'emp', 'rand'}
            Initial chain configurations (empirical sample or random).
        J0, h0 : ndarrays, optional
            Initial parameters when ``iicc='given'``.
        optimizer : {'adam', 'sgd'}
            ``'adam'`` uses the Adam optimizer (Kingma & Ba, 2014);
            ``'sgd'`` uses plain gradient descent (theta -= lr * grad).
        verbose : bool
            Print per-iteration loss / log-likelihood.
        n_workers : int
            Number of parallel worker processes for chain updates.
            Use 1 (default) for sequential execution.  Values > 1 spawn
            real OS processes via :class:`ProcessPoolExecutor`, bypassing
            the GIL.  On Windows the calling script must be
            protected with ``if __name__ == '__main__':``.
        """
        self.tau_PCD = tau_PCD
        myadam = Adam(lr=learning_rate) if optimizer == 'adam' else None

        n_samples, n_nodes = X.shape
        n_j_params = n_nodes * (n_nodes - 1) // 2
        self.x_mean = np.mean(X, axis=0)
        self.xxdag_mean = X.T @ X / n_samples

        self.ncopies = ncopies
        self.sigmas = np.random.choice(self.states, size=(self.ncopies, n_nodes))

        theta = np.zeros(n_nodes + n_j_params)
        match iicc:
            case 'meanfield':
                C_X = np.cov(X.T)
                mu_X = np.mean(X, axis=0)
                J0 = -np.linalg.inv(C_X)
                J0[np.diag_indices(n_nodes)] = np.zeros(n_nodes)
                h0 = -J0 @ mu_X
            case 'given':
                pass
            case _:
                J0 = np.zeros((n_nodes, n_nodes))
                h0 = np.zeros(n_nodes)

        assert J0 is not None and h0 is not None
        theta[:n_nodes] = h0
        tri_indices = np.triu_indices(n_nodes, k=1)
        theta[n_nodes:] = J0[tri_indices]

        # Build initial thermalization args
        therm_args = []
        for n in range(self.ncopies):
            if iicc_configurations == 'emp':
                index_sub = np.random.choice(range(n_samples))
                sigmas0_n = np.copy(X[index_sub])
            else:
                sigmas0_n = np.random.choice(self.states, size=n_nodes)
            therm_args.append((self.Q, sigmas0_n, J0, h0, tau_therm))

        # Create persistent worker pool (once for all iterations)
        self._pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=_MP_CONTEXT) if n_workers > 1 else None

        try:
            # Initial thermalization
            if self._pool is not None:
                therm_results = list(self._pool.map(_thermalize_chain_ising, therm_args))
            else:
                therm_results = [_thermalize_chain_ising(a) for a in therm_args]
            for n, sigmas_n in enumerate(therm_results):
                self.sigmas[n] = sigmas_n

            self.losses = np.zeros((niterations, 3))

            for n in tqdm(range(niterations)):
                if tau_c is not None and n % tau_c == 0:
                    for k in range(self.ncopies):
                        idx = np.random.choice(n_samples)
                        self.sigmas[k] = np.copy(X[idx])

                loss, grad = self._objective(theta, X)
                if myadam is not None:
                    theta = myadam.update(theta, grad)
                else:
                    theta -= learning_rate * grad

                self.J_fit, self.h_fit = self._unpack_params(theta, n_nodes)
                loglik = self.loglikelihood(X) / n_samples
                self.losses[n] = [loglik, self.lossh, self.lossJ]
                if verbose:
                    print(n, loss, loglik)

        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None

        return self.J_fit, self.h_fit

    def loglikelihood(self, X, compute_std=False):
        """Per-sample log pseudo-likelihood (used as a learning monitor)."""
        n_samples, n_nodes = X.shape
        J, h = self.J_fit, self.h_fit
        A = X @ J + h
        exponent = A[:, :, np.newaxis] * self.states[np.newaxis, np.newaxis, :]
        max_exp = np.max(exponent, axis=2, keepdims=True)
        log_z = np.log(np.sum(np.exp(exponent - max_exp), axis=2)) + max_exp.squeeze()
        log_lik = np.mean(X * A - log_z)
        if not compute_std:
            return log_lik
        return np.array([log_lik, np.std(X * A - log_z) / np.sqrt(n_samples)])

    def completion_error_1(self, x):
        """Mean completion error (pseudo-likelihood approximation) for one vector."""
        J, h, states = self.J_fit, self.h_fit, self.states
        phi = (h + J @ x)
        probabilities = np.exp(phi[:, np.newaxis] * states)
        probabilities = (probabilities.T / np.sum(probabilities, axis=1)).T
        return np.mean(np.abs(np.sum(probabilities * states, axis=1) - x))

    def completion_error_1_manyvectors(self, X):
        """Batched version of :meth:`completion_error_1`."""
        J, h, states = self.J_fit, self.h_fit, self.states
        phi = J @ X.T
        phi = (h + phi.T).T
        probabilities = np.exp(phi[:, :, np.newaxis] * states)
        probabilities = (probabilities.T / np.sum(probabilities, axis=2).T).T
        averages = np.sum(probabilities * states, axis=2).T
        return np.mean(np.abs(averages - X))


class generalizedBC_inferencePCD:
    """PCD inference for the Blume--Capel model."""

    def __init__(self, Q=3, l2_lambda=0.01):
        self.Q = Q
        self.l2_lambda = l2_lambda
        self.states = possible_states(Q)

    def _unpack_params(self, theta, n_nodes):
        h = theta[:n_nodes]
        J_flat = theta[n_nodes:]
        J = np.zeros((n_nodes, n_nodes))
        tri_indices = np.triu_indices(n_nodes, k=0)
        J[tri_indices] = J_flat
        J = J + J.T
        diag_indices = np.diag_indices(n_nodes)
        J[diag_indices] = 0.5 * J[diag_indices]
        return J, h

    def _objective(self, theta, X):
        n_samples, n_nodes = X.shape
        J, h = self._unpack_params(theta, n_nodes)

        theoretical_mean = np.zeros(n_nodes)
        theoretical_xxdag = np.zeros((n_nodes, n_nodes))

        args_list = [
            (self.Q, np.copy(self.sigmas[n]), J, h, self.tau_PCD)
            for n in range(self.ncopies)
        ]
        if self._pool is not None:
            results = list(self._pool.map(_run_chain_BC, args_list))
        else:
            results = [_run_chain_BC(a) for a in args_list]

        for n, (sigmas, dict_obs) in enumerate(results):
            self.sigmas[n] = sigmas
            theoretical_mean += np.mean(dict_obs['x'], axis=0)
            theoretical_xxdag += np.mean(dict_obs['xxdag'], axis=0)

        theoretical_mean /= self.ncopies
        theoretical_xxdag /= self.ncopies
        grad_h = -(self.x_mean - theoretical_mean - self.l2_lambda * h)
        diff_xxdag = self.xxdag_mean - theoretical_xxdag
        grad_J = -(diff_xxdag - self.l2_lambda * J)
        diag_idx = np.diag_indices(n_nodes)
        grad_J[diag_idx] = -(0.5 * diff_xxdag[diag_idx] - self.l2_lambda * np.diag(J))

        tri_indices = np.triu_indices(n_nodes, k=0)
        grad_J_flat = grad_J[tri_indices]

        self.lossh = np.sum(grad_h ** 2)
        self.lossJ = np.sum(grad_J ** 2)
        loss = (self.lossh + self.lossJ) \
               + 0.5 * self.l2_lambda * (np.sum(J ** 2) / 2 + np.sum(h ** 2))

        self.theoretical_xxdag = theoretical_xxdag
        self.theoretical_mean = theoretical_mean
        return loss, np.concatenate([grad_h, grad_J_flat])

    def fit(self, X, niterations=1000, learning_rate=1.0E-3, ncopies=20,
            tau_PCD=20, tau_therm=100, tau_c=None,
            iicc='meanfield', iicc_configurations='emp',
            J0=None, h0=None, optimizer='adam', verbose=False, n_workers=1):
        """Fit (J, h) via PCD.  See :meth:`generalizedIsing_inferencePCD.fit`."""
        self.tau_PCD = tau_PCD
        myadam = Adam(lr=learning_rate) if optimizer == 'adam' else None

        n_samples, n_nodes = X.shape
        n_j_params = n_nodes * (n_nodes + 1) // 2
        self.x_mean = np.mean(X, axis=0)
        self.xxdag_mean = X.T @ X / n_samples

        self.ncopies = ncopies
        self.sigmas = np.random.choice(self.states, size=(self.ncopies, n_nodes))

        theta = np.zeros(n_nodes + n_j_params)
        match iicc:
            case 'meanfield':
                C_X = np.cov(X.T)
                mu_X = np.mean(X, axis=0)
                J0 = -np.linalg.inv(C_X)
                h0 = -J0 @ mu_X
            case 'given':
                pass
            case _:
                J0 = np.zeros((n_nodes, n_nodes))
                h0 = np.zeros(n_nodes)

        assert J0 is not None and h0 is not None
        theta[:n_nodes] = h0
        tri_indices = np.triu_indices(n_nodes, k=0)
        theta[n_nodes:] = J0[tri_indices]

        therm_args = []
        for n in range(self.ncopies):
            if iicc_configurations == 'emp':
                index_sub = np.random.choice(range(n_samples))
                sigmas0_n = np.copy(X[index_sub])
            else:
                sigmas0_n = np.random.choice(self.states, size=n_nodes)
            therm_args.append((self.Q, sigmas0_n, J0, h0, tau_therm))

        self._pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=_MP_CONTEXT) if n_workers > 1 else None

        try:
            if self._pool is not None:
                therm_results = list(self._pool.map(_thermalize_chain_BC, therm_args))
            else:
                therm_results = [_thermalize_chain_BC(a) for a in therm_args]
            for n, sigmas_n in enumerate(therm_results):
                self.sigmas[n] = sigmas_n

            self.losses = np.zeros((niterations, 3))

            for n in tqdm(range(niterations)):
                if tau_c is not None and n % tau_c == 0:
                    for k in range(self.ncopies):
                        idx = np.random.choice(n_samples)
                        self.sigmas[k] = np.copy(X[idx])

                loss, grad = self._objective(theta, X)
                if myadam is not None:
                    theta = myadam.update(theta, grad)
                else:
                    theta -= learning_rate * grad

                self.J_fit, self.h_fit = self._unpack_params(theta, n_nodes)
                loglik = self.loglikelihood(X) / n_samples
                self.losses[n] = [loglik, self.lossh, self.lossJ]
                if verbose:
                    print(n, loss, loglik)

        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None

        return self.J_fit, self.h_fit

    def loglikelihood(self, X, compute_std=False):
        n_samples, n_nodes = X.shape
        J, h = self.J_fit, self.h_fit
        A = X @ J + h - X * np.diag(J)
        exponent = A[:, :, np.newaxis] * self.states[np.newaxis, np.newaxis, :] + \
                   0.5 * np.diag(J)[np.newaxis, :, np.newaxis] \
                       * self.states[np.newaxis, np.newaxis, :] ** 2
        max_exp = np.max(exponent, axis=2, keepdims=True)
        log_z = np.log(np.sum(np.exp(exponent - max_exp), axis=2)) + max_exp.squeeze()
        argument = X * A + 0.5 * X ** 2 * np.diag(J) - log_z
        log_lik = np.mean(argument)
        if not compute_std:
            return log_lik
        return np.array([log_lik, np.std(argument) / np.sqrt(n_samples)])

    def conditional_probability(self, i, x):
        J, h, states = self.J_fit, self.h_fit, self.states
        phi = (h + J @ x - np.diag(J) * x)[i]
        energies = phi * states + 0.5 * states ** 2 * J[i, i]
        probabilities = np.exp(energies)
        probabilities /= np.sum(probabilities)
        return probabilities

    def completion_error_1(self, x):
        J, h, states = self.J_fit, self.h_fit, self.states
        phi = (h + J @ x - x * np.diag(J))
        probabilities = np.exp(phi[:, np.newaxis] * states
                               + 0.5 * np.diag(J)[:, np.newaxis] * states ** 2)
        probabilities = (probabilities.T / np.sum(probabilities, axis=1)).T
        return np.mean(np.abs(np.sum(probabilities * states, axis=1) - x))

    def completion_error_1_manyvectors(self, X):
        J, h, states = self.J_fit, self.h_fit, self.states
        phi = J @ X.T - (X * np.diag(J)).T
        phi = (h + phi.T).T
        qq = 0.5 * np.diag(J)
        energies = phi[:, :, np.newaxis] * states + qq[:, np.newaxis, np.newaxis] * states ** 2
        probabilities = np.exp(energies)
        probabilities = (probabilities.T / np.sum(probabilities, axis=2).T).T
        averages = np.sum(probabilities * states, axis=2).T
        return np.mean(np.abs(averages - X))


class generalizedBEG_inferencePCD:
    """PCD inference for the Blume--Emery--Griffiths model."""

    def __init__(self, Q=3, l2_lambda=0.01):
        self.Q = Q
        self.l2_lambda = l2_lambda
        self.states = possible_states(Q)

    def _unpack_params(self, theta, n_nodes):
        tridiag_indices = np.triu_indices(n_nodes, k=0)
        trinodiag_indices = np.triu_indices(n_nodes, k=1)
        diag_indices = np.diag_indices(n_nodes)
        n_tridiag = int(n_nodes * (n_nodes + 1) / 2)

        h = theta[:n_nodes]
        J_flat = theta[n_nodes:n_nodes + n_tridiag]
        J = np.zeros((n_nodes, n_nodes))
        J[tridiag_indices] = J_flat
        J = J + J.T
        J[diag_indices] = 0.5 * J[diag_indices]

        K_flat = theta[n_nodes + n_tridiag:]
        K = np.zeros((n_nodes, n_nodes))
        K[trinodiag_indices] = K_flat
        K = K + K.T
        return J, h, K

    def _objective(self, theta, X):
        n_samples, n_nodes = X.shape
        J, h, K = self._unpack_params(theta, n_nodes)

        theoretical_mean = np.zeros(n_nodes)
        theoretical_xxdag = np.zeros((n_nodes, n_nodes))
        theoretical_x2x2dag = np.zeros((n_nodes, n_nodes))

        args_list = [
            (self.Q, np.copy(self.sigmas[n]), J, h, K, self.tau_PCD)
            for n in range(self.ncopies)
        ]
        if self._pool is not None:
            results = list(self._pool.map(_run_chain_BEG, args_list))
        else:
            results = [_run_chain_BEG(a) for a in args_list]

        for n, (sigmas, dict_obs) in enumerate(results):
            self.sigmas[n] = sigmas
            theoretical_mean += np.mean(dict_obs['x'], axis=0)
            theoretical_xxdag += np.mean(dict_obs['xxdag'], axis=0)
            theoretical_x2x2dag += np.mean(dict_obs['x2x2dag'], axis=0)

        theoretical_mean /= self.ncopies
        theoretical_xxdag /= self.ncopies
        theoretical_x2x2dag /= self.ncopies
        grad_h = -(self.x_mean - theoretical_mean - self.l2_lambda * h)
        diff_xxdag = self.xxdag_mean - theoretical_xxdag
        grad_J = -(diff_xxdag - self.l2_lambda * J)
        diag_idx = np.diag_indices(n_nodes)
        grad_J[diag_idx] = -(0.5 * diff_xxdag[diag_idx] - self.l2_lambda * np.diag(J))
        grad_K = -((self.x2x2dag_mean - theoretical_x2x2dag) - self.l2_lambda * K)

        tri_indices = np.triu_indices(n_nodes, k=0)
        grad_J_flat = grad_J[tri_indices]
        triu_indices = np.triu_indices(n_nodes, k=1)
        grad_K_flat = grad_K[triu_indices]

        self.lossh = np.sum(grad_h ** 2)
        self.lossJ = np.sum(grad_J ** 2)
        self.lossK = np.sum(grad_K ** 2)
        loss = (self.lossh + self.lossJ + self.lossK) \
               + 0.5 * self.l2_lambda * (np.sum(J ** 2) / 2 + np.sum(h ** 2))

        self.theoretical_mean = theoretical_mean
        self.theoretical_xxdag = theoretical_xxdag
        self.theoretical_x2x2dag = theoretical_x2x2dag
        return loss, np.concatenate([grad_h, grad_J_flat, grad_K_flat])

    def fit(self, X, niterations=1000, learning_rate=1.0E-3, ncopies=20,
            tau_PCD=20, tau_therm=100, tau_c=None,
            iicc='meanfield', iicc_configurations='emp',
            J0=None, h0=None, K0=None, optimizer='adam', verbose=False, n_workers=1):
        """Fit (J, h, K) via PCD.  See :meth:`generalizedIsing_inferencePCD.fit`."""
        self.tau_PCD = tau_PCD
        myadam = Adam(lr=learning_rate) if optimizer == 'adam' else None

        n_samples, n_nodes = X.shape
        n_j_params = n_nodes * (n_nodes + 1) // 2
        n_k_params = n_nodes * (n_nodes - 1) // 2
        self.x_mean = np.mean(X, axis=0)
        self.xxdag_mean = X.T @ X / n_samples
        self.x2x2dag_mean = (X ** 2).T @ (X ** 2) / n_samples

        self.ncopies = ncopies
        self.sigmas = np.random.choice(self.states, size=(self.ncopies, n_nodes))

        theta = np.zeros(n_nodes + n_j_params + n_k_params)
        match iicc:
            case 'meanfield':
                C_X = np.cov(X.T)
                mu_X = np.mean(X, axis=0)
                J0 = -np.linalg.inv(C_X)
                h0 = -J0 @ mu_X
                K0 = np.zeros((n_nodes, n_nodes))
            case 'given':
                pass
            case _:
                J0 = np.zeros((n_nodes, n_nodes))
                h0 = np.zeros(n_nodes)
                K0 = np.zeros((n_nodes, n_nodes))

        assert J0 is not None and h0 is not None and K0 is not None
        theta[:n_nodes] = h0
        tri_indices = np.triu_indices(n_nodes, k=0)
        theta[n_nodes:n_nodes + n_j_params] = J0[tri_indices]
        triu_indices = np.triu_indices(n_nodes, k=1)
        theta[n_nodes + n_j_params:] = K0[triu_indices]

        therm_args = []
        for n in range(self.ncopies):
            if iicc_configurations == 'emp':
                index_sub = np.random.choice(range(n_samples))
                sigmas0_n = np.copy(X[index_sub])
            else:
                sigmas0_n = np.random.choice(self.states, size=n_nodes)
            therm_args.append((self.Q, sigmas0_n, J0, h0, K0, tau_therm))

        self._pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=_MP_CONTEXT) if n_workers > 1 else None

        try:
            if self._pool is not None:
                therm_results = list(self._pool.map(_thermalize_chain_BEG, therm_args))
            else:
                therm_results = [_thermalize_chain_BEG(a) for a in therm_args]
            for n, sigmas_n in enumerate(therm_results):
                self.sigmas[n] = sigmas_n

            self.losses = np.zeros((niterations, 4))

            for n in tqdm(range(niterations)):
                if tau_c is not None and n % tau_c == 0:
                    for k in range(self.ncopies):
                        idx = np.random.choice(n_samples)
                        self.sigmas[k] = np.copy(X[idx])

                loss, grad = self._objective(theta, X)
                if myadam is not None:
                    theta = myadam.update(theta, grad)
                else:
                    theta -= learning_rate * grad

                self.J_fit, self.h_fit, self.K_fit = self._unpack_params(theta, n_nodes)
                loglik = self.loglikelihood(X) / n_samples
                self.losses[n] = [loglik, self.lossh, self.lossJ, self.lossK]
                if verbose:
                    print(n, loss, loglik)

        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None

        return self.J_fit, self.h_fit, self.K_fit

    def loglikelihood(self, X, compute_std=False):
        n_samples, n_nodes = X.shape
        J, h, K = self.J_fit, self.h_fit, self.K_fit
        phi = X @ J + h - X * np.diag(J)
        varphi = (X ** 2) @ K

        exponent = phi[:, :, np.newaxis] * self.states[np.newaxis, np.newaxis, :] + \
                   0.5 * np.diag(J)[np.newaxis, :, np.newaxis] \
                       * self.states[np.newaxis, np.newaxis, :] ** 2 + \
                   varphi[:, :, np.newaxis] * self.states[np.newaxis, np.newaxis, :] ** 2
        max_exp = np.max(exponent, axis=2, keepdims=True)
        log_z = np.log(np.sum(np.exp(exponent - max_exp), axis=2)) + max_exp.squeeze()
        argument = X * phi + 0.5 * X ** 2 * np.diag(J) + X ** 2 * varphi - log_z
        log_lik = np.mean(argument)
        if not compute_std:
            return log_lik
        return np.array([log_lik, np.std(argument) / np.sqrt(n_samples)])

    def conditional_probability(self, i, x):
        J, h, states = self.J_fit, self.h_fit, self.states
        phi = (h + J @ x - np.diag(J) * x)[i]
        varphi = (self.K_fit @ (x ** 2))[i]
        energies = phi * states + 0.5 * states ** 2 * J[i, i] + varphi * states ** 2
        probabilities = np.exp(energies)
        probabilities /= np.sum(probabilities)
        return probabilities

    def completion_error_1(self, x):
        J, h, states = self.J_fit, self.h_fit, self.states
        phi = (h + J @ x - x * np.diag(J))
        varphi = self.K_fit @ (x ** 2)
        probabilities = np.exp(phi[:, np.newaxis] * states
                               + 0.5 * np.diag(J)[:, np.newaxis] * states ** 2
                               + varphi[:, np.newaxis] * states ** 2)
        probabilities = (probabilities.T / np.sum(probabilities, axis=1)).T
        return np.mean(np.abs(np.sum(probabilities * states, axis=1) - x))

    def completion_error_1_manyvectors(self, X):
        J, h, K, states = self.J_fit, self.h_fit, self.K_fit, self.states
        phi = J @ X.T - (X * np.diag(J)).T
        phi = (h + phi.T).T
        varphi = ((X ** 2) @ K).T
        qq = 0.5 * np.diag(J)
        energies = phi[:, :, np.newaxis] * states \
                   + qq[:, np.newaxis, np.newaxis] * states ** 2 \
                   + varphi[:, :, np.newaxis] * states ** 2
        probabilities = np.exp(energies)
        probabilities = (probabilities.T / np.sum(probabilities, axis=2).T).T
        averages = np.sum(probabilities * states, axis=2).T
        return np.mean(np.abs(averages - X))
