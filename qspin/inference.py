"""
Pseudo-likelihood inference for generalized Ising models.

Three model families are provided.  Let x in {gauge-fixed states} and let
H(x) denote the energy:

* :class:`generalizedIsing_inference`
      H(x) = -h.x - (1/2) x^T J x,        diag(J) = 0
* :class:`generalizedBC_inference` (Blume--Capel)
      H(x) = -h.x - (1/2) x^T J x,        diag(J) free (anisotropy term)
* :class:`generalizedBEG_inference` (Blume--Emery--Griffiths)
      H(x) = -h.x - (1/2) x^T J x - (1/2) sum_{i!=j} K_{ij} x_i^2 x_j^2,
      diag(J) free, diag(K) = 0

Each class fits ``J`` (and ``h``, and ``K`` for BEG) by maximizing the
pseudo-likelihood with L-BFGS-B.  Data must be gauge-fixed
(see :func:`isingq.gauge.gaugefixing_data`) before calling ``fit``.
"""

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm


class generalizedIsing_inference:
    """Pseudo-likelihood inference for the gauge-fixed generalized Ising model.

    Parameters
    ----------
    Q : int
        Number of spin states.
    l2_lambda : float
        L2 regularization strength on (J, h).
    """

    def __init__(self, Q=3, l2_lambda=0.01):
        self.Q = Q
        self.l2_lambda = l2_lambda
        if Q % 2 == 1:
            self.states = np.arange(1, Q + 1)
            self.states = self.states - np.mean(self.states)
        else:
            self.states = np.arange(1, Q + 1)
            self.states = 2 * (self.states - np.mean(self.states))

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

        # effective fields
        phi = X @ J + h

        # log pseudo-likelihood
        exponent = phi[:, :, np.newaxis] * self.states[np.newaxis, np.newaxis, :]
        max_exp = np.max(exponent, axis=2, keepdims=True)
        log_z = np.log(np.sum(np.exp(exponent - max_exp), axis=2)) + max_exp.squeeze()

        log_lik = np.sum(X * phi - log_z)

        # gradients
        probs = np.exp(exponent - max_exp
                       - np.log(np.sum(np.exp(exponent - max_exp), axis=2, keepdims=True)))
        expected_x = np.sum(probs * self.states, axis=2)

        diff = X - expected_x
        grad_h = -(np.sum(diff, axis=0) - self.l2_lambda * h)

        grad_J_full = -((X.T @ diff) - self.l2_lambda * J)
        grad_J_sym = grad_J_full + grad_J_full.T
        tri_indices = np.triu_indices(n_nodes, k=1)
        grad_J_flat = grad_J_sym[tri_indices]

        loss = -(log_lik - 0.5 * self.l2_lambda * (np.sum(J ** 2) / 2 + np.sum(h ** 2)))
        return loss, np.concatenate([grad_h, grad_J_flat])

    def fit(self, X, iicc='meanfield', J0=None, h0=None, verbose=True):
        """Fit (J, h) by maximizing the pseudo-likelihood with L-BFGS-B.

        Parameters
        ----------
        X : ndarray, shape (N, M)
            Gauge-fixed data.
        iicc : {'meanfield', 'given'}
            Initial-condition strategy.  ``'meanfield'`` uses the naive mean-field
            inverse covariance; ``'given'`` requires ``J0`` and ``h0``.
        J0, h0 : ndarrays, optional
            Initial parameters (only used when ``iicc='given'``).
        verbose : bool
            If True, print L-BFGS-B termination message.

        Returns
        -------
        (J_fit, h_fit) : tuple of ndarrays
        """
        n_samples, n_nodes = X.shape
        n_j_params = n_nodes * (n_nodes - 1) // 2
        tri_indices = np.triu_indices(n_nodes, k=1)
        initial_theta = np.zeros(n_nodes + n_j_params)

        match iicc:
            case 'meanfield':
                C_X = np.cov(X.T)
                mu_X = np.mean(X, axis=0)
                J0 = -np.linalg.inv(C_X)
                J0[np.diag_indices(n_nodes)] = np.zeros(n_nodes)
                h0 = -J0 @ mu_X
                initial_theta[:n_nodes] = h0
                initial_theta[n_nodes:] = J0[tri_indices]
            case 'given':
                assert h0 is not None and J0 is not None
                initial_theta[:n_nodes] = h0
                initial_theta[n_nodes:] = J0[tri_indices]

        res = minimize(
            fun=self._objective,
            x0=initial_theta,
            args=(X,),
            method='L-BFGS-B', tol=1.0E-8,
            jac=True,
            options={'disp': True, 'maxiter': 1000}
        )

        if verbose:
            print(res.message)

        self.J_fit, self.h_fit = self._unpack_params(res.x, n_nodes)
        return self.J_fit, self.h_fit

    def naif_fit_euler(self, X, niterations=1000, learning_rate=1E-3,
                       iicc='meanfield', J0=None, h0=None):
        """Naive gradient-descent variant of :meth:`fit`."""
        n_samples, n_nodes = X.shape
        n_j_params = n_nodes * (n_nodes - 1) // 2
        tri_indices = np.triu_indices(n_nodes, k=1)
        initial_theta = np.zeros(n_nodes + n_j_params)

        match iicc:
            case 'meanfield':
                C_X = np.cov(X.T)
                mu_X = np.mean(X, axis=0)
                J0 = -np.linalg.inv(C_X)
                J0[np.diag_indices(n_nodes)] = np.zeros(n_nodes)
                h0 = -J0 @ mu_X
                initial_theta[:n_nodes] = h0
                initial_theta[n_nodes:] = J0[tri_indices]
            case 'given':
                assert h0 is not None and J0 is not None
                initial_theta[:n_nodes] = h0
                initial_theta[n_nodes:] = J0[tri_indices]

        theta = np.copy(initial_theta)
        for n in tqdm(range(niterations)):
            loss, grad = self._objective(theta, X)
            theta = theta - learning_rate * grad
            self.J_fit, self.h_fit = self._unpack_params(theta, n_nodes)
            print(n, self.loglikelihood(X) / n_samples)
        return self.J_fit, self.h_fit

    def loglikelihood(self, X):
        """Per-sample log pseudo-likelihood evaluated at (J_fit, h_fit)."""
        n_samples, n_nodes = X.shape
        J, h = self.J_fit, self.h_fit
        A = X @ J + h
        exponent = A[:, :, np.newaxis] * self.states[np.newaxis, np.newaxis, :]
        max_exp = np.max(exponent, axis=2, keepdims=True)
        log_z = np.log(np.sum(np.exp(exponent - max_exp), axis=2)) + max_exp.squeeze()
        log_lik = np.sum(X * A - log_z)
        return log_lik / n_samples

    def conditional_probability(self, i, x):
        """P(x_i | x_{\\i}) under the fitted model."""
        J, h = self.J_fit, self.h_fit
        phi = (h + J @ x)[i]
        probabilities = np.exp(phi * self.states)
        probabilities /= np.sum(probabilities)
        return probabilities

    def completion_error_1(self, x):
        """Mean completion error  sum_i |x_i - <x_i|x_{\\i}>| / N  for one vector."""
        J, h, states = self.J_fit, self.h_fit, self.states
        phi = (h + J @ x)
        probabilities = np.exp(phi[:, np.newaxis] * states)
        probabilities = (probabilities.T / np.sum(probabilities, axis=1)).T
        return np.mean(np.abs(np.sum(probabilities * states, axis=1) - x))

    def completion_error_1_manyvectors(self, X):
        """Vectorized version of :meth:`completion_error_1` over a batch."""
        J, h, states = self.J_fit, self.h_fit, self.states
        phi = J @ X.T
        phi = (h + phi.T).T
        probabilities = np.exp(phi[:, :, np.newaxis] * states)
        probabilities = (probabilities.T / np.sum(probabilities, axis=2).T).T
        averages = np.sum(probabilities * states, axis=2).T
        return np.mean(np.abs(averages - X))


class generalizedBC_inference:
    """Pseudo-likelihood inference for the Blume--Capel (BC) model.

    Like :class:`generalizedIsing_inference` but with non-zero ``diag(J)``,
    which encodes a per-site anisotropy on ``x_i^2``.
    """

    def __init__(self, Q=3, l2_lambda=0.01):
        self.Q = Q
        self.l2_lambda = l2_lambda
        if Q % 2 == 1:
            self.states = np.arange(1, Q + 1)
            self.states = self.states - np.mean(self.states)
        else:
            self.states = np.arange(1, Q + 1)
            self.states = 2 * (self.states - np.mean(self.states))

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

        # effective fields (linear part); the diagonal contribution is handled separately
        phi = X @ J + h - X * np.diag(J)

        exponent = phi[:, :, np.newaxis] * self.states[np.newaxis, np.newaxis, :]
        exponent = exponent + 0.5 * np.diag(J)[np.newaxis, :, np.newaxis] \
                              * self.states[np.newaxis, np.newaxis, :] ** 2

        max_exp = np.max(exponent, axis=2, keepdims=True)
        log_z = np.log(np.sum(np.exp(exponent - max_exp), axis=2)) + max_exp.squeeze()

        log_lik = np.sum(X * phi + 0.5 * X ** 2 * np.diag(J) - log_z)

        probs = np.exp(exponent - max_exp
                       - np.log(np.sum(np.exp(exponent - max_exp), axis=2, keepdims=True)))
        expected_x = np.sum(probs * self.states, axis=2)

        diff = X - expected_x
        grad_h = -(np.sum(diff, axis=0) - self.l2_lambda * h)

        grad_J_full = -((X.T @ diff) - self.l2_lambda * J)
        grad_J_sym = grad_J_full + grad_J_full.T

        expected_x2 = np.sum(probs * self.states ** 2, axis=2)
        diff_x2 = X ** 2 - expected_x2
        grad_J_diag = -(0.5 * np.sum(diff_x2, axis=0) - self.l2_lambda * np.diag(J))
        diag_indices = np.diag_indices(n_nodes)
        grad_J_sym[diag_indices] = grad_J_diag

        tri_indices = np.triu_indices(n_nodes, k=0)
        grad_J_flat = grad_J_sym[tri_indices]

        loss = -(log_lik - 0.5 * self.l2_lambda * (np.sum(J ** 2) / 2 + np.sum(h ** 2)))
        return loss, np.concatenate([grad_h, grad_J_flat])

    def fit(self, X, iicc='meanfield', J0=None, h0=None, verbose=False):
        """Fit (J, h) by maximizing the pseudo-likelihood with L-BFGS-B."""
        n_samples, n_nodes = X.shape
        n_j_params = n_nodes * (n_nodes + 1) // 2
        tri_indices = np.triu_indices(n_nodes, k=0)
        initial_theta = np.zeros(n_nodes + n_j_params)

        match iicc:
            case 'meanfield':
                mu_X = np.mean(X, axis=0)
                C_X = np.cov(X.T)
                J0 = -np.linalg.inv(C_X)
                h0 = -J0 @ mu_X
                initial_theta[:n_nodes] = h0
                initial_theta[n_nodes:] = J0[tri_indices]
            case 'given':
                assert h0 is not None and J0 is not None
                initial_theta[:n_nodes] = h0
                initial_theta[n_nodes:] = J0[tri_indices]

        res = minimize(
            fun=self._objective,
            x0=initial_theta,
            args=(X,),
            method='L-BFGS-B', tol=1.0E-8,
            jac=True,
            options={'disp': True, 'maxiter': 1000}
        )

        if verbose:
            print(res.message)

        self.J_fit, self.h_fit = self._unpack_params(res.x, n_nodes)
        return self.J_fit, self.h_fit

    def naif_fit_euler(self, X, niterations=1000, learning_rate=1E-3,
                       iicc='meanfield', J0=None, h0=None):
        """Naive gradient-descent variant of :meth:`fit`."""
        n_samples, n_nodes = X.shape
        n_j_params = n_nodes * (n_nodes + 1) // 2
        initial_theta = np.zeros(n_nodes + n_j_params)
        tri_indices = np.triu_indices(n_nodes, k=0)

        match iicc:
            case 'meanfield':
                mu_X = np.mean(X, axis=0)
                C_X = np.cov(X.T)
                J0 = -np.linalg.inv(C_X)
                J0[np.diag_indices(n_nodes)] = 0.
                h0 = -J0 @ mu_X
                initial_theta[:n_nodes] = h0
                initial_theta[n_nodes:] = J0[tri_indices]
            case 'given':
                assert h0 is not None and J0 is not None
                initial_theta[:n_nodes] = h0
                initial_theta[n_nodes:] = J0[tri_indices]

        theta = np.copy(initial_theta)
        for n in tqdm(range(niterations)):
            loss, grad = self._objective(theta, X)
            theta = theta - learning_rate * grad
            self.J_fit, self.h_fit = self._unpack_params(theta, n_nodes)
            print(n, self.loglikelihood(X) / n_samples, loss)
        return self.J_fit, self.h_fit

    def loglikelihood(self, X):
        n_samples, n_nodes = X.shape
        J, h = self.J_fit, self.h_fit
        A = X @ J + h - X * np.diag(J)
        exponent = A[:, :, np.newaxis] * self.states[np.newaxis, np.newaxis, :] + \
                   0.5 * np.diag(J)[np.newaxis, :, np.newaxis] \
                       * self.states[np.newaxis, np.newaxis, :] ** 2
        max_exp = np.max(exponent, axis=2, keepdims=True)
        log_z = np.log(np.sum(np.exp(exponent - max_exp), axis=2)) + max_exp.squeeze()
        log_lik = np.sum(X * A + 0.5 * X ** 2 * np.diag(J) - log_z)
        return log_lik / n_samples

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


class generalizedBEG_inference:
    """Pseudo-likelihood inference for the Blume--Emery--Griffiths (BEG) model.

    Adds a biquadratic coupling K with zero diagonal:
        H(x) = -h.x - (1/2) x^T J x - (1/2) sum_{i!=j} K_{ij} x_i^2 x_j^2
    """

    def __init__(self, Q=3, l2_lambda=0.01):
        self.Q = Q
        self.l2_lambda = l2_lambda
        if Q % 2 == 1:
            self.states = np.arange(1, Q + 1)
            self.states = self.states - np.mean(self.states)
        else:
            self.states = np.arange(1, Q + 1)
            self.states = 2 * (self.states - np.mean(self.states))

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

        phi = X @ J + h - X * np.diag(J)
        varphi = (X ** 2) @ K

        exponent = phi[:, :, np.newaxis] * self.states[np.newaxis, np.newaxis, :]
        exponent = exponent + 0.5 * np.diag(J)[np.newaxis, :, np.newaxis] \
                              * self.states[np.newaxis, np.newaxis, :] ** 2
        exponent = exponent + varphi[:, :, np.newaxis] \
                              * self.states[np.newaxis, np.newaxis, :] ** 2

        max_exp = np.max(exponent, axis=2, keepdims=True)
        log_z = np.log(np.sum(np.exp(exponent - max_exp), axis=2)) + max_exp.squeeze()

        log_lik = np.sum(X * phi + X ** 2 * varphi + 0.5 * X ** 2 * np.diag(J) - log_z)

        probs = np.exp(exponent - max_exp
                       - np.log(np.sum(np.exp(exponent - max_exp), axis=2, keepdims=True)))
        expected_x = np.sum(probs * self.states, axis=2)

        diff = X - expected_x
        grad_h = -(np.sum(diff, axis=0) - self.l2_lambda * h)

        grad_J_full = -((X.T @ diff) - self.l2_lambda * J)
        grad_J_sym = grad_J_full + grad_J_full.T

        expected_x2 = np.sum(probs * self.states ** 2, axis=2)
        diff_x2 = X ** 2 - expected_x2
        grad_J_diag = -(0.5 * np.sum(diff_x2, axis=0) - self.l2_lambda * np.diag(J))
        diag_indices = np.diag_indices(n_nodes)
        grad_J_sym[diag_indices] = grad_J_diag

        tri_indices = np.triu_indices(n_nodes, k=0)
        grad_J_flat = grad_J_sym[tri_indices]

        grad_K_full = -(((X ** 2).T @ diff_x2) - self.l2_lambda * K)
        grad_K_sym = grad_K_full + grad_K_full.T
        trinodiag_indices = np.triu_indices(n_nodes, k=1)
        grad_K_flat = grad_K_sym[trinodiag_indices]

        loss = -(log_lik - 0.5 * self.l2_lambda
                                * (np.sum(K ** 2) / 2 + np.sum(J ** 2) / 2 + np.sum(h ** 2)))
        return loss, np.concatenate([grad_h, grad_J_flat, grad_K_flat])

    def fit(self, X, iicc='meanfield', J0=None, h0=None, K0=None, verbose=False):
        """Fit (J, h, K) by maximizing the pseudo-likelihood with L-BFGS-B."""
        n_samples, n_nodes = X.shape
        n_j_params = n_nodes * (n_nodes + 1) // 2
        n_k_params = n_nodes * (n_nodes - 1) // 2
        tridiag_indices = np.triu_indices(n_nodes, k=0)
        initial_theta = np.zeros(n_nodes + n_j_params + n_k_params)

        match iicc:
            case 'meanfield':
                mu_X = np.mean(X, axis=0)
                C_X = np.cov(X.T)
                J0 = -np.linalg.inv(C_X)
                h0 = -J0 @ mu_X
                initial_theta[:n_nodes] = h0
                initial_theta[n_nodes:n_nodes + n_j_params] = J0[tridiag_indices]
            case 'given':
                assert h0 is not None and J0 is not None and K0 is not None
                initial_theta[:n_nodes] = h0
                initial_theta[n_nodes:n_nodes + n_j_params] = J0[tridiag_indices]
                initial_theta[n_nodes + n_j_params:] = K0[np.triu_indices(n_nodes, k=1)]

        res = minimize(
            fun=self._objective,
            x0=initial_theta,
            args=(X,),
            method='L-BFGS-B', tol=1.0E-8,
            jac=True,
            options={'disp': True, 'maxiter': 1000}
        )

        if verbose:
            print(res.message)

        self.J_fit, self.h_fit, self.K_fit = self._unpack_params(res.x, n_nodes)
        return self.J_fit, self.h_fit, self.K_fit

    def naif_fit_euler(self, X, niterations=1000, learning_rate=1E-3,
                       iicc='meanfield', J0=None, h0=None, K0=None):
        """Naive gradient-descent variant of :meth:`fit`."""
        n_samples, n_nodes = X.shape
        n_j_params = n_nodes * (n_nodes + 1) // 2
        n_k_params = n_nodes * (n_nodes - 1) // 2
        initial_theta = np.zeros(n_nodes + n_j_params + n_k_params)
        tri_indices = np.triu_indices(n_nodes, k=0)

        match iicc:
            case 'meanfield':
                mu_X = np.mean(X, axis=0)
                C_X = np.cov(X.T)
                J0 = -np.linalg.inv(C_X)
                h0 = -J0 @ mu_X
                initial_theta[:n_nodes] = h0
                initial_theta[n_nodes:n_nodes + n_j_params] = J0[tri_indices]
            case 'given':
                assert h0 is not None and J0 is not None and K0 is not None
                initial_theta[:n_nodes] = h0
                initial_theta[n_nodes:n_nodes + n_j_params] = J0[tri_indices]
                initial_theta[n_nodes + n_j_params:] = K0[np.triu_indices(n_nodes, k=1)]

        theta = np.copy(initial_theta)
        for n in tqdm(range(niterations)):
            loss, grad = self._objective(theta, X)
            theta = theta - learning_rate * grad
            self.J_fit, self.h_fit, self.K_fit = self._unpack_params(theta, n_nodes)
            print(n, self.loglikelihood(X) / n_samples, loss)
        return self.J_fit, self.h_fit, self.K_fit

    def loglikelihood(self, X):
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
        log_lik = np.sum(X * phi + 0.5 * X ** 2 * np.diag(J) + X ** 2 * varphi - log_z)
        return log_lik / n_samples

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
