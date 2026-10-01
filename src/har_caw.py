"""
HAR-CAW model: Heterogeneous Autoregressive Realized Covariance.
Custom implementation — Context7 found no library equivalent.
Follows Chiriac & Voev (2010) Section 3.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
import scipy.linalg as sla
from typing import Tuple, Optional, Dict, List

from matrix_ops import vech, vech_to_matrix, elimination_matrix, duplication_matrix
from fast_recursions import har_compute_St_vectorized


class HARCAWModel:
    """
    HAR-CAW model:
      S_t = CC' + A R_{t-1} A' + A_w R_bar_w_{t-1} A_w'
            + A_bw R_bar_bw_{t-1} A_bw' + A_m R_bar_m_{t-1} A_m'

    where:
      R_bar_w_{t-1}  = sum_{tau=t-5}^{t-1} R_tau  (5-day sum)
      R_bar_bw_{t-1} = sum_{tau=t-10}^{t-1} R_tau (10-day sum)
      R_bar_m_{t-1}  = sum_{tau=t-20}^{t-1} R_tau (20-day sum)

    R_t | F_{t-1} ~ W_n(nu, S_t / nu)

    Parameters
    ----------
    n        : number of assets
    diagonal : if True, coefficient matrices A, A_w, A_bw, A_m are diagonal
    """

    def __init__(self, n: int, diagonal: bool = False):
        self.n = n
        self.diagonal = diagonal
        self.k = n * (n + 1) // 2

        # Set after fitting
        self.psi_hat: Optional[np.ndarray] = None
        self.nu_hat: Optional[float] = None
        self.C_hat: Optional[np.ndarray] = None
        self.A_hat: Optional[np.ndarray] = None
        self.A_w_hat: Optional[np.ndarray] = None
        self.A_bw_hat: Optional[np.ndarray] = None
        self.A_m_hat: Optional[np.ndarray] = None
        self.S_t_hat: Optional[np.ndarray] = None
        self.fitted = False

    # ---------------------------------------------------------------
    # Parameter count
    # ---------------------------------------------------------------

    def _n_params(self) -> int:
        n = self.n
        n_nu = 1
        n_C  = n * (n + 1) // 2
        stride = n if self.diagonal else n * n
        n_coef = 4 * stride  # A, A_w, A_bw, A_m
        return n_nu + n_C + n_coef

    # ---------------------------------------------------------------
    # Pack / unpack
    # ---------------------------------------------------------------

    def _unpack_params(self, psi: np.ndarray) -> Tuple:
        """Unpack flat parameter vector into (nu, C, A, A_w, A_bw, A_m)."""
        n = self.n
        idx = 0

        nu = (n + 1.0) + np.exp(np.clip(psi[idx], -30.0, 30.0))
        idx += 1

        n_C = n * (n + 1) // 2
        c_raw = psi[idx: idx + n_C].copy()
        idx += n_C
        C = np.zeros((n, n))
        tril = np.tril_indices(n)
        C[tril] = c_raw
        for d in range(n):
            C[d, d] = np.exp(np.clip(C[d, d], -40.0, 40.0))

        stride = n if self.diagonal else n * n

        def _get_matrix(start: int) -> np.ndarray:
            raw = psi[start: start + stride]
            if self.diagonal:
                return np.diag(raw)
            return raw.reshape(n, n)

        A     = _get_matrix(idx); idx += stride
        A_w   = _get_matrix(idx); idx += stride
        A_bw  = _get_matrix(idx); idx += stride
        A_m   = _get_matrix(idx); idx += stride

        return nu, C, A, A_w, A_bw, A_m

    def _pack_params(self, nu: float, C: np.ndarray,
                     A: np.ndarray, A_w: np.ndarray,
                     A_bw: np.ndarray, A_m: np.ndarray) -> np.ndarray:
        """Pack model components into flat parameter vector."""
        n = self.n
        parts: List[np.ndarray] = []

        parts.append(np.array([np.log(max(nu - (n + 1.0), 1e-8))]))

        tril = np.tril_indices(n)
        rows, cols = tril
        c_vec = C[tril].copy()
        diag_mask = rows == cols
        c_vec[diag_mask] = np.log(np.maximum(c_vec[diag_mask], 1e-8))
        parts.append(c_vec)

        def _flatten(M: np.ndarray) -> np.ndarray:
            return np.diag(M) if self.diagonal else M.flatten()

        parts.append(_flatten(A))
        parts.append(_flatten(A_w))
        parts.append(_flatten(A_bw))
        parts.append(_flatten(A_m))

        return np.concatenate(parts)

    # ---------------------------------------------------------------
    # S_t computation (direct HAR equation — no recursion needed)
    # ---------------------------------------------------------------

    def _compute_St(self, psi: np.ndarray, R_t: np.ndarray, har_sums: dict) -> Optional[np.ndarray]:
        """Compute S_t with a vectorized einsum (no sequential dependency in HAR).

        S_t = CC' + A R_{t-1} A' + A_w R_bar_w A_w' + A_bw R_bar_bw A_bw'
              + A_m R_bar_m A_m'

        Because S_t does NOT depend on S_{t-1}, all t are computed simultaneously
        via `har_compute_St_vectorized`.

        Returns
        -------
        S_t : (T, n, n) or None if any S_t is not positive definite
        """
        nu, C, A, A_w, A_bw, A_m = self._unpack_params(psi)
        T, n, _ = R_t.shape
        CC = C @ C.T

        # Build lagged daily R: R_lag1[t] = R_t[t-1]; t=0 → sample mean
        R_init = np.mean(R_t[:max(1, 20)], axis=0)
        R_init = (R_init + R_init.T) / 2.0
        R_lag1 = np.empty_like(R_t)
        R_lag1[0] = R_init
        R_lag1[1:] = R_t[:-1]

        # Replace zero-filled fallback entries in har_sums with multiples of R_init
        R_w = har_sums['w5'].copy()
        R_bw = har_sums['w10'].copy()
        R_m = har_sums['w20'].copy()
        for t in range(T):
            if not np.any(R_w[t]):
                R_w[t] = R_init
            if not np.any(R_bw[t]):
                R_bw[t] = 2.0 * R_init
            if not np.any(R_m[t]):
                R_m[t] = 4.0 * R_init

        # Stack into (K, T, n, n) for the vectorized kernel
        A_mats  = np.stack([A, A_w, A_bw, A_m], axis=0)         # (4, n, n)
        R_lags  = np.stack([R_lag1, R_w, R_bw, R_m], axis=0)    # (4, T, n, n)

        S_t = har_compute_St_vectorized(CC, A_mats, R_lags)

        if np.linalg.eigvalsh(S_t).min() <= 0:
            return None
        return S_t

    # ---------------------------------------------------------------
    # Wishart log-likelihood
    # ---------------------------------------------------------------

    def _wishart_loglik(self, psi: np.ndarray, R_t: np.ndarray, har_sums: dict) -> float:
        """Compute negative Wishart log-likelihood.

        Vectorised over T: batched slogdet / inv / einsum replace the Python loop,
        matching the implementation in caw_model.py.
        """
        nu, C, A, A_w, A_bw, A_m = self._unpack_params(psi)
        T, n, _ = R_t.shape

        S_t = self._compute_St(psi, R_t, har_sums)
        if S_t is None:
            return 1e10

        # --- Batched log-determinants ---
        signs_S, ld_S = np.linalg.slogdet(S_t)
        signs_R, ld_R = np.linalg.slogdet(R_t)
        if (signs_S <= 0).any() or (signs_R <= 0).any():
            return 1e10

        # --- Batched trace: tr(nu S_t^{-1} R_t) ---
        S_inv = np.linalg.inv(S_t)                          # (T, n, n)
        traces = nu * np.einsum('tij,tji->t', S_inv, R_t)  # (T,)

        const = (
            -(nu * n / 2.0) * np.log(2.0)
            - (n * (n - 1) / 4.0) * np.log(np.pi)
            - sum(gammaln((nu + 1.0 - i) / 2.0) for i in range(1, n + 1))
        )

        ld_S_over_nu = ld_S - n * np.log(nu)

        loglik = (
            T * const
            - (nu / 2.0) * ld_S_over_nu.sum()
            + ((nu - n - 1.0) / 2.0) * ld_R.sum()
            - 0.5 * traces.sum()
        )

        if not np.isfinite(loglik):
            return 1e10

        return -loglik

    # ---------------------------------------------------------------
    # Fitting
    # ---------------------------------------------------------------

    def _init_params(self, R_t: np.ndarray) -> np.ndarray:
        """Initialize parameter vector from data."""
        n = self.n
        nu0 = float(n) + 10.0
        log_nu = np.log(max(nu0 - (n + 1.0), 1e-4))

        R_mean = np.mean(R_t, axis=0)
        try:
            C0 = np.linalg.cholesky(R_mean)
        except np.linalg.LinAlgError:
            C0 = np.eye(n) * np.sqrt(np.mean(np.diag(R_mean)))

        tril = np.tril_indices(n)
        rows, cols = tril
        c_vec = C0[tril].copy()
        diag_mask = rows == cols
        c_vec[diag_mask] = np.log(np.maximum(c_vec[diag_mask], 1e-8))

        psi0 = [log_nu] + list(c_vec)

        # Initialize A, A_w, A_bw, A_m with small diagonal values
        init_diag = 0.2
        for _ in range(4):
            if self.diagonal:
                psi0.extend([init_diag] * n)
            else:
                psi0.extend((np.eye(n) * init_diag).flatten())

        return np.array(psi0)

    def fit(self, R_t: np.ndarray, har_sums: dict, verbose: bool = True,
            warm_start_psi: Optional[np.ndarray] = None,
            maxiter: int = 300,
            maxiter_warm: Optional[int] = None) -> Dict:
        """Fit HAR-CAW model by maximum likelihood.

        Parameters
        ----------
        R_t            : ndarray (T, n, n)
        har_sums       : dict {'w5': ..., 'w10': ..., 'w20': ...}, each (T, n, n)
        verbose        : print optimization progress
        warm_start_psi : optional parameter vector from a previous fit
        maxiter        : max L-BFGS-B iterations for cold start
        maxiter_warm   : max iterations when warm_start_psi is given;
                         defaults to min(60, maxiter)

        Returns
        -------
        dict with loglik, n_params, psi, S_t, converged
        """
        n_params = self._n_params()
        if warm_start_psi is not None and len(warm_start_psi) == n_params:
            psi0 = warm_start_psi.copy()
            _maxiter = maxiter_warm if maxiter_warm is not None else min(60, maxiter)
        else:
            psi0 = self._init_params(R_t)
            _maxiter = maxiter

        result = minimize(
            self._wishart_loglik,
            psi0,
            args=(R_t, har_sums),
            method='L-BFGS-B',
            options={'maxiter': _maxiter, 'maxfun': _maxiter * 3,
                     'ftol': 1e-8, 'gtol': 1e-5, 'disp': False}
        )

        self.psi_hat = result.x
        nu, C, A, A_w, A_bw, A_m = self._unpack_params(result.x)
        self.nu_hat = nu
        self.C_hat = C
        self.A_hat = A
        self.A_w_hat = A_w
        self.A_bw_hat = A_bw
        self.A_m_hat = A_m
        self.S_t_hat = self._compute_St(result.x, R_t, har_sums)
        self.R_t_fit = R_t
        self.har_sums_fit = har_sums
        self.fitted = True

        loglik = -result.fun

        return {
            'loglik': loglik,
            'n_params': n_params,
            'psi': result.x,
            'nu': nu,
            'S_t': self.S_t_hat,
            'converged': result.success
        }

    # ---------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------

    def standardized_residuals(self, R_t: np.ndarray, har_sums: dict) -> np.ndarray:
        """Compute vech-space standardized residuals."""
        assert self.fitted
        T, n, _ = R_t.shape
        k = self.k
        nu = self.nu_hat
        tril = np.tril_indices(n)
        rows, cols = tril

        resids = np.zeros((T, k))
        for t in range(T):
            St = self.S_t_hat[t]
            Rt = R_t[t]
            r_t = vech(Rt)
            s_t = vech(St)

            Var_t = np.zeros((k, k))
            for p_idx in range(k):
                i, j = rows[p_idx], cols[p_idx]
                for q_idx in range(k):
                    l, m = rows[q_idx], cols[q_idx]
                    Var_t[p_idx, q_idx] = (
                        (1.0 / nu) * (St[i, l] * St[j, m] + St[i, m] * St[j, l])
                    )

            try:
                chol_Var = np.linalg.cholesky(Var_t)
                resids[t] = np.linalg.solve(chol_Var, r_t - s_t)
            except np.linalg.LinAlgError:
                resids[t] = r_t - s_t

        return resids

    def schwarz_criterion(self, loglik: float, T: int) -> float:
        """Schwarz (BIC) = -2 * L + k * ln(T)."""
        return -2.0 * loglik + self._n_params() * np.log(T)

    # ---------------------------------------------------------------
    # Forecasting
    # ---------------------------------------------------------------

    def forecast(self, R_t: np.ndarray, har_sums: dict, h: int = 1) -> np.ndarray:
        """Forecast E(R_{t+h} | F_t) using iterative HAR recursion.

        For multi-step ahead, replace future unknown lags with forecasts.

        Parameters
        ----------
        R_t      : (T, n, n)
        har_sums : aligned rolling sums
        h        : horizon

        Returns
        -------
        forecast matrix (n, n)
        """
        assert self.fitted
        T, n, _ = R_t.shape
        nu = self.nu_hat

        # For h=1, S_{T+1} = CC' + A R_T A' + A_w R_bar_w_T A_w' + ...
        # For h>1, we iterate: E(R_{t+h}|F_t) = E(S_{t+h}|F_t)
        # Simplified: use S_T (the last fitted value) as anchor for all steps
        # because HAR is a direct (non-recursive GARCH) model
        S_last = self.S_t_hat[-1]

        # Build extended history for multi-step recursion
        R_hist = list(R_t)
        for step in range(1, h + 1):
            # The 'forecast' at step is E(R_{T+step}|F_T) = E(S_{T+step}|F_T)
            # Use last available R for lags that exceed known data
            R_lag1 = R_hist[-1]  # Best proxy for R_{T+step-1}

            # Rolling sums: extend with last known value as proxy
            if step == 1:
                R_w_use   = har_sums['w5'][-1]
                R_bw_use  = har_sums['w10'][-1]
                R_m_use   = har_sums['w20'][-1]
            else:
                # For multi-step, approximate by using last known S as proxy
                R_w_use  = 5  * S_last
                R_bw_use = 10 * S_last
                R_m_use  = 20 * S_last

            St = self.C_hat @ self.C_hat.T
            St += self.A_hat @ R_lag1 @ self.A_hat.T
            St += self.A_w_hat @ R_w_use @ self.A_w_hat.T
            St += self.A_bw_hat @ R_bw_use @ self.A_bw_hat.T
            St += self.A_m_hat @ R_m_use @ self.A_m_hat.T
            St = (St + St.T) / 2

            R_hist.append(St)

        return R_hist[-1]
