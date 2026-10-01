"""
Benchmark models: EWMA realized covariance and BEKK-GARCH.
Custom implementation — Context7 confirmed arch library only provides univariate GARCH.
Follows Chiriac & Voev (2010) Section 5 benchmarks.
"""
import numpy as np
from scipy.optimize import minimize
import scipy.linalg as sla
from typing import Optional, Dict, Tuple

try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

# ---------------------------------------------------------------------------
# Numba-JIT BEKK recursion kernels
# ---------------------------------------------------------------------------

if _HAS_NUMBA:
    from numba import njit as _njit, float64

    @_njit(cache=True)
    def _bekk_sigma_diag(
        Omega: np.ndarray,    # (n, n)
        G_diag: np.ndarray,   # (p, n) diagonal entries of G matrices
        F_diag: np.ndarray,   # (q, n) diagonal entries of F matrices
        returns: np.ndarray,  # (T, n)
        S0: np.ndarray,       # (n, n) initialisation
    ) -> np.ndarray:
        """Fast diagonal-BEKK Sigma_t recursion (numba JIT).

        For diagonal G and F: G @ S @ G.T = g² ⊙ S (outer product scale).
        """
        T, n = returns.shape
        p = G_diag.shape[0]
        q = F_diag.shape[0]
        Sigma = np.empty((T, n, n))
        Sigma[0] = S0

        for t in range(1, T):
            S = Omega.copy()
            # GARCH terms
            for i in range(p):
                lag = t - 1 - i
                Sl = Sigma[lag] if lag >= 0 else S0
                g = G_diag[i]
                for r in range(n):
                    for c in range(n):
                        S[r, c] += g[r] * g[c] * Sl[r, c]
            # ARCH terms
            for j in range(q):
                lag = t - 1 - j
                if lag >= 0:
                    y = returns[lag]
                else:
                    y = np.zeros(n)   # conservative init
                f = F_diag[j]
                for r in range(n):
                    for c in range(n):
                        S[r, c] += f[r] * f[c] * y[r] * y[c]
            # Symmetrise
            for r in range(n):
                for c in range(r):
                    v = (S[r, c] + S[c, r]) * 0.5
                    S[r, c] = v
                    S[c, r] = v
            Sigma[t] = S

        return Sigma

    @_njit(cache=True)
    def _bekk_sigma_full(
        Omega: np.ndarray,    # (n, n)
        G_arr: np.ndarray,    # (p, n, n)
        F_arr: np.ndarray,    # (q, n, n)
        returns: np.ndarray,  # (T, n)
        S0: np.ndarray,       # (n, n)
    ) -> np.ndarray:
        """Fast full-BEKK Sigma_t recursion (numba JIT)."""
        T, n = returns.shape
        p = G_arr.shape[0]
        q = F_arr.shape[0]
        Sigma = np.empty((T, n, n))
        Sigma[0] = S0

        for t in range(1, T):
            S = Omega.copy()
            # GARCH terms: G @ Sl @ G.T
            for i in range(p):
                lag = t - 1 - i
                Sl = Sigma[lag] if lag >= 0 else S0
                G = G_arr[i]
                GS = np.zeros((n, n))
                for r in range(n):
                    for c in range(n):
                        for k in range(n):
                            GS[r, c] += G[r, k] * Sl[k, c]
                for r in range(n):
                    for c in range(n):
                        v = 0.0
                        for k in range(n):
                            v += GS[r, k] * G[c, k]
                        S[r, c] += v
            # ARCH terms: F @ yy' @ F.T
            for j in range(q):
                lag = t - 1 - j
                y = returns[lag] if lag >= 0 else np.zeros(n)
                F = F_arr[j]
                Fy = np.zeros(n)
                for r in range(n):
                    for k in range(n):
                        Fy[r] += F[r, k] * y[k]
                for r in range(n):
                    for c in range(n):
                        S[r, c] += Fy[r] * Fy[c]
            # Symmetrise
            for r in range(n):
                for c in range(r):
                    v = (S[r, c] + S[c, r]) * 0.5
                    S[r, c] = v
                    S[c, r] = v
            Sigma[t] = S

        return Sigma

else:
    def _bekk_sigma_diag(Omega, G_diag, F_diag, returns, S0):
        """Pure-Python fallback for diagonal BEKK."""
        T, n = returns.shape
        p, q = G_diag.shape[0], F_diag.shape[0]
        Sigma = np.empty((T, n, n))
        Sigma[0] = S0
        for t in range(1, T):
            S = Omega.copy()
            for i in range(p):
                lag = t - 1 - i
                Sl = Sigma[lag] if lag >= 0 else S0
                g = G_diag[i]
                S += np.outer(g, g) * Sl
            for j in range(q):
                lag = t - 1 - j
                y = returns[lag] if lag >= 0 else np.zeros(n)
                f = F_diag[j]
                S += np.outer(f * y, f * y)
            Sigma[t] = (S + S.T) * 0.5
        return Sigma

    def _bekk_sigma_full(Omega, G_arr, F_arr, returns, S0):
        """Pure-Python fallback for full BEKK."""
        T, n = returns.shape
        p, q = G_arr.shape[0], F_arr.shape[0]
        Sigma = np.empty((T, n, n))
        Sigma[0] = S0
        for t in range(1, T):
            S = Omega.copy()
            for i in range(p):
                lag = t - 1 - i
                Sl = Sigma[lag] if lag >= 0 else S0
                G = G_arr[i]
                S += G @ Sl @ G.T
            for j in range(q):
                lag = t - 1 - j
                y = returns[lag] if lag >= 0 else np.zeros(n)
                F = F_arr[j]
                Fy = F @ y
                S += np.outer(Fy, Fy)
            Sigma[t] = (S + S.T) * 0.5
        return Sigma


# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------

class EWMAModel:
    """
    EWMA realized covariance: E(R_{t+1} | F_t) = (1-lambda)*R_t + lambda*E(R_t | F_{t-1})
    Multi-step forecast: E(R_{t+h} | F_t) = S_{t+1} for all h >= 1 (random-walk in expectation).
    Default lambda = 0.94 (RiskMetrics standard).
    """

    def __init__(self, lam: float = 0.94):
        self.lam = lam
        self.fitted = False

    def fit(self, R_t: np.ndarray) -> Dict:
        """Compute EWMA filtered covariance path.

        Parameters
        ----------
        R_t : (T, n, n) realized covariance matrices

        Returns
        -------
        dict with 'S_t' — EWMA filtered covariance path (T, n, n)
        """
        T, n, _ = R_t.shape
        S_t = np.empty_like(R_t)
        S_t[0] = R_t[0]
        lam = self.lam
        for t in range(1, T):
            S_t[t] = (1.0 - lam) * R_t[t - 1] + lam * S_t[t - 1]
        self.S_t = S_t
        self.R_t = R_t
        self.fitted = True
        return {'S_t': S_t}

    def forecast(self, t_origin: int, h: int) -> np.ndarray:
        """Forecast E(R_{t_origin + h} | F_{t_origin}).

        EWMA h-step forecast equals the one-step-ahead EWMA level for all h >= 1
        (the filtered level S_{t_origin+1} is the conditional expectation at any horizon).

        Parameters
        ----------
        t_origin : last observed index (0-based)
        h        : horizon (>= 1, not used — same forecast for all h)

        Returns
        -------
        (n, n) symmetric forecast matrix
        """
        assert self.fitted, 'Call fit() first.'
        n = self.R_t.shape[1]
        lam = self.lam
        # One additional EWMA step from the last observed matrix
        last_S = self.S_t[t_origin]
        last_R = self.R_t[t_origin]
        S_next = (1.0 - lam) * last_R + lam * last_S
        return (S_next + S_next.T) * 0.5


# ---------------------------------------------------------------------------
# BEKK-GARCH(p, q) on daily returns
# ---------------------------------------------------------------------------

class BEKKGARCHModel:
    """
    BEKK-GARCH model on daily returns:

        Sigma_t = D0 D0' + sum_{i=1}^p G_i Sigma_{t-i} G_i'
                         + sum_{j=1}^q F_j (y_{t-j} y_{t-j}') F_j'

    D0 is lower-triangular (positive diagonal), G_i and F_j are full or diagonal (n,n).
    Estimated by Gaussian quasi-maximum likelihood.

    Custom implementation — Context7 confirmed no library provides full multivariate BEKK.
    """

    def __init__(self, n: int, p: int = 1, q: int = 1, diagonal: bool = False):
        self.n = n
        self.p = p   # GARCH lags
        self.q = q   # ARCH lags
        self.diagonal = diagonal
        self.fitted = False

    # ------------------------------------------------------------------
    # Parameter counting
    # ------------------------------------------------------------------

    def _n_params(self) -> int:
        n = self.n
        n_D = n * (n + 1) // 2
        stride = n if self.diagonal else n * n
        return n_D + (self.p + self.q) * stride

    # ------------------------------------------------------------------
    # Parameter packing / unpacking
    # ------------------------------------------------------------------

    def _unpack_params(self, psi: np.ndarray) -> Tuple:
        n = self.n
        idx = 0

        # D0 — lower triangular with log-transformed positive diagonal
        n_D = n * (n + 1) // 2
        d_vec = psi[idx: idx + n_D]; idx += n_D
        D0 = np.zeros((n, n))
        tril_rows, tril_cols = np.tril_indices(n)
        D0[tril_rows, tril_cols] = d_vec
        # Replace diagonal with exp so D0[i,i] > 0
        diag_r, diag_c = np.diag_indices(n)
        D0[diag_r, diag_c] = np.exp(D0[diag_r, diag_c])

        # G matrices (GARCH)
        stride = n if self.diagonal else n * n
        G_list = []
        for _ in range(self.p):
            raw = psi[idx: idx + stride]; idx += stride
            G_list.append(np.diag(raw) if self.diagonal else raw.reshape(n, n))

        # F matrices (ARCH)
        F_list = []
        for _ in range(self.q):
            raw = psi[idx: idx + stride]; idx += stride
            F_list.append(np.diag(raw) if self.diagonal else raw.reshape(n, n))

        return D0, G_list, F_list

    def _pack_D0(self, D0: np.ndarray) -> np.ndarray:
        """Pack D0 into the raw parameter vector (log-diagonal)."""
        n = self.n
        raw = D0.copy()
        diag_r, diag_c = np.diag_indices(n)
        raw[diag_r, diag_c] = np.log(np.maximum(np.diag(D0), 1e-9))
        return raw[np.tril_indices(n)]

    # ------------------------------------------------------------------
    # Covariance recursion
    # ------------------------------------------------------------------

    def _compute_Sigma(self, psi: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Recursively compute the conditional covariance path.

        Dispatches to numba-JIT kernels when available for a ~10× speed-up.

        Parameters
        ----------
        psi     : flat parameter vector
        returns : (T, n) daily returns

        Returns
        -------
        Sigma_t : (T, n, n) conditional covariance matrices
        """
        T, n = returns.shape
        D0, G_list, F_list = self._unpack_params(psi)
        Omega = D0 @ D0.T   # (n, n) unconditional base

        # Initialise with sample covariance from early observations
        init_len = min(max(20, n + 1), T)
        S0 = np.cov(returns[:init_len].T)
        if S0.ndim == 0:
            S0 = np.atleast_2d(S0)
        ev_min = np.linalg.eigvalsh(S0).min()
        if ev_min <= 0:
            S0 += (-ev_min + 1e-8) * np.eye(n)

        p, q = len(G_list), len(F_list)
        ret_c = np.ascontiguousarray(returns, dtype=np.float64)

        if self.diagonal:
            # Extract diagonals → (p, n) and (q, n) arrays
            G_diag = np.array([np.diag(G) for G in G_list], dtype=np.float64)
            F_diag = np.array([np.diag(F) for F in F_list], dtype=np.float64)
            return _bekk_sigma_diag(
                np.ascontiguousarray(Omega, dtype=np.float64),
                G_diag,
                F_diag,
                ret_c,
                np.ascontiguousarray(S0, dtype=np.float64),
            )
        else:
            G_arr = np.array([G for G in G_list], dtype=np.float64)
            F_arr = np.array([F for F in F_list], dtype=np.float64)
            return _bekk_sigma_full(
                np.ascontiguousarray(Omega, dtype=np.float64),
                G_arr,
                F_arr,
                ret_c,
                np.ascontiguousarray(S0, dtype=np.float64),
            )

    # ------------------------------------------------------------------
    # Gaussian quasi-log-likelihood
    # ------------------------------------------------------------------

    def _gaussian_loglik(self, psi: np.ndarray, returns: np.ndarray) -> float:
        """Negative multivariate Gaussian log-likelihood (for minimisation).

        Vectorised over T using batched numpy slogdet / solve / einsum to
        avoid per-timestep Python overhead (key BEKK speed-up).
        """
        T, n = returns.shape
        try:
            Sigma_t = self._compute_Sigma(psi, returns)
        except Exception:
            return 1e10

        log2pi_n = n * np.log(2.0 * np.pi)

        # --- Batched log-determinant (T,) ---
        signs, logdets = np.linalg.slogdet(Sigma_t)
        if np.any(signs <= 0) or not np.all(np.isfinite(logdets)):
            return 1e10

        # --- Batched quadratic form: (T,n,n) solve (T,n) → (T,n) ---
        try:
            # np.linalg.solve broadcasts over batch dim
            S_inv_y = np.linalg.solve(Sigma_t, returns)   # (T, n)
        except np.linalg.LinAlgError:
            return 1e10

        quads = np.einsum('ti,ti->t', returns, S_inv_y)   # (T,)
        if not np.all(np.isfinite(quads)):
            return 1e10

        loglik = float(np.sum(-0.5 * (log2pi_n + logdets + quads)))
        return -loglik   # negative for minimisation

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray, verbose: bool = False,
            warm_start_psi: Optional[np.ndarray] = None,
            maxiter_cold: int = 300, maxiter_warm: int = 50) -> Dict:
        """Fit BEKK-GARCH model by Gaussian QMLE.

        Parameters
        ----------
        returns        : (T, n) daily returns
        verbose        : print optimiser progress
        warm_start_psi : optional parameter vector from a previous fit for
            warm-starting the optimiser (dramatically speeds up OOS loops)
        maxiter_cold   : L-BFGS-B max iterations for cold (no warm-start) fits
        maxiter_warm   : L-BFGS-B max iterations for warm-started fits

        Returns
        -------
        dict with loglik, n_params, psi, Sigma_t, converged
        """
        n = self.n
        n_params = self._n_params()

        # --- Initial values ---
        if warm_start_psi is not None and len(warm_start_psi) == n_params:
            psi0 = warm_start_psi.copy()
        else:
            psi0 = np.zeros(n_params)
            idx = 0
            n_D = n * (n + 1) // 2

            sample_cov = np.cov(returns.T)
            ev_min = np.linalg.eigvalsh(sample_cov).min()
            if ev_min <= 0:
                sample_cov += (-ev_min + 1e-8) * np.eye(n)
            try:
                D0_init = np.linalg.cholesky(sample_cov * 0.1)
            except np.linalg.LinAlgError:
                D0_init = np.eye(n) * 0.01

            psi0[idx: idx + n_D] = self._pack_D0(D0_init); idx += n_D

            stride = n if self.diagonal else n * n
            for _ in range(self.p):
                if self.diagonal:
                    psi0[idx: idx + stride] = 0.85
                else:
                    psi0[idx: idx + stride] = np.eye(n).flatten() * 0.85
                idx += stride
            for _ in range(self.q):
                if self.diagonal:
                    psi0[idx: idx + stride] = 0.15
                else:
                    psi0[idx: idx + stride] = np.eye(n).flatten() * 0.15
                idx += stride

        def objective(psi):
            return self._gaussian_loglik(psi, returns)

        maxiter = maxiter_warm if warm_start_psi is not None else maxiter_cold
        res = minimize(
            objective, psi0,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-7, 'gtol': 1e-5, 'disp': verbose},
        )

        self.psi_hat = res.x
        self.Sigma_t = self._compute_Sigma(res.x, returns)
        self.returns = returns
        self.fitted = True
        loglik = -res.fun

        return {
            'loglik': loglik,
            'psi': res.x,
            'n_params': n_params,
            'Sigma_t': self.Sigma_t,
            'converged': res.success,
        }

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(self, t_origin: int, h: int) -> np.ndarray:
        """Recursive h-step-ahead BEKK conditional covariance forecast.

        For h=1: uses actual returns and Sigma at origin.
        For h>1: E[y_{t+k} y_{t+k}' | F_t] = E[Sigma_{t+k} | F_t] (zero-mean returns).

        Parameters
        ----------
        t_origin : last observed index (0-based)
        h        : forecast horizon

        Returns
        -------
        (n, n) symmetric forecast covariance matrix
        """
        assert self.fitted, 'Call fit() first.'
        D0, G_list, F_list = self._unpack_params(self.psi_hat)
        Omega = D0 @ D0.T
        n = self.n

        T_obs = len(self.returns)
        # Build mutable history lists (observed + forecasted)
        Sigma_hist = list(self.Sigma_t[: t_origin + 1])   # [0 .. t_origin]
        y_hist     = list(self.returns[: t_origin + 1])   # [0 .. t_origin]

        for step in range(1, h + 1):
            # Forecast Sigma at t_origin + step
            S = Omega.copy()
            for i, G in enumerate(G_list):
                lag = t_origin + step - 1 - i   # index of lagged Sigma
                if lag >= 0:
                    Sl = Sigma_hist[lag] if lag < len(Sigma_hist) else Sigma_hist[-1]
                else:
                    Sl = Sigma_hist[0]
                S += G @ Sl @ G.T
            for j, F in enumerate(F_list):
                lag = t_origin + step - 1 - j   # index of lagged y
                if lag >= 0 and lag < len(y_hist):
                    # Observed return: use y y'
                    y_l = y_hist[lag]
                    S += F @ np.outer(y_l, y_l) @ F.T
                elif lag >= len(y_hist):
                    # Future return: replace y y' with E[Sigma_{lag}|F_t]
                    Sl = Sigma_hist[lag] if lag < len(Sigma_hist) else Sigma_hist[-1]
                    S += F @ Sl @ F.T
                else:
                    # lag < 0 (before sample): use first available Sigma
                    S += F @ Sigma_hist[0] @ F.T
            S = (S + S.T) * 0.5
            Sigma_hist.append(S)

        return Sigma_hist[-1]
