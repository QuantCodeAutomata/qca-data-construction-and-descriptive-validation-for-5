"""
MIDAS-CAW model: Mixed-Data Sampling augmented Conditional Autoregressive Wishart.
Custom implementation — Context7 found no library equivalent.
Follows Chiriac & Voev (2010) Section 3.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
import scipy.linalg as sla
from typing import Tuple, Optional, Dict, List

from fast_recursions import (midas_compute_St_star, midas_loglik_star_loop,
                             midas_loglik_fused)


class MIDASCAWModel:
    """
    Decomposition: S_t = C_t S_t^* C_t'
    Long-run: M_t = C_t C_t' = C_bar C_bar' + theta * sum_{ell=1}^L phi_ell(omega) * R_bar_{t,ell}
    Short-run: S_t^* recursion with E(S_t^*) = I_n imposed via intercept
    R_t | F_{t-1} ~ W_n(nu, S_t / nu)

    Parameters: m=20, L=12, common theta>=0, common omega>1
    """

    def __init__(self, n: int, p: int, q: int, diagonal: bool = False,
                 m: int = 20, L: int = 12):
        self.n = n
        self.p = p   # GARCH lags (B matrices, acting on past S_t^*)
        self.q = q   # ARCH lags  (A matrices, acting on past R_t^*)
        self.diagonal = diagonal
        self.m = m
        self.L = L
        # Set after fitting
        self.psi_hat: Optional[np.ndarray] = None
        self.fitted = False

    # ------------------------------------------------------------------
    # MIDAS weights
    # ------------------------------------------------------------------

    def _midas_weights(self, omega: float) -> np.ndarray:
        """Normalised Beta weights phi_ell(omega) = (1 - ell/L)^(omega-1) / sum."""
        ells = np.arange(1, self.L + 1)
        raw = (1.0 - ells / self.L) ** (omega - 1.0)
        total = raw.sum()
        if total <= 0.0 or not np.isfinite(total):
            return np.ones(self.L) / self.L
        return raw / total

    # ------------------------------------------------------------------
    # Long-run component
    # ------------------------------------------------------------------

    def _compute_long_run(self, C_bar: np.ndarray, theta: float, omega: float,
                          midas_blocks: np.ndarray) -> np.ndarray:
        """Compute M_t = C_bar C_bar' + theta * sum_ell phi_ell(omega) * R_bar_{t,ell}.

        Parameters
        ----------
        C_bar        : (n, n) lower-triangular matrix with positive diagonal
        theta        : scalar >= 0
        omega        : scalar > 1
        midas_blocks : (T, L, n, n) block sums

        Returns
        -------
        M_t : (T, n, n) long-run component matrices
        """
        base = C_bar @ C_bar.T                    # (n, n)
        phi  = self._midas_weights(omega)          # (L,)
        # Fully vectorised: MIDAS sum over lags for all t at once
        # midas_blocks: (T, L, n, n); phi: (L,) → midas_sum: (T, n, n)
        midas_sum = np.einsum('l,tlij->tij', phi, midas_blocks)
        M_t = base[np.newaxis] + theta * midas_sum
        return (M_t + M_t.swapaxes(-2, -1)) * 0.5

    # ------------------------------------------------------------------
    # Short-run component
    # ------------------------------------------------------------------

    def _compute_St_star(self, A_list: List[np.ndarray], B_list: List[np.ndarray],
                         R_t_star: np.ndarray) -> np.ndarray:
        """Recursively compute short-run S_t^* via numba-JIT kernel.

        Intercept = I - sum_j A_j A_j' - sum_i B_i B_i'  (ensures E[S_t^*] = I)
        Delegates to `midas_compute_St_star` from fast_recursions.

        Parameters
        ----------
        A_list   : list of q (n,n) matrices
        B_list   : list of p (n,n) matrices
        R_t_star : (T, n, n) standardised realised covariances

        Returns
        -------
        S_star : (T, n, n) short-run components
        """
        n = R_t_star.shape[1]
        I_n = np.eye(n)

        # Intercept ensures E[S_t^*] = I
        Intercept = I_n.copy()
        for A in A_list:
            Intercept -= A @ A.T
        for B in B_list:
            Intercept -= B @ B.T

        p = len(B_list)
        q = len(A_list)
        # Build (p, n, n) and (q, n, n) arrays; handle p=0 case
        B_arr = np.stack(B_list, axis=0) if p > 0 else np.empty((0, n, n))
        A_arr = np.stack(A_list, axis=0) if q > 0 else np.empty((0, n, n))

        return midas_compute_St_star(Intercept, B_arr, A_arr, R_t_star, I_n)

    # ------------------------------------------------------------------
    # Parameter packing / unpacking
    # ------------------------------------------------------------------

    def _n_params(self) -> int:
        n = self.n
        n_nu   = 1
        n_Cbar = n * (n + 1) // 2
        n_th   = 1
        n_om   = 1
        stride = n if self.diagonal else n * n
        return n_nu + n_Cbar + n_th + n_om + (self.p + self.q) * stride

    def _unpack_params(self, psi: np.ndarray):
        """Unpack flat parameter vector into model components."""
        n = self.n
        idx = 0

        # nu > n+1 via nu = (n+1) + exp(psi[0])
        nu = (n + 1.0) + np.exp(psi[idx]); idx += 1

        # C_bar: lower triangular, positive diagonal stored as log
        n_C = n * (n + 1) // 2
        c_raw = psi[idx: idx + n_C].copy(); idx += n_C
        C_bar = np.zeros((n, n))
        tril = np.tril_indices(n)
        C_bar[tril] = c_raw
        diag = np.diag_indices(n)
        C_bar[diag] = np.exp(np.clip(C_bar[diag], -40.0, 40.0))   # positive diagonal

        # theta = exp(2 * log_sqrt_theta) >= 0
        log_sqrt_theta = psi[idx]; idx += 1
        theta = np.exp(2.0 * log_sqrt_theta)

        # omega = 1 + exp(log_omega_m1) > 1
        log_omega_m1 = psi[idx]; idx += 1
        omega = 1.0 + np.exp(log_omega_m1)

        # B matrices (GARCH)
        B_list = []
        stride = n if self.diagonal else n * n
        for _ in range(self.p):
            raw = psi[idx: idx + stride]; idx += stride
            B_list.append(np.diag(raw) if self.diagonal else raw.reshape(n, n))

        # A matrices (ARCH)
        A_list = []
        for _ in range(self.q):
            raw = psi[idx: idx + stride]; idx += stride
            A_list.append(np.diag(raw) if self.diagonal else raw.reshape(n, n))

        return nu, C_bar, theta, omega, B_list, A_list

    def _pack_params(self, nu: float, C_bar: np.ndarray,
                     log_sqrt_theta: float, log_omega_m1: float,
                     B_list: List[np.ndarray], A_list: List[np.ndarray]) -> np.ndarray:
        """Pack model components into flat parameter vector."""
        n = self.n
        parts = []

        # nu
        parts.append(np.log(max(nu - n - 1.0, 1e-6)))

        # C_bar (log-diagonal)
        C_raw = C_bar.copy()
        C_raw[np.diag_indices(n)] = np.log(np.maximum(np.diag(C_bar), 1e-9))
        parts.append(C_raw[np.tril_indices(n)])

        # theta and omega
        parts.append(np.atleast_1d(log_sqrt_theta))
        parts.append(np.atleast_1d(log_omega_m1))

        # B and A matrices
        for B in B_list:
            parts.append(np.diag(B) if self.diagonal else B.flatten())
        for A in A_list:
            parts.append(np.diag(A) if self.diagonal else A.flatten())

        return np.concatenate([np.atleast_1d(p) for p in parts])

    # ------------------------------------------------------------------
    # Wishart log-likelihood helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_multivariate_gamma(a: float, n: int) -> float:
        """log Gamma_n(a) = n*(n-1)/4*log(pi) + sum_{j=1}^n gammaln(a - (j-1)/2)."""
        result = n * (n - 1) / 4.0 * np.log(np.pi)
        for j in range(1, n + 1):
            result += gammaln(a - (j - 1) / 2.0)
        return result

    # ------------------------------------------------------------------
    # Full log-likelihood (negative, for minimisation)
    # ------------------------------------------------------------------

    def _wishart_loglik(self, psi: np.ndarray,
                        R_t: np.ndarray,
                        midas_blocks: np.ndarray) -> float:
        """Negative Wishart log-likelihood (minimised by scipy).

        Uses the fused JIT kernel `midas_loglik_fused` which combines, per t:
          1. C_t^{-1} via forward substitution (eliminates batch np.linalg.inv)
          2. Standardised RC  R*_t = C_t^{-1} R_t C_t^{-T}
          3. S*_t recursion
          4. Wishart loglik contribution
        Only the batch Cholesky of M_t remains outside the kernel.
        """
        try:
            nu, C_bar, theta, omega, B_list, A_list = self._unpack_params(psi)
        except Exception:
            return 1e10

        n, T = self.n, len(R_t)
        I_n = np.eye(n)

        # --- Intercept validity check ---
        Intercept = I_n.copy()
        for A in A_list:
            Intercept -= A @ A.T
        for B in B_list:
            Intercept -= B @ B.T
        if np.any(np.diag(Intercept) < -1e-6):
            return 1e10
        if np.linalg.eigvalsh(Intercept).min() < -1e-6:
            return 1e10

        # --- Long-run Cholesky C_t — only batch op remaining outside the kernel ---
        M_t = self._compute_long_run(C_bar, theta, omega, midas_blocks)
        try:
            C_t_all = np.linalg.cholesky(M_t)
        except np.linalg.LinAlgError:
            return 1e10
        if not np.all(np.isfinite(C_t_all)):
            return 1e10

        # --- t-independent constants (depend on nu only) ---
        a = nu / 2.0
        log_mgamma = self._log_multivariate_gamma(a, n)
        const_per_t = (nu * n / 2.0) * np.log(nu / 2.0) - log_mgamma
        half_nu_minus_n_minus_1 = (nu - n - 1.0) / 2.0

        # --- Contiguous arrays for the JIT kernel ---
        B_arr = np.array(B_list, dtype=np.float64) if B_list else np.empty((0, n, n))
        A_arr = np.array(A_list, dtype=np.float64) if A_list else np.empty((0, n, n))

        # --- Fused kernel: C_t^{-1} + R* + S*_t + loglik (all per-t, in one pass) ---
        ll = midas_loglik_fused(C_t_all, R_t, self._log_det_Rt_all,
                                Intercept, B_arr, A_arr,
                                nu, half_nu_minus_n_minus_1, const_per_t)
        if ll <= -1e9 or not np.isfinite(ll):
            return 1e10

        return -ll   # positive value for minimisation

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, R_t: np.ndarray, midas_blocks: np.ndarray,
            verbose: bool = True,
            warm_start_psi: Optional[np.ndarray] = None,
            maxiter: int = 200,
            maxiter_warm: Optional[int] = None) -> Dict:
        """Fit MIDAS-CAW model by maximum likelihood.

        Parameters
        ----------
        R_t            : (T, n, n) realised covariance matrices
        midas_blocks   : (T, L, n, n) pre-computed MIDAS block sums
        verbose        : print optimiser progress
        warm_start_psi : optional parameter vector from a previous fit
        maxiter        : max L-BFGS-B iterations for a cold start
        maxiter_warm   : max L-BFGS-B iterations when warm_start_psi is given;
                         defaults to min(50, maxiter) to reduce OOS re-estimation cost

        Returns
        -------
        dict with loglik, n_params, psi, converged
        """
        n, T = self.n, len(R_t)
        n_params = self._n_params()

        # --- Cache log|R_t|: data-only, parameter-independent ---
        signs, log_det_Rt = np.linalg.slogdet(R_t)
        if np.any(signs <= 0):
            raise ValueError("Input R_t contains non-positive-definite matrices.")
        self._log_det_Rt_all = log_det_Rt   # (T,) consumed by _wishart_loglik

        # --- Use warm start if provided and length matches ---
        if warm_start_psi is not None and len(warm_start_psi) == n_params:
            psi0 = warm_start_psi.copy()
            _maxiter = maxiter_warm if maxiter_warm is not None else min(50, maxiter)
        else:
            _maxiter = maxiter
            # --- Default initial parameter values ---
            sample_mean = R_t.mean(axis=0)
            try:
                C_bar0 = np.linalg.cholesky(sample_mean)
            except np.linalg.LinAlgError:
                C_bar0 = np.eye(n) * np.sqrt(np.diag(sample_mean).mean())

            nu0    = n + 5.0
            theta0 = 0.5
            omega0 = 3.0   # omega - 1 = 2

            # Small initial A and B values (diagonal / full)
            if self.diagonal:
                B0_diag = np.full(n, 0.5 if self.p > 0 else 0.0)
                A0_diag = np.full(n, 0.3 if self.q > 0 else 0.0)
                B_list0 = [np.diag(B0_diag)] * self.p
                A_list0 = [np.diag(A0_diag)] * self.q
            else:
                B0 = np.eye(n) * (0.5 if self.p > 0 else 0.0)
                A0 = np.eye(n) * (0.3 if self.q > 0 else 0.0)
                B_list0 = [B0.copy() for _ in range(self.p)]
                A_list0 = [A0.copy() for _ in range(self.q)]

            psi0 = self._pack_params(
                nu0, C_bar0,
                np.log(np.sqrt(theta0)),
                np.log(omega0 - 1.0),
                B_list0, A_list0,
            )

        best_val = np.inf
        best_psi = psi0.copy()

        def objective(psi):
            return self._wishart_loglik(psi, R_t, midas_blocks)

        # --- First optimisation pass (L-BFGS-B with numerical gradients) ---
        res = minimize(
            objective, psi0,
            method='L-BFGS-B',
            options={'maxiter': _maxiter, 'maxfun': _maxiter * 3,
                     'ftol': 1e-8, 'gtol': 1e-5, 'disp': False},
        )
        if res.fun < best_val:
            best_val = res.fun
            best_psi = res.x.copy()

        # --- Second pass via Nelder-Mead only when first pass is clearly stuck ---
        # (fun > 1e8 ↔ loglik < −1e8, indicative of a degenerate parameter region)
        if res.fun > 1e8:
            res2 = minimize(
                objective, best_psi,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5},
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_psi = res2.x.copy()

        self.psi_hat = best_psi
        self.fitted = True
        self._R_t = R_t
        self._midas_blocks = midas_blocks

        # --- Cache filter quantities ---
        self._cache_filter(R_t, midas_blocks)

        loglik = -best_val
        return {
            'loglik': loglik,
            'n_params': self._n_params(),
            'psi': best_psi,
            'converged': res.success,
        }

    def _cache_filter(self, R_t: np.ndarray, midas_blocks: np.ndarray) -> None:
        """Run forward filter and cache S_t^*, C_t, R_t^* for diagnostics/forecast."""
        nu, C_bar, theta, omega, B_list, A_list = self._unpack_params(self.psi_hat)
        n, T = self.n, len(R_t)
        I_n = np.eye(n)

        M_t = self._compute_long_run(C_bar, theta, omega, midas_blocks)

        Intercept = I_n.copy()
        for A in A_list:
            Intercept -= A @ A.T
        for B in B_list:
            Intercept -= B @ B.T

        C_t_all     = np.empty((T, n, n))
        R_star_all  = np.empty((T, n, n))
        S_star_all  = np.empty((T, n, n))

        R_star_hist = [I_n.copy()] * max(self.q, 1)
        S_star_hist = [I_n.copy()] * max(self.p, 1)

        for t in range(T):
            try:
                C_t = np.linalg.cholesky(M_t[t])
            except np.linalg.LinAlgError:
                C_t = np.eye(n)
            C_t_all[t] = C_t

            try:
                C_t_inv = sla.solve_triangular(C_t, I_n, lower=True)
            except Exception:
                C_t_inv = np.linalg.pinv(C_t)
            R_star = C_t_inv @ R_t[t] @ C_t_inv.T
            R_star_all[t] = R_star

            S_star = Intercept.copy()
            for i, B in enumerate(B_list):
                S_star += B @ S_star_hist[-(i + 1)] @ B.T
            for j, A in enumerate(A_list):
                S_star += A @ R_star_hist[-(j + 1)] @ A.T
            S_star = (S_star + S_star.T) * 0.5
            S_star_all[t] = S_star

            R_star_hist.append(R_star)
            S_star_hist.append(S_star)

        self._C_t_all    = C_t_all
        self._R_star_all = R_star_all
        self._S_star_all = S_star_all
        self._nu         = nu
        self._B_list     = B_list
        self._A_list     = A_list
        self._Intercept  = Intercept
        self._C_bar      = C_bar
        self._theta      = theta
        self._omega      = omega

    # ------------------------------------------------------------------
    # Persistence matrix
    # ------------------------------------------------------------------

    def compute_Psi1_shortrun(self) -> np.ndarray:
        """Companion persistence matrix for the short-run S_t^* recursion.

        For p=0 returns an (n^2, n^2) zero matrix.
        For p=1 returns B_1 ⊗ B_1.
        For p>1 returns the (p*n^2) × (p*n^2) companion matrix.
        """
        assert self.fitted, 'Model must be fitted first.'
        n   = self.n
        n2  = n * n
        p   = self.p
        B_list = self._B_list

        if p == 0:
            return np.zeros((n2, n2))

        KronB = [np.kron(B, B) for B in B_list]   # each (n^2, n^2)

        if p == 1:
            return KronB[0]

        # Companion form for p > 1
        size = p * n2
        Psi = np.zeros((size, size))
        for i, KB in enumerate(KronB):
            Psi[:n2, i * n2: (i + 1) * n2] = KB
        Psi[n2:, :n2 * (p - 1)] = np.eye(n2 * (p - 1))
        return Psi

    # ------------------------------------------------------------------
    # Information criterion
    # ------------------------------------------------------------------

    def schwarz_criterion(self, loglik: float, T: int) -> float:
        """BIC/Schwarz criterion: -2*loglik + k*log(T)."""
        return -2.0 * loglik + self._n_params() * np.log(T)

    # ------------------------------------------------------------------
    # Standardised residuals
    # ------------------------------------------------------------------

    def standardized_residuals(self, R_t: np.ndarray,
                               midas_blocks: np.ndarray) -> np.ndarray:
        """Wishart-standardised residuals Z_t = L_t^{-1} R_t^* L_t^{-T}.

        where L_t L_t' = S_t^* (Cholesky), giving E[Z_t | F_{t-1}] = I_n.

        Returns
        -------
        resids : (T, n*(n+1)//2) array of vech(Z_t) elements
        """
        assert self.fitted
        T, n = len(R_t), self.n
        k_vech = n * (n + 1) // 2
        resids = np.empty((T, k_vech))

        for t in range(T):
            S_star = self._S_star_all[t]
            R_star = self._R_star_all[t]
            try:
                L = np.linalg.cholesky(S_star)
                L_inv = sla.solve_triangular(L, np.eye(n), lower=True)
                Z = L_inv @ R_star @ L_inv.T
            except np.linalg.LinAlgError:
                Z = R_star
            # vech of Z
            idx = 0
            for j in range(n):
                for i in range(j, n):
                    resids[t, idx] = Z[i, j]
                    idx += 1

        return resids

    # ------------------------------------------------------------------
    # Multi-step Monte Carlo forecast
    # ------------------------------------------------------------------

    def forecast_mc(
        self,
        R_t: np.ndarray,
        midas_blocks: np.ndarray,
        h: int = 1,
        n_paths: int = 10_000,
        seed: int = 42,
    ) -> np.ndarray:
        """MC forecast E(R_{t+h} | F_t) from the last observation t = T-1.

        For h=1 the forecast is deterministic (no simulation needed).
        For h>1, n_paths Wishart-path simulations are averaged.

        Parameters
        ----------
        R_t          : (T, n, n) training realised covariances
        midas_blocks : (T, L, n, n) training MIDAS block sums
        h            : forecast horizon
        n_paths      : number of MC paths (for h > 1)
        seed         : random seed

        Returns
        -------
        (n, n) forecast matrix E(R_{t+h} | F_t)
        """
        assert self.fitted
        rng = np.random.default_rng(seed)
        n   = self.n
        nu  = self._nu
        T   = len(R_t)
        I_n = np.eye(n)

        B_list   = self._B_list
        A_list   = self._A_list
        Intercept = self._Intercept

        # --- Build future M_{T}, M_{T+1}, ..., M_{T+h-1} ---
        # MIDAS blocks for step t=T+k depend on R_{T+k-m*L:T+k-1},
        # all of which lie within the observed R_t.  We compute them
        # directly from R_t using the same sliding-window formula.
        future_M = self._compute_future_long_run(R_t, h)  # list of h (n,n) matrices

        # --- Collect the last max(p,q) entries of the filter ---
        max_lag = max(self.p, self.q, 1)
        S_star_recent = list(self._S_star_all[max(0, T - max_lag): T])
        R_star_recent = list(self._R_star_all[max(0, T - max_lag): T])
        # Pad front with I_n if history shorter than max_lag
        while len(S_star_recent) < max_lag:
            S_star_recent.insert(0, I_n.copy())
        while len(R_star_recent) < max_lag:
            R_star_recent.insert(0, I_n.copy())

        # --- h=1: deterministic one-step-ahead ---
        C_1 = self._chol_safe(future_M[0])
        S_star_1 = Intercept.copy()
        for i, B in enumerate(B_list):
            S_star_1 += B @ S_star_recent[-(i + 1)] @ B.T
        for j, A in enumerate(A_list):
            S_star_1 += A @ R_star_recent[-(j + 1)] @ A.T
        S_star_1 = (S_star_1 + S_star_1.T) * 0.5
        S_1 = C_1 @ S_star_1 @ C_1.T

        if h == 1:
            return (S_1 + S_1.T) * 0.5

        # --- h>1: Monte Carlo ---
        # Degrees of freedom for Wishart draws
        df = int(np.round(nu))

        # Accumulator for E[R_{T+h}]
        forecast_acc = np.zeros((n, n))

        for _ in range(n_paths):
            # Each path maintains a rolling buffer of S^* and R^* values
            s_hist = list(S_star_recent)   # copies of starting history
            r_hist = list(R_star_recent)

            # Step 1 is deterministic (same for all paths)
            s_new = S_star_1.copy()
            C_new = C_1.copy()

            for step in range(1, h + 1):
                if step == 1:
                    # Already computed
                    pass
                else:
                    # Draw R_{T+step-1} ~ W_n(nu, S_{T+step-1} / nu)
                    # S_{T+step-1} = C_{T+step-1} @ S*_{T+step-1} @ C_{T+step-1}.T
                    S_prev = C_new @ s_new @ C_new.T
                    scale_mat = S_prev / nu
                    # Make scale_mat symmetric PD
                    scale_mat = (scale_mat + scale_mat.T) * 0.5
                    try:
                        R_draw = _wishart_rvs(df, scale_mat, rng)
                    except Exception:
                        R_draw = scale_mat * nu   # fallback: use mean

                    # R^*_{T+step-1} = C^{-1} R C^{-T}
                    C_inv = sla.solve_triangular(C_new, I_n, lower=True)
                    R_star_new = C_inv @ R_draw @ C_inv.T

                    # Update histories
                    s_hist.append(s_new)
                    r_hist.append(R_star_new)

                    # Compute next S^* and C
                    M_next = future_M[step - 1]   # M_{T+step}
                    C_new = self._chol_safe(M_next)
                    s_new = Intercept.copy()
                    for i, B in enumerate(B_list):
                        s_new += B @ s_hist[-(i + 1)] @ B.T
                    for j, A in enumerate(A_list):
                        s_new += A @ r_hist[-(j + 1)] @ A.T
                    s_new = (s_new + s_new.T) * 0.5

            # Final R_{T+h} draw (the realisation at the target horizon)
            C_final = self._chol_safe(future_M[h - 1])
            S_final = C_final @ s_new @ C_final.T
            scale_final = (S_final + S_final.T) * 0.5 / nu
            try:
                R_final = _wishart_rvs(df, scale_final, rng)
            except Exception:
                R_final = scale_final * nu
            forecast_acc += R_final

        forecast = forecast_acc / n_paths
        return (forecast + forecast.T) * 0.5

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chol_safe(self, M: np.ndarray) -> np.ndarray:
        """Cholesky of M; falls back to regularised version if needed."""
        M_sym = (M + M.T) * 0.5
        try:
            return np.linalg.cholesky(M_sym)
        except np.linalg.LinAlgError:
            eig_min = np.linalg.eigvalsh(M_sym).min()
            M_sym += (-eig_min + 1e-8) * np.eye(M_sym.shape[0])
            return np.linalg.cholesky(M_sym)

    def _compute_future_long_run(self, R_t: np.ndarray, h: int) -> List[np.ndarray]:
        """Compute M_{T}, M_{T+1}, ..., M_{T+h-1} from available R_t history.

        The MIDAS block for time t+k (k=1,...,h) requires R_{t+k-m*L:t+k-1},
        which are all contained in R_t when t+k-m*L >= 0 (true for T >> m*L).
        """
        T = len(R_t)
        n = self.n
        m = self.m
        L = self.L
        phi = self._midas_weights(self._omega)
        base = self._C_bar @ self._C_bar.T

        future_M = []
        for k in range(1, h + 1):
            t_future = T + k - 1   # 0-indexed future time step
            midas_sum = np.zeros((n, n))
            for ell in range(1, L + 1):
                start_idx = t_future - m * ell
                end_idx   = t_future - m * (ell - 1)
                # Clamp to available R_t
                a_start = max(start_idx, 0)
                a_end   = min(end_idx, T)
                if a_start < a_end:
                    block = R_t[a_start:a_end].sum(axis=0)
                else:
                    block = np.zeros((n, n))
                midas_sum += phi[ell - 1] * block
            M = base + self._theta * midas_sum
            M = (M + M.T) * 0.5
            future_M.append(M)

        return future_M


# ------------------------------------------------------------------
# Module-level Wishart sampler (Bartlett decomposition)
# ------------------------------------------------------------------

def _wishart_rvs(df: int, scale: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one sample from W_n(df, scale) via Bartlett decomposition.

    Parameters
    ----------
    df    : degrees of freedom (integer)
    scale : (n, n) symmetric PD scale matrix
    rng   : numpy Generator

    Returns
    -------
    (n, n) symmetric PD Wishart realisation
    """
    n = scale.shape[0]
    L = np.linalg.cholesky(scale)

    # Bartlett decomposition of the standard Wishart W_n(df, I)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = np.sqrt(rng.chisquare(df - i))
        if i > 0:
            A[i, :i] = rng.standard_normal(i)

    X = L @ A   # X X' ~ W_n(df, scale)
    W = X @ X.T
    return (W + W.T) * 0.5
