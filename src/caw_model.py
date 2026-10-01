"""
Conditional Autoregressive Wishart (CAW) model.
Custom implementation — Context7 found no library equivalent.
Follows Chiriac & Voev (2010) Sections 2 and 3.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
import scipy.linalg as sla
from typing import Tuple, Optional, Dict, List

from matrix_ops import (
    vech, vech_to_matrix, duplication_matrix, elimination_matrix, commutation_matrix
)
from fast_recursions import caw_compute_St


class CAWModel:
    """
    CAW(p, q) model:
      R_t | F_{t-1} ~ W_n(nu, S_t / nu)
      S_t = CC' + sum_{i=1}^p B_i S_{t-i} B_i' + sum_{j=1}^q A_j R_{t-j} A_j'

    Parameters
    ----------
    n        : number of assets
    p        : GARCH lag order (B matrices, past S)
    q        : ARCH lag order  (A matrices, past R)
    diagonal : if True, A_j and B_i are diagonal; C stays lower triangular
    """

    def __init__(self, n: int, p: int, q: int, diagonal: bool = False):
        self.n = n
        self.p = p
        self.q = q
        self.diagonal = diagonal
        self.k = n * (n + 1) // 2  # vech dimension

        # Set after fitting
        self.psi_hat: Optional[np.ndarray] = None
        self.nu_hat: Optional[float] = None
        self.C_hat: Optional[np.ndarray] = None
        self.B_hat: Optional[List[np.ndarray]] = None
        self.A_hat: Optional[List[np.ndarray]] = None
        self.S_t_hat: Optional[np.ndarray] = None
        self.fitted = False

    # ---------------------------------------------------------------
    # Parameter count
    # ---------------------------------------------------------------

    def _n_params(self) -> int:
        n = self.n
        n_nu = 1
        n_C  = n * (n + 1) // 2   # lower-triangular elements
        stride = n if self.diagonal else n * n
        return n_nu + n_C + (self.p + self.q) * stride

    # ---------------------------------------------------------------
    # Pack / unpack
    # ---------------------------------------------------------------

    def _unpack_params(self, psi: np.ndarray) -> Tuple:
        """Unpack flat parameter vector into (nu, C, B_list, A_list)."""
        n = self.n
        idx = 0

        # nu: must satisfy nu > n+1 → parameterise as nu = (n+1) + exp(psi[0])
        nu = (n + 1.0) + np.exp(np.clip(psi[idx], -30.0, 30.0))
        idx += 1

        # C: lower triangular, positive diagonal stored as log
        n_C = n * (n + 1) // 2
        c_raw = psi[idx: idx + n_C].copy()
        idx += n_C
        C = np.zeros((n, n))
        tril = np.tril_indices(n)
        C[tril] = c_raw
        # Exponentiate diagonal to enforce positivity
        for d in range(n):
            C[d, d] = np.exp(np.clip(C[d, d], -40.0, 40.0))

        # B matrices (GARCH lags on S)
        B_list: List[np.ndarray] = []
        for _ in range(self.p):
            if self.diagonal:
                b_diag = psi[idx: idx + n]
                idx += n
                B_list.append(np.diag(b_diag))
            else:
                B = psi[idx: idx + n * n].reshape(n, n)
                idx += n * n
                B_list.append(B)

        # A matrices (ARCH lags on R)
        A_list: List[np.ndarray] = []
        for _ in range(self.q):
            if self.diagonal:
                a_diag = psi[idx: idx + n]
                idx += n
                A_list.append(np.diag(a_diag))
            else:
                A = psi[idx: idx + n * n].reshape(n, n)
                idx += n * n
                A_list.append(A)

        return nu, C, B_list, A_list

    def _pack_params(self, nu: float, C: np.ndarray,
                     B_list: List[np.ndarray], A_list: List[np.ndarray]) -> np.ndarray:
        """Pack model components into flat parameter vector."""
        n = self.n
        parts = []

        # nu: store log(nu - (n+1))
        parts.append([np.log(max(nu - (n + 1.0), 1e-8))])

        # C: lower triangular, diagonal stored as log
        C_copy = C.copy()
        tril = np.tril_indices(n)
        c_vec = C_copy[tril].copy()
        # Locate diagonal positions in tril ordering
        diag_locs = [j * (n - j // 2 + n - 1) // 2 + 0 for j in range(n)]
        # Simpler: find indices where row==col in tril
        rows, cols = tril
        diag_mask = rows == cols
        c_vec[diag_mask] = np.log(np.maximum(c_vec[diag_mask], 1e-8))
        parts.append(c_vec)

        for B in B_list:
            if self.diagonal:
                parts.append(np.diag(B))
            else:
                parts.append(B.flatten())

        for A in A_list:
            if self.diagonal:
                parts.append(np.diag(A))
            else:
                parts.append(A.flatten())

        return np.concatenate(parts)

    # ---------------------------------------------------------------
    # S_t recursion
    # ---------------------------------------------------------------

    def _compute_St(self, psi: np.ndarray, R_t: np.ndarray) -> Optional[np.ndarray]:
        """Recursively compute S_t using the numba-JIT kernel (caw_compute_St).

        Returns None if any S_t is not positive definite (triggers penalty).
        """
        nu, C, B_list, A_list = self._unpack_params(psi)
        T, n, _ = R_t.shape
        CC = C @ C.T

        # Pre-sample initialiser: sample mean of first few observations
        S_init = np.mean(R_t[:max(1, self.p + self.q)], axis=0)
        S_init = (S_init + S_init.T) / 2.0
        min_ev = np.linalg.eigvalsh(S_init).min()
        if min_ev <= 0:
            S_init += (-min_ev + 1e-6) * np.eye(n)

        # Stack B and A lists into 3-D arrays for the numba kernel
        p, q = self.p, self.q
        B_arr = np.stack(B_list, axis=0) if p > 0 else np.empty((0, n, n))
        A_arr = np.stack(A_list, axis=0) if q > 0 else np.empty((0, n, n))

        S_t = caw_compute_St(CC, B_arr, A_arr, R_t, S_init)

        if np.linalg.eigvalsh(S_t).min() <= 0:
            return None
        return S_t

    # ---------------------------------------------------------------
    # Wishart log-likelihood
    # ---------------------------------------------------------------

    def _wishart_loglik(self, psi: np.ndarray, R_t: np.ndarray) -> float:
        """Compute negative Wishart log-likelihood (for minimization).

        Vectorised over T via batched numpy slogdet / inv / einsum so that the
        only sequential work is the S_t recursion in _compute_St.

        ell_t = -(nu*n/2)*ln 2 - n(n-1)/4*ln pi
                - sum_i lnGamma((nu+1-i)/2)
                - (nu/2)*ln|S_t/nu| + ((nu-n-1)/2)*ln|R_t|
                - (1/2)*tr(nu S_t^{-1} R_t)
        """
        nu, C, B_list, A_list = self._unpack_params(psi)
        T, n, _ = R_t.shape

        S_t = self._compute_St(psi, R_t)
        if S_t is None:
            return 1e10

        # --- Batched log-determinants (T,) ---
        signs_S, ld_S = np.linalg.slogdet(S_t)   # S_t already PD from _compute_St
        signs_R, ld_R = np.linalg.slogdet(R_t)
        if (signs_S <= 0).any() or (signs_R <= 0).any():
            return 1e10

        # --- Batched trace term: tr(nu S_t^{-1} R_t) for all t ---
        # S_inv shape (T, n, n); einsum gives tr for each t
        S_inv = np.linalg.inv(S_t)                            # (T, n, n)
        traces = nu * np.einsum('tij,tji->t', S_inv, R_t)    # (T,)

        # --- Scalar constant (depends only on nu, n) ---
        const = (
            -(nu * n / 2.0) * np.log(2.0)
            - (n * (n - 1) / 4.0) * np.log(np.pi)
            - sum(gammaln((nu + 1.0 - i) / 2.0) for i in range(1, n + 1))
        )

        # log|S_t / nu| = log|S_t| - n*log(nu)
        ld_S_over_nu = ld_S - n * np.log(nu)

        loglik = (
            T * const
            - (nu / 2.0) * ld_S_over_nu.sum()
            + ((nu - n - 1.0) / 2.0) * ld_R.sum()
            - 0.5 * traces.sum()
        )

        if not np.isfinite(loglik):
            return 1e10

        return -loglik  # negative for minimization

    # ---------------------------------------------------------------
    # Fitting
    # ---------------------------------------------------------------

    def _init_params(self, R_t: np.ndarray) -> np.ndarray:
        """Build initial parameter vector from data."""
        n = self.n

        # nu: start at n+10
        nu0 = float(n) + 10.0
        log_nu_shifted = np.log(max(nu0 - (n + 1.0), 1e-4))

        # C: Cholesky of sample covariance
        R_mean = np.mean(R_t, axis=0)
        try:
            C0 = np.linalg.cholesky(R_mean)
        except np.linalg.LinAlgError:
            C0 = np.eye(n) * np.sqrt(np.mean(np.diag(R_mean)))

        tril = np.tril_indices(n)
        c_vec = C0[tril].copy()
        rows, cols = tril
        diag_mask = rows == cols
        c_vec[diag_mask] = np.log(np.maximum(c_vec[diag_mask], 1e-8))

        psi0 = [log_nu_shifted] + list(c_vec)

        # B: small positive diagonal
        for _ in range(self.p):
            if self.diagonal:
                psi0.extend([0.3] * n)
            else:
                B0 = np.eye(n) * 0.3
                psi0.extend(B0.flatten())

        # A: small positive diagonal
        for _ in range(self.q):
            if self.diagonal:
                psi0.extend([0.3] * n)
            else:
                A0 = np.eye(n) * 0.3
                psi0.extend(A0.flatten())

        return np.array(psi0)

    def fit(self, R_t: np.ndarray, verbose: bool = True,
            warm_start_psi: Optional[np.ndarray] = None,
            maxiter: int = 300,
            maxiter_warm: Optional[int] = None) -> Dict:
        """Fit the CAW model by maximum likelihood using L-BFGS-B.

        Parameters
        ----------
        R_t            : ndarray (T, n, n) — realized covariance matrices
        verbose        : print progress
        warm_start_psi : optional starting parameter vector (e.g. from a
                         previous fit) to speed up re-estimation
        maxiter        : max L-BFGS-B iterations for a cold start
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
        assert len(psi0) == n_params, f"Param count mismatch: {len(psi0)} vs {n_params}"

        result = minimize(
            self._wishart_loglik,
            psi0,
            args=(R_t,),
            method='L-BFGS-B',
            options={'maxiter': _maxiter, 'maxfun': _maxiter * 3,
                     'ftol': 1e-8, 'gtol': 1e-5, 'disp': False}
        )

        self.psi_hat = result.x
        nu, C, B_list, A_list = self._unpack_params(result.x)
        self.nu_hat = nu
        self.C_hat = C
        self.B_hat = B_list
        self.A_hat = A_list
        self.S_t_hat = self._compute_St(result.x, R_t)
        self.R_t_fit = R_t
        self.fitted = True

        loglik = -result.fun

        return {
            'loglik': loglik,
            'n_params': n_params,
            'psi': result.x,
            'nu': nu,
            'C': C,
            'B_list': B_list,
            'A_list': A_list,
            'S_t': self.S_t_hat,
            'converged': result.success
        }

    # ---------------------------------------------------------------
    # Moment diagnostics
    # ---------------------------------------------------------------

    def _cal_matrix(self, M: np.ndarray) -> np.ndarray:
        """Compute L_n (M ⊗ M) D_n for a 5x5 matrix M.

        Result shape: (k, k) where k = n*(n+1)/2
        """
        n = self.n
        L_n = elimination_matrix(n)
        D_n = duplication_matrix(n)
        return L_n @ np.kron(M, M) @ D_n

    def compute_Psi1(self) -> np.ndarray:
        """Compute first-moment matrix Psi_1 = sum_i (B_cal_i + A_cal_j).

        A_cal_j = L_n (A_j ⊗ A_j) D_n
        B_cal_i = L_n (B_i ⊗ B_i) D_n
        """
        assert self.fitted
        k = self.k
        Psi1 = np.zeros((k, k))
        for A in self.A_hat:
            Psi1 += self._cal_matrix(A)
        for B in self.B_hat:
            Psi1 += self._cal_matrix(B)
        return Psi1

    def _compute_Omega(self) -> np.ndarray:
        """Compute Omega = (1/nu) (L_n⊗L_n)[I_{n^2}⊗(I_{n^2}+K_nn)](I_n⊗K_nn⊗I_n)(D_n⊗D_n).

        Shape: (k^2, k^2) is too large; return the effective (k, k) Wishart variance operator.
        For implementation, use the simplified form:
          Omega_{(ij),(lm)} = (1/nu)(s_{il}s_{jm} + s_{im}s_{jl})
        in vech-space, evaluated at S = unconditional mean (sample mean of S_t).
        """
        assert self.fitted
        n, nu = self.n, self.nu_hat
        k = self.k

        # Use mean S_t as reference
        S_bar = np.mean(self.S_t_hat, axis=0)

        # Build vech-space conditional covariance matrix of r_t
        # Cov(r_{ij}, r_{lm}) = (1/nu) (s_{il} s_{jm} + s_{im} s_{jl})
        # mapped to vech space
        tril = np.tril_indices(n)
        rows, cols = tril
        Omega = np.zeros((k, k))
        for p_idx in range(k):
            i, j = rows[p_idx], cols[p_idx]
            for q_idx in range(k):
                l, m = rows[q_idx], cols[q_idx]
                Omega[p_idx, q_idx] = (
                    (1.0 / nu) * (S_bar[i, l] * S_bar[j, m] + S_bar[i, m] * S_bar[j, l])
                )
        return Omega

    def compute_Psi2(self, truncation: int = 100) -> np.ndarray:
        """Compute second-moment matrix Psi_2 = sum_{i=1}^∞ (Phi_i ⊗ Phi_i) Omega.

        VMA recursion: Phi_0 = I_k
        Phi_i = -B_cal_i + sum_{j=1}^i (A_cal_j + B_cal_j) Phi_{i-j}
        """
        assert self.fitted
        k = self.k
        Omega = self._compute_Omega()

        # Precompute cal matrices
        A_cals = [self._cal_matrix(A) for A in self.A_hat]
        B_cals = [self._cal_matrix(B) for B in self.B_hat]

        max_lag = max(self.p, self.q, 1)
        Phi = [np.zeros((k, k)) for _ in range(truncation + 1)]
        Phi[0] = np.eye(k)

        for i in range(1, truncation + 1):
            # -B_cal_i term
            if i <= self.p:
                Phi[i] = -B_cals[i - 1].copy()
            else:
                Phi[i] = np.zeros((k, k))

            # + sum_{j=1}^i (A_cal_j + B_cal_j) Phi[i-j]
            for j in range(1, i + 1):
                AB_j = np.zeros((k, k))
                if j <= self.q:
                    AB_j += A_cals[j - 1]
                if j <= self.p:
                    AB_j += B_cals[j - 1]
                Phi[i] += AB_j @ Phi[i - j]

        # Psi_2 = sum_{i=1}^trunc (Phi_i ⊗ Phi_i) Omega
        # Since k=15, Phi_i is (15,15), Phi_i⊗Phi_i is (225,225) — too large
        # Instead compute Psi_2 as a (k,k) matrix using the identity:
        # (Phi ⊗ Phi) vec(Omega) = vec(Phi Omega Phi') for each row-col pair
        # Simplified: Psi_2 = sum_i Phi_i Omega Phi_i'  (k×k matrix)
        Psi2 = np.zeros((k, k))
        for i in range(1, truncation + 1):
            Psi2 += Phi[i] @ Omega @ Phi[i].T
            # Early stopping if contribution is negligible
            if np.max(np.abs(Phi[i])) < 1e-10:
                break

        return Psi2

    # ---------------------------------------------------------------
    # Standardized residuals
    # ---------------------------------------------------------------

    def standardized_residuals(self, R_t: np.ndarray) -> np.ndarray:
        """Compute standardized vech residuals e_t* = Var(r_t|F_{t-1})^{-1/2} [r_t - s_t].

        Var(r_{ij}, r_{lm} | F_{t-1}) = (1/nu)(s_{il} s_{jm} + s_{im} s_{jl}) in vech space.
        """
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

            # Build conditional covariance matrix of vech(R_t)
            Var_t = np.zeros((k, k))
            for p_idx in range(k):
                i, j = rows[p_idx], cols[p_idx]
                for q_idx in range(k):
                    l, m = rows[q_idx], cols[q_idx]
                    Var_t[p_idx, q_idx] = (
                        (1.0 / nu) * (St[i, l] * St[j, m] + St[i, m] * St[j, l])
                    )

            # Standardize: e_t* = Var^{-1/2} (r_t - s_t)
            try:
                chol_Var = np.linalg.cholesky(Var_t)
                resids[t] = np.linalg.solve(chol_Var, r_t - s_t)
            except np.linalg.LinAlgError:
                resids[t] = r_t - s_t

        return resids

    # ---------------------------------------------------------------
    # Forecasting
    # ---------------------------------------------------------------

    def forecast(self, R_t: np.ndarray, h: int = 1) -> np.ndarray:
        """Forecast E(R_{t+h} | F_t) using vectorized recursion.

        For h=1: returns S_{T+1} from the recursion.
        For h>1: iterates the expected recursion forward.

        Parameters
        ----------
        R_t : ndarray (T, n, n) — training data used to build state
        h   : forecast horizon

        Returns
        -------
        forecast matrix (n, n)
        """
        assert self.fitted
        T, n, _ = R_t.shape
        k = self.k

        # Build cal matrices for recursion
        A_cals = [self._cal_matrix(A) for A in self.A_hat]
        B_cals = [self._cal_matrix(B) for B in self.B_hat]

        # Historical vech series (realized + fitted)
        r_hist = np.array([vech(R_t[t]) for t in range(T)])  # shape (T, k)
        s_hist = np.array([vech(self.S_t_hat[t]) for t in range(T)])  # shape (T, k)

        # Constant term
        C_cal_vec = vech(self.C_hat @ self.C_hat.T)

        # Forecast recursion: build augmented history
        # E(r_{t+h}|F_t) = E(s_{t+h}|F_t)
        # E(s_{t+h}|F_t) = c_cal + sum_i B_cal_i E(s_{t+h-i}|F_t) + sum_j A_cal_j E(r_{t+h-j}|F_t)
        # For indices <= T, use observed/fitted values; for indices > T, use forecasts

        # Store forecasts for steps 1..h
        forecasts_vech = {}

        for step in range(1, h + 1):
            fc = C_cal_vec.copy()
            for i, B_cal in enumerate(B_cals):
                lag_idx = T + step - 1 - (i + 1)  # T + step - 1 - 1 - i (0-indexed lag i+1)
                if lag_idx < T:
                    val = s_hist[lag_idx]
                else:
                    # Use previously computed forecast
                    fc_step = lag_idx - T + 1
                    val = forecasts_vech.get(fc_step, s_hist[-1])
                fc += B_cal @ val

            for j, A_cal in enumerate(A_cals):
                lag_idx = T + step - 1 - (j + 1)
                if lag_idx < T:
                    # Use realized for past, fitted for future
                    val = r_hist[lag_idx]
                else:
                    fc_step = lag_idx - T + 1
                    val = forecasts_vech.get(fc_step, s_hist[-1])
                fc += A_cal @ val

            forecasts_vech[step] = fc

        # Convert vech forecast back to matrix
        fc_vech = forecasts_vech[h]
        return vech_to_matrix(fc_vech, self.n)

    # ---------------------------------------------------------------
    # Model selection
    # ---------------------------------------------------------------

    def schwarz_criterion(self, loglik: float, T: int) -> float:
        """Schwarz (BIC) = -2 * L + k * ln(T)."""
        return -2.0 * loglik + self._n_params() * np.log(T)
