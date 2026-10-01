"""
Numba-JIT-compiled recursion kernels for CAW and MIDAS-CAW S_t loops.

The plain Python loops in _compute_St / _compute_St_star are the numerical
bottleneck during maximum-likelihood estimation.  Moving them to numba gives
~10-20x speedup without changing any mathematical logic.

Usage
-----
Import the JIT functions; they compile on first call (cached to __pycache__).
"""
import numpy as np

try:
    import numba
    from numba import njit

    @njit(cache=True)
    def _caw_recursion(CC: np.ndarray,
                       B_arr: np.ndarray,
                       A_arr: np.ndarray,
                       R_t: np.ndarray,
                       S_init: np.ndarray) -> np.ndarray:
        """Numba-JIT CAW S_t recursion.

        Parameters
        ----------
        CC      : (n, n) intercept = C @ C.T
        B_arr   : (p, n, n) GARCH coefficient matrices (p may be 0)
        A_arr   : (q, n, n) ARCH coefficient matrices
        R_t     : (T, n, n) observed realized covariance matrices
        S_init  : (n, n) pre-sample initializer for t < 0

        Returns
        -------
        S_t : (T, n, n)
        """
        T, n, _ = R_t.shape
        p = B_arr.shape[0]
        q = A_arr.shape[0]
        S_t = np.empty((T, n, n))

        for t in range(T):
            St = CC.copy()
            for i in range(p):
                lag = t - 1 - i
                prev_S = S_t[lag] if lag >= 0 else S_init
                St += B_arr[i] @ prev_S @ B_arr[i].T
            for j in range(q):
                lag = t - 1 - j
                prev_R = R_t[lag] if lag >= 0 else S_init
                St += A_arr[j] @ prev_R @ A_arr[j].T
            S_t[t] = (St + St.T) * 0.5

        return S_t

    @njit(cache=True)
    def _midas_loglik_star_loop(Intercept: np.ndarray,
                                B_arr: np.ndarray,
                                A_arr: np.ndarray,
                                R_star_all: np.ndarray,
                                nu: float) -> float:
        """Fused JIT: S*_t recursion + Wishart loglik accumulation.

        Replaces the Python-level sequential loop in MIDASCAWModel._wishart_loglik.
        Pre-sample values of S*_t and R*_t are initialised to I_n.
        A tiny jitter (1e-10 * I) is added to each S*_t before Cholesky so
        the decomposition always succeeds for nearly-PSD matrices.

        Parameters
        ----------
        Intercept   : (n, n) I - sum_j A_j A_j' - sum_i B_i B_i'
        B_arr       : (p, n, n)
        A_arr       : (q, n, n)
        R_star_all  : (T, n, n) standardised RC  C_t^{-1} R_t C_t^{-T}
        nu          : Wishart degrees-of-freedom

        Returns
        -------
        neg_loglik_contribution : float (positive = to minimise)
            Only the terms involving S*_t; caller adds t-independent terms.
            Returns 1e10 on numerical failure.
        """
        T, n, _ = R_star_all.shape
        p = B_arr.shape[0]
        q = A_arr.shape[0]
        max_lag = p if p > q else q
        if max_lag < 1:
            max_lag = 1

        I_n = np.eye(n)
        eps = 1e-10

        # Padded history: indices 0..max_lag-1 hold pre-sample I_n
        S_hist = np.empty((max_lag + T, n, n))
        R_hist = np.empty((max_lag + T, n, n))
        for i in range(max_lag):
            for r in range(n):
                for c in range(n):
                    S_hist[i, r, c] = I_n[r, c]
                    R_hist[i, r, c] = I_n[r, c]

        loglik = 0.0

        for t in range(T):
            R_star = R_star_all[t]
            idx = max_lag + t

            # --- S*_t recursion ---
            S = Intercept.copy()
            for i in range(p):
                prev_S = S_hist[idx - 1 - i]
                S = S + B_arr[i] @ prev_S @ B_arr[i].T
            for j in range(q):
                prev_R = R_hist[idx - 1 - j]
                S = S + A_arr[j] @ prev_R @ A_arr[j].T
            S = (S + S.T) * 0.5

            # Jitter for numerical stability
            for k in range(n):
                S[k, k] += eps

            # --- Cholesky ---
            L = np.linalg.cholesky(S)

            # --- log|S*| = 2 * sum(log diag(L)) ---
            log_det = 0.0
            for k in range(n):
                if L[k, k] <= 0.0:
                    return 1e10
                log_det += np.log(L[k, k])
            log_det *= 2.0

            # --- tr(S*^{-1} R*) via two triangular solves ---
            # X = L^{-1} R*  (forward substitution)
            X = np.linalg.solve(L, R_star)
            # Y = L^{-T} X   (backward substitution)
            Y = np.linalg.solve(L.T, X)
            tr_term = 0.0
            for k in range(n):
                tr_term += Y[k, k]

            if not (np.isfinite(log_det) and np.isfinite(tr_term)):
                return 1e10

            loglik -= (nu / 2.0) * (tr_term + log_det)

            # Store to history
            for r in range(n):
                for c in range(n):
                    S_hist[idx, r, c] = S[r, c]
                    R_hist[idx, r, c] = R_star[r, c]

        return -loglik   # positive value for minimisation

    @njit(cache=True)
    def _midas_star_recursion(Intercept: np.ndarray,
                              B_arr: np.ndarray,
                              A_arr: np.ndarray,
                              R_t_star: np.ndarray,
                              I_n: np.ndarray) -> np.ndarray:
        """Numba-JIT MIDAS-CAW short-run S_t^* recursion.

        Parameters
        ----------
        Intercept : (n, n) I - sum A_j A_j' - sum B_i B_i'
        B_arr     : (p, n, n)
        A_arr     : (q, n, n)
        R_t_star  : (T, n, n) standardised RC  C_{t-j}^{-1} R_{t-j} C_{t-j}^{-T}
        I_n       : (n, n) identity (pre-sample initializer)

        Returns
        -------
        S_star : (T, n, n)
        """
        T, n, _ = R_t_star.shape
        p = B_arr.shape[0]
        q = A_arr.shape[0]
        S_star = np.empty((T, n, n))

        for t in range(T):
            S = Intercept.copy()
            for i in range(p):
                lag = t - 1 - i
                prev = S_star[lag] if lag >= 0 else I_n
                S += B_arr[i] @ prev @ B_arr[i].T
            for j in range(q):
                lag = t - 1 - j
                prev = R_t_star[lag] if lag >= 0 else I_n
                S += A_arr[j] @ prev @ A_arr[j].T
            S_star[t] = (S + S.T) * 0.5

        return S_star

    @njit(cache=True)
    def _midas_loglik_fused(C_t_all: np.ndarray,
                            R_t: np.ndarray,
                            log_det_Rt_all: np.ndarray,
                            Intercept: np.ndarray,
                            B_arr: np.ndarray,
                            A_arr: np.ndarray,
                            nu: float,
                            half_nu_minus_n_minus_1: float,
                            const_per_t: float) -> float:
        """Fused JIT: per-t C_t inversion + R_star + S*_t recursion + loglik.

        Eliminates the batch np.linalg.inv(C_t_all) call and the separate
        R_star_all array by computing each C_t^{-1} via forward substitution
        (O(n^2), negligible for n=5) inside the sequential loop.

        Parameters
        ----------
        C_t_all                 : (T, n, n) lower-triangular Cholesky factors of M_t
        R_t                     : (T, n, n) observed daily RC matrices
        log_det_Rt_all          : (T,) precomputed log|R_t| (data-only, cached)
        Intercept               : (n, n) I - sum A_j A_j' - sum B_i B_i'
        B_arr                   : (p, n, n)
        A_arr                   : (q, n, n)
        nu                      : Wishart degrees of freedom
        half_nu_minus_n_minus_1 : (nu - n - 1) / 2  (precomputed)
        const_per_t             : t-independent loglik constant per observation

        Returns
        -------
        loglik : float  raw log-likelihood (positive = good; -1e10 on failure)
        """
        T, n, _ = R_t.shape
        p = B_arr.shape[0]
        q = A_arr.shape[0]
        max_lag = p if p > q else q
        if max_lag < 1:
            max_lag = 1

        eps = 1e-10
        loglik = 0.0

        # Pre-sample histories initialised to I_n
        S_hist = np.empty((max_lag + T, n, n))
        R_hist = np.empty((max_lag + T, n, n))
        for i in range(max_lag):
            for r in range(n):
                for c in range(n):
                    S_hist[i, r, c] = 1.0 if r == c else 0.0
                    R_hist[i, r, c] = 1.0 if r == c else 0.0

        for t in range(T):
            idx = max_lag + t
            C_t = C_t_all[t]   # (n, n) lower triangular Cholesky factor of M_t

            # --- C_t^{-1} via forward substitution: C_t x = e_j for each j ---
            C_inv = np.zeros((n, n))
            for j in range(n):
                C_inv[j, j] = 1.0 / C_t[j, j]
                for i in range(j + 1, n):
                    s = 0.0
                    for k in range(j, i):
                        s -= C_t[i, k] * C_inv[k, j]
                    C_inv[i, j] = s / C_t[i, i]

            # log|C_t| = sum(log diag(C_t))  [log|M_t| = 2 * log|C_t|]
            log_det_Ct = 0.0
            for k in range(n):
                if C_t[k, k] <= 0.0:
                    return -1e10
                log_det_Ct += np.log(C_t[k, k])

            # Standardised RC: R*_t = C_t^{-1} R_t C_t^{-T}
            tmp = C_inv @ R_t[t]
            R_star = tmp @ C_inv.T
            R_star = (R_star + R_star.T) * 0.5

            # --- S*_t recursion ---
            S = Intercept.copy()
            for i in range(p):
                prev_S = S_hist[idx - 1 - i]
                S = S + B_arr[i] @ prev_S @ B_arr[i].T
            for j in range(q):
                prev_R = R_hist[idx - 1 - j]
                S = S + A_arr[j] @ prev_R @ A_arr[j].T
            S = (S + S.T) * 0.5
            for k in range(n):
                S[k, k] += eps

            # --- Cholesky of S* ---
            L = np.linalg.cholesky(S)

            # log|S*| = 2 * sum log diag(L)
            log_det_S = 0.0
            for k in range(n):
                if L[k, k] <= 0.0:
                    return -1e10
                log_det_S += np.log(L[k, k])
            log_det_S *= 2.0

            # tr(S*^{-1} R*) via two triangular solves
            X = np.linalg.solve(L, R_star)
            Y = np.linalg.solve(L.T, X)
            tr_term = 0.0
            for k in range(n):
                tr_term += Y[k, k]

            if not (np.isfinite(log_det_S) and np.isfinite(tr_term)):
                return -1e10

            # Full loglik contribution for this t:
            #   const_per_t + (nu-n-1)/2 * log|R_t|
            #   - nu * log|C_t|          (= -nu/2 * log|M_t|)
            #   - nu/2 * (log|S*| + tr(S*^{-1} R*))
            loglik += (const_per_t
                       + half_nu_minus_n_minus_1 * log_det_Rt_all[t]
                       - nu * log_det_Ct
                       - (nu / 2.0) * (log_det_S + tr_term))

            # Store histories
            for r in range(n):
                for c in range(n):
                    S_hist[idx, r, c] = S[r, c]
                    R_hist[idx, r, c] = R_star[r, c]

        return loglik   # raw log-likelihood; caller negates for minimisation

    _NUMBA_AVAILABLE = True

except ImportError:
    _NUMBA_AVAILABLE = False

    def _caw_recursion(CC, B_arr, A_arr, R_t, S_init):
        """Pure-numpy fallback when numba is unavailable."""
        T, n, _ = R_t.shape
        p = B_arr.shape[0]
        q = A_arr.shape[0]
        S_t = np.empty((T, n, n))
        for t in range(T):
            St = CC.copy()
            for i in range(p):
                lag = t - 1 - i
                prev_S = S_t[lag] if lag >= 0 else S_init
                St += B_arr[i] @ prev_S @ B_arr[i].T
            for j in range(q):
                lag = t - 1 - j
                prev_R = R_t[lag] if lag >= 0 else S_init
                St += A_arr[j] @ prev_R @ A_arr[j].T
            S_t[t] = (St + St.T) * 0.5
        return S_t

    def _midas_star_recursion(Intercept, B_arr, A_arr, R_t_star, I_n):
        """Pure-numpy fallback when numba is unavailable."""
        T, n, _ = R_t_star.shape
        p = B_arr.shape[0]
        q = A_arr.shape[0]
        S_star = np.empty((T, n, n))
        for t in range(T):
            S = Intercept.copy()
            for i in range(p):
                lag = t - 1 - i
                prev = S_star[lag] if lag >= 0 else I_n
                S += B_arr[i] @ prev @ B_arr[i].T
            for j in range(q):
                lag = t - 1 - j
                prev = R_t_star[lag] if lag >= 0 else I_n
                S += A_arr[j] @ prev @ A_arr[j].T
            S_star[t] = (S + S.T) * 0.5
        return S_star

    def _midas_loglik_star_loop(Intercept, B_arr, A_arr, R_star_all, nu):
        """Pure-numpy fallback for fused S* loop + loglik accumulation."""
        import scipy.linalg as _sla
        T, n, _ = R_star_all.shape
        p = B_arr.shape[0]
        q = A_arr.shape[0]
        I_n = np.eye(n)
        eps = 1e-10
        R_star_hist = [I_n] * max(q, 1)
        S_star_hist = [I_n] * max(p, 1)
        loglik = 0.0
        for t in range(T):
            R_star = R_star_all[t]
            S = Intercept.copy()
            for i, B in enumerate(B_arr):
                S += B @ S_star_hist[-(i + 1)] @ B.T
            for j, A in enumerate(A_arr):
                S += A @ R_star_hist[-(j + 1)] @ A.T
            S = (S + S.T) * 0.5
            np.fill_diagonal(S, np.diag(S) + eps)
            try:
                L = np.linalg.cholesky(S)
            except np.linalg.LinAlgError:
                return 1e10
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            tr_term = np.trace(_sla.cho_solve((L, True), R_star))
            if not (np.isfinite(log_det) and np.isfinite(tr_term)):
                return 1e10
            loglik -= (nu / 2.0) * (tr_term + log_det)
            R_star_hist.append(R_star)
            S_star_hist.append(S)
        return -loglik

    def _midas_loglik_fused(C_t_all, R_t, log_det_Rt_all, Intercept,
                            B_arr, A_arr, nu, half_nu_minus_n_minus_1, const_per_t):
        """Pure-numpy fallback for fused MIDAS loglik kernel."""
        import scipy.linalg as _sla
        T, n, _ = R_t.shape
        p = B_arr.shape[0]
        q = A_arr.shape[0]
        I_n = np.eye(n)
        eps = 1e-10
        R_hist = [I_n.copy()] * max(q, 1)
        S_hist = [I_n.copy()] * max(p, 1)
        loglik = 0.0
        for t in range(T):
            C_t = C_t_all[t]
            C_inv = np.linalg.solve(C_t, I_n)  # lower-tri forward sub
            log_det_Ct = np.sum(np.log(np.maximum(np.diag(C_t), 1e-300)))
            R_star = C_inv @ R_t[t] @ C_inv.T
            R_star = (R_star + R_star.T) * 0.5
            S = Intercept.copy()
            for i, B in enumerate(B_arr):
                S += B @ S_hist[-(i + 1)] @ B.T
            for j, A in enumerate(A_arr):
                S += A @ R_hist[-(j + 1)] @ A.T
            S = (S + S.T) * 0.5
            np.fill_diagonal(S, np.diag(S) + eps)
            try:
                L = np.linalg.cholesky(S)
            except np.linalg.LinAlgError:
                return -1e10
            log_det_S = 2.0 * np.sum(np.log(np.diag(L)))
            tr_term = np.trace(_sla.cho_solve((L, True), R_star))
            if not (np.isfinite(log_det_S) and np.isfinite(tr_term)):
                return -1e10
            loglik += (const_per_t
                       + half_nu_minus_n_minus_1 * log_det_Rt_all[t]
                       - nu * log_det_Ct
                       - (nu / 2.0) * (log_det_S + tr_term))
            R_hist.append(R_star)
            S_hist.append(S)
        return loglik


def caw_compute_St(CC: np.ndarray,
                   B_arr: np.ndarray,
                   A_arr: np.ndarray,
                   R_t: np.ndarray,
                   S_init: np.ndarray) -> np.ndarray:
    """Public wrapper: calls numba kernel if available, numpy fallback otherwise.

    Parameters
    ----------
    CC     : (n, n) C @ C.T
    B_arr  : (p, n, n), may be empty (shape (0, n, n)) for p=0
    A_arr  : (q, n, n)
    R_t    : (T, n, n)
    S_init : (n, n)

    Returns
    -------
    S_t : (T, n, n)
    """
    return _caw_recursion(CC, B_arr, A_arr, R_t, S_init)


def midas_compute_St_star(Intercept: np.ndarray,
                          B_arr: np.ndarray,
                          A_arr: np.ndarray,
                          R_t_star: np.ndarray,
                          I_n: np.ndarray) -> np.ndarray:
    """Public wrapper for MIDAS-CAW short-run recursion."""
    return _midas_star_recursion(Intercept, B_arr, A_arr, R_t_star, I_n)


def midas_loglik_star_loop(Intercept: np.ndarray,
                           B_arr: np.ndarray,
                           A_arr: np.ndarray,
                           R_star_all: np.ndarray,
                           nu: float) -> float:
    """Public wrapper: fused S*_t recursion + Wishart loglik accumulation.

    Returns the *negative* loglik contribution from the sequential loop so
    it can be added directly to the scalar penalty being minimised.

    Parameters
    ----------
    Intercept   : (n, n)  I - sum A A' - sum B B'
    B_arr       : (p, n, n)
    A_arr       : (q, n, n)
    R_star_all  : (T, n, n)  standardised RC   C_t^{-1} R_t C_t^{-T}
    nu          : Wishart df

    Returns
    -------
    neg_loop_ll : float  (positive value; 1e10 on numerical failure)
    """
    return _midas_loglik_star_loop(Intercept, B_arr, A_arr, R_star_all, nu)


def midas_loglik_fused(C_t_all: np.ndarray,
                       R_t: np.ndarray,
                       log_det_Rt_all: np.ndarray,
                       Intercept: np.ndarray,
                       B_arr: np.ndarray,
                       A_arr: np.ndarray,
                       nu: float,
                       half_nu_minus_n_minus_1: float,
                       const_per_t: float) -> float:
    """Public wrapper: fused per-t C_t^{-1} + R* + S*_t recursion + loglik.

    Eliminates the batch np.linalg.inv(C_t_all) bottleneck by computing each
    C_t^{-1} inside the sequential loop via O(n^2) forward substitution.

    Parameters
    ----------
    C_t_all                 : (T, n, n) lower-triangular Cholesky factors of M_t
    R_t                     : (T, n, n) observed daily RC matrices
    log_det_Rt_all          : (T,) precomputed log|R_t| (data-only, cached)
    Intercept               : (n, n) I - sum A_j A_j' - sum B_i B_i'
    B_arr                   : (p, n, n)
    A_arr                   : (q, n, n)
    nu                      : Wishart degrees of freedom
    half_nu_minus_n_minus_1 : (nu - n - 1) / 2
    const_per_t             : t-independent loglik constant per observation

    Returns
    -------
    loglik : float  raw log-likelihood (positive = good; -1e10 on failure)
    """
    return _midas_loglik_fused(C_t_all, R_t, log_det_Rt_all, Intercept,
                               B_arr, A_arr, nu, half_nu_minus_n_minus_1,
                               const_per_t)


def har_compute_St_vectorized(CC: np.ndarray,
                               A_mats: np.ndarray,
                               R_lags: np.ndarray) -> np.ndarray:
    """Fully vectorized HAR-CAW S_t computation (no sequential dependency).

    HAR-CAW: S_t = CC' + sum_k A_k @ R_k_{t-1} @ A_k.T
    Since S_t does NOT depend on past S_t values, this is embarrassingly
    parallel and can be vectorized across t entirely.

    Parameters
    ----------
    CC      : (n, n) intercept C @ C.T
    A_mats  : (K, n, n) coefficient matrices for K terms (daily, 5d, 10d, 20d)
    R_lags  : (K, T, n, n) lagged realized covariance arrays for each term

    Returns
    -------
    S_t : (T, n, n)
    """
    # S_t[t] = CC + sum_k A_k @ R_lags[k, t] @ A_k.T
    # Use einsum for batched matrix multiply: (T, n, n)
    T = R_lags.shape[1]
    n = CC.shape[0]
    K = A_mats.shape[0]

    S_t = np.broadcast_to(CC, (T, n, n)).copy()
    for k in range(K):
        # A_mats[k]: (n,n), R_lags[k]: (T,n,n)
        # tmp[t] = A @ R_lags[k,t] @ A.T
        tmp = np.einsum('ij,tjk,lk->til', A_mats[k], R_lags[k], A_mats[k])
        S_t += tmp

    S_t = (S_t + S_t.transpose(0, 2, 1)) * 0.5
    return S_t
