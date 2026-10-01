"""
Unit tests for src/fast_recursions.py.

Tests cover output shape / symmetry guarantees, boundary conditions (p=0, q=0),
and numerical consistency with the equivalent pure-numpy loops.
"""
import numpy as np
import pytest
from fast_recursions import caw_compute_St, midas_compute_St_star, har_compute_St_vectorized


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pd(n: int, scale: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)) * scale
    M = A @ A.T + np.eye(n) * scale
    return (M + M.T) / 2


def _make_rc(T: int, n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    R = np.empty((T, n, n))
    for t in range(T):
        A = rng.standard_normal((n, n)) * 0.05
        M = A @ A.T + np.eye(n) * 0.01
        R[t] = (M + M.T) / 2
    return R


def _numpy_caw_recursion(CC, B_arr, A_arr, R_t, S_init):
    """Reference pure-numpy CAW recursion."""
    T, n, _ = R_t.shape
    p, q = B_arr.shape[0], A_arr.shape[0]
    S_t = np.empty_like(R_t)
    for t in range(T):
        St = CC.copy()
        for i in range(p):
            lag = t - 1 - i
            prev = S_t[lag] if lag >= 0 else S_init
            St += B_arr[i] @ prev @ B_arr[i].T
        for j in range(q):
            lag = t - 1 - j
            prev = R_t[lag] if lag >= 0 else S_init
            St += A_arr[j] @ prev @ A_arr[j].T
        S_t[t] = (St + St.T) * 0.5
    return S_t


def _numpy_midas_star_recursion(Intercept, B_arr, A_arr, R_t_star, I_n):
    """Reference pure-numpy MIDAS star recursion."""
    T, n, _ = R_t_star.shape
    p, q = B_arr.shape[0], A_arr.shape[0]
    S_star = np.empty_like(R_t_star)
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


# ---------------------------------------------------------------------------
# caw_compute_St
# ---------------------------------------------------------------------------

class TestCawComputeSt:
    n = 5

    def test_output_shape(self):
        T = 30
        n = self.n
        R_t = _make_rc(T, n)
        CC = _make_pd(n)
        A_arr = np.stack([np.eye(n) * 0.3], axis=0)
        B_arr = np.stack([np.eye(n) * 0.5], axis=0)
        S_init = _make_pd(n)
        S_t = caw_compute_St(CC, B_arr, A_arr, R_t, S_init)
        assert S_t.shape == (T, n, n)

    def test_output_is_symmetric(self):
        T, n = 20, self.n
        R_t = _make_rc(T, n)
        CC = _make_pd(n)
        A_arr = np.stack([np.eye(n) * 0.3], axis=0)
        B_arr = np.empty((0, n, n))
        S_init = _make_pd(n)
        S_t = caw_compute_St(CC, B_arr, A_arr, R_t, S_init)
        for t in range(T):
            np.testing.assert_allclose(S_t[t], S_t[t].T, atol=1e-12)

    def test_matches_numpy_reference(self):
        """Numba kernel must match pure-numpy reference implementation."""
        T, n = 25, self.n
        R_t = _make_rc(T, n)
        CC = _make_pd(n)
        A_arr = np.stack([np.eye(n) * 0.3, np.eye(n) * 0.1], axis=0)
        B_arr = np.stack([np.eye(n) * 0.4], axis=0)
        S_init = np.eye(n) * 0.01
        S_ref = _numpy_caw_recursion(CC, B_arr, A_arr, R_t, S_init)
        S_jit = caw_compute_St(CC, B_arr, A_arr, R_t, S_init)
        np.testing.assert_allclose(S_jit, S_ref, atol=1e-10)

    def test_p_zero(self):
        """CAW(0,q): no B matrices — S_t depends only on lagged R_t."""
        T, n = 20, self.n
        R_t = _make_rc(T, n)
        CC = _make_pd(n)
        A_arr = np.stack([np.eye(n) * 0.4], axis=0)
        B_arr = np.empty((0, n, n))
        S_init = np.eye(n) * 0.01
        S_t = caw_compute_St(CC, B_arr, A_arr, R_t, S_init)
        assert S_t.shape == (T, n, n)
        # All S_t should be positive (CC + A R A' is at least PSD)
        assert np.linalg.eigvalsh(S_t).min() >= 0

    def test_output_is_positive_semidefinite(self):
        T, n = 30, self.n
        R_t = _make_rc(T, n)
        CC = _make_pd(n)
        A_arr = np.stack([np.eye(n) * 0.2], axis=0)
        B_arr = np.stack([np.eye(n) * 0.3], axis=0)
        S_init = _make_pd(n)
        S_t = caw_compute_St(CC, B_arr, A_arr, R_t, S_init)
        min_eig = np.linalg.eigvalsh(S_t).min()
        assert min_eig >= -1e-8

    def test_intercept_only_constant_output(self):
        """A=B=0 ⟹ S_t = CC' for all t."""
        T, n = 10, self.n
        R_t = _make_rc(T, n)
        CC = _make_pd(n)
        A_arr = np.empty((0, n, n))
        B_arr = np.empty((0, n, n))
        S_init = np.eye(n)
        S_t = caw_compute_St(CC, B_arr, A_arr, R_t, S_init)
        for t in range(T):
            np.testing.assert_allclose(S_t[t], CC, atol=1e-12)

    def test_pre_sample_initializer_used_at_t0(self):
        """For CAW(1,0), t=0 should use S_init because no S_t[-1] exists."""
        T, n = 5, 3
        CC = np.eye(n) * 0.005
        B_arr = np.stack([np.eye(n) * 0.5], axis=0)
        A_arr = np.empty((0, n, n))
        R_t = np.zeros((T, n, n))
        S_init = np.eye(n) * 0.1
        S_t = caw_compute_St(CC, B_arr, A_arr, R_t, S_init)
        # At t=0: S_t[0] = CC + B @ S_init @ B.T
        expected_0 = CC + B_arr[0] @ S_init @ B_arr[0].T
        expected_0 = (expected_0 + expected_0.T) * 0.5
        np.testing.assert_allclose(S_t[0], expected_0, atol=1e-12)


# ---------------------------------------------------------------------------
# midas_compute_St_star
# ---------------------------------------------------------------------------

class TestMidasComputeStStar:
    n = 5

    def test_output_shape(self):
        T, n = 30, self.n
        I_n = np.eye(n)
        A_arr = np.stack([np.eye(n) * 0.3], axis=0)
        B_arr = np.stack([np.eye(n) * 0.4], axis=0)
        Intercept = I_n - A_arr[0] @ A_arr[0].T - B_arr[0] @ B_arr[0].T
        R_star = np.tile(I_n, (T, 1, 1))
        S_star = midas_compute_St_star(Intercept, B_arr, A_arr, R_star, I_n)
        assert S_star.shape == (T, n, n)

    def test_output_is_symmetric(self):
        T, n = 20, self.n
        I_n = np.eye(n)
        A_arr = np.stack([np.eye(n) * 0.2], axis=0)
        B_arr = np.stack([np.eye(n) * 0.3], axis=0)
        Intercept = I_n - A_arr[0] @ A_arr[0].T - B_arr[0] @ B_arr[0].T
        rng = np.random.default_rng(5)
        R_star = np.array([I_n + rng.standard_normal((n, n)) * 0.01 for _ in range(T)])
        R_star = (R_star + R_star.transpose(0, 2, 1)) / 2
        S_star = midas_compute_St_star(Intercept, B_arr, A_arr, R_star, I_n)
        for t in range(T):
            np.testing.assert_allclose(S_star[t], S_star[t].T, atol=1e-12)

    def test_matches_numpy_reference(self):
        T, n = 25, self.n
        I_n = np.eye(n)
        A_arr = np.stack([np.eye(n) * 0.2], axis=0)
        B_arr = np.stack([np.eye(n) * 0.3], axis=0)
        Intercept = I_n - A_arr[0] @ A_arr[0].T - B_arr[0] @ B_arr[0].T
        rng = np.random.default_rng(9)
        R_star = np.array([I_n + rng.standard_normal((n, n)) * 0.05 for _ in range(T)])
        R_star = (R_star + R_star.transpose(0, 2, 1)) / 2
        S_ref = _numpy_midas_star_recursion(Intercept, B_arr, A_arr, R_star, I_n)
        S_jit = midas_compute_St_star(Intercept, B_arr, A_arr, R_star, I_n)
        np.testing.assert_allclose(S_jit, S_ref, atol=1e-10)

    def test_intercept_implies_unit_mean(self):
        """If R_t^* = I_n for all t and A, B small, S_t^* should converge near I_n."""
        T, n = 200, 3
        I_n = np.eye(n)
        a, b = 0.2, 0.3
        A_arr = np.stack([I_n * a], axis=0)
        B_arr = np.stack([I_n * b], axis=0)
        Intercept = I_n - a**2 * I_n - b**2 * I_n
        R_star = np.tile(I_n, (T, 1, 1))
        S_star = midas_compute_St_star(Intercept, B_arr, A_arr, R_star, I_n)
        # At stationarity, E[S_t^*] = I_n
        np.testing.assert_allclose(S_star[-1], I_n, atol=1e-2)


# ---------------------------------------------------------------------------
# har_compute_St_vectorized
# ---------------------------------------------------------------------------

class TestHarComputeStVectorized:
    n = 5

    def test_output_shape(self):
        T, n = 30, self.n
        CC = np.eye(n) * 0.01
        K = 4  # daily, w5, w10, w20
        rng = np.random.default_rng(2)
        A_mats = np.stack([rng.standard_normal((n, n)) * 0.1 for _ in range(K)], axis=0)
        R_lags = np.zeros((K, T, n, n))
        for k in range(K):
            for t in range(T):
                r = rng.standard_normal((n, n)) * 0.05
                R_lags[k, t] = (r + r.T) / 2 + np.eye(n) * 0.01
        S_t = har_compute_St_vectorized(CC, A_mats, R_lags)
        assert S_t.shape == (T, n, n)

    def test_output_is_symmetric(self):
        T, n = 20, self.n
        rng = np.random.default_rng(3)
        CC = np.eye(n) * 0.01
        K = 4
        A_mats = np.stack([np.eye(n) * 0.1 for _ in range(K)], axis=0)
        R_lags = np.zeros((K, T, n, n))
        for k in range(K):
            for t in range(T):
                r = rng.standard_normal((n, n)) * 0.05
                R_lags[k, t] = (r + r.T) / 2 + np.eye(n) * 0.01
        S_t = har_compute_St_vectorized(CC, A_mats, R_lags)
        for t in range(T):
            np.testing.assert_allclose(S_t[t], S_t[t].T, atol=1e-12)

    def test_no_sequential_dependency(self):
        """HAR S_t depends only on R_lags, not on previous S. Permuting time should
        give the permuted S."""
        T, n = 15, 3
        rng = np.random.default_rng(7)
        CC = np.eye(n) * 0.002
        K = 2
        A_mats = np.stack([np.eye(n) * 0.3, np.eye(n) * 0.2], axis=0)
        R_lags = np.zeros((K, T, n, n))
        for k in range(K):
            for t in range(T):
                r = rng.standard_normal((n, n)) * 0.03
                R_lags[k, t] = (r + r.T) / 2 + np.eye(n) * 0.005
        S_t_orig = har_compute_St_vectorized(CC, A_mats, R_lags)
        perm = np.random.default_rng(8).permutation(T)
        R_lags_perm = R_lags[:, perm, :, :]
        S_t_perm = har_compute_St_vectorized(CC, A_mats, R_lags_perm)
        np.testing.assert_allclose(S_t_perm, S_t_orig[perm], atol=1e-12)

    def test_intercept_only_gives_constant(self):
        """A_mats = 0 ⟹ S_t[t] = CC for all t."""
        T, n = 10, self.n
        CC = np.eye(n) * 0.007
        K = 4
        A_mats = np.zeros((K, n, n))
        R_lags = np.ones((K, T, n, n)) * 99.0  # large values that should be ignored
        S_t = har_compute_St_vectorized(CC, A_mats, R_lags)
        for t in range(T):
            np.testing.assert_allclose(S_t[t], CC, atol=1e-12)

    def test_positive_semidefinite(self):
        """With diagonal A_mats and PSD R_lags, result must be PSD."""
        T, n = 25, self.n
        rng = np.random.default_rng(4)
        CC = np.eye(n) * 0.005
        K = 4
        A_mats = np.stack([np.eye(n) * 0.2 for _ in range(K)], axis=0)
        R_lags = np.zeros((K, T, n, n))
        for k in range(K):
            for t in range(T):
                a = rng.standard_normal((n, n)) * 0.02
                R_lags[k, t] = a @ a.T + np.eye(n) * 0.001
        S_t = har_compute_St_vectorized(CC, A_mats, R_lags)
        min_eig = np.linalg.eigvalsh(S_t).min()
        assert min_eig >= -1e-8
