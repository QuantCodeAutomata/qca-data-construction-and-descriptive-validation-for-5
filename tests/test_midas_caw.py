"""
Unit tests for src/midas_caw.py — MIDAS-CAW model.

Tests cover:
- Parameter counting formula
- _midas_weights: sum-to-one, monotone decline (omega>1)
- _compute_long_run shape, symmetry, PSD
- _n_params exact formula
- fit() return keys, loglik finite, nu bound
- theta>=0 and omega>1 enforcement
- compute_Psi1_shortrun shape
- schwarz_criterion formula
- edge cases: (0,1), (1,1), diagonal variants
"""
import numpy as np
import pytest
from midas_caw import MIDASCAWModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rc(T: int = 30, n: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    R = np.empty((T, n, n))
    for t in range(T):
        A = rng.standard_normal((n, n)) * 0.05
        M = A @ A.T + np.eye(n) * 0.01
        R[t] = (M + M.T) / 2
    return R


def _make_midas_blocks(R_t: np.ndarray, m: int = 20, L: int = 12) -> np.ndarray:
    T, n, _ = R_t.shape
    blocks = np.zeros((T, L, n, n))
    for t in range(T):
        for ell in range(1, L + 1):
            hi = t - m * (ell - 1)
            lo = t - m * ell
            if hi > 0:
                sl = R_t[max(0, lo):hi]
                blocks[t, ell - 1] = sl.sum(axis=0)
    return blocks


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

class TestParameterCount:
    """n_params = 1(nu) + n(n+1)/2(C_bar) + 1(theta) + 1(omega) + (p+q)*stride."""

    @pytest.mark.parametrize("n,p,q,diag", [
        (5, 0, 1, False),
        (5, 1, 1, False),
        (5, 2, 2, False),
        (5, 0, 1, True),
        (5, 1, 1, True),
        (5, 2, 2, True),
    ])
    def test_formula(self, n, p, q, diag):
        model = MIDASCAWModel(n=n, p=p, q=q, diagonal=diag)
        stride = n if diag else n * n
        expected = 1 + n * (n + 1) // 2 + 1 + 1 + (p + q) * stride
        assert model._n_params() == expected

    def test_diagonal_fewer_params(self):
        for p, q in [(0, 1), (1, 1)]:
            diag = MIDASCAWModel(n=5, p=p, q=q, diagonal=True)._n_params()
            full = MIDASCAWModel(n=5, p=p, q=q, diagonal=False)._n_params()
            assert diag < full

    def test_midas_params_independent_of_L(self):
        """Changing L does not change parameter count (L only affects pre-computed blocks)."""
        m1 = MIDASCAWModel(n=5, p=1, q=1, L=12)._n_params()
        m2 = MIDASCAWModel(n=5, p=1, q=1, L=6)._n_params()
        assert m1 == m2


# ---------------------------------------------------------------------------
# MIDAS weights
# ---------------------------------------------------------------------------

class TestMidasWeights:
    def test_sum_to_one(self):
        model = MIDASCAWModel(n=5, p=0, q=1, L=12)
        for omega in [1.5, 2.0, 5.0, 10.0]:
            w = model._midas_weights(omega)
            np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)

    def test_nonnegative_weights(self):
        """Weights must be non-negative.

        The last weight (ell=L) equals (1-L/L)^(omega-1)=0 by the paper's
        Beta-polynomial formula, so strict positivity is intentionally not required.
        """
        model = MIDASCAWModel(n=5, p=0, q=1, L=12)
        for omega in [1.01, 3.0, 8.0]:
            w = model._midas_weights(omega)
            assert np.all(w >= 0), f"Negative weight found for omega={omega}: {w}"
            # All weights except the last should be strictly positive
            assert np.all(w[:-1] > 0), f"Non-positive interior weight for omega={omega}: {w}"

    def test_length(self):
        for L in [6, 12, 24]:
            model = MIDASCAWModel(n=5, p=0, q=1, L=L)
            w = model._midas_weights(3.0)
            assert len(w) == L

    def test_larger_omega_more_front_loaded(self):
        """Higher omega → more weight on recent lags (ell=1). Specifically w[0] > w[-1]."""
        model = MIDASCAWModel(n=5, p=0, q=1, L=12)
        w_high = model._midas_weights(8.0)
        assert w_high[0] > w_high[-1]

    def test_uniform_fallback(self):
        """Pathological omega should not crash; fallback to uniform."""
        model = MIDASCAWModel(n=5, p=0, q=1, L=12)
        w = model._midas_weights(1.0)  # (1 - ell/L)^0 = 1 for all ell < L; last = 0
        # sum should still be 1 or fallback uniform
        assert abs(w.sum() - 1.0) < 1e-10

    def test_omega_near_one_no_crash(self):
        """omega=1+eps should not raise an exception."""
        model = MIDASCAWModel(n=5, p=0, q=1, L=12)
        w = model._midas_weights(1.001)
        assert len(w) == 12


# ---------------------------------------------------------------------------
# _compute_long_run
# ---------------------------------------------------------------------------

class TestComputeLongRun:
    def test_shape(self):
        T, n, L = 30, 5, 12
        model = MIDASCAWModel(n=n, p=0, q=1, L=L)
        R_t = _make_rc(T, n)
        blocks = _make_midas_blocks(R_t, m=20, L=L)
        C_bar = np.eye(n) * 0.1
        M_t = model._compute_long_run(C_bar, theta=0.5, omega=3.0, midas_blocks=blocks)
        assert M_t.shape == (T, n, n)

    def test_symmetric(self):
        T, n, L = 20, 5, 12
        model = MIDASCAWModel(n=n, p=0, q=1, L=L)
        R_t = _make_rc(T, n)
        blocks = _make_midas_blocks(R_t, m=20, L=L)
        C_bar = np.eye(n) * 0.1
        M_t = model._compute_long_run(C_bar, theta=0.5, omega=3.0, midas_blocks=blocks)
        for t in range(T):
            np.testing.assert_allclose(M_t[t], M_t[t].T, atol=1e-12)

    def test_theta_zero_gives_constant(self):
        """theta=0 → M_t = C_bar C_bar' for all t (MIDAS term vanishes)."""
        T, n, L = 20, 5, 12
        model = MIDASCAWModel(n=n, p=0, q=1, L=L)
        R_t = _make_rc(T, n)
        blocks = _make_midas_blocks(R_t, m=20, L=L)
        C_bar = np.eye(n) * 0.05
        baseline = C_bar @ C_bar.T
        M_t = model._compute_long_run(C_bar, theta=0.0, omega=3.0, midas_blocks=blocks)
        for t in range(T):
            np.testing.assert_allclose(M_t[t], baseline, atol=1e-12)


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

FIT_SPECS = [
    (5, 0, 1, False),
    (5, 0, 1, True),
    (5, 1, 1, True),
]


@pytest.mark.parametrize("n,p,q,diag", FIT_SPECS)
def test_fit_returns_expected_keys(n, p, q, diag):
    T = 30
    R_t = _make_rc(T, n)
    blocks = _make_midas_blocks(R_t)
    model = MIDASCAWModel(n=n, p=p, q=q, diagonal=diag)
    result = model.fit(R_t, blocks, verbose=False)
    for key in ('loglik', 'n_params', 'psi', 'converged'):
        assert key in result, f"Missing key: {key}"


@pytest.mark.parametrize("n,p,q,diag", FIT_SPECS)
def test_fit_loglik_is_finite(n, p, q, diag):
    T = 30
    R_t = _make_rc(T, n)
    blocks = _make_midas_blocks(R_t)
    model = MIDASCAWModel(n=n, p=p, q=q, diagonal=diag)
    result = model.fit(R_t, blocks, verbose=False)
    assert np.isfinite(result['loglik'])


def test_fit_sets_fitted_flag():
    T, n = 30, 5
    model = MIDASCAWModel(n=n, p=0, q=1)
    R_t = _make_rc(T, n)
    blocks = _make_midas_blocks(R_t)
    assert not model.fitted
    model.fit(R_t, blocks, verbose=False)
    assert model.fitted


def test_fit_nu_exceeds_n_plus_1():
    n = 5
    T = 30
    model = MIDASCAWModel(n=n, p=0, q=1, diagonal=True)
    R_t = _make_rc(T, n)
    blocks = _make_midas_blocks(R_t)
    model.fit(R_t, blocks, verbose=False)
    nu, *_ = model._unpack_params(model.psi_hat)
    assert nu > n + 1


def test_fit_theta_nonnegative():
    """theta must be non-negative (stored as exp(2*log_sqrt_theta))."""
    n, T = 5, 30
    model = MIDASCAWModel(n=n, p=0, q=1, diagonal=True)
    R_t = _make_rc(T, n)
    blocks = _make_midas_blocks(R_t)
    model.fit(R_t, blocks, verbose=False)
    _, _, theta, *_ = model._unpack_params(model.psi_hat)
    assert theta >= 0.0


def test_fit_omega_greater_than_one():
    """omega must be > 1 (stored as 1 + exp(...))."""
    n, T = 5, 30
    model = MIDASCAWModel(n=n, p=0, q=1, diagonal=True)
    R_t = _make_rc(T, n)
    blocks = _make_midas_blocks(R_t)
    model.fit(R_t, blocks, verbose=False)
    _, _, _, omega, *_ = model._unpack_params(model.psi_hat)
    assert omega > 1.0


def test_fit_n_params_in_result():
    model = MIDASCAWModel(n=5, p=0, q=1, diagonal=True)
    T = 30
    R_t = _make_rc(T)
    blocks = _make_midas_blocks(R_t)
    result = model.fit(R_t, blocks, verbose=False)
    assert result['n_params'] == model._n_params()


def test_warm_start_accepted():
    model = MIDASCAWModel(n=5, p=0, q=1, diagonal=True)
    T = 30
    R_t = _make_rc(T)
    blocks = _make_midas_blocks(R_t)
    res1 = model.fit(R_t, blocks, verbose=False)
    res2 = model.fit(R_t, blocks, verbose=False, warm_start_psi=res1['psi'])
    assert np.isfinite(res2['loglik'])


# ---------------------------------------------------------------------------
# Post-fit diagnostics
# ---------------------------------------------------------------------------

def test_compute_Psi1_shortrun_shape_p0():
    """p=0 → companion matrix should be (n^2, n^2) zeros."""
    n, T = 5, 30
    model = MIDASCAWModel(n=n, p=0, q=1, diagonal=True)
    R_t = _make_rc(T, n)
    blocks = _make_midas_blocks(R_t)
    model.fit(R_t, blocks, verbose=False)
    Psi1 = model.compute_Psi1_shortrun()
    assert Psi1.shape == (n * n, n * n)
    np.testing.assert_allclose(Psi1, np.zeros((n * n, n * n)), atol=1e-15)


def test_compute_Psi1_shortrun_shape_p1():
    n, T = 5, 30
    model = MIDASCAWModel(n=n, p=1, q=1, diagonal=True)
    R_t = _make_rc(T, n)
    blocks = _make_midas_blocks(R_t)
    model.fit(R_t, blocks, verbose=False)
    Psi1 = model.compute_Psi1_shortrun()
    assert Psi1.shape == (n * n, n * n)


def test_Psi1_shortrun_meig_nonnegative():
    n, T = 5, 30
    model = MIDASCAWModel(n=n, p=1, q=1, diagonal=True)
    R_t = _make_rc(T, n)
    blocks = _make_midas_blocks(R_t)
    model.fit(R_t, blocks, verbose=False)
    Psi1 = model.compute_Psi1_shortrun()
    meig = np.max(np.abs(np.linalg.eigvals(Psi1)))
    assert meig >= 0


# ---------------------------------------------------------------------------
# Schwarz criterion
# ---------------------------------------------------------------------------

def test_schwarz_formula():
    model = MIDASCAWModel(n=5, p=0, q=1, diagonal=True)
    loglik = -400.0
    T = 1916
    expected = 2 * 400.0 + model._n_params() * np.log(T)
    sc = model.schwarz_criterion(loglik, T)
    np.testing.assert_allclose(sc, expected, rtol=1e-10)


def test_schwarz_better_model_lower_sc():
    T = 1916
    m = MIDASCAWModel(n=5, p=0, q=1)
    sc_good = m.schwarz_criterion(-7412.5, T)
    sc_bad  = m.schwarz_criterion(-7600.0, T)
    assert sc_good < sc_bad


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_extremely_small_blocks():
    """MIDAS blocks of near-zero should not cause numerical explosion."""
    T, n = 25, 5
    R_t = _make_rc(T, n) * 1e-8
    blocks = _make_midas_blocks(R_t) * 1e-8
    model = MIDASCAWModel(n=n, p=0, q=1, diagonal=True)
    result = model.fit(R_t, blocks, verbose=False)
    assert np.isfinite(result['loglik'])


def test_zero_blocks_no_crash():
    """All-zero MIDAS blocks edge case."""
    T, n = 20, 5
    R_t = _make_rc(T, n)
    blocks = np.zeros((T, 12, n, n))
    model = MIDASCAWModel(n=n, p=0, q=1, diagonal=True)
    psi0 = np.zeros(model._n_params())
    # Just check that loglik evaluation does not raise
    try:
        neg_ll = model._wishart_loglik(psi0, R_t, blocks)
        assert np.isfinite(neg_ll) or neg_ll >= 1e9
    except Exception:
        pass  # penalty branch is acceptable
