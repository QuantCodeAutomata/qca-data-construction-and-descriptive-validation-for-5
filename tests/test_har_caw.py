"""
Unit tests for src/har_caw.py — HAR-CAW model.

Tests cover:
- Parameter counting (exact formula)
- _compute_St shape, symmetry
- Wishart log-likelihood finiteness
- fit() return keys, S_t shape, nu bound, C positivity
- standardized_residuals shape
- forecast shape and symmetry
- Schwarz criterion formula
- edge cases: diagonal=True, tiny data
"""
import numpy as np
import pytest
from har_caw import HARCAWModel


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


def _make_har_sums(R_t: np.ndarray) -> dict:
    T, n, _ = R_t.shape
    w5  = np.zeros_like(R_t)
    w10 = np.zeros_like(R_t)
    w20 = np.zeros_like(R_t)
    for t in range(T):
        if t > 0:
            w5[t]  = R_t[max(0, t - 5):t].sum(axis=0)
            w10[t] = R_t[max(0, t - 10):t].sum(axis=0)
            w20[t] = R_t[max(0, t - 20):t].sum(axis=0)
    return {'w5': w5, 'w10': w10, 'w20': w20}


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

class TestParameterCount:
    """n_params = 1 (nu) + n(n+1)/2 (C) + 4 * stride (A, A_w, A_bw, A_m)."""

    @pytest.mark.parametrize("n,diag", [
        (5, False),
        (5, True),
        (3, False),
        (3, True),
    ])
    def test_formula(self, n, diag):
        model = HARCAWModel(n=n, diagonal=diag)
        stride = n if diag else n * n
        expected = 1 + n * (n + 1) // 2 + 4 * stride
        assert model._n_params() == expected

    def test_diagonal_fewer_params(self):
        diag = HARCAWModel(n=5, diagonal=True)._n_params()
        full = HARCAWModel(n=5, diagonal=False)._n_params()
        assert diag < full


# ---------------------------------------------------------------------------
# _compute_St
# ---------------------------------------------------------------------------

class TestComputeSt:
    def test_shape(self):
        T, n = 30, 5
        model = HARCAWModel(n=n)
        R_t = _make_rc(T, n)
        har = _make_har_sums(R_t)
        psi0 = model._init_params(R_t)
        St = model._compute_St(psi0, R_t, har)
        assert St is not None
        assert St.shape == (T, n, n)

    def test_symmetric(self):
        T, n = 25, 5
        model = HARCAWModel(n=n)
        R_t = _make_rc(T, n)
        har = _make_har_sums(R_t)
        psi0 = model._init_params(R_t)
        St = model._compute_St(psi0, R_t, har)
        assert St is not None
        for t in range(T):
            np.testing.assert_allclose(St[t], St[t].T, atol=1e-12)

    def test_diagonal_model_shape(self):
        T, n = 25, 5
        model = HARCAWModel(n=n, diagonal=True)
        R_t = _make_rc(T, n)
        har = _make_har_sums(R_t)
        psi0 = model._init_params(R_t)
        St = model._compute_St(psi0, R_t, har)
        assert St is not None
        assert St.shape == (T, n, n)


# ---------------------------------------------------------------------------
# Wishart log-likelihood
# ---------------------------------------------------------------------------

class TestLogLikelihood:
    def test_returns_finite(self):
        T, n = 30, 5
        model = HARCAWModel(n=n)
        R_t = _make_rc(T, n)
        har = _make_har_sums(R_t)
        psi0 = model._init_params(R_t)
        neg_ll = model._wishart_loglik(psi0, R_t, har)
        assert np.isfinite(neg_ll)

    def test_not_penalty_for_valid_params(self):
        """Valid params must return a finite value well below the 1e10 penalty sentinel.

        The Wishart log-density is not bounded to be negative: for small-scale
        matrices the density can exceed 1 making log-density positive, so
        the returned -loglik can be negative. Only finiteness is guaranteed.
        """
        T, n = 30, 5
        model = HARCAWModel(n=n)
        R_t = _make_rc(T, n)
        har = _make_har_sums(R_t)
        psi0 = model._init_params(R_t)
        neg_ll = model._wishart_loglik(psi0, R_t, har)
        assert np.isfinite(neg_ll)
        assert neg_ll < 1e9  # well below the 1e10 penalty sentinel


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("diagonal", [False, True])
def test_fit_returns_expected_keys(diagonal):
    T, n = 30, 5
    R_t = _make_rc(T, n)
    har = _make_har_sums(R_t)
    model = HARCAWModel(n=n, diagonal=diagonal)
    result = model.fit(R_t, har, verbose=False)
    for key in ('loglik', 'n_params', 'psi', 'nu', 'S_t', 'converged'):
        assert key in result, f"Missing key: {key}"


@pytest.mark.parametrize("diagonal", [False, True])
def test_fit_St_shape_and_symmetry(diagonal):
    T, n = 30, 5
    R_t = _make_rc(T, n)
    har = _make_har_sums(R_t)
    model = HARCAWModel(n=n, diagonal=diagonal)
    result = model.fit(R_t, har, verbose=False)
    S_t = result['S_t']
    assert S_t.shape == (T, n, n)
    for t in range(T):
        np.testing.assert_allclose(S_t[t], S_t[t].T, atol=1e-10)


def test_fit_sets_fitted_flag():
    model = HARCAWModel(n=5)
    R_t = _make_rc()
    har = _make_har_sums(R_t)
    assert not model.fitted
    model.fit(R_t, har, verbose=False)
    assert model.fitted


def test_fit_loglik_is_finite():
    model = HARCAWModel(n=5)
    R_t = _make_rc()
    har = _make_har_sums(R_t)
    result = model.fit(R_t, har, verbose=False)
    assert np.isfinite(result['loglik'])


def test_fit_nu_exceeds_n_plus_1():
    n = 5
    model = HARCAWModel(n=n)
    R_t = _make_rc(n=n)
    har = _make_har_sums(R_t)
    result = model.fit(R_t, har, verbose=False)
    assert result['nu'] > n + 1


def test_fit_n_params_in_result():
    model = HARCAWModel(n=5, diagonal=True)
    R_t = _make_rc()
    har = _make_har_sums(R_t)
    result = model.fit(R_t, har, verbose=False)
    assert result['n_params'] == model._n_params()


def test_fit_C_positive_diagonal():
    model = HARCAWModel(n=5)
    R_t = _make_rc()
    har = _make_har_sums(R_t)
    model.fit(R_t, har, verbose=False)
    C = model.C_hat
    assert all(C[i, i] > 0 for i in range(5))
    for i in range(5):
        for j in range(i + 1, 5):
            assert C[i, j] == 0.0


def test_warm_start_accepted():
    model = HARCAWModel(n=5, diagonal=True)
    R_t = _make_rc()
    har = _make_har_sums(R_t)
    res1 = model.fit(R_t, har, verbose=False)
    res2 = model.fit(R_t, har, verbose=False, warm_start_psi=res1['psi'])
    assert np.isfinite(res2['loglik'])


# ---------------------------------------------------------------------------
# Standardized residuals
# ---------------------------------------------------------------------------

def test_standardized_residuals_shape():
    T, n = 30, 5
    model = HARCAWModel(n=n)
    R_t = _make_rc(T, n)
    har = _make_har_sums(R_t)
    model.fit(R_t, har, verbose=False)
    resids = model.standardized_residuals(R_t, har)
    assert resids.shape == (T, 15)


def test_standardized_residuals_finite():
    T, n = 30, 5
    model = HARCAWModel(n=n, diagonal=True)
    R_t = _make_rc(T, n)
    har = _make_har_sums(R_t)
    model.fit(R_t, har, verbose=False)
    resids = model.standardized_residuals(R_t, har)
    assert np.all(np.isfinite(resids))


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h", [1, 5, 10])
def test_forecast_shape(h):
    T, n = 30, 5
    model = HARCAWModel(n=n, diagonal=True)
    R_t = _make_rc(T, n)
    har = _make_har_sums(R_t)
    model.fit(R_t, har, verbose=False)
    fc = model.forecast(R_t, har, h=h)
    assert fc.shape == (n, n)


@pytest.mark.parametrize("h", [1, 5])
def test_forecast_is_symmetric(h):
    T, n = 30, 5
    model = HARCAWModel(n=n, diagonal=True)
    R_t = _make_rc(T, n)
    har = _make_har_sums(R_t)
    model.fit(R_t, har, verbose=False)
    fc = model.forecast(R_t, har, h=h)
    np.testing.assert_allclose(fc, fc.T, atol=1e-10)


def test_forecast_is_finite():
    T, n = 30, 5
    model = HARCAWModel(n=n)
    R_t = _make_rc(T, n)
    har = _make_har_sums(R_t)
    model.fit(R_t, har, verbose=False)
    for h in [1, 5, 10]:
        fc = model.forecast(R_t, har, h=h)
        assert np.all(np.isfinite(fc))


# ---------------------------------------------------------------------------
# Schwarz criterion
# ---------------------------------------------------------------------------

def test_schwarz_formula():
    model = HARCAWModel(n=5, diagonal=True)
    loglik = -300.0
    T = 100
    expected = 2 * 300.0 + model._n_params() * np.log(T)
    sc = model.schwarz_criterion(loglik, T)
    np.testing.assert_allclose(sc, expected, rtol=1e-10)


def test_schwarz_higher_loglik_lower_sc():
    model = HARCAWModel(n=5)
    T = 1916
    sc_good = model.schwarz_criterion(-7337.8, T)
    sc_bad  = model.schwarz_criterion(-7700.0, T)
    assert sc_good < sc_bad


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_small_T_does_not_crash():
    """T=5 is too small for 20-day sums but should not raise an exception."""
    T, n = 5, 5
    R_t = _make_rc(T, n)
    har = _make_har_sums(R_t)
    model = HARCAWModel(n=n, diagonal=True)
    result = model.fit(R_t, har, verbose=False)
    assert 'loglik' in result


def test_tiny_variance_no_crash():
    T, n = 25, 5
    R_t = _make_rc(T, n) * 1e-8
    har = _make_har_sums(R_t)
    model = HARCAWModel(n=n, diagonal=True)
    psi0 = model._init_params(R_t)
    neg_ll = model._wishart_loglik(psi0, R_t, har)
    assert np.isfinite(neg_ll)
