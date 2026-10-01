"""
Unit tests for src/benchmarks.py — EWMA and BEKK-GARCH.

Tests cover:
- EWMA: fit path shape, symmetry, lambda persistence, one-step forecast
- EWMA: forecast at all horizons equals S_{t+1} (flat beyond h=1)
- BEKK: parameter count formula, fit keys, Sigma_t shape/symmetry, nu bound
- BEKK: forecast shape and symmetry
- BEKK: diagonal variant
- Edge cases: T=1, extreme lambda
"""
import numpy as np
import pytest
from benchmarks import EWMAModel, BEKKGARCHModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rc(T: int = 30, n: int = 5, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    R = np.empty((T, n, n))
    for t in range(T):
        A = rng.standard_normal((n, n)) * 0.05
        M = A @ A.T + np.eye(n) * 0.01
        R[t] = (M + M.T) / 2
    return R


def _make_returns(T: int = 30, n: int = 5, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, n)) * 0.01


# ===========================================================================
# EWMAModel
# ===========================================================================

class TestEWMAFit:
    def test_fit_St_shape(self):
        T, n = 30, 5
        R_t = _make_rc(T, n)
        model = EWMAModel(lam=0.94)
        res = model.fit(R_t)
        assert res['S_t'].shape == (T, n, n)

    def test_fit_St_symmetric(self):
        T, n = 30, 5
        R_t = _make_rc(T, n)
        model = EWMAModel()
        res = model.fit(R_t)
        for t in range(T):
            np.testing.assert_allclose(res['S_t'][t], res['S_t'][t].T, atol=1e-12)

    def test_fit_sets_fitted_flag(self):
        model = EWMAModel()
        assert not model.fitted
        model.fit(_make_rc())
        assert model.fitted

    def test_first_step_equals_R0(self):
        R_t = _make_rc()
        model = EWMAModel()
        res = model.fit(R_t)
        np.testing.assert_allclose(res['S_t'][0], R_t[0], atol=1e-15)

    def test_lambda_one_persistent(self):
        """With lambda=1, S_t stays at R_t[0] for all t > 0."""
        R_t = _make_rc()
        model = EWMAModel(lam=1.0)
        res = model.fit(R_t)
        for t in range(1, len(R_t)):
            np.testing.assert_allclose(res['S_t'][t], R_t[0], atol=1e-12)

    def test_lambda_zero_is_naive(self):
        """With lambda=0, S_t = R_{t-1} for all t >= 1."""
        R_t = _make_rc()
        model = EWMAModel(lam=0.0)
        res = model.fit(R_t)
        for t in range(1, len(R_t)):
            np.testing.assert_allclose(res['S_t'][t], R_t[t - 1], atol=1e-12)

    def test_finite_St(self):
        R_t = _make_rc()
        model = EWMAModel()
        res = model.fit(R_t)
        assert np.all(np.isfinite(res['S_t']))


class TestEWMAForecast:
    def _fitted_model(self, T: int = 30, n: int = 5) -> EWMAModel:
        model = EWMAModel()
        model.fit(_make_rc(T, n))
        return model

    def test_forecast_shape(self):
        model = self._fitted_model()
        fc = model.forecast(t_origin=25, h=1)
        assert fc.shape == (5, 5)

    def test_forecast_symmetric(self):
        model = self._fitted_model()
        for h in [1, 5, 10]:
            fc = model.forecast(t_origin=20, h=h)
            np.testing.assert_allclose(fc, fc.T, atol=1e-12)

    def test_forecast_flat_across_horizons(self):
        """EWMA multi-step == single-step forecast for all h."""
        model = self._fitted_model()
        fc1 = model.forecast(t_origin=20, h=1)
        for h in [2, 5, 10]:
            fc_h = model.forecast(t_origin=20, h=h)
            np.testing.assert_allclose(fc_h, fc1, atol=1e-12)

    def test_forecast_is_finite(self):
        model = self._fitted_model()
        for h in [1, 5, 10]:
            assert np.all(np.isfinite(model.forecast(t_origin=20, h=h)))

    def test_unfitted_raises(self):
        model = EWMAModel()
        with pytest.raises(AssertionError):
            model.forecast(t_origin=0, h=1)


# ===========================================================================
# BEKKGARCHModel
# ===========================================================================

class TestBEKKParamCount:
    """n_params = n(n+1)/2 + (p+q)*stride."""

    @pytest.mark.parametrize("n,p,q,diag", [
        (5, 1, 1, False),
        (5, 1, 1, True),
        (5, 2, 1, False),
        (3, 1, 1, True),
    ])
    def test_formula(self, n, p, q, diag):
        model = BEKKGARCHModel(n=n, p=p, q=q, diagonal=diag)
        stride = n if diag else n * n
        expected = n * (n + 1) // 2 + (p + q) * stride
        assert model._n_params() == expected

    def test_diagonal_fewer_params(self):
        diag = BEKKGARCHModel(n=5, p=1, q=1, diagonal=True)._n_params()
        full = BEKKGARCHModel(n=5, p=1, q=1, diagonal=False)._n_params()
        assert diag < full


class TestBEKKFit:
    def test_fit_returns_expected_keys(self):
        T, n = 30, 5
        rets = _make_returns(T, n)
        model = BEKKGARCHModel(n=n, p=1, q=1, diagonal=True)
        result = model.fit(rets, verbose=False)
        for key in ('loglik', 'n_params', 'psi', 'Sigma_t', 'converged'):
            assert key in result

    def test_fit_Sigma_t_shape(self):
        T, n = 30, 5
        rets = _make_returns(T, n)
        model = BEKKGARCHModel(n=n, p=1, q=1, diagonal=True)
        result = model.fit(rets, verbose=False)
        assert result['Sigma_t'].shape == (T, n, n)

    def test_fit_Sigma_t_symmetric(self):
        T, n = 30, 5
        rets = _make_returns(T, n)
        model = BEKKGARCHModel(n=n, p=1, q=1, diagonal=True)
        result = model.fit(rets, verbose=False)
        Sigma = result['Sigma_t']
        for t in range(T):
            np.testing.assert_allclose(Sigma[t], Sigma[t].T, atol=1e-10)

    def test_fit_loglik_finite(self):
        rets = _make_returns()
        model = BEKKGARCHModel(n=5, p=1, q=1, diagonal=True)
        result = model.fit(rets, verbose=False)
        assert np.isfinite(result['loglik'])

    def test_fit_sets_fitted_flag(self):
        model = BEKKGARCHModel(n=5, p=1, q=1, diagonal=True)
        assert not model.fitted
        model.fit(_make_returns(), verbose=False)
        assert model.fitted

    def test_fit_n_params_correct(self):
        model = BEKKGARCHModel(n=5, p=1, q=1, diagonal=True)
        result = model.fit(_make_returns(), verbose=False)
        assert result['n_params'] == model._n_params()

    def test_full_model_fit(self):
        """Unrestricted BEKK should also fit without error."""
        T, n = 25, 5
        rets = _make_returns(T, n)
        model = BEKKGARCHModel(n=n, p=1, q=1, diagonal=False)
        result = model.fit(rets, verbose=False)
        assert 'loglik' in result
        assert np.isfinite(result['loglik'])


class TestBEKKForecast:
    def _fitted_model(self, T: int = 30, n: int = 5, diag: bool = True) -> BEKKGARCHModel:
        model = BEKKGARCHModel(n=n, p=1, q=1, diagonal=diag)
        model.fit(_make_returns(T, n), verbose=False)
        return model

    def test_forecast_shape(self):
        model = self._fitted_model()
        fc = model.forecast(t_origin=20, h=1)
        assert fc.shape == (5, 5)

    @pytest.mark.parametrize("h", [1, 5, 10])
    def test_forecast_symmetric(self, h):
        model = self._fitted_model()
        fc = model.forecast(t_origin=20, h=h)
        np.testing.assert_allclose(fc, fc.T, atol=1e-10)

    def test_forecast_is_finite(self):
        model = self._fitted_model()
        for h in [1, 5, 10]:
            fc = model.forecast(t_origin=20, h=h)
            assert np.all(np.isfinite(fc)), f"Forecast h={h} has non-finite values"

    def test_unfitted_raises(self):
        model = BEKKGARCHModel(n=5, p=1, q=1)
        with pytest.raises(AssertionError):
            model.forecast(t_origin=10, h=1)


# ===========================================================================
# Edge cases
# ===========================================================================

def test_ewma_single_obs():
    R_t = _make_rc(T=1)
    model = EWMAModel()
    res = model.fit(R_t)
    assert res['S_t'].shape == (1, 5, 5)


def test_bekk_minimum_T():
    """T must be > n+1 for a valid sample covariance in initialisation."""
    n, T = 5, 20
    rets = _make_returns(T, n)
    model = BEKKGARCHModel(n=n, p=1, q=1, diagonal=True)
    result = model.fit(rets, verbose=False)
    assert 'Sigma_t' in result


def test_ewma_default_lambda():
    assert EWMAModel().lam == 0.94
