"""
Shared pytest fixtures for the CAW realized-covariance test suite.

All fixtures use small, fast, deterministic datasets (T=50, n=5).
No external data or network access is required.
"""
import sys
import os
import numpy as np
import pytest

# Ensure src/ is on the path for every test module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spd(n: int, scale: float = 1.0) -> np.ndarray:
    """Return a random symmetric positive-definite n×n matrix."""
    A = np.random.randn(n, n) * scale
    M = A @ A.T + np.eye(n) * 0.5
    return (M + M.T) / 2


def _make_rc_sequence(T: int = 50, n: int = 5, seed: int = 0) -> np.ndarray:
    """Simulate T positive-definite n×n realized covariance matrices."""
    rng = np.random.default_rng(seed)
    R_t = np.empty((T, n, n))
    base = np.eye(n) * 0.01
    for t in range(T):
        A = rng.standard_normal((n, n)) * 0.05
        M = base + A @ A.T
        R_t[t] = (M + M.T) / 2
    return R_t


def _make_har_sums(R_t: np.ndarray) -> dict:
    """Build w5 / w10 / w20 rolling-sum dicts aligned with R_t."""
    T, n, _ = R_t.shape
    w5  = np.zeros_like(R_t)
    w10 = np.zeros_like(R_t)
    w20 = np.zeros_like(R_t)
    for t in range(T):
        w5[t]  = R_t[max(0, t-5):t].sum(axis=0) if t > 0 else np.zeros((n, n))
        w10[t] = R_t[max(0, t-10):t].sum(axis=0) if t > 0 else np.zeros((n, n))
        w20[t] = R_t[max(0, t-20):t].sum(axis=0) if t > 0 else np.zeros((n, n))
    return {'w5': w5, 'w10': w10, 'w20': w20}


def _make_midas_blocks(R_t: np.ndarray, m: int = 20, L: int = 12) -> np.ndarray:
    """Build (T, L, n, n) MIDAS block sums for all t in R_t."""
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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def n() -> int:
    return 5


@pytest.fixture(scope='session')
def T_small() -> int:
    return 50


@pytest.fixture(scope='session')
def R_t_small(T_small, n) -> np.ndarray:
    """Small (T=50, n=5) realized covariance tensor — used by most unit tests."""
    return _make_rc_sequence(T=T_small, n=n, seed=42)


@pytest.fixture(scope='session')
def har_sums_small(R_t_small) -> dict:
    return _make_har_sums(R_t_small)


@pytest.fixture(scope='session')
def midas_blocks_small(R_t_small) -> np.ndarray:
    return _make_midas_blocks(R_t_small)


@pytest.fixture(scope='session')
def daily_returns_small(T_small, n) -> np.ndarray:
    """Small (T=50, n=5) daily return matrix for BEKK tests."""
    rng = np.random.default_rng(7)
    return rng.standard_normal((T_small, n)) * 0.01
