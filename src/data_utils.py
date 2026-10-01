"""
Data construction utilities for CAW realized covariance analysis.

Generates synthetic 5-asset realized covariance matrices whose statistical
properties (persistence, right-skewness, leptokurtosis) are calibrated to
match the descriptive statistics in Chiriac & Voev (2010) for AXP, C, GE,
HD, IBM (NYSE 2000-2008).

All rolling computations use strictly past data (no look-ahead).
"""
import os

import numpy as np
import pandas as pd
from typing import Tuple, Optional


ASSETS = ['AXP', 'C', 'GE', 'HD', 'IBM']
N_ASSETS = 5
K_VECH = 15          # n*(n+1)/2 for n=5
MIDAS_INIT_DAYS = 240  # Days reserved for MIDAS weight initialization
EFFECTIVE_T = 1916     # Effective analysis sample size (paper: 2156 - 240)


# ---------------------------------------------------------------------------
# Primary data generation
# ---------------------------------------------------------------------------

def generate_synthetic_realized_covariances(
    n_days: int = 2156,
    n_assets: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Generate synthetic realized covariance matrices matching paper's statistical properties.

    Simulates persistent, right-skewed, leptokurtic positive-definite covariance
    matrices consistent with Chiriac & Voev (2010) descriptive statistics.

    The log-variance of each asset follows an AR(1) process with
        log(σ²_t) = μ + φ·(log(σ²_{t-1}) − μ) + σ_η·ε_t,  ε_t ~ N(0,1)
    Cross-sectional correlations are held approximately constant, with small
    daily noise added to the covariance structure to break exact factor structure.

    Parameters
    ----------
    n_days : int
        Total number of trading days to generate (default 2156, matching paper).
    n_assets : int
        Number of assets (default 5).
    seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    R_t : ndarray of shape (n_days, n_assets, n_assets)
        Realized covariance matrices; symmetric positive definite.
    dates : DatetimeIndex
        NYSE trading dates from 2000-01-03 to approximately 2008-07-30.
    """
    np.random.seed(seed)
    dates = generate_trading_dates('2000-01-03', '2008-07-30', n_days)

    # AR(1) log-variance process — parameters calibrated to paper's descriptive stats.
    # Unconditional mean log(var) ~ log(2e-4) ≈ -8.5; typical large-cap daily RC.
    phi = 0.97                              # strong persistence
    sigma_eta = 0.15                        # innovation vol for log-variance
    mu_logvar = np.array([-8.0, -7.8, -7.9, -8.1, -7.7])   # asset-specific means

    log_vars = np.zeros((n_days, n_assets))
    log_vars[0] = mu_logvar
    innovations = np.random.randn(n_days, n_assets)
    for t in range(1, n_days):
        log_vars[t] = mu_logvar + phi * (log_vars[t - 1] - mu_logvar) + sigma_eta * innovations[t]

    vars_t = np.exp(log_vars)   # daily realized variances, shape (n_days, n_assets)

    # Realistic correlation structure for large-cap US stocks, 2000-2008
    base_corr = np.array([
        [1.00, 0.55, 0.50, 0.40, 0.45],
        [0.55, 1.00, 0.48, 0.38, 0.42],
        [0.50, 0.48, 1.00, 0.45, 0.50],
        [0.40, 0.38, 0.45, 1.00, 0.42],
        [0.45, 0.42, 0.50, 0.42, 1.00],
    ])

    R_t = np.zeros((n_days, n_assets, n_assets))
    for t in range(n_days):
        std_t = np.sqrt(vars_t[t])
        D = np.diag(std_t)
        cov_t = D @ base_corr @ D

        # Small symmetric noise breaks exact factor structure, adding realistic variation
        noise = np.random.randn(n_assets, n_assets) * 0.02
        noise_sym = (noise + noise.T) / 2
        cov_t = cov_t + np.diag(vars_t[t]) * noise_sym * 0.1

        # Guarantee positive definiteness
        eigvals = np.linalg.eigvalsh(cov_t)
        if eigvals.min() <= 0:
            cov_t += (-eigvals.min() + 1e-8) * np.eye(n_assets)

        R_t[t] = (cov_t + cov_t.T) / 2     # enforce exact symmetry

    return R_t, dates


def generate_trading_dates(start: str, end: str, target_n: int) -> pd.DatetimeIndex:
    """Generate approximately *target_n* business days between *start* and *end*.

    Uses pandas business-day range (Mon–Fri, no holiday adjustment) and
    truncates to exactly *target_n* observations.

    Parameters
    ----------
    start : str — ISO date string, e.g. '2000-01-03'
    end   : str — ISO date string, e.g. '2008-07-30'
    target_n : int — desired number of dates

    Returns
    -------
    DatetimeIndex of length min(target_n, available business days)
    """
    dates = pd.bdate_range(start=start, end=end)
    if len(dates) > target_n:
        dates = dates[:target_n]
    return dates


# ---------------------------------------------------------------------------
# Return simulation
# ---------------------------------------------------------------------------

def generate_daily_returns(
    R_t: np.ndarray,
    dates: pd.DatetimeIndex,
    seed: int = 123,
) -> pd.DataFrame:
    """Generate daily returns consistent with the realized covariance matrices.

    Each day's return is drawn from N(0, R_t) via Cholesky factorisation.
    Falls back to diagonal scaling if R_t[t] is not positive definite.

    Parameters
    ----------
    R_t : ndarray of shape (T, n_assets, n_assets)
    dates : DatetimeIndex of length T
    seed : int

    Returns
    -------
    DataFrame of shape (T, n_assets) with columns = ASSETS
    """
    n_days, n_assets, _ = R_t.shape
    np.random.seed(seed)
    returns = np.zeros((n_days, n_assets))
    for t in range(n_days):
        try:
            chol = np.linalg.cholesky(R_t[t])
            returns[t] = chol @ np.random.randn(n_assets)
        except np.linalg.LinAlgError:
            returns[t] = np.random.randn(n_assets) * np.sqrt(np.diag(R_t[t]))
    return pd.DataFrame(returns, index=dates, columns=ASSETS)


# ---------------------------------------------------------------------------
# MIDAS block sums
# ---------------------------------------------------------------------------

def compute_rolling_midas_blocks(
    R_t: np.ndarray,
    dates: pd.DatetimeIndex,
    m: int = 20,
    L: int = 12,
) -> np.ndarray:
    """Compute MIDAS monthly block sums for each observation t.

    For lag ell ∈ {1, …, L}, the block sum is

        R̄[t, ell] = Σ_{τ = t − m·ell}^{t − m·(ell−1) − 1}  R_τ

    which covers the ell-th most-recent non-overlapping m-day window
    *strictly before* t (no look-ahead).

    Parameters
    ----------
    R_t  : ndarray of shape (T, n, n)
    dates : DatetimeIndex of length T (unused in computation; kept for API consistency)
    m    : int — days per MIDAS block (default 20 ≈ one trading month)
    L    : int — number of MIDAS lags (default 12 ≈ one year)

    Returns
    -------
    blocks : ndarray of shape (T, L, n, n)
        blocks[t, ell-1] is the block sum for lag ell at time t.
        Zero-filled where insufficient history exists.
    """
    T, n, _ = R_t.shape
    blocks = np.zeros((T, L, n, n))
    for t in range(T):
        for ell in range(1, L + 1):
            start_idx = t - m * ell
            end_idx   = t - m * (ell - 1)   # exclusive; equal to t for ell=1
            if start_idx >= 0 and end_idx > 0:
                actual_start = max(start_idx, 0)
                actual_end   = min(end_idx, t)   # strict past: never include t
                if actual_start < actual_end:
                    blocks[t, ell - 1] = np.sum(R_t[actual_start:actual_end], axis=0)
    return blocks


# ---------------------------------------------------------------------------
# HAR rolling sums
# ---------------------------------------------------------------------------

def compute_rolling_har_sums(
    R_t: np.ndarray,
    windows: list = [5, 10, 20],
) -> dict:
    """Compute rolling sums of realized covariance matrices for HAR specification.

    For each window w, computes

        R̄_w[t] = Σ_{τ = max(0, t−w)}^{t−1}  R_τ   (strictly past, no look-ahead)

    Parameters
    ----------
    R_t     : ndarray of shape (T, n, n)
    windows : list of int — window lengths (default [5, 10, 20])

    Returns
    -------
    dict with keys 'w5', 'w10', 'w20' (or 'w{window}' for custom windows);
    each value is ndarray of shape (T, n, n).
    """
    T, n, _ = R_t.shape
    sums = {}
    for window in windows:
        rolling = np.zeros((T, n, n))
        for t in range(1, T):             # t=0 stays zero (no past data)
            start_idx = max(0, t - window)
            rolling[t] = np.sum(R_t[start_idx:t], axis=0)
        sums[f'w{window}'] = rolling
    return sums


# ---------------------------------------------------------------------------
# Persistence (I/O)
# ---------------------------------------------------------------------------

def save_data_artifacts(
    R_t: np.ndarray,
    returns_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    output_dir: str = 'data',
) -> None:
    """Save dataset artifacts in reproducible compressed format.

    Writes:
    - ``realized_covariances.npz`` — R_t array and ISO-format date strings
    - ``daily_returns.parquet``    — returns DataFrame

    Parameters
    ----------
    R_t        : ndarray of shape (T, n, n)
    returns_df : DataFrame of shape (T, n_assets)
    dates      : DatetimeIndex of length T
    output_dir : str — target directory (created if absent)
    """
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dir, 'realized_covariances.npz'),
        R_t=R_t,
        dates=dates.strftime('%Y-%m-%d').values,
    )
    returns_df.to_parquet(os.path.join(output_dir, 'daily_returns.parquet'))
    print(f'Saved {len(R_t)} realized covariance matrices and returns to {output_dir}/')


def load_data_artifacts(
    data_dir: str = 'data',
) -> Tuple[np.ndarray, pd.DataFrame, pd.DatetimeIndex]:
    """Load previously saved dataset artifacts.

    Parameters
    ----------
    data_dir : str — directory containing saved .npz and .parquet files

    Returns
    -------
    R_t        : ndarray of shape (T, n, n)
    returns_df : DataFrame of shape (T, n_assets)
    dates      : DatetimeIndex of length T
    """
    npz = np.load(os.path.join(data_dir, 'realized_covariances.npz'), allow_pickle=True)
    R_t = npz['R_t']
    dates = pd.DatetimeIndex(npz['dates'])
    returns_df = pd.read_parquet(os.path.join(data_dir, 'daily_returns.parquet'))
    return R_t, returns_df, dates
