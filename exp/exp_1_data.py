"""
Experiment 1: Data construction and descriptive validation for 5-asset realized
covariance matrices.  Follows Chiriac & Voev (2010) methodology for AXP, C, GE,
HD, IBM (2000-2008).

Run from the project root:
    python exp/exp_1_data.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from matrix_ops import vech, vech_to_matrix, duplication_matrix, elimination_matrix, commutation_matrix
from data_utils import (
    ASSETS, N_ASSETS, K_VECH, MIDAS_INIT_DAYS, EFFECTIVE_T,
    generate_synthetic_realized_covariances,
    generate_daily_returns,
    compute_rolling_midas_blocks,
    compute_rolling_har_sums,
    save_data_artifacts,
)

# Absolute paths so the script works regardless of CWD
_EXP_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_EXP_DIR, '..')
RESULTS_DIR  = os.path.join(_PROJECT_DIR, 'results')
DATA_DIR     = os.path.join(_PROJECT_DIR, 'data')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_exp1() -> dict:
    """Execute all data-construction and validation steps for Experiment 1."""
    print('=' * 60)
    print('Experiment 1: Data Construction and Descriptive Validation')
    print('=' * 60)

    # ------------------------------------------------------------------
    # Step 1–5: Generate synthetic realized covariance matrices
    # ------------------------------------------------------------------
    print('\nGenerating synthetic realized covariance matrices...')
    print('Note: Using synthetic data with paper-consistent statistical properties')
    print('(Chiriac & Voev 2010 precomputed dataset not available; 30-subgrid')
    print(' procedure is an implementation assumption)')

    n_total = 2156  # Paper total: 2156 trading days (Jan 2000 – Jul 2008)
    R_t, dates = generate_synthetic_realized_covariances(n_days=n_total, seed=42)
    print(f'Generated {len(R_t)} realized covariance matrices')
    print(f'Date range: {dates[0].date()} to {dates[-1].date()}')
    print(f'Shape: {R_t.shape}')

    # ------------------------------------------------------------------
    # Step 6: Verify symmetry and positive definiteness
    # ------------------------------------------------------------------
    n_pd = 0
    for t in range(len(R_t)):
        R_t[t] = (R_t[t] + R_t[t].T) / 2      # enforce exact symmetry
        try:
            np.linalg.cholesky(R_t[t])
            n_pd += 1
        except np.linalg.LinAlgError:
            pass
    print(f'\nSymmetry enforced.  Positive definite: {n_pd}/{len(R_t)}')

    # ------------------------------------------------------------------
    # Step 7: Generate daily returns
    # ------------------------------------------------------------------
    returns_df = generate_daily_returns(R_t, dates, seed=123)
    print(f'Daily returns shape: {returns_df.shape}')

    # ------------------------------------------------------------------
    # Step 8: Convert to vech form [T, 15]
    # ------------------------------------------------------------------
    r_t = np.array([vech(R_t[t]) for t in range(len(R_t))])
    print(f'vech representation shape: {r_t.shape}')

    # ------------------------------------------------------------------
    # Step 9: Reserve first 240 days for MIDAS weight initialization
    # ------------------------------------------------------------------
    R_eff      = R_t[MIDAS_INIT_DAYS:]
    r_eff      = r_t[MIDAS_INIT_DAYS:]
    dates_eff  = dates[MIDAS_INIT_DAYS:]
    print(f'\nMIDAS initialization period: {MIDAS_INIT_DAYS} days')
    print(f'Effective sample size: {len(R_eff)} (target: {EFFECTIVE_T})')

    # Human-readable vech element names following paper notation
    vech_names = [
        f'r{i+1}{j+1}'
        for j in range(N_ASSETS)
        for i in range(j, N_ASSETS)
    ]

    # ------------------------------------------------------------------
    # Step 10: Descriptive statistics
    # ------------------------------------------------------------------
    print('\nDescriptive Statistics for vech(R_t) elements:')
    desc_stats = pd.DataFrame(index=vech_names)
    desc_stats['Mean']     = r_eff.mean(axis=0)
    desc_stats['Std']      = r_eff.std(axis=0)
    desc_stats['Min']      = r_eff.min(axis=0)
    desc_stats['Max']      = r_eff.max(axis=0)
    desc_stats['Skewness'] = stats.skew(r_eff, axis=0)
    desc_stats['Kurtosis'] = stats.kurtosis(r_eff, axis=0)   # excess kurtosis
    print(desc_stats.to_string())
    desc_stats.to_csv(os.path.join(RESULTS_DIR, 'exp1_descriptive_stats.csv'))

    # ------------------------------------------------------------------
    # Step 11: Time-series and ACF plots
    # ------------------------------------------------------------------
    _plot_diagnostics(r_eff, dates_eff, vech_names)

    # ------------------------------------------------------------------
    # Step 12: Compute rolling sums for MIDAS and HAR
    # ------------------------------------------------------------------
    print('\nComputing MIDAS block sums (m=20, L=12)...')
    midas_blocks = compute_rolling_midas_blocks(R_t, dates, m=20, L=12)
    print(f'MIDAS blocks shape: {midas_blocks.shape}')

    print('Computing HAR rolling sums (5, 10, 20-day)...')
    har_sums = compute_rolling_har_sums(R_t, windows=[5, 10, 20])

    # ------------------------------------------------------------------
    # Save all artifacts
    # ------------------------------------------------------------------
    save_data_artifacts(R_t, returns_df, dates, output_dir=DATA_DIR)
    np.savez_compressed(os.path.join(DATA_DIR, 'midas_blocks.npz'), blocks=midas_blocks)
    np.savez_compressed(
        os.path.join(DATA_DIR, 'har_sums.npz'),
        w5=har_sums['w5'],
        w10=har_sums['w10'],
        w20=har_sums['w20'],
    )

    # ------------------------------------------------------------------
    # Matrix operator validation
    # ------------------------------------------------------------------
    print('\nMatrix operator validation:')
    n = 5
    D_n  = duplication_matrix(n)
    L_n  = elimination_matrix(n)
    K_nn = commutation_matrix(n)
    print(f'D_5 shape: {D_n.shape},  L_5 shape: {L_n.shape},  K_55 shape: {K_nn.shape}')

    A = R_t[0]
    assert np.allclose(D_n @ vech(A), A.flatten(order='F')), \
        'D_n @ vech(A) != vec(A)'
    assert np.allclose(L_n @ A.flatten(order='F'), vech(A)), \
        'L_n @ vec(A) != vech(A)'

    rng = np.random.default_rng(0)
    B = rng.standard_normal((n, n))
    assert np.allclose(K_nn @ B.flatten(order='F'), B.T.flatten(order='F')), \
        'K_nn @ vec(B) != vec(B.T)'

    print('All matrix operator assertions passed!')

    _print_replication_note()

    return {
        'R_t': R_t, 'r_t': r_t, 'returns_df': returns_df, 'dates': dates,
        'R_eff': R_eff, 'r_eff': r_eff, 'dates_eff': dates_eff,
        'midas_blocks': midas_blocks, 'har_sums': har_sums,
        'desc_stats': desc_stats, 'vech_names': vech_names,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plot_diagnostics(
    r_eff: np.ndarray,
    dates_eff: pd.DatetimeIndex,
    vech_names: list,
) -> None:
    """Plot time series and ACF for three key vech elements and save to results/."""
    # r11 = AXP variance, r22 = C variance, r21 = AXP-C covariance
    selected_idx   = [0, 5, 1]
    selected_names = [vech_names[i] for i in selected_idx]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for row, (idx, name) in enumerate(zip(selected_idx, selected_names)):
        series = r_eff[:, idx]

        # Time-series panel
        axes[row, 0].plot(dates_eff, series, lw=0.5, color='steelblue')
        axes[row, 0].set_title(f'Time Series: {name}')
        axes[row, 0].set_xlabel('Date')
        axes[row, 0].set_ylabel('Realized Cov')
        axes[row, 0].tick_params(axis='x', rotation=30)

        # ACF panel
        sm.graphics.tsa.plot_acf(series, lags=40, ax=axes[row, 1], title=f'ACF: {name}')

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'exp1_time_series_acf.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')


def _print_replication_note() -> None:
    note = """
============================================================
REPLICATION NOTE (Exp 1)
============================================================
PAPER-SPECIFIED STEPS:
  - Asset universe  : AXP, C, GE, HD, IBM
  - Trading window  : 09:30–16:00 New York time
  - 5-minute intervals (~78 per day)
  - Raw RC          : R_t = Σ_j  y_{t,j} y_{t,j}'
  - Subsampling     : 30 subgrids (MIDAS init: 240 days)
  - Effective sample: 1916 days  (total: 2156)
  - vech ordering   : r11,r21,r31,r41,r51,r22,...,r55

IMPLEMENTATION ASSUMPTIONS:
  - Data source : synthetic (paper uses Chiriac-Voev precomputed
    dataset; 5-minute historical data not available via current API)
  - 30-subgrid subsampling: not implementable without raw tick data;
    replaced by standard realised covariance simulation
  - Simulation parameters calibrated to reproduce paper's qualitative
    properties: persistence (φ≈0.97), right-skewness, leptokurtosis
  - Missing data / holiday handling: not applicable (synthetic)
============================================================
"""
    print(note)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    results = run_exp1()
    print('\nExperiment 1 complete.')
