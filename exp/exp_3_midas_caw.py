"""
Experiment 3: In-sample estimation and diagnostics of MIDAS-CAW models.
Follows Chiriac & Voev (2010) Table 2.

Run from the project root:
    python exp/exp_3_midas_caw.py
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

_EXP_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_EXP_DIR, '..')
sys.path.insert(0, os.path.join(_PROJECT_DIR, 'src'))
sys.path.insert(0, _EXP_DIR)

from data_utils import load_data_artifacts, MIDAS_INIT_DAYS, EFFECTIVE_T, ASSETS
from midas_caw import MIDASCAWModel

try:
    from exp_2_caw import compute_residual_predictability
except ImportError:
    def compute_residual_predictability(resids, n_lags=50):
        """Fallback: return NaN pvals when exp_2_caw is unavailable."""
        T, k = resids.shape
        f_stats = np.full(k, np.nan)
        f_pvals = np.full(k, np.nan)
        return f_stats, f_pvals

RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')
DATA_DIR    = os.path.join(_PROJECT_DIR, 'data')
os.makedirs(RESULTS_DIR, exist_ok=True)

# (p, q) grid: p = GARCH lags on S_t^*, q = ARCH lags on R_t^*
LAG_GRID = [(0, 1), (1, 1), (1, 2), (2, 1), (2, 2)]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_exp3(R_t: np.ndarray = None, midas_blocks: np.ndarray = None) -> dict:
    """Run full MIDAS-CAW in-sample estimation and diagnostics.

    Parameters
    ----------
    R_t          : pre-loaded full (T, 5, 5) realized covariance array; loaded
                   from disk when None
    midas_blocks : pre-computed (T, L, 5, 5) MIDAS block sums; loaded from
                   disk when None

    Returns
    -------
    dict mapping model names to estimation results
    """
    print('=' * 60)
    print('Experiment 3: MIDAS-CAW Model Estimation')
    print('=' * 60)

    # --- Load data if not supplied ---
    if R_t is None:
        R_t_full, _, _ = load_data_artifacts(DATA_DIR)
        blocks_npz = np.load(os.path.join(DATA_DIR, 'midas_blocks.npz'))
        midas_blocks = blocks_npz['blocks']
    else:
        R_t_full = R_t

    # Effective sample: drop first MIDAS_INIT_DAYS
    R_eff          = R_t_full[MIDAS_INIT_DAYS:]
    midas_eff      = midas_blocks[MIDAS_INIT_DAYS:]
    T              = len(R_eff)
    print(f'Effective sample: {T} days  '
          f'(target {EFFECTIVE_T});  MIDAS blocks shape: {midas_eff.shape}')

    results = {}

    for diagonal in [False, True]:
        model_type = 'Diagonal' if diagonal else 'Unrestricted'
        print(f'\n--- {model_type} MIDAS-CAW Models ---')

        for p, q in LAG_GRID:
            model_name = f'{"Diag" if diagonal else "Unres"}_MIDAS-CAW({p},{q})'
            print(f'Estimating {model_name} ...', flush=True)

            model = MIDASCAWModel(n=5, p=p, q=q, diagonal=diagonal, m=20, L=12)
            try:
                result = model.fit(R_eff, midas_eff, verbose=False)
                loglik  = result['loglik']
                n_pars  = result['n_params']
                sc      = model.schwarz_criterion(loglik, T)

                # Short-run persistence
                try:
                    Psi1   = model.compute_Psi1_shortrun()
                    meig1  = float(np.max(np.abs(np.linalg.eigvals(Psi1))))
                except Exception:
                    meig1  = np.nan

                # Residual predictability (F-test p-values on vech residuals)
                try:
                    resids = model.standardized_residuals(R_eff, midas_eff)
                    _, f_pvals = compute_residual_predictability(resids, n_lags=50)
                    avg_pval   = float(np.nanmean(f_pvals))
                except Exception:
                    avg_pval = np.nan

                results[model_name] = {
                    'p': p, 'q': q, 'diagonal': diagonal,
                    'loglik': loglik,
                    'SC': sc,
                    'n_params': n_pars,
                    'meig1': meig1,
                    'avg_f_pval': avg_pval,
                    'model': model,
                    'result': result,
                }
                meig_str = f'{meig1:.4f}' if np.isfinite(meig1) else 'nan'
                pval_str = f'{avg_pval:.4f}' if np.isfinite(avg_pval) else 'nan'
                print(f'  LogLik={loglik:.2f}  SC={sc:.2f}  '
                      f'k={n_pars}  meig1={meig_str}  avg_Fpval={pval_str}')

            except Exception as exc:
                print(f'  FAILED: {exc}')
                import traceback
                traceback.print_exc()

    # --- Summary table ---
    rows = []
    for name, v in results.items():
        rows.append({
            'Model':    name,
            'p':        v['p'],
            'q':        v['q'],
            'Diagonal': v['diagonal'],
            'LogLik':   f"{v['loglik']:.2f}",
            'SC':       f"{v['SC']:.2f}",
            'Params':   v['n_params'],
            'meig1':    f"{v['meig1']:.4f}" if np.isfinite(v['meig1']) else 'nan',
            'AvgFpval': f"{v['avg_f_pval']:.4f}" if np.isfinite(v['avg_f_pval']) else 'nan',
        })

    summary_df = pd.DataFrame(rows)
    print('\nMIDAS-CAW Estimation Summary:')
    print(summary_df.to_string(index=False))

    csv_path = os.path.join(RESULTS_DIR, 'exp3_midas_caw_comparison.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    # --- Plot log-likelihoods ---
    _plot_loglik_comparison(results)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plot_loglik_comparison(results: dict) -> None:
    """Bar chart of log-likelihoods for all estimated MIDAS-CAW specs."""
    if not results:
        return

    names  = list(results.keys())
    logliks = [results[n]['loglik'] for n in names]
    colors  = ['steelblue' if not results[n]['diagonal'] else 'darkorange'
               for n in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(names)), logliks, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Log-Likelihood')
    ax.set_title('MIDAS-CAW In-Sample Log-Likelihoods\n'
                 '(blue = unrestricted, orange = diagonal)')
    plt.tight_layout()

    out = os.path.join(
        os.path.join(_PROJECT_DIR, 'results'),
        'exp3_midas_caw_logliks.png',
    )
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    run_exp3()
    print('\nExperiment 3 complete.')
