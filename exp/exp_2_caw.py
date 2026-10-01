"""
Experiment 2: In-sample estimation and diagnostics of baseline CAW(p,q) models.
Follows Chiriac & Voev (2010) Section 2 / Table 1 (unrestricted and diagonal CAW).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from matrix_ops import vech
from data_utils import load_data_artifacts, MIDAS_INIT_DAYS, EFFECTIVE_T
from caw_model import CAWModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

LAG_GRID = [(0, 1), (1, 1), (1, 2), (2, 1), (2, 2)]


def compute_residual_predictability(resids: np.ndarray, n_lags: int = 50):
    """F-test for residual predictability using n_lags lagged residuals.

    For each scalar residual series j, regresses e_{j,t} on a constant and
    50 lagged residual vectors; returns F-statistic and p-value.

    Parameters
    ----------
    resids : ndarray (T, k) — standardized residuals
    n_lags : int — number of lags to include as regressors

    Returns
    -------
    f_stats : ndarray (k,)
    f_pvals : ndarray (k,)
    """
    from scipy.stats import f as f_dist
    from statsmodels.regression.linear_model import OLS

    T, k = resids.shape
    f_stats = []
    f_pvals = []

    if T <= n_lags + k + 1:
        return np.full(k, np.nan), np.full(k, np.nan)

    for j in range(k):
        y = resids[n_lags:, j]
        # Stack 50 lags of the full residual vector as predictors
        X = np.column_stack([
            resids[n_lags - lag: T - lag, :]
            for lag in range(1, n_lags + 1)
        ])
        X = np.column_stack([np.ones(len(y)), X])
        try:
            res = OLS(y, X).fit()
            f_stats.append(res.fvalue)
            f_pvals.append(res.f_pvalue)
        except Exception:
            f_stats.append(np.nan)
            f_pvals.append(np.nan)

    return np.array(f_stats), np.array(f_pvals)


def run_exp2(R_t: np.ndarray = None) -> dict:
    """Run baseline CAW estimation across lag grid and parameterizations.

    Parameters
    ----------
    R_t : optional pre-loaded array (T_total, 5, 5)

    Returns
    -------
    results : dict mapping model name → estimation results
    """
    print('=' * 60)
    print('Experiment 2: Baseline CAW Model Estimation')
    print('=' * 60)

    if R_t is None:
        R_t_full, _, _ = load_data_artifacts('data')
    else:
        R_t_full = R_t

    R_eff = R_t_full[MIDAS_INIT_DAYS:]
    T = len(R_eff)
    print(f'Effective sample: {T} observations (target: {EFFECTIVE_T})')

    results = {}

    for diagonal in [False, True]:
        model_type = 'Diagonal' if diagonal else 'Unrestricted'
        print(f'\n--- {model_type} CAW Models ---')

        for p, q in LAG_GRID:
            model_name = f'{"Diag" if diagonal else "Unres"}_CAW({p},{q})'
            print(f'Estimating {model_name}...', end=' ', flush=True)

            model = CAWModel(n=5, p=p, q=q, diagonal=diagonal)
            try:
                result = model.fit(R_eff, verbose=False)
                loglik = result['loglik']
                n_params = result['n_params']
                sc = model.schwarz_criterion(loglik, T)

                # Stationarity diagnostics
                Psi1 = model.compute_Psi1()
                meig1 = float(np.max(np.abs(np.linalg.eigvals(Psi1))))

                Psi2 = model.compute_Psi2(truncation=100)
                meig2 = float(np.max(np.abs(np.linalg.eigvals(Psi2))))

                # Residual predictability diagnostics
                resids = model.standardized_residuals(R_eff)
                _, f_pvals = compute_residual_predictability(resids, n_lags=50)
                avg_pval = float(np.nanmean(f_pvals))

                results[model_name] = {
                    'p': p, 'q': q, 'diagonal': diagonal,
                    'loglik': loglik, 'SC': sc, 'n_params': n_params,
                    'meig1': meig1, 'meig2': meig2,
                    'avg_f_pval': avg_pval,
                    'model': model
                }
                print(f'LogLik={loglik:.1f}, SC={sc:.1f}, '
                      f'meig1={meig1:.3f}, meig2={meig2:.3f}')
            except Exception as e:
                print(f'FAILED: {e}')
                import traceback
                traceback.print_exc()

    # Build summary table
    rows = []
    for name, res in results.items():
        rows.append({
            'Model': name,
            'p': res['p'], 'q': res['q'],
            'LogLik': round(res['loglik'], 1),
            'SC': round(res['SC'], 1),
            'Params': res['n_params'],
            'meig1': round(res['meig1'], 3),
            'meig2': round(res['meig2'], 3),
            'AvgFpval': round(res['avg_f_pval'], 3)
        })

    summary_df = pd.DataFrame(rows)
    print('\nCAW Model Comparison Table:')
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'exp2_caw_comparison.csv'), index=False)

    # Plot fitted S_t for best model
    if results:
        best_name = min(
            results.keys(),
            key=lambda k: results[k]['SC']
        )
        plot_fitted_st(results[best_name]['model'], R_eff, best_name)

    return results


def plot_fitted_st(model: CAWModel, R_eff: np.ndarray, model_name: str):
    """Plot realized vs fitted conditional variance for AXP."""
    T = len(R_eff)
    realized_var = np.array([R_eff[t][0, 0] for t in range(T)])
    if model.S_t_hat is not None:
        fitted_var = np.array([model.S_t_hat[t][0, 0] for t in range(T)])
    else:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(realized_var, lw=0.6, label='Realized Var (AXP)', color='steelblue', alpha=0.7)
    ax.plot(fitted_var, lw=1.2, label=f'Fitted S_t ({model_name})', color='darkorange')
    ax.set_title(f'Realized vs Fitted Variance — AXP ({model_name})')
    ax.set_xlabel('Day')
    ax.set_ylabel('Variance')
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f'exp2_fitted_vs_realized.png')
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


if __name__ == '__main__':
    run_exp2()
    print('\nExperiment 2 complete.')
