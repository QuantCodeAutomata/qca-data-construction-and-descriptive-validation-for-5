"""
Experiment 4: In-sample estimation and diagnostics of HAR-CAW models.
Follows Chiriac & Voev (2010) Section 3 / HAR-CAW specification.
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
from data_utils import load_data_artifacts, MIDAS_INIT_DAYS, compute_rolling_har_sums
from har_caw import HARCAWModel
from exp_2_caw import compute_residual_predictability

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_exp4(R_t: np.ndarray = None) -> dict:
    """Run HAR-CAW estimation for unrestricted and diagonal parameterizations.

    Parameters
    ----------
    R_t : optional pre-loaded array (T_total, 5, 5)

    Returns
    -------
    results : dict mapping model name → estimation results
    """
    print('=' * 60)
    print('Experiment 4: HAR-CAW Model Estimation')
    print('=' * 60)

    if R_t is None:
        R_t_full, _, _ = load_data_artifacts('data')
    else:
        R_t_full = R_t

    # Compute HAR sums over full dataset for proper initialization
    all_sums = compute_rolling_har_sums(R_t_full, windows=[5, 10, 20])

    # Slice to effective sample
    R_eff = R_t_full[MIDAS_INIT_DAYS:]
    T = len(R_eff)
    har_sums_eff = {
        'w5':  all_sums['w5'][MIDAS_INIT_DAYS:],
        'w10': all_sums['w10'][MIDAS_INIT_DAYS:],
        'w20': all_sums['w20'][MIDAS_INIT_DAYS:]
    }

    print(f'Effective sample: {T} observations')
    print(f'HAR sum shapes: {har_sums_eff["w5"].shape}')

    results = {}

    for diagonal in [False, True]:
        model_type = 'Diagonal' if diagonal else 'Unrestricted'
        print(f'\n--- {model_type} HAR-CAW ---')

        model = HARCAWModel(n=5, diagonal=diagonal)
        model_name = f'{"Diag" if diagonal else "Unres"}_HAR-CAW'
        print(f'Estimating {model_name}...', end=' ', flush=True)

        try:
            result = model.fit(R_eff, har_sums_eff, verbose=False)
            loglik = result['loglik']
            n_params = result['n_params']
            sc = model.schwarz_criterion(loglik, T)

            # Residual diagnostics
            resids = model.standardized_residuals(R_eff, har_sums_eff)
            _, f_pvals = compute_residual_predictability(resids, n_lags=50)
            avg_pval = float(np.nanmean(f_pvals))

            results[model_name] = {
                'diagonal': diagonal,
                'loglik': loglik,
                'SC': sc,
                'n_params': n_params,
                'avg_f_pval': avg_pval,
                'model': model
            }
            print(f'LogLik={loglik:.1f}, SC={sc:.1f}, Params={n_params}')

        except Exception as e:
            print(f'FAILED: {e}')
            import traceback
            traceback.print_exc()

    # Summary table
    rows = []
    for name, res in results.items():
        rows.append({
            'Model': name,
            'LogLik': round(res['loglik'], 1),
            'SC': round(res['SC'], 1),
            'Params': res['n_params'],
            'AvgFpval': round(res['avg_f_pval'], 3)
        })

    summary_df = pd.DataFrame(rows)
    print('\nHAR-CAW Comparison:')
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'exp4_har_caw_comparison.csv'), index=False)

    # Compare with paper targets
    print('\nPaper approximate targets:')
    print('  Unrestricted HAR-CAW: LogLik ~ -7337.8, SC ~ 15552.3')
    print('  Diagonal HAR-CAW:     LogLik ~ -7606.8, SC ~ 15485.7')

    return results


if __name__ == '__main__':
    run_exp4()
    print('\nExperiment 4 complete.')
