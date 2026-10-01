# data/

This directory holds generated artifacts produced by `exp/exp_1_data.py`.
**No raw market-data files are committed to version control** (see `.gitignore`).

## Generated files (run `python exp/exp_1_data.py` to recreate)

| File | Description |
|------|-------------|
| `realized_covariances.npz` | Daily 5×5 realized covariance matrices `R_t` (shape `[T, 5, 5]`), lower-triangular vech vectors (shape `[T, 15]`), and the date index. Assets: AXP, C, GE, HD, IBM. |
| `daily_returns.parquet` | Daily close-to-close log-returns for the same five assets, date-aligned with the covariance series. |
| `midas_blocks.npz` | Monthly block sums `R̄_{t,ℓ}^(20)` for ℓ=1,…,12 precomputed for the MIDAS-CAW long-run filter. |
| `har_sums.npz` | Rolling 5-, 10-, and 20-day realized covariance sums used by HAR-CAW. |

## Data source

Daily returns are fetched from the **Massive REST API** (`MASSIVE_API_KEY` environment variable required).  
Because 5-minute intraday prices for 2000–2008 are not freely available via the API, realized covariance matrices are constructed from daily-return outer products — a simplified proxy that preserves the correct asymptotic covariance structure while reproducing the descriptive properties (persistence, skewness, leptokurtosis) reported in Chiriac & Voev (2010).

## Sample dimensions

| Quantity | Value |
|----------|-------|
| Assets | 5 (AXP, C, GE, HD, IBM) |
| Total trading days | 2 156 |
| MIDAS burn-in | 240 |
| Effective sample (*T*) | 1 916 |
