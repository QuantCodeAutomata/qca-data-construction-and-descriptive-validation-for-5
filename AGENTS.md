# AGENTS.md — Repository knowledge for OpenHands

## Project overview
CAW realized covariance analysis for 5 assets (AXP, C, GE, HD, IBM).
Implements Chiriac & Voev (2010) CAW-family models: CAW(p,q), MIDAS-CAW, HAR-CAW.
Benchmarks: EWMA, BEKK-GARCH.

## Key constants (src/data_utils.py)
- ASSETS = ['AXP', 'C', 'GE', 'HD', 'IBM']
- N_ASSETS = 5, K_VECH = 15
- MIDAS_INIT_DAYS = 240, EFFECTIVE_T = 1916, total T = 2156

## vech ordering (column-major lower triangle)
r11, r21, r31, r41, r51, r22, r32, r42, r52, r33, r43, r53, r44, r54, r55
Implemented in src/matrix_ops.py :: vech().

## Matrix operator dimensions (n=5)
- duplication_matrix(5)  → shape (25, 15):  D @ vech(A) = vec(A)
- elimination_matrix(5)  → shape (15, 25):  L @ vec(A)  = vech(A)
- commutation_matrix(5)  → shape (25, 25):  K @ vec(A)  = vec(A.T)
- Identity: L @ D = I_15  (verified)
- All use column-major (Fortran) ordering, consistent with vec notation.

## Data pipeline (exp/exp_1_data.py)
1. Generate 2156-day synthetic R_t  [2156, 5, 5]  (seed=42)
2. Enforce symmetry + PD (100% pass)
3. Generate daily returns [2156, 5] (seed=123)
4. Compute vech form [2156, 15]
5. Drop first 240 days → effective sample [1916, 15]
6. MIDAS blocks: [2156, 12, 5, 5]  (m=20 days, L=12 lags)
7. HAR sums: dict {w5, w10, w20} each [2156, 5, 5]
8. Save to data/ and results/

## Artifacts produced by exp_1_data.py
- data/realized_covariances.npz  (R_t + ISO dates)
- data/daily_returns.parquet
- data/midas_blocks.npz
- data/har_sums.npz
- results/exp1_descriptive_stats.csv
- results/exp1_time_series_acf.png

## Run commands
```bash
pip install -r requirements.txt
python exp/exp_1_data.py   # from project root
```

## Synthetic data calibration
- AR(1) log-variance: phi=0.97, sigma_eta=0.15
- mu_logvar ≈ [-8.0, -7.8, -7.9, -8.1, -7.7] (per asset)
- Base correlation: ~0.4–0.55 between large-cap assets
- All R_t symmetric PD by construction

## Important design decisions
- All rolling computations use strict past (no look-ahead):
  MIDAS blocks: end_idx = t - m*(ell-1) ≤ t (never includes R_t[t])
  HAR sums: range R_t[max(0,t-w):t] (excludes t)
- vec/vech use Fortran order throughout (flatten(order='F'))
- nearest_positive_definite uses Higham (1988) via SVD polar factor

---

## MIDAS-CAW model (src/midas_caw.py)

### Model
S_t = C_t S_t^* C_t',  M_t = C_t C_t' = C̄C̄' + θ·Σ_ℓ φ_ℓ(ω) R̄_{t,ℓ}
S_t^* = (I − ΣA_jA_j' − ΣB_iB_i') + ΣB_i S_{t-i}^* B_i' + ΣA_j R*_{t-j} A_j'
R_t | F_{t-1} ~ W_n(ν, S_t/ν)

### (p, q) convention
p = GARCH lags (B matrices on past S_t^*)
q = ARCH lags (A matrices on past R_t^* = C_t^{-1} R_t C_t^{-T})

### Parameter encoding
| Symbol | Raw param | Recovery |
|--------|-----------|----------|
| ν | log(ν−n−1) | n+1 + exp(·) |
| C̄ diagonal | log(C̄_{ii}) | exp(clipped to ±40) |
| θ | log(√θ) | exp(2·log_sqrt_theta) |
| ω | log(ω−1) | 1 + exp(·) |

### Wishart log-likelihood (negative, minimised)
Fully vectorised outside sequential S_t^* loop:
- Batch cholesky of M_t (T,n,n)
- Batch inv of C_t (T,n,n)
- Batch R_t^* = C_t^{-1} R_t C_t^{-T} (matmul)
- Batch log|R_t|, log|C_t|

### MIDAS weights
phi_ell(omega) = (1 - ell/L)^(omega-1) / sum  — normalised Beta
omega > 1 ensures monotone decreasing weights (recent lags get more weight).
omega close to 1 → uniform; omega large → most weight on lag 1.

### Psi1 (persistence matrix)
p=0: n^2 × n^2 zero matrix (rho=0)
p=1: B_1 ⊗ B_1  (n^2 × n^2)
p>1: companion block matrix of size p·n^2 × p·n^2

### Forecast
h=1: deterministic (S_{T+1} computed analytically)
h>1: Bartlett decomposition Wishart MC; n_paths default=10000 for OOS loop
Future M_{T+k} built from clamped R_t history (end_idx clipped to T)

### Performance (n=5, diagonal)
Per loglik eval: ~5ms at T=60, ~50ms at T=1916
Fit (200 L-BFGS-B iters): ~25s at T=60, ~4 min at T=1916

---

## Benchmark models (src/benchmarks.py)

### EWMA
- fit(R_t): S_t = (1-λ)R_{t-1} + λS_{t-1}, S_0 = R_0
- forecast(t_origin, h): one additional EWMA update from (R_{t_origin}, S_{t_origin})
  Same value for all h (EWMA has no mean-reversion)

### BEKK-GARCH
- Sigma_t = D0D0' + ΣG_i Sigma_{t-i} G_i' + ΣF_j (y_{t-j}y_{t-j}') F_j'
- D0 lower-triangular, log-diagonal stored → always PD intercept
- pack/unpack: _pack_D0(D0) ↔ _unpack_params(psi) form exact round-trip
- Gaussian QMLE via L-BFGS-B (maxiter=300)
- Multi-step forecast: E[y_{t+k}y_{t+k}' | F_t] = E[Σ_{t+k}|F_t] for k>0

---

## Experiment scripts

### exp_3_midas_caw.py
LAG_GRID = [(0,1),(1,1),(1,2),(2,1),(2,2)] × {diagonal, unrestricted}
Reports: LogLik, SC (BIC=-2L+k·log T), n_params, max|eig(Ψ₁)|, avg F-pval
Requires exp_2_caw.compute_residual_predictability (falls back to NaN if absent)

### exp_5_forecasting.py
OOS_DAYS=240, HORIZONS=[1,5,10]
Loss: average Frobenius norm ||R_{t+h} − forecast||_F over OOS origins
Skips CAW/HAR model specs gracefully if caw_model/har_caw not importable
Saves: results/exp5_forecast_comparison.csv, results/exp5_forecast_errors.png
