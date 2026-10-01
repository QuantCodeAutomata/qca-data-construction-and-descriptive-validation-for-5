# CAW Realized Covariance Analysis

Implementation of Conditional Autoregressive Wishart (CAW) family models for 5-asset realized covariance matrices, following Chiriac & Voev (2010).

## Assets
AXP, C, GE, HD, IBM (NYSE, 2000–2008)

## Models
- **CAW(p,q)**: Baseline Conditional Autoregressive Wishart model
- **MIDAS-CAW**: Long-run MIDAS component + short-run CAW dynamics
- **HAR-CAW**: Heterogeneous Autoregressive Realized Covariance
- **EWMA**: Exponentially Weighted Moving Average benchmark
- **BEKK-GARCH**: Multivariate GARCH benchmark on daily returns

## Structure
```
data/       - Data artifacts (realized covariances, returns)
src/        - Core model implementations
exp/        - Experiment scripts
tests/      - pytest tests
results/    - Metrics, plots, RESULTS.md
```

## Usage
```bash
pip install -r requirements.txt
python exp/exp_1_data.py
python exp/exp_2_caw.py
python exp/exp_3_midas_caw.py
python exp/exp_4_har_caw.py
python exp/exp_5_forecasting.py
```

## References
Chiriac, R., & Voev, V. (2010). Modelling and Forecasting Multivariate Realized Volatility. *Journal of Applied Econometrics*.
