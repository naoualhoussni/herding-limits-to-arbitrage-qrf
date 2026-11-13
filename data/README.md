# Datasets

## Raw Granular Data (>6.4 Million Rows)
Because the raw data files (`LA-SHA.pkl` and `LA-SZA.pkl`) exceed 600 MB each, they are excluded from Git tracking via `.gitignore`.
They can be downloaded directly from the official project Google Drive:
🔗 https://drive.google.com/drive/folders/1T8BkHvuxBGO8WkFhMmEEHHLpVM3rkTIQ?usp=sharing

## Daily Aggregated Datasets (Included in Repo)
- `daily_market_sha.csv`: Aggregated daily time-series for the Shanghai Stock Exchange A-shares (3,886 trading sessions, 2003–2018).
- `daily_market_sza.csv`: Aggregated daily time-series for the Shenzhen Stock Exchange A-shares (3,886 trading sessions, 2003–2018).

### Columns Specification:
- `Date`: Trading day timestamp (YYYY-MM-DD).
- `Rm`: Equal-weighted aggregate market return ($R_{m,t}$).
- `LoA`: Composite Limits-to-Arbitrage index (standardized across 6 market friction metrics).
- `CSAD`: Cross-Sectional Absolute Deviation of returns ($CSAD_t$).
- `abs_Rm`: Absolute market return ($|R_{m,t}|$).
- `Rm2`: Squared market return ($R_{m,t}^2$).
- `CSAD_lag1`: 1-day lagged dispersion ($CSAD_{t-1}$).
- `Period`: Subperiod label (`BFC` = Before Financial Crisis 2003–2007; `FC` = Financial Crisis 2007–2008; `AFC` = After Financial Crisis 2009–2018).
- `LoA_group`: Categorical LoA group (`1` = Low LoA bottom 30%, `2` = Medium LoA, `3` = High LoA top 30%).
