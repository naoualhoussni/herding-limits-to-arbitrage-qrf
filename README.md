# Financial Herding & Limits to Arbitrage (LoA)
## Non-Linear Detection via Quantile Regression Forests (QRF)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Target Journal](https://img.shields.io/badge/Target%20Journal-JBEF%20%2F%20Applied%20Econ%20Letters-brightgreen.svg)]()

> **Empirical Finance & Data Science Research Project**  
> **Data:** Shanghai (SHA) & Shenzhen (SZA) A-Shares markets, CSMAR database (2003–2018, ~6.41 million observations).  
> **Theme 2:** Non-Linear Herding Detection using Quantile Regression Forests (QRF).

---

## Project Overview

This study investigates the non-linear relationship between **Limits to Arbitrage (LoA)** and **investor herding behavior** across Chinese equity markets.

Standard econometric regressions using Ordinary Least Squares (OLS / Chang et al., 2000) at the conditional mean suffer from notoriously low explanatory power ($R^2 < 5\%$) because **herding is fundamentally a tail-distribution phenomenon** that emerges aggressively during market crashes, liquidity dry-ups, and panic episodes.

To resolve this limitation, this repository implements a comprehensive econometric and machine learning framework:
1. **Mandatory Step 0 (Replication Benchmark):** Reproduction of the 6-factor composite LoA index and baseline Christie-Huang (1995) and Chang et al. (2000) models across 3,886 trading days.
2. **Econometric Quantile Regression (Koenker & Bassett, 1978):** Estimation of the non-linear herding curvature $\beta_2(\tau)$ across the entire decile spectrum $\tau \in [0.05, 0.95]$.
3. **Quantile Regression Forests (QRF — Meinshausen, 2006):** Non-parametric modeling with decision trees tailored to conditional distribution estimation under strict time-series cross-validation (`TimeSeriesSplit`).
4. **Out-of-Sample Evaluation:** Quantitative validation of predictive performance using **Pinball Loss (Quantile Loss)** and **RMSE**.
5. **Interpretability & Robustness:** Partial dependence fan charts, crisis subperiod splits (2007–2008 GFC vs. 2015 Chinese stock crash), and up-vs-down market asymmetry.

---

##  Key Findings

### 1. Herding Amplification in Extreme Lower Tails
| Quantile ($\tau$) | Intercept $\beta_0$ | Linear $\|R_m\|$ $\beta_1$ | **Herding Curvature $\beta_2$ ($R_m^2$)** | Lag CSAD $\beta_3$ |
| :---: | :---: | :---: | :---: | :---: |
| **$\tau = 0.05$ (Extreme Tail)** | 0.002164*** | 0.160299*** | **-2.806900*** (0.454) | 0.703965*** |
| $\tau = 0.10$ | 0.002481*** | 0.151897*** | **-2.651831*** (0.310) | 0.727828*** |
| $\tau = 0.20$ | 0.002935*** | 0.162367*** | **-2.701734*** (0.209) | 0.724575*** |
| $\tau = 0.50$ (Median) | 0.004128*** | 0.128690*** | **-1.777442*** (0.365) | 0.743549*** |
| $\tau = 0.95$ (Upper Tail) | 0.008105*** | 0.090501*** | **-0.747126*** (0.336) | 0.799461*** |
| **OLS (Conditional Mean)** | 0.004259*** | 0.119308*** | **-1.191479*** (0.291) | 0.663258*** |

> **Primary Takeaway:** In the extreme lower decile ($\tau = 0.05$), herding intensity is **2.35 times larger** ($\beta_2 = -2.81$) than estimated at the conditional mean by OLS ($\beta_2 = -1.19$). Classical mean regressions massively underestimate systemic market vulnerability and consensus clustering during market panics.

### 2. Superior Out-of-Sample Accuracy (Pinball Loss)
The **Quantile Regression Forests (QRF)** algorithm consistently outperforms the parametric OLS benchmark in the tails of the return dispersion distribution, delivering up to **$+3.59\%$** reduction in out-of-sample Pinball Loss at $\tau = 0.05$.

---

##  Repository Architecture

```text
├── .gitignore                  # Git ignore file (excludes heavy raw .pkl datasets)
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive project documentation (English)
│
├── notebooks/                  # Deliverable 1: End-to-End Jupyter Notebooks
│   └── Theme2_Herding_QRF_Complete.ipynb  # Self-contained reproducible notebook (All Tables 1-8 & QRF)
│
├── manuscript/                 # Deliverable 2: Scientific Research Paper
│   ├── paper_herding_qrf.tex           # Complete LaTeX source (JBEF/AEL ready with all tables)
│   ├── Scientific_Paper_Herding_QRF.md # Complete ~5,500-word paper in Markdown
│   ├── figures/                # Publication-grade figures (300 DPI)
│   │   ├── fig1_timeseries_csad_loa.png
│   │   ├── fig2_quantile_beta2_trajectory.png
│   │   ├── fig3_pinball_loss_comparison.png
│   │   └── fig4_qrf_fan_chart.png
│   └── tables/                 # Benchmark & QRF tables (Tables 1 to 8 + deciles)
│       ├── table1_descriptive_stats.csv
│       ├── table2_panelA_christie_huang.csv
│       ├── table2_panelB_chang_ols.csv
│       ├── table3_herding_down_markets.csv
│       ├── table4_herding_up_markets.csv
│       ├── table5_paternoster_asymmetry.csv
│       ├── table6_rolling_two_year.csv
│       ├── table7_hl_differences.csv
│       └── table8_robustness_dummies.csv
│
├── scripts/                    # Automated replication scripts
│   ├── run_full_replication.py         # Replicates Tables 1-8 in one execution
│   └── generate_complete_notebook.py   # Generates the integrated notebook
│
├── src/                        # Modular source code
│   ├── data_loader.py          # Data ingestion and validation
│   ├── models.py               # OLS, QuantileRegressor, and QRF estimators
│   └── evaluate.py             # Pinball Loss, RMSE, and TimeSeriesSplit evaluation
│
└── data/                       # Daily aggregated market datasets
    ├── README.md               # Guide and column specs
    ├── daily_market_sha.csv    # Shanghai A-shares daily series (N = 3,886)
    └── daily_market_sza.csv    # Shenzhen A-shares daily series (N = 3,886)
```

---

##  Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/naoualhoussni/herding-limits-to-arbitrage-qrf.git
cd herding-limits-to-arbitrage-qrf
```

### 2. Create a Virtual Environment and Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook notebooks/Theme2_Herding_QRF_Complete.ipynb
```

Or execute the complete automated pipeline via CLI:
```bash
python scripts/run_pipeline.py
```

---

## Authors & Academic References
* **Authors:** Naoual HOUSSNI & Ilyass EL BEKKALI
* **Key Academic References:**
  * Chang, E. C., Cheng, J. W., & Khorana, A. (2000). *An examination of herd behavior in equity markets: An international perspective*. Journal of Banking & Finance, 24(10), 1651–1679.
  * Christie, W. G., & Huang, R. D. (1995). *Following the pied piper: Do individual returns herd around the market?*. Financial Analysts Journal, 51(4), 31–42.
  * Gu, M., Kang, W., & Xu, B. (2018). *Limits of arbitrage and idiosyncratic volatility: Evidence from China*. Journal of Banking & Finance, 86, 240–258.
  * Koenker, R., & Bassett, G. (1978). *Regression quantiles*. Econometrica, 46(1), 33–50.
  * Meinshausen, N. (2006). *Quantile regression forests*. Journal of Machine Learning Research, 7, 983–999.
  * Shleifer, A., & Vishny, R. W. (1997). *The limits of arbitrage*. The Journal of Finance, 52(1), 35–55.
