# S&P 500 Realized Volatility Forecasting via Ensemble Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FL-28C976?style=flat-square&logo=xgboost&logoColor=white)

> **Authors:** Alec Reynen, Carl Roussel, Louis Roze
> **Context:** Machine Learning for Finance

---

## 📖 Abstract

This project addresses the challenge of forecasting the **10-day realized volatility** of the S&P 500 index. Volatility forecasting is a critical component of financial risk management, yet it presents specific challenges regarding non-stationarity and temporal causality.

Our approach implements a rigorous pipeline designed to eliminate **Look-Ahead Bias**. We propose a **Weighted Voting Regressor** that combines the stability of regularized linear models with the non-linear capabilities of gradient boosting. The model's primary application is to drive dynamic risk metrics, such as Value at Risk (VaR) and volatility-targeted portfolio allocation.

---

## 🎯 Modeling Objective

The goal is to predict the future volatility of the market over a two-week horizon, using only information available at the time of prediction.

### The Target Variable ($Y$)
The target is the **10-day Realized Volatility**, computed as the rolling standard deviation of daily log-returns:

$$\text{Target}_t = \sigma(r_{t-9}, \dots, r_t)$$

*Where the daily log-return is defined as:*

$$r_t = \ln(P_t) - \ln(P_{t-1})$$

---

## 📂 Data & Feature Engineering

The dataset aggregates multi-asset financial data and macroeconomic indicators from 2015 to 2025.

### 1. Anti-Leakage Protocol
To ensure the model is viable for real-world trading, we applied a strict **Blocklist**:
* **Excluded:** All contemporaneous variables (e.g., `VIX_Today`, `Vol_10d_Today`) are removed from the feature matrix $X$.
* **Included:** Only lagged versions ($t-1$) and macro-economic indicators known prior to the market open are retained.

### 2. Feature Dictionary ($X$)
The final model utilizes a set of 27 predictors grouped into four categories:

| Category | Variables | Description & Rationale |
| :--- | :--- | :--- |
| **Endogenous** | `roll_std_10_l1_log` | **Log-transformed lagged volatility**. The strongest predictor due to volatility clustering/persistence. |
| | `ewma_10d_l1`, `ewma_20d_l1` | Exponentially Weighted Moving Averages for reactive trend detection. |
| | `meanabs_10d_l1` | Robust volatility proxy based on absolute returns. |
| **Market Regime** | `ret_neg_sq_l1` | **Downside Leverage**: Squared returns considering only negative days (captures panic/asymmetry). |
| | `vrp_ratio` | **Volatility Risk Premium**: Ratio of Implied Volatility (VIX) to Realized Volatility. |
| **Exogenous** | `vix` | CBOE Volatility Index (Market sentiment/Fear gauge). |
| | `brent_ret`, `gold_ret` | Log-returns of Commodities (Oil & Gold). |
| | `eurusd_ret` | Log-returns of the EUR/USD exchange rate. |
| **Macro** | `SPREAD_10_2` | Yield Curve slope (10Y - 2Y Treasury yields), a leading recession indicator. |
| | `dCPI_YoY`, `dUNEMP_YoY` | Year-over-Year changes in Inflation and Unemployment (stationarized). |

---

## ⚙️ Methodology

### Validation Strategy: Walk-Forward
Standard K-Fold cross-validation is unsuitable for time series as it shuffles temporal order. We employ **`TimeSeriesSplit` (Expanding Window)**:
* The model is trained on past data $[T_0, T_k]$ and evaluated on future data $[T_{k+1}]$.
* This simulates a realistic production environment where the model is periodically retrained.

### Model Architecture
We explored a hierarchy of four algorithms to characterize market dynamics before constructing the final ensemble:

1.  **Lasso Regression (L1):** Acts as a feature selector and captures high-magnitude volatility spikes (High Bias, Low Variance).
2.  **XGBoost (Gradient Boosting):** Used for surgical precision during normal market regimes, actively reducing bias by correcting residual errors.
3.  **Ridge Regression (L2):** Used as a conservative linear baseline.
4.  **Random Forest (Bagging):** Evaluated as a non-linear benchmark to assess stability and the extrapolation limits ("Ceiling Effect") of tree models.

> **🏆 Final Engine:** The Champion Model is a **Weighted Voting Regressor** combining **Lasso (50%)**, **XGBoost (40%)**, and **Ridge (10%)**, selected for their complementary behavior during the validation phase.

---

## 💼 Business Application

The predictive output is converted into actionable financial metrics:

1.  **Risk Management (VaR):** Estimation of the **Value at Risk (95%)** for a theoretical portfolio, allowing for dynamic capital provisioning during crisis periods (e.g., March 2020).
2.  **Volatility Targeting:** Simulation of an active investment strategy that scales exposure inversely to predicted volatility ($w_t \propto 1/\hat{\sigma}_t$), aiming to improve the Sharpe Ratio by avoiding drawdown periods.
