# GMF Investments – Time Series Forecasting for Portfolio Management Optimization

## Overview

This project applies time series forecasting and Modern Portfolio Theory (MPT) to optimize portfolio management for **Guide Me in Finance (GMF) Investments**. Using historical financial data for TSLA, BND, and SPY sourced from YFinance, we build predictive models to forecast market trends and recommend optimal portfolio allocations that maximize returns while managing risk.

## Business Context

GMF Investments provides personalized portfolio management strategies. This project leverages:
- **TSLA** – High-growth, high-risk equity
- **BND** – Low-risk bond ETF for stability
- **SPY** – Moderate-risk broad market exposure (S&P 500)

> Note: Per the Efficient Market Hypothesis (EMH), these models are used to forecast volatility and inform portfolio allocation decisions — not as standalone price prediction oracles.

## Project Structure

```
portfolio-optimization/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   ├── raw/              # Raw downloaded data (gitignored)
│   └── processed/        # Cleaned & feature-engineered data (gitignored)
├── notebooks/
│   ├── __init__.py
│   ├── README.md
│   ├── task1_eda.ipynb
│   ├── task2_forecasting.ipynb
│   ├── task3_forecast_trends.ipynb
│   ├── task4_portfolio_optimization.ipynb
│   └── task5_backtesting.ipynb
├── src/
│   └── __init__.py       # Shared utilities and helper modules
├── tests/
│   └── __init__.py       # Unit tests
└── scripts/
    └── __init__.py       # Standalone scripts
```

## Tasks

| Task | Description | Branch |
|---|---|---|
| Task 1 | Data extraction, EDA, stationarity testing, risk metrics | `task-1` |
| Task 2 | ARIMA/SARIMA & LSTM forecasting models | `task-2` |
| Task 3 | Future market trend forecasting | `task-3` |
| Task 4 | Portfolio optimization via Efficient Frontier | `task-4` |
| Task 5 | Strategy backtesting vs. benchmark | `task-5` |

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd portfolio-optimization

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab
```

## Data

Data is fetched automatically via the `yfinance` library for the period **January 1, 2015 – June 30, 2026**. Raw and processed data files are excluded from version control (see `.gitignore`).

## Key Dependencies

- `yfinance` – Financial data API
- `pandas`, `numpy` – Data manipulation
- `statsmodels`, `pmdarima` – ARIMA/SARIMA modeling
- `tensorflow` / `keras` – LSTM deep learning
- `PyPortfolioOpt` – Portfolio optimization
- `matplotlib`, `seaborn`, `plotly` – Visualization

## Team

| Name | Role |
|---|---|
| Kerod | Team Lead / Data Analyst |
| Mahbubah | Modeling & Optimization |
| Feven | Backtesting & Visualization |

## Key Dates

| Milestone | Date |
|---|---|
| Challenge Introduction | Wed, 01 Jul 2026, 10:30 AM UTC |
| Interim Submission | Sun, 05 Jul 2026, 8:00 PM UTC |
| **Final Submission** | **Tue, 07 Jul 2026, 8:00 PM UTC** |

## License

This project is developed for educational purposes as part of the 10 Academy Week 9 Challenge.
