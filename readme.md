# Beta Tracker (Finance)

## Overview

**Beta Tracker** is a financial analytics project that calculates, tracks, and visualizes **beta values** of stocks or portfolios relative to a benchmark index (e.g., NIFTY 50, SENSEX, S&P 500). Beta measures the volatility or systematic risk of a security compared to the overall market.

This project is useful for **investors, finance students, and analysts** to understand market risk, compare stocks, and support portfolio decision‑making.

---

## Objectives

* Calculate beta for individual stocks
* Compute **rolling beta** over time to observe risk variation
* Compare stock volatility against a benchmark index
* Analyze risk exposure across different time windows
* Provide clear visualizations for better interpretation
* Support learning and basic investment analysis

---

## What is Beta?

* **Beta = 1** → Stock moves with the market
* **Beta > 1** → More volatile than the market (higher risk)
* **Beta < 1** → Less volatile than the market (lower risk)
* **Beta < 0** → Moves opposite to the market

Formula:

> **β = Covariance(stock returns, market returns) / Variance(market returns)**

---

## Features

* Fetch historical price data for stocks and benchmark indices
* Compute daily / weekly / monthly returns
* Calculate beta using statistical methods
* **Rolling beta calculation** (e.g., 30‑day, 60‑day, 90‑day windows)
* Time‑series analysis of changing market risk
* Interactive charts and summary tables

---

## Tech Stack

* **Backend**: FastAPI
* **Language**: Python
* **Data Processing**: Pandas, NumPy
* **Visualization**: Matplotlib / Plotly
* **API Docs**: Swagger (OpenAPI)
* **Storage**: CSV / Local DB (extensible)

---

## How It Works
1. Ingest OHLCV stock and index data via API
2. Compute returns from price data
3. Calculate beta and rolling beta
4. Store computed results
5. Serve beta values and visualization data via APIs

---

## Sample Output
- Stock beta value (numeric)
- Rolling beta time‑series plot
- Risk interpretation over time (increasing / decreasing volatility)
- Line charts comparing stock vs market returns





