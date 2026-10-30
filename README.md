# BlackSwan Lab: Portfolio Stress Testing & Risk Intelligence

Quantactive: Code Conflux (MNNIT Allahabad) - Problem Statement 02

A dashboard that shows how a stock portfolio behaves in a crisis. It measures risk, estimates
potential losses (historical VaR and CVaR), simulates market crashes, sector shocks and
rising correlations, finds which stocks and sectors drive the risk, and compares alternative
allocations under identical shocks.

## Quick start

```
pip install yfinance pandas matplotlib streamlit
python download_data.py        # downloads prices once into ./data
python test_engine.py          # 27 automated checks of the maths (all should say PASS)
streamlit run app.py           # opens the dashboard in the browser
```

The dashboard also accepts your own files (sidebar -> "Upload my own files"):

| File | Format |
|---|---|
| Prices CSV | First column = dates, then one column of prices per ticker. An optional `^NSEI` column is used as the market index; without it, betas use the average of your stocks. |
| Sectors CSV | Columns `ticker`, `sector` |
| Portfolio CSV | Columns `ticker`, `weight` (rescaled to 100% if needed) |

Bad input produces a clear message instead of a crash (unknown tickers, too little history,
negative weights, missing sectors, and so on).

## What it does, against the problem statement

| Requirement | Where |
|---|---|
| Import prices, holdings, weights | Built-in data or CSV upload |
| Returns, volatility, correlations, max drawdown | Overview tab |
| Historical VaR and CVaR | VaR & CVaR tab (90-99% confidence, rupee amounts) |
| Market-wide and sector price shocks | Stress tests tab (beta-based market shock + per-sector shocks) |
| Largest contributors to losses / risk | Overview and Stress tests tabs |
| Dashboard | Streamlit app, 6 tabs |
| Combined multi-sector scenarios | Market slider + any number of sector sliders at once |
| Correlation changes in stress | Correlation & Monte Carlo tab (0-100% stress slider) |
| Historical stress periods vs hypothetical | Stress tests tab (2008 and COVID replays next to your scenario) |
| Alternative allocations under the same shocks | Rebalancing tab (5 allocations, same shocks) |
| Monte Carlo | Correlation & Monte Carlo tab (bell-curve or fat-tailed) |

Brownie points: before/after rebalancing, plain-English VaR vs CVaR explanation written from
the portfolio's own numbers, a year-by-year view of when diversification failed, judge-adjustable
shocks and correlation, and a Method & limitations tab.

## Method in brief

- **VaR / CVaR:** the worst-day cutoff at a chosen confidence level, and the average loss beyond it.
- **Beta stress test:** each stock moves by (its beta x the market move) plus any sector shock, floored at -100%.
- **Correlation stress:** the correlation matrix is blended toward all-ones (volatilities unchanged),
  so it always stays valid. At 100% stress, portfolio volatility equals the weighted-average stock volatility.
- **Monte Carlo:** correlated daily returns drawn from the historical covariance (zero drift), optionally
  with Student-t fat tails scaled to keep the same volatility.
- **Risk contribution:** each stock's share of portfolio variance; shares add to exactly 100%.
- **Allocations compared:** as in file, equal weight, inverse volatility, low-beta tilt, sector-equal.

## Findings on the built-in data (18 large Indian stocks, 2007-2026, equal weight)

- Diversification cut volatility from a 32.1% weighted average to 19.7% for the portfolio.
- One-day 95% VaR is 1.77% and CVaR is 2.84%. A calm year (2017) gives a VaR of about 1.0%; 2008 gives about 4.3%.
- Average correlation between the stocks was 0.13 in 2017, 0.46 in 2008 and 0.65 in the COVID crash.
- Banks and metals (5 of 18 stocks, 28% of the money) carry about 37% of the risk.
- The maximum drawdown was -50.6% (October 2008).
- The beta model predicted -54.4% for 2008 (actual -50.4%) and -34.6% for the COVID crash (actual -35.3%).
- A low-beta tilt reduced the 2008 replay loss from -50.4% to -43.4%, but this is an in-sample result.

## Limitations (also in the app)

History is not destiny and tail estimates rest on little data; bad days cluster but the simulation draws
days independently; beta is an average relationship; correlation stress leaves each stock's own volatility
unchanged; the built-in stocks are survivors (survivorship bias); weights are assumed reset daily with no
costs or liquidity effects; rebalancing comparisons are in-sample.

## Files

| File | Purpose |
|---|---|
| `download_data.py` | Downloads and saves prices, sectors and the starting portfolio |
| `riskengine.py` | All calculations (no interface code) |
| `app.py` | The Streamlit dashboard |
| `test_engine.py` | 27 self-contained checks of the maths |
| `step2` to `step6_*.py` | The step-by-step scripts the engine was built from |
