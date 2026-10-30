"""
Step 2 of BlackSwan Lab: a first look at our data.

Run:   pip install matplotlib
       python step2_explore.py

Needs the ./data folder made by download_data.py.
Prints a few tables and saves a chart to ./output/overview.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRADING_DAYS = 252          # trading days in a year (used to annualize)
os.makedirs("output", exist_ok=True)

# ---- 1. Load the files made by download_data.py ----
prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
sectors = pd.read_csv("data/sectors.csv").set_index("ticker")["sector"]
weights = pd.read_csv("data/portfolio.csv").set_index("ticker")["weight"]
stocks = list(weights.index)                       # the 18 stocks in our portfolio

# ---- 2. Daily returns = today's price / yesterday's price - 1 ----
returns = prices[stocks].pct_change().dropna()

# ---- 3. Statistics per stock ----
stock_vol = returns.std() * np.sqrt(TRADING_DAYS)  # annualized volatility
stats = pd.DataFrame({
    "sector": sectors[stocks],
    "avg_return_per_year_%": returns.mean() * TRADING_DAYS * 100,
    "volatility_per_year_%": stock_vol * 100,
})
print("\nPer-stock statistics (most volatile first):")
print(stats.sort_values("volatility_per_year_%", ascending=False).round(1).to_string())

# ---- 4. The portfolio ----
# Daily portfolio return = weighted sum of the stocks' daily returns.
# Simplification: this assumes weights are reset to the targets every day.
port_ret = (returns * weights[stocks]).sum(axis=1)
growth = (1 + port_ret).cumprod()                  # value of 1 rupee invested on day 1
drawdown = growth / growth.cummax() - 1            # fall from the highest point so far

port_vol = port_ret.std() * np.sqrt(TRADING_DAYS)
avg_stock_vol = (stock_vol * weights[stocks]).sum()
print("\nWeighted-average volatility of the individual stocks:", round(avg_stock_vol * 100, 1), "%")
print("Volatility of the portfolio as a whole:             ", round(port_vol * 100, 1), "%")
print("Maximum drawdown:", round(drawdown.min() * 100, 1), "% (lowest point on",
      drawdown.idxmin().date(), ")")

# ---- 5. Do stocks move together more in a crash? ----
def avg_correlation(r):
    """Average correlation between all pairs of stocks."""
    c = r.corr().values
    n = len(c)
    return (c.sum() - n) / (n * (n - 1))           # average of the off-diagonal entries

windows = {
    "Whole period": (None, None),
    "Calm 2017": ("2017-01-01", "2017-12-31"),
    "2008 crash": ("2008-01-01", "2008-12-31"),
    "COVID crash": ("2020-02-15", "2020-04-15"),
}
print("\nAverage correlation between our stocks:")
for name, (start, end) in windows.items():
    print(f"  {name:13s} {avg_correlation(returns.loc[start:end]):.2f}")

# ---- 6. Charts ----
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

ax[0].plot(growth)
ax[0].set_yscale("log")
ax[0].set_title("Growth of 1 rupee (equal-weight portfolio)")

ax[1].fill_between(drawdown.index, drawdown * 100, 0, color="crimson")
ax[1].set_title("Drawdown (%): how far below its previous peak")

im = ax[2].imshow(returns.corr(), cmap="RdYlGn_r", vmin=0, vmax=1)
labels = [s.replace(".NS", "") for s in stocks]
ax[2].set_xticks(range(len(stocks)))
ax[2].set_xticklabels(labels, rotation=90, fontsize=7)
ax[2].set_yticks(range(len(stocks)))
ax[2].set_yticklabels(labels, fontsize=7)
ax[2].set_title("Correlation between stocks (red = move together)")
fig.colorbar(im, ax=ax[2], fraction=0.046)

plt.tight_layout()
plt.savefig("output/overview.png", dpi=120)
print("\nChart saved to output/overview.png")
plt.show()
