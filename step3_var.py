"""
Step 3 of BlackSwan Lab: Value at Risk (VaR) and Conditional VaR (CVaR).

Run:   python step3_var.py
Needs the ./data folder made by download_data.py.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PORTFOLIO_VALUE = 1_000_000      # pretend we invest Rs 10 lakh
os.makedirs("output", exist_ok=True)


def var_cvar(returns, confidence=0.95):
    """Historical VaR and CVaR, both returned as positive loss fractions.

    VaR  : the loss that was exceeded on only (1 - confidence) of days.
    CVaR : the average loss on those worst days.
    """
    cutoff = np.percentile(returns, (1 - confidence) * 100)   # e.g. the 5th percentile
    tail = returns[returns <= cutoff]                         # the worst days
    return -cutoff, -tail.mean()


def self_test():
    """Check our function against a tiny example we can verify by hand."""
    r = pd.Series([-4, -3, -2, -2, -1, -1, -1, 0, 0, 0,
                   0, 1, 1, 1, 2, 2, 2, 3, 3, 4]) / 100       # 20 days, in %
    var, cvar = var_cvar(r, confidence=0.90)
    # Worst 10% of 20 days = 2 days: -4% and -3%  ->  average = -3.5%
    assert abs(cvar - 0.035) < 1e-9, cvar
    # The cutoff lies between the 2nd worst (-3%) and 3rd worst (-2%) day
    assert 0.02 < var < 0.03, var
    print(f"Self-test passed: VaR(90%) = {var:.1%}, CVaR(90%) = {cvar:.1%}")


self_test()

# ---- Load data and build the portfolio's daily returns ----
prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
weights = pd.read_csv("data/portfolio.csv").set_index("ticker")["weight"]
stocks = list(weights.index)
returns = prices[stocks].pct_change().dropna()
port_ret = (returns * weights[stocks]).sum(axis=1)    # daily portfolio return

# ---- VaR and CVaR at two confidence levels ----
print(f"\nPortfolio of Rs {PORTFOLIO_VALUE:,} - one-day historical risk "
      f"({len(port_ret)} days of history)\n")
for conf in (0.95, 0.99):
    var, cvar = var_cvar(port_ret, conf)
    n_tail = int((port_ret <= -var).sum())
    print(f"{conf:.0%} confidence:")
    print(f"  VaR  = {var:6.2%}  (Rs {var * PORTFOLIO_VALUE:,.0f})  "
          f"- exceeded on only {n_tail} days")
    print(f"  CVaR = {cvar:6.2%}  (Rs {cvar * PORTFOLIO_VALUE:,.0f})  "
          f"- average loss on those days")

# ---- The answer depends on which period you look at ----
last_date = port_ret.index[-1]
windows = {
    "Whole period": port_ret,
    "Calm 2017": port_ret.loc["2017-01-01":"2017-12-31"],
    "2008 crash year": port_ret.loc["2008-01-01":"2008-12-31"],
    "Last 3 years": port_ret.loc[last_date - pd.DateOffset(years=3):],
}
print("\n95% one-day risk depends on which period you look at:")
for name, r in windows.items():
    var, cvar = var_cvar(r, 0.95)
    print(f"  {name:16s} VaR {var:6.2%}   CVaR {cvar:6.2%}   ({len(r)} days)")

# ---- Chart ----
var95, cvar95 = var_cvar(port_ret, 0.95)
var99, _ = var_cvar(port_ret, 0.99)

plt.figure(figsize=(11, 5))
plt.hist(port_ret * 100, bins=120, color="steelblue")
plt.axvline(-var95 * 100, color="orange", label=f"VaR 95%: -{var95:.2%}")
plt.axvline(-var99 * 100, color="red", label=f"VaR 99%: -{var99:.2%}")
plt.axvline(-cvar95 * 100, color="black", linestyle="--", label=f"CVaR 95%: -{cvar95:.2%}")
plt.title("Daily portfolio returns (%) and the worst-day cutoffs")
plt.xlabel("Daily return (%)")
plt.ylabel("Number of days")
plt.legend()
plt.tight_layout()
plt.savefig("output/var_chart.png", dpi=120)
print("\nChart saved to output/var_chart.png")
plt.show()
