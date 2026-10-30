"""
Step 6 of BlackSwan Lab: where does the risk come from, and what does rebalancing change?

Run:   python step6_rebalance.py
Needs the ./data folder made by download_data.py.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRADING_DAYS = 252
os.makedirs("output", exist_ok=True)

# ---- Load data ----
prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
sectors = pd.read_csv("data/sectors.csv").set_index("ticker")["sector"]
weights = pd.read_csv("data/portfolio.csv").set_index("ticker")["weight"]
stocks = list(weights.index)
sectors = sectors[stocks]

returns = prices[stocks].pct_change().dropna()
nifty = prices["^NSEI"].pct_change(fill_method=None).rename("NIFTY")
cov = returns.cov().values
corr = returns.corr().values
vols = np.sqrt(np.diag(cov))
vol_s = returns.std()

# Betas (same method as step 4)
data = pd.concat([returns, nifty], axis=1, sort=True).dropna()
betas = data[stocks].apply(lambda s: s.cov(data["NIFTY"]) / data["NIFTY"].var())


def var_cvar(x, confidence=0.95):
    cutoff = np.percentile(x, (1 - confidence) * 100)
    return -cutoff, -x[x <= cutoff].mean()


# ============ PART 1: where does the risk come from? ============
def risk_shares(w):
    """Each stock's share of total portfolio variance. The shares add up to 100%."""
    w = np.asarray(w)
    return w * (cov @ w) / (w @ cov @ w)


rs = pd.Series(risk_shares(weights[stocks].values), index=stocks)
assert abs(rs.sum() - 1) < 1e-9, "risk shares must add to 100%"

by_sector = pd.DataFrame({
    "weight_%": weights[stocks].groupby(sectors).sum() * 100,
    "risk_share_%": rs.groupby(sectors).sum() * 100,
})
by_sector["risk_minus_weight"] = by_sector["risk_share_%"] - by_sector["weight_%"]
print("RISK vs MONEY by sector (equal-weight portfolio):")
print(by_sector.sort_values("risk_share_%", ascending=False).round(1).to_string())

print("\nTop 5 individual risk contributors:")
for t, share in rs.sort_values(ascending=False).head(5).items():
    print(f"  {t.replace('.NS', ''):14s} weight {weights[t]:5.1%}   risk share {share:5.1%}")


# ============ PART 2: alternative allocations ============
def normalize(x):
    return x / x.sum()


sector_counts = sectors.value_counts()
allocations = {
    "Equal": weights[stocks],
    "InvVol": normalize(1 / vol_s),
    "LowBeta": normalize(1 / betas),
    "SectorEq": 1 / (len(sector_counts) * sectors.map(sector_counts)),
}
for name, wts in allocations.items():
    assert abs(wts.sum() - 1) < 1e-9, f"{name} weights do not sum to 100%"


def run_scenario(wts, market_move=0.0, sector_moves=None):
    """Portfolio gain/loss = sum of weight * (beta * market move + sector shock)."""
    sector_part = sectors.map(sector_moves or {}).fillna(0.0)
    return (wts * (betas * market_move + sector_part)).sum()


CRASHES = {"crash08": ("2008-01-01", "2008-10-27"), "covid": ("2020-02-14", "2020-03-23")}
ones = np.ones((len(stocks), len(stocks)))
stress_corr = 0.5 * corr + 0.5 * ones
stress_cov = np.outer(vols, vols) * stress_corr

results = {}
for name, wts in allocations.items():
    w = wts[stocks].values
    pr = returns @ w                                   # daily portfolio returns
    growth = (1 + pr).cumprod()
    v95, c95 = var_cvar(pr.values, 0.95)
    res = {
        "max_w": w.max(),
        "eff_n": 1 / (w ** 2).sum(),
        "beta": (wts * betas).sum(),
        "vol": pr.std() * np.sqrt(TRADING_DAYS),
        "vol_stress": np.sqrt(w @ stress_cov @ w) * np.sqrt(TRADING_DAYS),
        "var95": v95,
        "cvar95": c95,
        "mdd": (growth / growth.cummax() - 1).min(),
        "mkt20": run_scenario(wts, -0.20),
        "banks30": run_scenario(wts, 0.0, {"Banks": -0.30}),
        "combined": run_scenario(wts, -0.20, {"Banks": -0.15, "Metals": -0.15}),
    }
    for key, (start, end) in CRASHES.items():
        res[key] = (1 + pr.loc[start:end]).prod() - 1
    results[name] = res

rows = [
    ("Biggest single-stock weight", "max_w", "{:.1%}"),
    ("Effective number of stocks", "eff_n", "{:.1f}"),
    ("Portfolio beta", "beta", "{:.2f}"),
    ("Annual volatility", "vol", "{:.1%}"),
    ("  ...with correlations 50% stressed", "vol_stress", "{:.1%}"),
    ("Daily VaR 95%", "var95", "{:.2%}"),
    ("Daily CVaR 95%", "cvar95", "{:.2%}"),
    ("Max drawdown (full history)", "mdd", "{:.1%}"),
    ("Scenario: market -20%", "mkt20", "{:.1%}"),
    ("Scenario: banks -30%", "banks30", "{:.1%}"),
    ("Scenario: combined", "combined", "{:.1%}"),
    ("2008 crash (actual replay)", "crash08", "{:.1%}"),
    ("COVID crash (actual replay)", "covid", "{:.1%}"),
]
print("\nALLOCATIONS COMPARED UNDER THE SAME SHOCKS")
print(f"{'':38s}" + "".join(f"{name:>11s}" for name in allocations))
for label, key, fmt in rows:
    print(f"{label:38s}" + "".join(f"{fmt.format(results[n][key]):>11s}" for n in allocations))

print("\nTop 3 weights in each allocation:")
for name, wts in allocations.items():
    top = wts.sort_values(ascending=False).head(3)
    print(f"  {name:9s}" + ", ".join(f"{t.replace('.NS', '')} {x:.1%}" for t, x in top.items()))

# ============ Charts ============
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sec = by_sector.sort_values("risk_share_%")
y = np.arange(len(sec))
ax[0].barh(y - 0.2, sec["weight_%"], height=0.4, label="Share of money", color="steelblue")
ax[0].barh(y + 0.2, sec["risk_share_%"], height=0.4, label="Share of risk", color="crimson")
ax[0].set_yticks(y)
ax[0].set_yticklabels(sec.index)
ax[0].set_xlabel("%")
ax[0].set_title("Money vs risk by sector (equal-weight portfolio)")
ax[0].legend()

keys = ["mkt20", "banks30", "combined", "crash08", "covid"]
labels = ["Market -20%", "Banks -30%", "Combined", "2008 replay", "COVID replay"]
x = np.arange(len(keys))
width = 0.2
for i, name in enumerate(allocations):
    ax[1].bar(x + (i - 1.5) * width, [results[name][k] * 100 for k in keys], width, label=name)
ax[1].set_xticks(x)
ax[1].set_xticklabels(labels)
ax[1].axhline(0, color="black", linewidth=0.8)
ax[1].set_ylabel("Portfolio gain / loss (%)")
ax[1].set_title("Same shocks, different allocations")
ax[1].legend()

plt.tight_layout()
plt.savefig("output/rebalance_chart.png", dpi=120)
print("\nChart saved to output/rebalance_chart.png")
plt.show()
