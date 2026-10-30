"""
Step 4 of BlackSwan Lab: stress tests using beta (market shocks) and sector shocks.

Run:   python step4_stress.py
Needs the ./data folder made by download_data.py.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

PORTFOLIO_VALUE = 1_000_000      # pretend we invest Rs 10 lakh
os.makedirs("output", exist_ok=True)

# ---- Load data ----
prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
sectors = pd.read_csv("data/sectors.csv").set_index("ticker")["sector"]
weights = pd.read_csv("data/portfolio.csv").set_index("ticker")["weight"]
stocks = list(weights.index)

returns = prices[stocks].pct_change().dropna()
nifty = prices["^NSEI"].pct_change(fill_method=None).rename("NIFTY")
port_ret = (returns * weights[stocks]).sum(axis=1)

# ---- The extreme days in our history (so we know what the tails are made of) ----
print("Five worst days for the portfolio:")
for date, r in port_ret.nsmallest(5).items():
    print(f"  {date.date()}  {r:7.2%}")
best = port_ret.nlargest(1)
print(f"Best day: {best.index[0].date()}  {best.iloc[0]:.2%}")

# ---- Beta: how much does each stock move per 1% move in the Nifty? ----
data = pd.concat([returns, nifty], axis=1, sort=True).dropna()       # only days where everything exists
betas = data[stocks].apply(lambda s: s.cov(data["NIFTY"]) / data["NIFTY"].var())
portfolio_beta = (weights[stocks] * betas).sum()

print(f"\nBetas (estimated from {len(data)} days):")
beta_table = pd.DataFrame({"sector": sectors[stocks], "beta": betas.round(2)})
print(beta_table.sort_values("beta", ascending=False).to_string())
print(f"Portfolio beta: {portfolio_beta:.2f}")


# ---- The stress-test engine ----
def run_scenario(market_move=0.0, sector_moves=None):
    """Each stock's assumed move = beta * market move + any direct sector shock.
    Returns each stock's contribution to the portfolio's gain/loss."""
    sector_moves = sector_moves or {}
    sector_part = sectors[stocks].map(sector_moves).fillna(0.0)
    stock_moves = betas * market_move + sector_part
    return weights[stocks] * stock_moves


SCENARIOS = {
    "Market -10%": (-0.10, {}),
    "Market -20%": (-0.20, {}),
    "Market -60% (2008-sized)": (-0.60, {}),
    "Banks -30% (market flat)": (0.0, {"Banks": -0.30}),
    "IT -25% (market flat)": (0.0, {"IT": -0.25}),
    "Metals -35% (market flat)": (0.0, {"Metals": -0.35}),
    "Combined: market -20%, banks & metals a further -15%": (-0.20, {"Banks": -0.15, "Metals": -0.15}),
}

print("\n=== Hypothetical scenarios ===")
chart_labels, chart_values, chart_colors = [], [], []
for name, (market_move, sector_moves) in SCENARIOS.items():
    contrib = run_scenario(market_move, sector_moves)
    total = contrib.sum()
    worst3 = contrib.sort_values().head(3)
    print(f"\n{name}")
    print(f"   Portfolio: {total:7.1%}   (Rs {total * PORTFOLIO_VALUE:,.0f})")
    print("   Biggest hits: " + ", ".join(
        f"{t.replace('.NS', '')} {c:.1%}" for t, c in worst3.items()))
    chart_labels.append(name)
    chart_values.append(total * 100)
    chart_colors.append("darkorange")

# ---- Replay real crashes and check our beta model against reality ----
HISTORICAL = {
    "2008 crash (2 Jan - 27 Oct 2008)": ("2008-01-01", "2008-10-27"),
    "COVID crash (14 Feb - 23 Mar 2020)": ("2020-02-14", "2020-03-23"),
}
print("\n=== Real crashes vs what our beta model would have predicted ===")
for name, (start, end) in HISTORICAL.items():
    actual = (1 + port_ret.loc[start:end]).prod() - 1
    nifty_move = (1 + nifty.loc[start:end].dropna()).prod() - 1
    predicted = portfolio_beta * nifty_move
    print(f"\n{name}")
    print(f"   Nifty moved:               {nifty_move:7.1%}")
    print(f"   Beta model predicted:      {predicted:7.1%}")
    print(f"   Portfolio actually moved:  {actual:7.1%}")
    chart_labels.append("ACTUAL: " + name)
    chart_values.append(actual * 100)
    chart_colors.append("crimson")

# ---- Chart ----
plt.figure(figsize=(12, 6))
plt.barh(chart_labels, chart_values, color=chart_colors)
plt.gca().invert_yaxis()
plt.xlabel("Portfolio gain / loss (%)")
plt.title("Stress scenarios (orange) vs real crashes (red)")
plt.tight_layout()
plt.savefig("output/stress_chart.png", dpi=120)
print("\nChart saved to output/stress_chart.png")
plt.show()