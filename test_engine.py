"""
Tests for riskengine.py. They use made-up data, so no files are needed.

Run:   python test_engine.py
Every line should say PASS. If one says FAIL, the maths has a bug.
"""
import numpy as np
import pandas as pd

import riskengine as eng

# ---- Build a small fake market that covers 2008 and 2020 ----
rng = np.random.default_rng(1)
dates = pd.bdate_range("2006-01-02", "2021-12-31")
TICKERS = ["A1", "A2", "B1", "B2", "C1", "C2"]
SECTORS = ["Banks", "Banks", "IT", "IT", "Pharma", "Pharma"]
market = rng.normal(0.0004, 0.011, len(dates))
beta_true = [1.3, 1.2, 0.9, 0.8, 0.5, 0.6]
rets = {t: b * market + rng.normal(0, 0.008, len(dates)) for t, b in zip(TICKERS, beta_true)}
prices = 100 * (1 + pd.DataFrame(rets, index=dates)).cumprod()
prices["^NSEI"] = 100 * (1 + market).cumprod()
sectors_df = pd.DataFrame({"ticker": TICKERS, "sector": SECTORS})
portfolio_df = pd.DataFrame({"ticker": TICKERS, "weight": [1 / 6] * 6})
m = eng.build_market(prices, sectors_df, portfolio_df)
w = m.file_weights

passed = 0


def check(name, condition):
    global passed
    print(("PASS  " if condition else "FAIL  ") + name)
    passed += bool(condition)


# ---- VaR / CVaR against the hand-checkable example ----
x = pd.Series([-4, -3, -2, -2, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4]) / 100
v, c = eng.var_cvar(x, 0.90)
check("CVaR equals the average of the worst 10% of 20 days (3.5%)", abs(c - 0.035) < 1e-12)
check("CVaR is never smaller than VaR", c >= v)

# ---- Risk shares ----
check("Risk shares add up to 100%", abs(eng.risk_shares(m, w).sum() - 1) < 1e-9)

# ---- Correlation stress ----
cov0, _ = eng.stressed_cov(m, 0.0)
check("Stress 0% leaves the covariance matrix unchanged", np.allclose(cov0, m.cov))
vol_full = eng.annual_vol(m, w, eng.stressed_cov(m, 1.0)[0])
avg_vol = float((w[m.stocks] * m.vols).sum() * np.sqrt(252))
check("With correlation 1, portfolio volatility = weighted average volatility",
      abs(vol_full - avg_vol) < 1e-9)
check("Diversification works: real volatility < weighted average volatility",
      eng.annual_vol(m, w) < avg_vol)

# ---- Scenarios ----
total, _, _ = eng.run_scenario(m, w, market_move=-0.10)
check("Market -10% changes the portfolio by (portfolio beta x -10%)",
      abs(total - eng.portfolio_beta(m, w) * -0.10) < 1e-12)
total, _, _ = eng.run_scenario(m, w, 0.0, {"Banks": -0.30})
check("Banks -30% costs (banks' weight x 30%)", abs(total - (2 / 6) * -0.30) < 1e-12)
_, _, moves = eng.run_scenario(m, w, -0.60, {"Banks": -0.50})
check("No stock can lose more than 100%", moves.min() >= -1.0)

# ---- Monte Carlo ----
wv = w[m.stocks].values
analytic = np.sqrt(wv @ m.cov @ wv)
for fat in (False, True):
    one_day = eng.simulate(m.cov, wv, horizon=1, n_sims=20000, fat_tails=fat)
    check(f"Simulated daily volatility matches real (fat tails = {fat})",
          abs(one_day.std() / analytic - 1) < 0.04)
a = eng.simulate(m.cov, wv, 21, 4000)
b = eng.simulate(m.cov, wv, 21, 4000)
check("Same seed gives identical simulations", np.array_equal(a, b))
check("Fat-tailed simulation has a worse 99% month than bell-curve (more extreme)",
      eng.var_cvar(eng.simulate(m.cov, wv, 21, 20000, True), 0.99)[1]
      > eng.var_cvar(eng.simulate(m.cov, wv, 21, 20000, False), 0.99)[1] * 0.98)

# ---- Allocations ----
allocs = eng.make_allocations(m)
for name, wt in allocs.items():
    check(f"Allocation '{name}': weights sum to 100% and none negative",
          abs(wt.sum() - 1) < 1e-9 and (wt >= 0).all())
res = eng.evaluate_allocation(m, w, -0.2, {"Banks": -0.1})
check("evaluate_allocation reports the crash replays", len(eng.replay_windows(m)) == 2
      and all(k in res for k in eng.REPLAYS))

# ---- Diversification by year ----
reg = eng.regime_table(m, w)
check("Year-by-year table has one row per full year", len(reg) >= 15)
check("Average correlations are valid (between -1 and 1)",
      ((reg["avg_corr"] >= -1) & (reg["avg_corr"] <= 1)).all())
check("Diversification never makes a long-only portfolio riskier", (reg["div_benefit"] >= -1e-9).all())

# ---- Input validation ----
try:
    eng.build_market(prices, sectors_df, pd.DataFrame({"ticker": ["A1", "ZZZ"], "weight": [.5, .5]}))
    check("Unknown ticker is rejected", False)
except eng.DataError:
    check("Unknown ticker is rejected", True)

m2 = eng.build_market(prices, sectors_df, pd.DataFrame({"ticker": TICKERS, "weight": [10] * 6}))
check("Weights not summing to 1 are rescaled with a warning",
      abs(m2.file_weights.sum() - 1) < 1e-9 and any("rescaled" in n for n in m2.notes))

m3 = eng.build_market(prices.drop(columns="^NSEI"), sectors_df, portfolio_df)
check("Missing benchmark falls back to a proxy with a warning",
      any("proxy" in n.lower() or "instead of a market index" in n for n in m3.notes))

short = prices.loc["2015-01-01":"2015-12-31"]
m4 = eng.build_market(short.iloc[:260], sectors_df, portfolio_df)
check("A one-year history works but has no crash replays to show", eng.replay_windows(m4) == {})

try:
    eng.build_market(prices.iloc[:100], sectors_df, portfolio_df)
    check("Too little history is rejected", False)
except eng.DataError:
    check("Too little history is rejected", True)

print(f"\n{passed} checks passed.")