"""
Step 5 of BlackSwan Lab: Monte Carlo simulation and correlation stress.

Run:   python step5_montecarlo.py
Needs the ./data folder made by download_data.py. Takes a few seconds.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PORTFOLIO_VALUE = 1_000_000   # pretend we invest Rs 10 lakh
HORIZON = 21                  # trading days (about one month)
N_SIMS = 20_000               # number of simulated futures
DOF = 5                       # fat-tail setting (smaller = fatter tails; must be > 2)
TRADING_DAYS = 252
os.makedirs("output", exist_ok=True)
rng = np.random.default_rng(42)          # fixed seed so results are repeatable

# ---- Load data ----
prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
weights = pd.read_csv("data/portfolio.csv").set_index("ticker")["weight"]
stocks = list(weights.index)
returns = prices[stocks].pct_change().dropna()

w = weights[stocks].values               # portfolio weights as a plain array
cov = returns.cov().values               # daily covariance matrix of our stocks
vols = np.sqrt(np.diag(cov))             # each stock's daily volatility
corr = returns.corr().values             # correlation matrix
n = len(stocks)
port_ret = returns @ w                   # historical daily portfolio returns


def var_cvar(x, confidence=0.95):
    """Historical-style VaR and CVaR (positive loss fractions) of an array of returns."""
    cutoff = np.percentile(x, (1 - confidence) * 100)
    return -cutoff, -x[x <= cutoff].mean()


# ---- Random numbers are drawn ONCE and reused for every scenario, so differences
#      between scenarios come from our assumptions, not from luck. ----
n_draws = N_SIMS * HORIZON
base_normal = rng.standard_normal((n_draws, n))
base_shock = np.sqrt(rng.chisquare(DOF, n_draws) / DOF)[:, None]


def simulate(cov_matrix, fat_tails=True):
    """Simulate N_SIMS possible HORIZON-day portfolio returns.
    Zero average drift is assumed (standard for short-horizon risk)."""
    jitter = 1e-8 * np.eye(n) * np.mean(np.diag(cov_matrix))   # tiny fix for numerical safety
    chol = np.linalg.cholesky(cov_matrix + jitter)
    z = base_normal @ chol.T                                   # correlated daily stock returns
    if fat_tails:
        # Student-t: divide by a random 'shock size'. The scale factor keeps the
        # volatility equal to the historical one - only the TAILS get fatter.
        z = z * np.sqrt((DOF - 2) / DOF) / base_shock
    daily_port = (z @ w).reshape(N_SIMS, HORIZON)              # one row per simulated future
    return np.prod(1 + daily_port, axis=1) - 1, daily_port     # compound over the horizon


sim_normal, daily_normal = simulate(cov, fat_tails=False)
sim_fat, daily_fat = simulate(cov, fat_tails=True)

# ---- Sanity check: do the simulations reproduce the volatility in our data? ----
real_vol = port_ret.std()
print(f"Daily portfolio volatility: real data {real_vol:.3%} | "
      f"simulated bell-curve {daily_normal.std():.3%} | simulated fat-tail {daily_fat.std():.3%}")
assert abs(daily_normal.std() / real_vol - 1) < 0.05, "normal simulation is off"
assert abs(daily_fat.std() / real_vol - 1) < 0.05, "fat-tail simulation is off"

# ---- 1-month risk: three ways to estimate it ----
hist_h = (np.exp(np.log1p(port_ret).rolling(HORIZON).sum()) - 1).dropna().values
methods = {
    "Historical (overlapping months)": hist_h,
    "Monte Carlo, bell-curve": sim_normal,
    "Monte Carlo, fat tails": sim_fat,
}
print(f"\n{HORIZON}-day (about 1 month) risk on Rs {PORTFOLIO_VALUE:,}:")
print(f"{'':34s}{'VaR 95%':>9}{'CVaR 95%':>10}{'VaR 99%':>9}{'CVaR 99%':>10}")
for name, x in methods.items():
    v95, c95 = var_cvar(x, 0.95)
    v99, c99 = var_cvar(x, 0.99)
    print(f"{name:34s}{v95:9.2%}{c95:10.2%}{v99:9.2%}{c99:10.2%}")
print("(Overlapping months share most of their days, so the historical row is "
      "less independent evidence than it looks.)")

# ---- Correlation stress: pull every pair of stocks toward correlation = 1 ----
# stressed correlation = (1 - lam) * real correlation + lam * 1
# lam = 0 is reality, lam = 1 means all stocks move perfectly together.
ones = np.ones((n, n))
print("\nCorrelation stress (fat-tail simulation):")
print(f"{'stress':>7}{'avg corr':>10}{'annual vol':>12}{'VaR 95%':>9}{'CVaR 95%':>10}{'VaR 95% in Rs':>16}")
stress_rows = []
for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
    stressed_corr = (1 - lam) * corr + lam * ones
    stressed_cov = np.outer(vols, vols) * stressed_corr
    annual_vol = np.sqrt(w @ stressed_cov @ w) * np.sqrt(TRADING_DAYS)
    avg_corr = (stressed_corr.sum() - n) / (n * (n - 1))
    sim, _ = simulate(stressed_cov, fat_tails=True)
    v95, c95 = var_cvar(sim, 0.95)
    stress_rows.append((avg_corr, v95, c95))
    print(f"{lam:7.2f}{avg_corr:10.2f}{annual_vol:12.1%}{v95:9.2%}{c95:10.2%}{v95 * PORTFOLIO_VALUE:16,.0f}")

# ---- Charts ----
v95, c95 = var_cvar(sim_fat, 0.95)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].hist(sim_fat * 100, bins=100, color="steelblue")
ax[0].axvline(-v95 * 100, color="orange", label=f"VaR 95%: -{v95:.1%}")
ax[0].axvline(-c95 * 100, color="black", linestyle="--", label=f"CVaR 95%: -{c95:.1%}")
ax[0].set_title(f"{N_SIMS:,} simulated {HORIZON}-day portfolio returns (fat tails)")
ax[0].set_xlabel("Return over the month (%)")
ax[0].legend()

xs = [r[0] for r in stress_rows]
ax[1].plot(xs, [r[1] * 100 for r in stress_rows], marker="o", label="VaR 95%")
ax[1].plot(xs, [r[2] * 100 for r in stress_rows], marker="o", label="CVaR 95%")
ax[1].set_title("Monthly loss grows as correlations rise")
ax[1].set_xlabel("Average correlation between stocks")
ax[1].set_ylabel("Loss (%)")
ax[1].legend()

plt.tight_layout()
plt.savefig("output/montecarlo_chart.png", dpi=120)
print("\nChart saved to output/montecarlo_chart.png")
plt.show()
