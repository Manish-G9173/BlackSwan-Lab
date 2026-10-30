"""
BlackSwan Lab - calculation engine.

All the maths lives here. app.py only handles the user interface.
Everything is tested by test_engine.py.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252
BENCHMARK_COL = "^NSEI"

# Real crashes we can replay (only used if the price history covers them)
REPLAYS = {
    "2008 crash (Jan-Oct 2008)": ("2008-01-01", "2008-10-27"),
    "COVID crash (Feb-Mar 2020)": ("2020-02-14", "2020-03-23"),
}


class DataError(ValueError):
    """Raised with a friendly message when the input files are unusable."""


@dataclass
class MarketData:
    stocks: list
    sectors: pd.Series            # sector of each stock (index = ticker)
    file_weights: pd.Series       # weights from the portfolio file (sum to 1)
    returns: pd.DataFrame         # daily returns, only days where every stock has data
    benchmark: pd.Series          # daily returns of the market benchmark
    benchmark_name: str
    cov: np.ndarray               # daily covariance matrix
    corr: np.ndarray              # correlation matrix
    vols: np.ndarray              # daily volatility of each stock
    betas: pd.Series              # beta of each stock vs the benchmark
    notes: list = field(default_factory=list)   # warnings to show the user


# ---------------------------------------------------------------- loading
def build_market(prices, sectors_df, portfolio_df):
    """Validate the three input tables and compute everything we need from them."""
    notes = []
    for df, cols, label in ((sectors_df, {"ticker", "sector"}, "sectors"),
                            (portfolio_df, {"ticker", "weight"}, "portfolio")):
        if not cols.issubset(df.columns):
            raise DataError(f"The {label} file needs the columns: {', '.join(sorted(cols))}.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise DataError("The first column of the prices file must contain dates.")
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated()]

    pf = portfolio_df.copy()
    pf["ticker"] = pf["ticker"].astype(str).str.strip()
    if pf["ticker"].duplicated().any():
        raise DataError("Some tickers appear twice in the portfolio file.")
    pf["weight"] = pd.to_numeric(pf["weight"], errors="coerce")
    if pf["weight"].isna().any() or (pf["weight"] < 0).any():
        raise DataError("Weights must be numbers that are zero or positive (long-only portfolio).")
    pf = pf[pf["weight"] > 0]
    if len(pf) < 2:
        raise DataError("The portfolio needs at least 2 stocks with a positive weight.")
    weights = pf.set_index("ticker")["weight"]

    missing = [t for t in weights.index if t not in prices.columns]
    if missing:
        raise DataError("These tickers are in the portfolio file but not in the prices file: "
                        + ", ".join(missing))
    total = weights.sum()
    if abs(total - 1) > 1e-6:
        notes.append(f"The weights added up to {total:.4g}, so they were rescaled to 100%.")
        weights = weights / total
    stocks = list(weights.index)

    sec = (sectors_df.assign(ticker=sectors_df["ticker"].astype(str).str.strip())
           .drop_duplicates("ticker").set_index("ticker")["sector"])
    sectors = sec.reindex(stocks)
    if sectors.isna().any():
        lacking = ", ".join(sectors[sectors.isna()].index)
        notes.append(f"No sector was given for: {lacking}. They are grouped as 'Unclassified'.")
        sectors = sectors.fillna("Unclassified")

    # Daily returns on days where every stock has a price
    raw = prices[stocks].pct_change(fill_method=None)
    n_any = len(raw.dropna(how="all"))
    returns = raw.dropna()
    if len(returns) < 250:
        raise DataError(f"Only {len(returns)} days have prices for every stock. "
                        "At least 250 (about one year) are needed.")
    if len(returns) < 0.9 * n_any:
        notes.append(f"{n_any - len(returns)} days were dropped because at least one stock "
                     "had no price that day.")

    # Market benchmark for betas
    bench, bench_name = None, BENCHMARK_COL
    if BENCHMARK_COL in prices.columns:
        candidate = prices[BENCHMARK_COL].pct_change(fill_method=None).rename("BENCH")
        if len(pd.concat([returns, candidate], axis=1, sort=True).dropna()) >= 250:
            bench = candidate
    if bench is None:
        bench = returns.mean(axis=1).rename("BENCH")
        bench_name = "equal-weight average of the portfolio's stocks (proxy)"
        notes.append("No usable '^NSEI' benchmark column was found, so betas are measured against "
                     "the equal-weight average of your stocks instead of a market index.")

    data = pd.concat([returns, bench], axis=1, sort=True).dropna()
    betas = data[stocks].apply(lambda s: s.cov(data["BENCH"]) / data["BENCH"].var())

    cov = returns.cov().values
    return MarketData(
        stocks=stocks, sectors=sectors, file_weights=weights, returns=returns,
        benchmark=bench, benchmark_name=bench_name, cov=cov,
        corr=returns.corr().values, vols=np.sqrt(np.diag(cov)), betas=betas, notes=notes,
    )


# ---------------------------------------------------------------- basic risk
def var_cvar(x, confidence=0.95):
    """Historical VaR and CVaR as positive loss fractions.
    VaR  = the loss exceeded on only (1 - confidence) of periods.
    CVaR = the average loss on those worst periods."""
    x = np.asarray(x, dtype=float)
    cutoff = np.percentile(x, (1 - confidence) * 100)
    return -cutoff, -x[x <= cutoff].mean()


def portfolio_returns(m, w):
    return m.returns @ w[m.stocks].values


def growth_and_drawdown(pr):
    growth = (1 + pr).cumprod()
    return growth, growth / growth.cummax() - 1


def annual_vol(m, w, cov=None):
    cov = m.cov if cov is None else cov
    wv = w[m.stocks].values
    return float(np.sqrt(wv @ cov @ wv) * np.sqrt(TRADING_DAYS))


def risk_shares(m, w):
    """Each stock's share of total portfolio variance (adds up to 100%)."""
    wv = w[m.stocks].values
    return pd.Series(wv * (m.cov @ wv) / (wv @ m.cov @ wv), index=m.stocks)


def portfolio_beta(m, w):
    return float((w[m.stocks] * m.betas).sum())


# ---------------------------------------------------------------- stress tests
def stressed_cov(m, lam):
    """Pull every pair of stocks toward correlation = 1.
    lam = 0 is reality, lam = 1 is perfect correlation. Volatilities are unchanged.
    Returns (covariance matrix, average pairwise correlation)."""
    n = len(m.stocks)
    sc = (1 - lam) * m.corr + lam * np.ones((n, n))
    return np.outer(m.vols, m.vols) * sc, (sc.sum() - n) / (n * (n - 1))


def run_scenario(m, w, market_move=0.0, sector_moves=None):
    """Stock move = beta * market move + direct sector shock (floored at -100%).
    Returns (portfolio change, each stock's contribution, each stock's move)."""
    sector_part = m.sectors.map(sector_moves or {}).fillna(0.0)
    stock_moves = (m.betas * market_move + sector_part).clip(lower=-1.0)
    contrib = w[m.stocks] * stock_moves
    return float(contrib.sum()), contrib, stock_moves


def regime_table(m, w, confidence=0.95, min_days=100):
    """For each calendar year: average correlation between the stocks, how much
    diversification reduced risk, and the 1-day VaR. Shows when diversification stops helping.
    diversification benefit = 1 - portfolio volatility / weighted-average stock volatility."""
    pr = portfolio_returns(m, w)
    n = len(m.stocks)
    wv = w[m.stocks].values
    rows = {}
    for year, r in m.returns.groupby(m.returns.index.year):
        if len(r) < min_days:
            continue
        c = r.corr().values
        avg_vol = float((wv * r.std().values).sum())
        pr_y = pr.loc[r.index]
        rows[int(year)] = {
            "avg_corr": (c.sum() - n) / (n * (n - 1)),
            "div_benefit": 1 - pr_y.std() / avg_vol,
            "var": var_cvar(pr_y.values, confidence)[0],
        }
    return pd.DataFrame(rows).T


def replay_windows(m):
    """Real crash windows that our price history actually covers."""
    out = {}
    for name, (a, b) in REPLAYS.items():
        if len(m.returns.loc[a:b]) >= 0.7 * np.busday_count(a, b):
            out[name] = (a, b)
    return out


def replay(m, w, start, end):
    """(portfolio actual move, benchmark move, move our beta model would have predicted)."""
    pr = portfolio_returns(m, w)
    actual = float((1 + pr.loc[start:end]).prod() - 1)
    bench_move = float((1 + m.benchmark.loc[start:end].dropna()).prod() - 1)
    return actual, bench_move, portfolio_beta(m, w) * bench_move


# ---------------------------------------------------------------- Monte Carlo
def simulate(cov, w, horizon=21, n_sims=20000, fat_tails=True, dof=5, seed=42):
    """Simulate n_sims possible `horizon`-day portfolio returns (zero average drift).
    fat_tails=True uses a Student-t with the same volatility but fatter tails.
    The same seed gives the same random draws, so scenarios compare fairly."""
    w = np.asarray(w, dtype=float)
    n = len(w)
    rng = np.random.default_rng(seed)
    jitter = 1e-8 * np.eye(n) * np.mean(np.diag(cov))        # numerical safety
    chol = np.linalg.cholesky(cov + jitter)
    scale = np.sqrt((dof - 2) / dof)
    out = np.empty(n_sims)
    chunk = 2000                                             # keeps memory small
    for start in range(0, n_sims, chunk):
        k = min(chunk, n_sims - start)
        z = rng.standard_normal((k * horizon, n)) @ chol.T
        if fat_tails:
            z = z * scale / np.sqrt(rng.chisquare(dof, k * horizon) / dof)[:, None]
        out[start:start + k] = np.prod(1 + (z @ w).reshape(k, horizon), axis=1) - 1
    return out


# ---------------------------------------------------------------- allocations
def make_allocations(m):
    """The portfolio from the file, plus simple, explainable alternatives."""
    n = len(m.stocks)
    vol_s = pd.Series(m.vols, index=m.stocks)
    counts = m.sectors.value_counts()

    def normalize(x):
        return x / x.sum()

    return {
        "As in portfolio file": m.file_weights,
        "Equal weight": pd.Series(1 / n, index=m.stocks),
        "Inverse volatility": normalize(1 / vol_s),
        "Low-beta tilt": normalize(1 / m.betas.clip(lower=0.05)),
        "Sector-equal": 1 / (len(counts) * m.sectors.map(counts)),
    }


def evaluate_allocation(m, w, market_move=-0.20, sector_moves=None, confidence=0.95):
    """One row of metrics for an allocation, under the same shocks as all others."""
    wv = w[m.stocks].values
    pr = portfolio_returns(m, w)
    _, dd = growth_and_drawdown(pr)
    var, cvar = var_cvar(pr.values, confidence)
    vol = annual_vol(m, w)
    avg_return = float(pr.mean() * TRADING_DAYS)
    res = {
        "max_w": float(wv.max()),
        "eff_n": float(1 / (wv ** 2).sum()),
        "beta": portfolio_beta(m, w),
        "avg_return": avg_return,
        "vol": vol,
        "ret_per_risk": avg_return / vol,
        "vol_stress": annual_vol(m, w, stressed_cov(m, 0.5)[0]),
        "var": var,
        "cvar": cvar,
        "mdd": float(dd.min()),
        "mkt20": run_scenario(m, w, -0.20)[0],
        "scenario": run_scenario(m, w, market_move, sector_moves)[0],
    }
    for name, (a, b) in replay_windows(m).items():
        res[name] = replay(m, w, a, b)[0]
    return res