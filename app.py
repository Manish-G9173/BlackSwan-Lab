"""
BlackSwan Lab - interactive dashboard.

Run:   pip install streamlit
       streamlit run app.py
Needs riskengine.py next to this file, and the ./data folder from download_data.py
(or upload your own files in the sidebar).
"""
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import riskengine as eng

st.set_page_config(page_title="BlackSwan Lab", page_icon="\U0001F9A2", layout="wide")

# ---- Colour palette and a dark, colourful chart theme ----
CYAN, PINK, GREEN = "#4CC9F0", "#F72585", "#06D6A0"
ORANGE, YELLOW, PURPLE, MUTED = "#FF9E00", "#FFD166", "#9B5DE5", "#8E88B8"
from cycler import cycler  # noqa: E402

_theme = {
    "figure.facecolor": "#14112B", "axes.facecolor": "#14112B", "savefig.facecolor": "#14112B",
    "axes.edgecolor": "#3D3766", "axes.labelcolor": "#E9E5FF", "text.color": "#E9E5FF",
    "axes.titlecolor": "#FFFFFF", "axes.titleweight": "bold", "axes.titlesize": 11,
    "xtick.color": "#BDB7E6", "ytick.color": "#BDB7E6",
    "axes.grid": True, "grid.color": "#2B2650", "grid.alpha": 0.6, "axes.axisbelow": True,
    "legend.facecolor": "#1D1940", "legend.edgecolor": "#3D3766", "legend.labelcolor": "#E9E5FF",
    "axes.prop_cycle": cycler(color=[CYAN, PINK, GREEN, ORANGE, PURPLE]),
}
for _key, _val in _theme.items():          # skip any setting an older matplotlib lacks
    try:
        plt.rcParams[_key] = _val
    except KeyError:
        pass

st.markdown("""
<style>
.hero-title {font-size: 3.2rem; font-weight: 800; line-height: 1.1; margin: 0;
  background: linear-gradient(90deg, #4CC9F0, #9B5DE5 55%, #F72585);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.hero-sub {opacity: .8; margin: 2px 0 10px 0;}
div[data-testid="stMetric"] {
  background: linear-gradient(135deg, rgba(155,93,229,.24), rgba(247,37,133,.12));
  border: 1px solid rgba(155,93,229,.35); border-left: 5px solid #F72585;
  border-radius: 14px; padding: 14px 18px;}
.take-row {display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 6px 0 18px 0;}
.take-card {border-radius: 16px; padding: 16px 18px; border: 1px solid rgba(155,93,229,.35);
  background: linear-gradient(135deg, rgba(76,201,240,.16), rgba(155,93,229,.18));}
.take-card:nth-child(2) {background: linear-gradient(135deg, rgba(155,93,229,.20), rgba(247,37,133,.18));}
.take-card:nth-child(3) {background: linear-gradient(135deg, rgba(247,37,133,.18), rgba(255,158,0,.16));}
.take-num {font-size: 1.8rem; font-weight: 800;}
.take-txt {opacity: .88; font-size: .92rem; margin-top: 2px;}
button[data-baseweb="tab"] {font-weight: 600;}
@media (max-width: 800px) {.take-row {grid-template-columns: 1fr;}}
</style>
""", unsafe_allow_html=True)


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


@st.cache_data(show_spinner="Crunching the numbers...")
def get_market(prices_b, sectors_b, portfolio_b):
    prices = pd.read_csv(io.BytesIO(prices_b), index_col=0, parse_dates=True)
    sectors = pd.read_csv(io.BytesIO(sectors_b))
    portfolio = pd.read_csv(io.BytesIO(portfolio_b))
    return eng.build_market(prices, sectors, portfolio)


@st.cache_data(show_spinner="Simulating...")
def run_mc(cov, w, horizon, n_sims, fat_tails, dof):
    return eng.simulate(cov, w, horizon, n_sims, fat_tails, dof)


# =============================================================== sidebar
st.markdown("<div class='hero-title'>\U0001F9A2 BlackSwan Lab</div>"
            "<div class='hero-sub'>Portfolio stress testing &amp; risk intelligence"
            " &nbsp;|&nbsp; Quantactive: Code Conflux</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Data")
    source = st.radio("Portfolio data", ["Built-in (data folder)", "Upload my own files"])
    if source == "Upload my own files":
        up_prices = st.file_uploader("Prices CSV", type="csv",
                                     help="First column = dates, then one column per ticker "
                                          "(optional '^NSEI' column = market index).")
        up_sectors = st.file_uploader("Sectors CSV", type="csv",
                                      help="Columns: ticker, sector")
        up_port = st.file_uploader("Portfolio CSV", type="csv",
                                   help="Columns: ticker, weight (weights are rescaled to 100%)")
        if not (up_prices and up_sectors and up_port):
            st.info("Upload all three files to continue.")
            st.stop()
        files = (up_prices.getvalue(), up_sectors.getvalue(), up_port.getvalue())
    else:
        try:
            files = (read_bytes("data/prices.csv"), read_bytes("data/sectors.csv"),
                     read_bytes("data/portfolio.csv"))
        except FileNotFoundError:
            st.error("The data folder was not found. Run download_data.py first, "
                     "or upload your own files.")
            st.stop()

try:
    m = get_market(*files)
except eng.DataError as e:
    st.error(str(e))
    st.stop()
except Exception as e:                                   # unreadable file etc.
    st.error(f"Could not read the files: {e}")
    st.stop()

allocations = eng.make_allocations(m)
with st.sidebar:
    st.header("Settings")
    value = st.number_input("Portfolio value (Rs)", min_value=10_000, value=1_000_000,
                            step=100_000)
    conf = st.slider("Confidence level for VaR / CVaR", 90, 99, 95, format="%d%%") / 100
    choice = st.selectbox("Allocation to analyse", list(allocations), index=0)
    for note in m.notes:
        st.warning(note)

w = allocations[choice]
st.caption(f"{len(m.stocks)} stocks  |  {m.returns.index[0].date()} to "
           f"{m.returns.index[-1].date()}  |  {len(m.returns):,} trading days  |  "
           f"market benchmark: {m.benchmark_name}  |  allocation: **{choice}**")

# Numbers used by several tabs
pr = eng.portfolio_returns(m, w)
growth, dd = eng.growth_and_drawdown(pr)
var, cvar = eng.var_cvar(pr.values, conf)
port_vol = eng.annual_vol(m, w)
beta_p = eng.portfolio_beta(m, w)
rs = eng.risk_shares(m, w)
wv = w[m.stocks].values

tab_over, tab_var, tab_stress, tab_mc, tab_reb, tab_lim = st.tabs(
    ["\U0001F4CA Overview", "\u26A0\uFE0F VaR & CVaR", "\U0001F32A\uFE0F Stress tests",
     "\U0001F517 Correlation & Monte Carlo", "\u2696\uFE0F Rebalancing",
     "\U0001F4D8 Method & limitations"])

# =============================================================== overview
with tab_over:
    c = st.columns(5)
    c[0].metric("Annual volatility", f"{port_vol:.1%}")
    c[1].metric("Max drawdown", f"{dd.min():.1%}", help=f"Lowest point: {dd.idxmin().date()}")
    c[2].metric("Portfolio beta", f"{beta_p:.2f}",
                help="1.0 = moves with the market; below 1 = falls less than the market.")
    c[3].metric(f"1-day VaR ({conf:.0%})", f"{var:.2%}")
    c[4].metric(f"1-day CVaR ({conf:.0%})", f"{cvar:.2%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Money vs risk by sector")
        sec_df = pd.DataFrame({
            "Share of money (%)": w[m.stocks].groupby(m.sectors).sum() * 100,
            "Share of risk (%)": rs.groupby(m.sectors).sum() * 100,
        }).sort_values("Share of risk (%)", ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        xs = np.arange(len(sec_df))
        ax.bar(xs - 0.2, sec_df["Share of money (%)"], 0.4, label="Share of money", color=CYAN)
        ax.bar(xs + 0.2, sec_df["Share of risk (%)"], 0.4, label="Share of risk", color=PINK)
        ax.set_xticks(xs)
        ax.set_xticklabels(sec_df.index, rotation=45, ha="right")
        ax.set_ylabel("% of portfolio")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("A sector whose risk bar is taller than its money bar is adding more risk "
                   "than its size suggests.")
    with right:
        st.subheader("Stocks ranked by risk contribution")
        top = pd.DataFrame({
            "Sector": m.sectors,
            "Weight (%)": w[m.stocks] * 100,
            "Risk share (%)": rs * 100,
            "Beta": m.betas,
        }).sort_values("Risk share (%)", ascending=False)
        st.dataframe(top, column_config={
            "Weight (%)": st.column_config.NumberColumn(format="%.2f"),
            "Risk share (%)": st.column_config.NumberColumn(format="%.2f"),
            "Beta": st.column_config.NumberColumn(format="%.2f"),
        })

    st.subheader("Growth and drawdown")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].plot(growth.index, growth.values, color=CYAN)
    ax[0].set_yscale("log")
    ax[0].set_title("Growth of 1 rupee (log scale: equal heights = equal % changes)")
    ax[1].fill_between(dd.index, dd.values * 100, 0, color=PINK)
    ax[1].set_title("Drawdown: % below the previous peak")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# =============================================================== VaR & CVaR
with tab_var:
    rows = []
    for cl in (0.95, 0.99):
        v, cv = eng.var_cvar(pr.values, cl)
        rows.append({
            "Confidence": f"{cl:.0%}",
            "VaR": f"{v:.2%}", "VaR (Rs)": f"{v * value:,.0f}",
            "CVaR": f"{cv:.2%}", "CVaR (Rs)": f"{cv * value:,.0f}",
            "Days worse than VaR": int((pr.values <= -v).sum()),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Confidence"))

    n_tail = int((pr.values <= -var).sum())
    st.info(
        f"**VaR ({conf:.0%}) = {var:.2%} (Rs {var * value:,.0f}).** On {conf:.0%} of the "
        f"{len(pr):,} days in our history, the portfolio lost less than this.\n\n"
        f"**CVaR = {cvar:.2%} (Rs {cvar * value:,.0f}).** On the worst {1 - conf:.0%} of days "
        f"(the {n_tail} days beyond VaR) the *average* loss was {cvar / var:.1f} times the VaR.\n\n"
        "VaR marks the doorway to the bad days. CVaR tells you how bad it is once you are inside."
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(pr * 100, bins=100, color=CYAN)
    ax.axvline(-var * 100, color=ORANGE, label=f"VaR {conf:.0%}: -{var:.2%}")
    ax.axvline(-cvar * 100, color=YELLOW, linestyle="--", label=f"CVaR {conf:.0%}: -{cvar:.2%}")
    ax.set_xlabel("Daily portfolio return (%)")
    ax.set_ylabel("Number of days")
    ax.set_title("Every daily return in our history")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("The answer depends on the period you look at")
    by_year = pr.groupby(pr.index.year).apply(
        lambda x: eng.var_cvar(x.values, conf)[0] if len(x) >= 100 else np.nan).dropna() * 100
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(by_year.index.astype(int), by_year.values, color=CYAN)
    ax.set_ylabel(f"1-day VaR {conf:.0%} (%)")
    ax.set_xticks(by_year.index.astype(int))
    ax.set_xticklabels(by_year.index.astype(int), rotation=90)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"A risk estimate built from a calm year can be several times smaller than one "
               f"built from a crisis year. At 99% confidence only about {int(len(pr) * 0.01)} "
               "days sit in the tail, so the 99% numbers are less reliable than the 95% ones.")

# =============================================================== stress tests
with tab_stress:
    st.subheader("Build a scenario")
    market_move = st.slider("Market-wide move", -60, 20, -20, format="%d%%") / 100
    st.markdown("**Extra shock to individual sectors** (added on top of the market move)")
    sector_moves = {}
    cols = st.columns(4)
    for i, s in enumerate(sorted(m.sectors.unique())):
        sector_moves[s] = cols[i % 4].slider(s, -50, 20, 0, format="%d%%", key=f"sec_{s}") / 100
    active_moves = {k: v for k, v in sector_moves.items() if v != 0}

    total, contrib, stock_moves = eng.run_scenario(m, w, market_move, active_moves)
    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio gain / loss", f"{total:.1%}")
    c2.metric("In rupees", f"Rs {total * value:,.0f}")
    c3.metric("Worst-hit stock", stock_moves.idxmin().replace(".NS", ""),
              f"{stock_moves.min():.0%}", delta_color="off")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    cs = (contrib * 100).sort_values()
    ax[0].barh([t.replace(".NS", "") for t in cs.index], cs.values,
               color=[PINK if v < 0 else GREEN for v in cs.values])
    ax[0].set_title("Contribution of each stock to the portfolio change (% points)")
    sc = (contrib.groupby(m.sectors).sum() * 100).sort_values()
    ax[1].barh(sc.index, sc.values, color=[PINK if v < 0 else GREEN for v in sc.values])
    ax[1].set_title("Contribution of each sector (% points)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Each stock moves by (its beta x the market move) plus any sector shock, "
               "floored at -100%. Sector shocks have no spillover to other sectors.")

    st.subheader("Real crashes vs our beta model")
    windows = eng.replay_windows(m)
    if not windows:
        st.info("The price history does not cover the 2008 or 2020 crashes, so no replays "
                "are shown.")
    else:
        rep = []
        for name, (a, b) in windows.items():
            actual, bm, pred = eng.replay(m, w, a, b)
            rep.append({
                "Crash": name, "Benchmark moved": f"{bm:.1%}",
                "Beta model predicted": f"{pred:.1%}",
                "Portfolio actually moved": f"{actual:.1%}",
                "Actual minus predicted (pts)": f"{(actual - pred) * 100:+.1f}",
            })
        st.dataframe(pd.DataFrame(rep).set_index("Crash"))
        st.caption("Positive = the portfolio did better than the model predicted. Betas were "
                   "estimated on data that includes these crashes, so the fit looks better "
                   "than a true forecast would.")

        st.subheader("Your scenario vs real crashes")
        labels = ["Your scenario"] + [n.split(" (")[0] for n in windows]
        vals = [total * 100] + [eng.replay(m, w, a, b)[0] * 100 for a, b in windows.values()]
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.barh(labels[::-1], vals[::-1], color=[PINK] * len(windows) + [ORANGE])
        ax.set_xlabel("Portfolio gain / loss (%)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# =============================================================== correlation & Monte Carlo
with tab_mc:
    st.subheader("Correlation stress")
    lam = st.slider("0% = real history, 100% = all stocks move perfectly together",
                    0, 100, 50, step=5, format="%d%%") / 100
    cov_s, corr_s = eng.stressed_cov(m, lam)
    _, corr_0 = eng.stressed_cov(m, 0.0)
    vol_s = eng.annual_vol(m, w, cov_s)
    avg_vol = float((w[m.stocks] * m.vols).sum() * np.sqrt(eng.TRADING_DAYS))
    c1, c2, c3 = st.columns(3)
    c1.metric("Average correlation", f"{corr_s:.2f}", f"{corr_s - corr_0:+.2f} vs real",
              delta_color="inverse")
    c2.metric("Annual volatility", f"{vol_s:.1%}", f"{(vol_s - port_vol) * 100:+.1f} pts vs real",
              delta_color="inverse")
    c3.metric("Diversification benefit", f"{1 - vol_s / avg_vol:.0%}",
              f"{((1 - vol_s / avg_vol) - (1 - port_vol / avg_vol)) * 100:+.0f} pts vs real")
    st.caption("Diversification benefit = how much lower portfolio volatility is than the "
               "weighted average of the stocks' own volatilities. At 100% stress it is zero.")

    st.subheader("When has diversification actually failed?")
    reg = eng.regime_table(m, w, conf)
    if reg.empty:
        st.info("Not enough history per calendar year (100+ days needed) for this view.")
    else:
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
        ax[0].bar(reg.index, reg["avg_corr"], color=CYAN)
        ax[0].set_title("Average correlation between stocks")
        ax[1].bar(reg.index, reg["div_benefit"] * 100, color=GREEN)
        ax[1].set_title("Diversification benefit (%)")
        ax[2].bar(reg.index, reg["var"] * 100, color=PINK)
        ax[2].set_title(f"1-day VaR {conf:.0%} (%)")
        for a_ in ax:
            a_.set_xticks(reg.index)
            a_.set_xticklabels(reg.index.astype(int), rotation=90)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        hi, lo = reg["avg_corr"].idxmax(), reg["avg_corr"].idxmin()
        st.caption(
            f"In {hi} the average correlation reached {reg.loc[hi, 'avg_corr']:.2f} and "
            f"diversification removed only {reg.loc[hi, 'div_benefit']:.0%} of the risk. "
            f"In {lo} the correlation was {reg.loc[lo, 'avg_corr']:.2f} and diversification "
            f"removed {reg.loc[lo, 'div_benefit']:.0%}. Diversification is weakest exactly "
            "when markets are under stress.")

    st.subheader("Monte Carlo simulation")
    a, b, c_, d = st.columns(4)
    horizon = a.slider("Horizon (trading days)", 5, 42, 21)
    n_sims = b.select_slider("Simulations", options=[2000, 5000, 10000, 20000], value=10000)
    fat = c_.checkbox("Fat tails (Student-t)", value=True)
    dof = d.slider("Tail fatness (lower = fatter)", 3, 15, 5, disabled=not fat)

    base = run_mc(m.cov, wv, horizon, n_sims, fat, dof)
    stressed = run_mc(cov_s, wv, horizon, n_sims, fat, dof)
    hist = (np.exp(np.log1p(pr).rolling(horizon).sum()) - 1).dropna().values

    rows = []
    for label, xs in [("Historical (overlapping periods)", hist),
                      ("Monte Carlo, real correlations", base),
                      (f"Monte Carlo, correlations stressed {lam:.0%}", stressed)]:
        v95, c95 = eng.var_cvar(xs, 0.95)
        v99, c99 = eng.var_cvar(xs, 0.99)
        rows.append({"Method": label, "VaR 95%": f"{v95:.2%}", "CVaR 95%": f"{c95:.2%}",
                     "VaR 99%": f"{v99:.2%}", "CVaR 99%": f"{c99:.2%}",
                     "VaR 95% (Rs)": f"{v95 * value:,.0f}"})
    st.dataframe(pd.DataFrame(rows).set_index("Method"))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(base * 100, bins=100, alpha=0.6, label="Real correlations", color=CYAN)
    ax.hist(stressed * 100, bins=100, alpha=0.5, label="Stressed correlations", color=PINK)
    ax.axvline(-eng.var_cvar(base, conf)[0] * 100, color=CYAN, linestyle="--")
    ax.axvline(-eng.var_cvar(stressed, conf)[0] * 100, color=PINK, linestyle="--")
    ax.set_xlabel(f"Simulated {horizon}-day portfolio return (%)   (dashed = VaR {conf:.0%})")
    ax.set_ylabel("Number of simulated futures")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("The simulation assumes zero average return and independent days, so it cannot "
               "reproduce the multi-week streaks of bad days seen in real crashes. "
               "Overlapping historical periods are not independent evidence either.")

# =============================================================== rebalancing
with tab_reb:
    st.subheader("Before vs after rebalancing")
    names = list(allocations)
    c1, c2 = st.columns(2)
    before = c1.selectbox("Before", names, index=0, key="before")
    after = c2.selectbox("After", names, index=names.index("Low-beta tilt"), key="after")

    results = {n: eng.evaluate_allocation(m, wt, market_move, active_moves, conf)
               for n, wt in allocations.items()}
    ROWS = [
        ("Biggest single-stock weight", "max_w", "pct1"),
        ("Effective number of stocks", "eff_n", "num1"),
        ("Portfolio beta", "beta", "num2"),
        ("Average annual return (history)", "avg_return", "pct1"),
        ("Annual volatility", "vol", "pct1"),
        ("Return per unit of risk", "ret_per_risk", "num2"),
        ("Volatility if correlations 50% stressed", "vol_stress", "pct1"),
        (f"1-day VaR ({conf:.0%})", "var", "pct2"),
        (f"1-day CVaR ({conf:.0%})", "cvar", "pct2"),
        ("Max drawdown", "mdd", "pct1"),
        ("Scenario: market -20%", "mkt20", "pct1"),
        ("Scenario: yours (Stress tests tab)", "scenario", "pct1"),
    ] + [(f"Replay: {name}", name, "pct1") for name in windows]

    def fmt(x, kind):
        return {"pct1": f"{x:.1%}", "pct2": f"{x:.2%}", "num1": f"{x:.1f}", "num2": f"{x:.2f}"}[kind]

    def change(after_x, before_x, kind):
        diff = after_x - before_x
        if kind.startswith("pct"):
            return f"{diff * 100:+.{2 if kind == 'pct2' else 1}f} pts"
        return f"{diff:+.1f}" if kind == "num1" else f"{diff:+.2f}"

    table = pd.DataFrame([{
        "Measure": label,
        before: fmt(results[before][key], kind),
        after: fmt(results[after][key], kind),
        "Change": change(results[after][key], results[before][key], kind),
    } for label, key, kind in ROWS]).set_index("Measure")
    st.dataframe(table)
    st.caption("For scenario and replay rows, a less negative number is better. "
               "Look at the return rows too: lower risk often means giving up some return. "
               "A small return cost in this history does not guarantee a small cost in future, "
               "because the weights were chosen with hindsight.")

    wt_table = pd.DataFrame({
        "Sector": m.sectors,
        f"{before} (%)": allocations[before][m.stocks] * 100,
        f"{after} (%)": allocations[after][m.stocks] * 100,
    })
    st.dataframe(wt_table, column_config={
        c: st.column_config.NumberColumn(format="%.1f") for c in wt_table.columns if c != "Sector"
    })

    chart_keys = [("Market -20%", "mkt20"), ("Your scenario", "scenario")] + \
                 [(name.split(" (")[0], name) for name in windows]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(chart_keys))
    width = 0.8 / len(allocations)
    for i, name in enumerate(allocations):
        ax.bar(x + (i - (len(allocations) - 1) / 2) * width,
               [results[name][k] * 100 for _, k in chart_keys], width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _ in chart_keys])
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_ylabel("Portfolio gain / loss (%)")
    ax.set_title("Same shocks, different allocations")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.warning("These comparisons are in-sample: every allocation's weights were calculated from "
               "the same history it is tested on, including the crashes. A fair test would choose "
               "weights from earlier data and evaluate them on later data.")

# =============================================================== limitations
with tab_lim:
    st.markdown("""
### How to read the numbers
- **VaR (Value at Risk):** the loss that was exceeded on only a small share of days (5% for 95% confidence). It says nothing about how bad those days were.
- **CVaR (Conditional VaR):** the *average* loss on those worst days. Always at least as large as VaR, and the better measure of tail severity.
- **Beta stress test:** each stock moves by its beta times the market move, plus any direct sector shock.
- **Correlation stress:** pulls every pair of stocks toward moving together (volatilities unchanged).
- **Monte Carlo:** thousands of simulated possible futures drawn from the historical volatilities and correlations.

### Limitations: what could make these estimates wrong
1. **History is not destiny.** Risk estimates depend heavily on the period used. A calm year can show a fraction of the risk of a crisis year (see the VaR-by-year chart).
2. **Tail estimates are thin.** A 99% figure rests on about 1% of days, and overlapping multi-day windows are not independent evidence. On the built-in data, the historical one-month 99% CVaR (about 22%) was far above the simulated one (about 15%), mostly because of two crash episodes.
3. **Bad days cluster.** The simulation draws each day independently, so it understates crashes that build over weeks.
4. **Beta is an average relationship.** It can change in a crisis. Sector shocks do not spill over to other sectors, and no stock can lose more than 100%.
5. **Correlation stress changes only correlations.** Real crises also raise each stock's own volatility, so stressed results are probably optimistic.
6. **Survivorship bias.** The built-in stocks are large companies that exist today; failed companies are missing, which flatters the history.
7. **Simplified portfolio mechanics.** Weights are assumed reset to target every day, with no transaction costs, taxes, liquidity limits or market impact.
8. **Rebalancing comparisons are in-sample.** The weights were chosen with hindsight, so a small return cost in this history (see the return rows) is no guarantee for the future.
9. **Monte Carlo assumes zero average return** over the horizon.
10. **Data quality.** Prices are adjusted closing prices from a free source; days where any stock has no price are dropped.

### Using your own data
- **Prices CSV:** first column = dates, then one column of prices per ticker. An optional column named `^NSEI` is used as the market index; without it, betas use the average of your stocks.
- **Sectors CSV:** columns `ticker`, `sector`.
- **Portfolio CSV:** columns `ticker`, `weight` (rescaled to 100%).
""")