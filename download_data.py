"""
Step 1 of BlackSwan Lab: download historical prices once and save them as CSV files.

Run:   pip install yfinance pandas
       python download_data.py

Creates a ./data folder with:
  prices.csv     daily adjusted closing prices (18 stocks + Nifty 50 index)
  sectors.csv    which sector each stock belongs to (our own simple grouping)
  portfolio.csv  a starting portfolio (equal weights) - the format judges' files should follow
"""
import os
import pandas as pd
import yfinance as yf

START = "2007-01-01"      # early enough to include the 2008 crash
BENCHMARK = "^NSEI"       # Nifty 50 index

# ticker -> sector
SECTORS = {
    "RELIANCE.NS": "Energy",
    "ONGC.NS": "Energy",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "HDFCBANK.NS": "Banks",
    "ICICIBANK.NS": "Banks",
    "SBIN.NS": "Banks",
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG",
    "SUNPHARMA.NS": "Pharma",
    "DRREDDY.NS": "Pharma",
    "TATASTEEL.NS": "Metals",
    "HINDALCO.NS": "Metals",
    "MARUTI.NS": "Auto",
    "M&M.NS": "Auto",
    "LT.NS": "Industrials",
    "NTPC.NS": "Utilities",
    "BHARTIARTL.NS": "Telecom",
}

os.makedirs("data", exist_ok=True)

tickers = list(SECTORS) + [BENCHMARK]
raw = yf.download(tickers, start=START, auto_adjust=True, progress=False, threads=False)
prices = raw["Close"].dropna(how="all").copy()

# Retry any ticker that came back empty (e.g. a temporary 'database is locked' error)
for t in tickers:
    if t not in prices.columns or prices[t].isna().all():
        print(f"Retrying {t} ...")
        one = yf.download(t, start=START, auto_adjust=True, progress=False, threads=False)
        if len(one) > 0:
            close = one["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            prices[t] = close

prices.to_csv("data/prices.csv")

pd.DataFrame({"ticker": list(SECTORS), "sector": list(SECTORS.values())}).to_csv(
    "data/sectors.csv", index=False
)

weight = 1 / len(SECTORS)
pd.DataFrame({"ticker": list(SECTORS), "weight": [weight] * len(SECTORS)}).to_csv(
    "data/portfolio.csv", index=False
)

# Quick health check: when does each series start, and how many gaps does it have?
print(f"Rows: {len(prices)}   From {prices.index.min().date()} to {prices.index.max().date()}\n")
def first_date(s):
    idx = s.first_valid_index()
    return idx.date() if idx is not None else "NO DATA"

report = pd.DataFrame({
    "first_date": prices.apply(first_date),
    "missing_values": prices.isna().sum(),
})
print(report.to_string())