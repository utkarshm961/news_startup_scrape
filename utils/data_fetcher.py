"""
Utility: Fetch market data from Yahoo Finance API.
Provides functions for crude oil (WTI, Brent), Indian stocks, and indices.
"""

import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# ── Ticker Mappings ──────────────────────────────────────────────────────────

CRUDE_TICKERS = {
    "WTI Crude": "CL=F",
    "Brent Crude": "BZ=F",
}

INDEX_TICKERS = {
    "Nifty 50": "^NSEI",
}

UPSTREAM_TICKERS = {
    "ONGC": "ONGC.NS",
    "Oil India": "OIL.NS",
    "Reliance": "RELIANCE.NS",
}

DOWNSTREAM_TICKERS = {
    "IOC": "IOC.NS",
    "BPCL": "BPCL.NS",
    "HPCL": "HINDPETRO.NS",
    "MRPL": "MRPL.NS",
}

GAS_TICKERS = {
    "GAIL": "GAIL.NS",
    "Petronet LNG": "PETRONET.NS",
    "IGL": "IGL.NS",
    "MGL": "MGL.NS",
    "Gujarat Gas": "GUJGASLTD.NS",
    "Adani Total Gas": "ATGL.NS",
    "GSPL": "GSPL.NS",
}

OTHER_TICKERS = {
    "Castrol India": "CASTROLIND.NS",
}

ALL_OIL_GAS_TICKERS = {**UPSTREAM_TICKERS, **DOWNSTREAM_TICKERS, **GAS_TICKERS, **OTHER_TICKERS}

# ── Segment labels ───────────────────────────────────────────────────────────

SEGMENTS = {
    "Upstream": list(UPSTREAM_TICKERS.keys()),
    "Downstream": list(DOWNSTREAM_TICKERS.keys()),
    "Gas Distribution": list(GAS_TICKERS.keys()),
    "Others": list(OTHER_TICKERS.keys()),
}


# ── Data Fetching ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(ticker_map: dict, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily close prices for a dict of {label: yahoo_ticker}.
    Returns a DataFrame with Date index and one column per label.
    """
    frames = {}
    for label, ticker in ticker_map.items():
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                # Handle both single and multi-level columns
                if isinstance(df.columns, pd.MultiIndex):
                    close = df["Close"].iloc[:, 0]
                else:
                    close = df["Close"]
                frames[label] = close
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    result = pd.DataFrame(frames)
    result.index = pd.to_datetime(result.index)
    result.index.name = "Date"
    return result.dropna(how="all")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_war_data():
    """Fetch all data needed for the Iran-Israel war analysis (Jun 2025)."""
    start = "2025-06-01"
    end = "2025-07-02"

    crude = fetch_prices(CRUDE_TICKERS, start, end)
    index = fetch_prices(INDEX_TICKERS, start, end)
    upstream = fetch_prices(UPSTREAM_TICKERS, start, end)
    downstream = fetch_prices(DOWNSTREAM_TICKERS, start, end)
    gas = fetch_prices(GAS_TICKERS, start, end)
    others = fetch_prices(OTHER_TICKERS, start, end)

    return {
        "crude": crude,
        "index": index,
        "upstream": upstream,
        "downstream": downstream,
        "gas": gas,
        "others": others,
    }


def normalize_to_100(df: pd.DataFrame) -> pd.DataFrame:
    """Rebase all columns so the first row = 100 (percentage indexed)."""
    if df.empty:
        return df
    first = df.iloc[0]
    first = first.replace(0, float("nan"))
    return (df / first) * 100


def compute_segment_avg(df: pd.DataFrame, label: str = "Avg") -> pd.Series:
    """Compute the equal-weighted average across columns."""
    normed = normalize_to_100(df)
    return normed.mean(axis=1).rename(label)


# ── War event dates ──────────────────────────────────────────────────────────

WAR_EVENTS = [
    {"date": datetime(2025, 6, 13), "label": "Israel strikes Iran", "color": "red"},
    {"date": datetime(2025, 6, 24), "label": "Ceasefire announced", "color": "green"},
]

WAR_START = datetime(2025, 6, 13)
WAR_END = datetime(2025, 6, 24)
