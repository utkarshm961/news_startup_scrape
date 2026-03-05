"""
Page: Iran-Israel War Analysis — Oil & Gas ETF Constituents vs Crude Oil
Sidebar page in the Newsssyyy Streamlit app.
"""

import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.data_fetcher import (
    fetch_all_war_data, normalize_to_100, compute_segment_avg,
    WAR_START, WAR_END,
    UPSTREAM_TICKERS, DOWNSTREAM_TICKERS, GAS_TICKERS, OTHER_TICKERS,
    ALL_OIL_GAS_TICKERS, SEGMENTS,
)
from utils.charts import (
    build_sector_vs_crude_chart,
    build_individual_vs_crude_chart,
    build_single_stock_vs_crude_chart,
    build_performance_bar_chart,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="War Analysis | Oil & Gas", page_icon="⛽", layout="wide")

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stMetric { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 1rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); }
    .stMetric label { color: #90CAF9 !important; font-size: 0.85rem !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    div[data-testid="stHorizontalBlock"] > div { padding: 0 0.25rem; }
    .segment-header { background: linear-gradient(90deg, rgba(255,111,0,0.15), transparent);
                      padding: 0.5rem 1rem; border-left: 3px solid #FF6F00;
                      border-radius: 0 8px 8px 0; margin: 1.5rem 0 0.8rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
# ⛽ Oil & Gas vs Crude Oil — Iran-Israel War Analysis
<p style="color:#78909C; font-size:1.05rem; margin-top:-0.5rem;">
    Nifty Oil & Gas ETF constituents compared against WTI & Brent Crude during June 2025.<br>
    All prices indexed to <b>100 on Jun 2</b> for like-for-like comparison across INR stocks and USD crude.
</p>
""", unsafe_allow_html=True)

st.divider()

# ── Fetch data ───────────────────────────────────────────────────────────────
with st.spinner("📡 Fetching market data from Yahoo Finance..."):
    data = fetch_all_war_data()

crude = data["crude"]
index_df = data["index"]
upstream = data["upstream"]
downstream = data["downstream"]
gas = data["gas"]
others = data["others"]

if crude.empty:
    st.error("Could not fetch crude oil data. Please try again later.")
    st.stop()

# ── Normalize everything ────────────────────────────────────────────────────
crude_n = normalize_to_100(crude)
index_n = normalize_to_100(index_df)
upstream_n = normalize_to_100(upstream)
downstream_n = normalize_to_100(downstream)
gas_n = normalize_to_100(gas)

upstream_avg = compute_segment_avg(upstream, "Upstream Avg")
downstream_avg = compute_segment_avg(downstream, "Downstream Avg")
gas_avg = compute_segment_avg(gas, "Gas Avg")

# ── Key Metrics Row ─────────────────────────────────────────────────────────
st.markdown("### 📋 Period at a Glance — Jun 2 to Jun 30, 2025")

def total_return(normed_df_or_series):
    """Compute total return from normalized series."""
    if isinstance(normed_df_or_series, pd.DataFrame):
        if normed_df_or_series.empty:
            return {}
        return {col: round(normed_df_or_series[col].iloc[-1] - 100, 1) for col in normed_df_or_series.columns}
    elif isinstance(normed_df_or_series, pd.Series):
        if normed_df_or_series.empty:
            return 0
        return round(normed_df_or_series.iloc[-1] - 100, 1)
    return 0

crude_ret = total_return(crude_n)
idx_ret = total_return(index_n)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🛢️ WTI Crude", f"{crude_ret.get('WTI Crude', 'N/A')}%",
            delta=f"{crude_ret.get('WTI Crude', 0):+.1f}%")
col2.metric("🛢️ Brent Crude", f"{crude_ret.get('Brent Crude', 'N/A')}%",
            delta=f"{crude_ret.get('Brent Crude', 0):+.1f}%")
col3.metric("⬆️ Upstream Avg", f"{total_return(upstream_avg):+.1f}%",
            delta=f"{total_return(upstream_avg):+.1f}%")
col4.metric("⬇️ Downstream Avg", f"{total_return(downstream_avg):+.1f}%",
            delta=f"{total_return(downstream_avg):+.1f}%")
nifty_ret = list(idx_ret.values())[0] if idx_ret else "N/A"
col5.metric("📊 Nifty 50", f"{nifty_ret}%",
            delta=f"{nifty_ret:+.1f}%" if isinstance(nifty_ret, (int, float)) else "N/A")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Sector Averages vs Crude & Nifty50
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="segment-header">
    <h3 style="margin:0;">📊 Sector Averages vs Crude Oil & Nifty 50</h3>
    <p style="color:#90A4AE; margin:0; font-size:0.9rem;">
        Equal-weighted average of Upstream (ONGC, Oil India, Reliance),
        Downstream (IOC, BPCL, HPCL, MRPL), and Gas companies vs crude benchmarks.
    </p>
</div>
""", unsafe_allow_html=True)

fig_sector = build_sector_vs_crude_chart(
    crude_n, upstream_avg, downstream_avg, index_n,
    WAR_START, WAR_END, gas_avg=gas_avg,
)
st.plotly_chart(fig_sector, use_container_width=True, key="sector_chart")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Upstream Individual Stocks
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="segment-header">
    <h3 style="margin:0;">⬆️ Upstream — Oil Producers</h3>
    <p style="color:#90A4AE; margin:0; font-size:0.9rem;">
        Upstream companies benefit from rising crude — they sell oil at higher prices.
    </p>
</div>
""", unsafe_allow_html=True)

col_up1, col_up2 = st.columns([2, 1])

with col_up1:
    if not upstream_n.empty:
        fig_up = build_individual_vs_crude_chart(crude_n, upstream_n, "Upstream", WAR_START, WAR_END)
        st.plotly_chart(fig_up, use_container_width=True, key="upstream_chart")

with col_up2:
    st.markdown("**Pick a stock for detailed view:**")
    up_choice = st.selectbox(
        "Upstream Company", list(UPSTREAM_TICKERS.keys()),
        label_visibility="collapsed", key="up_select"
    )
    if up_choice and up_choice in upstream_n.columns:
        fig_up_single = build_single_stock_vs_crude_chart(
            crude_n, upstream_n[up_choice], up_choice, WAR_START, WAR_END
        )
        st.plotly_chart(fig_up_single, use_container_width=True, key="up_single")

        # Stats card
        val = upstream_n[up_choice]
        war_chg = round(val.iloc[val.index.get_indexer([WAR_START], method="nearest")[0]] - 100, 1) if len(val) > 0 else 0
        total_chg = round(val.iloc[-1] - 100, 1)
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | **Jun 2 → War Start** | {war_chg:+.1f}% |
        | **Jun 2 → Jun 30** | {total_chg:+.1f}% |
        """)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Downstream Individual Stocks
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="segment-header">
    <h3 style="margin:0;">⬇️ Downstream — Refiners & Marketers</h3>
    <p style="color:#90A4AE; margin:0; font-size:0.9rem;">
        Downstream companies get squeezed when crude spikes — their input cost rises but regulated fuel prices lag.
    </p>
</div>
""", unsafe_allow_html=True)

col_dn1, col_dn2 = st.columns([2, 1])

with col_dn1:
    if not downstream_n.empty:
        fig_dn = build_individual_vs_crude_chart(crude_n, downstream_n, "Downstream", WAR_START, WAR_END)
        st.plotly_chart(fig_dn, use_container_width=True, key="downstream_chart")

with col_dn2:
    st.markdown("**Pick a stock for detailed view:**")
    dn_choice = st.selectbox(
        "Downstream Company", list(DOWNSTREAM_TICKERS.keys()),
        label_visibility="collapsed", key="dn_select"
    )
    if dn_choice and dn_choice in downstream_n.columns:
        fig_dn_single = build_single_stock_vs_crude_chart(
            crude_n, downstream_n[dn_choice], dn_choice, WAR_START, WAR_END
        )
        st.plotly_chart(fig_dn_single, use_container_width=True, key="dn_single")

        val = downstream_n[dn_choice]
        war_chg = round(val.iloc[val.index.get_indexer([WAR_START], method="nearest")[0]] - 100, 1) if len(val) > 0 else 0
        total_chg = round(val.iloc[-1] - 100, 1)
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | **Jun 2 → War Start** | {war_chg:+.1f}% |
        | **Jun 2 → Jun 30** | {total_chg:+.1f}% |
        """)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Performance Leaderboard
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("""
<div class="segment-header">
    <h3 style="margin:0;">🏆 Performance Leaderboard</h3>
    <p style="color:#90A4AE; margin:0; font-size:0.9rem;">
        Total return for the full period (Jun 2 → Jun 30, 2025) across all constituents and crude benchmarks.
    </p>
</div>
""", unsafe_allow_html=True)

# Build performance table
all_normed = pd.concat([crude_n, index_n, upstream_n, downstream_n, gas_n, normalize_to_100(others)], axis=1)
perf_data = {}
for col in all_normed.columns:
    s = all_normed[col].dropna()
    if len(s) >= 2:
        perf_data[col] = round(s.iloc[-1] - 100, 1)

perf_df = pd.DataFrame({"Total Return %": perf_data}).sort_values("Total Return %", ascending=False)

col_bar, col_table = st.columns([3, 2])

with col_bar:
    fig_perf = build_performance_bar_chart(perf_df)
    st.plotly_chart(fig_perf, use_container_width=True, key="perf_bar")

with col_table:
    st.markdown("##### Sorted Returns")
    styled = perf_df.style.format({"Total Return %": "{:+.1f}%"}).background_gradient(
        subset=["Total Return %"], cmap="RdYlGn", vmin=-10, vmax=15
    )
    st.dataframe(styled, use_container_width=True, height=450)

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style="color:#546E7A; font-size:0.8rem; text-align:center;">
    Data sourced from Yahoo Finance API. Indian stocks (NSE) in INR, Crude in USD.
    All values indexed to 100 on Jun 2, 2025 for relative comparison.<br>
    Nifty Oil & Gas ETF constituents as per ICICI Prudential Nifty Oil & Gas ETF holdings.
</p>
""", unsafe_allow_html=True)
