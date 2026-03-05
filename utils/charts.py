"""
Utility: Plotly chart builders for Oil & Gas analysis.
"""

import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# ── Color Palettes ───────────────────────────────────────────────────────────

COLORS = {
    # Crude
    "WTI Crude": "#DC3545",
    "Brent Crude": "#AB003C",
    # Index
    "Nifty 50": "#7B1FA2",
    # Upstream
    "ONGC": "#E65100",
    "Oil India": "#F57C00",
    "Reliance": "#FFB300",
    "Upstream Avg": "#FF6F00",
    # Downstream
    "IOC": "#1565C0",
    "BPCL": "#1E88E5",
    "HPCL": "#42A5F5",
    "MRPL": "#64B5F6",
    "Downstream Avg": "#0D47A1",
    # Gas
    "GAIL": "#2E7D32",
    "Petronet LNG": "#388E3C",
    "IGL": "#43A047",
    "MGL": "#4CAF50",
    "Gujarat Gas": "#66BB6A",
    "Adani Total Gas": "#81C784",
    "GSPL": "#A5D6A7",
    "Gas Avg": "#1B5E20",
    # Others
    "Castrol India": "#6A1B9A",
}

FALLBACK_COLORS = [
    "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF",
    "#FF9F40", "#C9CBCF", "#E7E9ED", "#7CB342", "#F06292",
]


def get_color(name: str, idx: int = 0) -> str:
    return COLORS.get(name, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


# ── Chart Layout Defaults ────────────────────────────────────────────────────

def _base_layout(title: str, yaxis_title: str = "Indexed Price (Base = 100)") -> dict:
    return dict(
        title=dict(text=title, font=dict(size=20, color="#ECEFF1"), x=0.5, xanchor="center"),
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(family="Inter, sans-serif", color="#B0BEC5"),
        xaxis=dict(
            title="Date",
            gridcolor="rgba(255,255,255,0.06)",
            showgrid=True,
            tickformat="%b %d",
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor="rgba(255,255,255,0.06)",
            showgrid=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        hovermode="x unified",
        height=560,
        margin=dict(l=60, r=30, t=100, b=60),
    )


def add_war_annotations(fig, war_start, war_end, y_range=None):
    """Add war-period shading and event markers."""
    # Shaded war zone
    fig.add_vrect(
        x0=war_start, x1=war_end,
        fillcolor="rgba(244,67,54,0.08)", line_width=0,
        annotation_text="War Period",
        annotation_position="top left",
        annotation_font=dict(size=10, color="rgba(244,67,54,0.5)"),
    )
    # Strike line
    fig.add_vline(
        x=war_start, line_dash="dash", line_color="#EF5350", line_width=1.5,
        annotation_text="⚔️ Israel strikes Iran",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#EF5350"),
    )
    # Ceasefire line
    fig.add_vline(
        x=war_end, line_dash="dash", line_color="#66BB6A", line_width=1.5,
        annotation_text="🕊️ Ceasefire",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#66BB6A"),
    )
    # Base-100 reference
    fig.add_hline(
        y=100, line_dash="dot", line_color="rgba(255,255,255,0.2)", line_width=1,
    )


# ── Chart Builders ───────────────────────────────────────────────────────────

def build_sector_vs_crude_chart(
    crude_normed: pd.DataFrame,
    upstream_avg: pd.Series,
    downstream_avg: pd.Series,
    nifty_normed: pd.DataFrame,
    war_start: datetime,
    war_end: datetime,
    gas_avg: pd.Series = None,
) -> go.Figure:
    """Main comparison: Sector averages + Nifty50 vs WTI & Brent."""
    fig = go.Figure()

    # Crude lines (thick, dashed)
    for col in crude_normed.columns:
        fig.add_trace(go.Scatter(
            x=crude_normed.index, y=crude_normed[col],
            name=col, mode="lines+markers",
            line=dict(color=get_color(col), width=3, dash="dash"),
            marker=dict(size=5, symbol="diamond"),
            hovertemplate="%{y:.1f}",
        ))

    # Nifty 50
    if not nifty_normed.empty:
        for col in nifty_normed.columns:
            fig.add_trace(go.Scatter(
                x=nifty_normed.index, y=nifty_normed[col],
                name=col, mode="lines+markers",
                line=dict(color=get_color(col), width=2.5, dash="dot"),
                marker=dict(size=5, symbol="star"),
                hovertemplate="%{y:.1f}",
            ))

    # Sector averages
    traces = [
        (upstream_avg, "Upstream Avg", "circle"),
        (downstream_avg, "Downstream Avg", "square"),
    ]
    if gas_avg is not None:
        traces.append((gas_avg, "Gas Avg", "triangle-up"))

    for series, name, symbol in traces:
        fig.add_trace(go.Scatter(
            x=series.index, y=series,
            name=name, mode="lines+markers",
            line=dict(color=get_color(name), width=2.5),
            marker=dict(size=6, symbol=symbol),
            hovertemplate="%{y:.1f}",
        ))

    add_war_annotations(fig, war_start, war_end)
    fig.update_layout(**_base_layout(
        "📊 Sector Averages vs Crude Oil & Nifty 50<br>"
        "<sub>Indexed to 100 on Jun 2, 2025 — Iran-Israel War Period</sub>"
    ))
    return fig


def build_individual_vs_crude_chart(
    crude_normed: pd.DataFrame,
    stock_normed: pd.DataFrame,
    segment_label: str,
    war_start: datetime,
    war_end: datetime,
) -> go.Figure:
    """Individual stocks from a segment vs WTI & Brent."""
    fig = go.Figure()

    # Crude lines
    for col in crude_normed.columns:
        fig.add_trace(go.Scatter(
            x=crude_normed.index, y=crude_normed[col],
            name=col, mode="lines+markers",
            line=dict(color=get_color(col), width=3, dash="dash"),
            marker=dict(size=5, symbol="diamond"),
            hovertemplate="%{y:.1f}",
        ))

    # Stock lines
    symbols = ["circle", "square", "triangle-up", "triangle-down", "cross", "x", "pentagon"]
    for i, col in enumerate(stock_normed.columns):
        fig.add_trace(go.Scatter(
            x=stock_normed.index, y=stock_normed[col],
            name=col, mode="lines+markers",
            line=dict(color=get_color(col, i), width=2.5),
            marker=dict(size=6, symbol=symbols[i % len(symbols)]),
            hovertemplate="%{y:.1f}",
        ))

    add_war_annotations(fig, war_start, war_end)
    fig.update_layout(**_base_layout(
        f"📈 {segment_label} Stocks vs Crude Oil<br>"
        f"<sub>Indexed to 100 on Jun 2, 2025</sub>"
    ))
    return fig


def build_single_stock_vs_crude_chart(
    crude_normed: pd.DataFrame,
    stock_series: pd.Series,
    stock_name: str,
    war_start: datetime,
    war_end: datetime,
) -> go.Figure:
    """Single stock vs WTI & Brent — used for dropdown selection."""
    fig = go.Figure()

    # Crude
    for col in crude_normed.columns:
        fig.add_trace(go.Scatter(
            x=crude_normed.index, y=crude_normed[col],
            name=col, mode="lines+markers",
            line=dict(color=get_color(col), width=3, dash="dash"),
            marker=dict(size=6, symbol="diamond"),
            hovertemplate="%{y:.1f}",
        ))

    # Stock
    fig.add_trace(go.Scatter(
        x=stock_series.index, y=stock_series,
        name=stock_name, mode="lines+markers",
        line=dict(color=get_color(stock_name), width=3),
        marker=dict(size=7),
        hovertemplate="%{y:.1f}",
        fill="tozeroy",
        fillcolor=f"rgba({_hex_to_rgb(get_color(stock_name))}, 0.08)",
    ))

    add_war_annotations(fig, war_start, war_end)
    fig.update_layout(**_base_layout(
        f"🔍 {stock_name} vs WTI & Brent Crude<br>"
        f"<sub>Indexed to 100 on Jun 2, 2025</sub>"
    ))
    return fig


def build_performance_bar_chart(perf_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of total period returns."""
    fig = go.Figure()

    colors = [get_color(name, i) for i, name in enumerate(perf_df.index)]
    bar_colors = ["#66BB6A" if v >= 0 else "#EF5350" for v in perf_df["Total Return %"]]

    fig.add_trace(go.Bar(
        y=perf_df.index,
        x=perf_df["Total Return %"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.1f}%" for v in perf_df["Total Return %"]],
        textposition="outside",
        textfont=dict(size=12),
    ))

    fig.update_layout(
        **_base_layout("🏆 Total Return: Jun 2 → Jun 30, 2025", yaxis_title=""),
        height=max(400, len(perf_df) * 35 + 120),
        xaxis=dict(title="Return %", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return fig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> str:
    """Convert #RRGGBB to 'R,G,B'."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "255,255,255"
    return ",".join(str(int(hex_color[i:i+2], 16)) for i in (0, 2, 4))
