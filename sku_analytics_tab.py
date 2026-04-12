"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         BAZAAR PRIME  —  SKU ANALYTICS ADVANCED MODULE                      ║
║  5 deep-dive sections:                                                       ║
║   1. SKU per Bill Analysis                                                   ║
║   2. Basket Analysis  (co-purchase affinity)                                 ║
║   3. SKU-wise Profitability  (NMV, margin proxy, contribution)               ║
║   4. Slow / Fast Moving SKU  (velocity segmentation)                         ║
║   5. SKU Cannibalization Effect                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

INTEGRATION (3 lines in streamlit_app.py)
─────────────────────────────────────────
1. from sku_analytics_tab import render_sku_analytics_advanced_tab

2. Add tab in st.tabs():
   ..., tab_sku_adv = st.tabs([..., "📦 SKU Analytics"])

3. with tab_sku_adv:
       render_sku_analytics_advanced_tab(start_date, end_date, town_code)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from datetime import datetime, timedelta
from html import escape
from itertools import combinations
from collections import defaultdict

# ── pull shared utilities from host app ─────────────────────────────────────
_HAS_HOST = False
get_engine = None
read_sql_cached = None
render_unified_kpi_card = None
resolve_chart_color = None

CHART_COLORS = {
    "primary": "#5B5F97", "secondary": "#B8B8D1",
    "accent": "#FFC145", "danger": "#FF6B6C",
    "success": "#16A34A", "warning": "#D97706",
    "info": "#2563EB", "neutral": "#94A3B8",
    "target": "#6B7280", "text_dark": "#111827",
    "text_muted": "#64748B", "border": "#D9E3EF",
    "surface": "#FFFFFF", "transparent": "rgba(0,0,0,0)",
    "grid": "#E2E8F0",
}

C = CHART_COLORS


def _resolve_host_bindings():
    """Resolve host app symbols lazily to avoid circular import fallback."""
    global _HAS_HOST, CHART_COLORS, C
    global get_engine, read_sql_cached, render_unified_kpi_card, resolve_chart_color

    for module_name in ("__main__", "streamlit_app"):
        mod = sys.modules.get(module_name)
        if mod is None:
            continue

        engine_fn = getattr(mod, "get_engine", None)
        if not callable(engine_fn):
            continue

        get_engine = engine_fn
        read_sql_cached = getattr(mod, "read_sql_cached", None)
        render_unified_kpi_card = getattr(mod, "render_unified_kpi_card", None)
        resolve_chart_color = getattr(mod, "resolve_chart_color", None)

        host_colors = getattr(mod, "CHART_COLORS", None)
        if isinstance(host_colors, dict) and host_colors:
            CHART_COLORS = host_colors
            C = CHART_COLORS

        _HAS_HOST = True
        return True

    _HAS_HOST = False
    return False


# ── Inline CSS injected once ─────────────────────────────────────────────────
_CSS = """
<style>
/* ── section header pill ── */
.sku-section-pill {
    display:inline-flex; align-items:center; gap:8px;
    background:linear-gradient(135deg,#5B5F97 0%,#B8B8D1 100%);
    color:#fff; padding:6px 18px 6px 14px; border-radius:999px;
    font-size:13px; font-weight:700; letter-spacing:.5px;
    margin-bottom:18px; box-shadow:0 2px 8px rgba(91,95,151,.25);
}
/* ── metric card override (matching bazaarprime style) ── */
.sku-kpi-wrap .stMarkdown { margin-bottom:0 !important; }
/* ── table card ── */
.sku-table-card {
    border:1px solid #D9E3EF; border-radius:12px;
    background:#fff; overflow:hidden;
    box-shadow:0 2px 8px rgba(15,23,42,.06);
}
.sku-table-card table { width:100%; border-collapse:separate; border-spacing:0; }
.sku-table-card th {
    position:sticky; top:0; z-index:2;
    background:#F8FAFC; border-bottom:2px solid #E2E8F0;
    padding:9px 12px; text-align:left;
    font-size:11px; letter-spacing:.8px; text-transform:uppercase;
    color:#5B5F97; font-weight:700; white-space:nowrap;
}
.sku-table-card th.num { text-align:right; }
.sku-table-card td {
    padding:8px 12px; font-size:12px;
    color:#0F172A; border-bottom:1px solid #EEF2F7;
    white-space:nowrap;
}
.sku-table-card td.num { text-align:right; color:#334155; }
.sku-table-card tr:hover td { background:#F8FAFC; }
/* ── velocity badge ── */
.vel-badge {
    display:inline-block; padding:2px 9px; border-radius:999px;
    font-size:10px; font-weight:700; letter-spacing:.5px;
}
.vel-star  { background:#FEF9C3; color:#854D0E; }
.vel-fast  { background:#DCFCE7; color:#166534; }
.vel-avg   { background:#DBEAFE; color:#1E40AF; }
.vel-slow  { background:#FEE2E2; color:#991B1B; }
.vel-dead  { background:#F1F5F9; color:#64748B; }
/* ── cannibal badge ── */
.cb-high { background:#FEE2E2; color:#991B1B; }
.cb-med  { background:#FEF9C3; color:#854D0E; }
.cb-low  { background:#DCFCE7; color:#166534; }
/* ── chart container ── */
div[data-testid="stPlotlyChart"] {
    background:#fff; border:1px solid #D9E3EF;
    border-radius:12px;
    box-shadow:0 2px 8px rgba(15,23,42,.06);
    padding:0; overflow-x:hidden !important;
}
</style>
"""


# ════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════

def _inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def _pill(icon, label):
    st.markdown(
        f"<div class='sku-section-pill'>{icon} {label}</div>",
        unsafe_allow_html=True,
    )


def _sql(query):
    """Run SQL via host engine; return DataFrame or raise."""
    if _resolve_host_bindings() and callable(get_engine):
        try:
            return pd.read_sql(query, get_engine())
        except Exception as e:
            raise e
    raise RuntimeError("No DB connection")


def _layout(fig, h=340, t=12, title=""):
    fig.update_layout(
        height=h,
        margin=dict(t=t+24 if title else t, b=10, l=10, r=10),
        paper_bgcolor=C["transparent"],
        plot_bgcolor=C["transparent"],
        font=dict(color=C["text_dark"], size=12),
        title=dict(text=title, font=dict(size=13)) if title else {},
        xaxis=dict(showgrid=False, linecolor=C["border"], linewidth=1),
        yaxis=dict(gridcolor="rgba(148,163,184,.15)", linecolor=C["border"], linewidth=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _card(col, label, value,
          delta="", delta_color=None, gradient=None, tooltip="", sparkline=None):
    """Render a KPI card — uses host render_unified_kpi_card if available."""
    _resolve_host_bindings()
    g = gradient or f"linear-gradient(90deg,{C['primary']},{C['secondary']})"
    dc = delta_color or C["success"]
    with col:
        if callable(render_unified_kpi_card):
            try:
                render_unified_kpi_card(
                    label=label, value=value,
                    line_gradient=g,
                    delta_primary=delta,
                    delta_primary_color=dc,
                    tooltip=tooltip,
                    sparkline_points=sparkline,
                    sparkline_color=dc,
                )
                return
            except Exception:
                pass
        st.metric(label=label, value=value, delta=delta or None)


def _delta(curr, prev, fmt=".1f", pct=True):
    """Return (text, color) comparing curr vs prev."""
    try:
        d = float(curr) - float(prev)
        arrow = "▲" if d >= 0 else "▼"
        col = C["success"] if d >= 0 else C["danger"]
        if pct:
            p = abs(d / prev * 100) if prev else 0
            return f"{arrow} {p:.1f}%", col
        return f"{arrow} {abs(d):{fmt}}", col
    except Exception:
        return "", C["neutral"]


def _no_data(msg="No data available for the selected filters."):
    st.info(msg, icon="📭")


# ════════════════════════════════════════════════════════════════════════
#  DATA FETCHERS
# ════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_sku_core(start_date, end_date, town_code):
    q = f"""
    SELECT
        o.`SKU Code`                                                   AS SKU_Code,
        COALESCE(m.sku, o.`SKU Code`)                         AS SKU_Name,
        COALESCE(m.brand,  'Unknown')                                  AS Brand,
        COALESCE(o.`category name`,'Unknown')                                 AS Category,
        ROUND(SUM(o.`Delivered Amount`),0)                            AS NMV,
        ROUND(SUM(o.`Delivered Amount` + o.`Total Discount`),0)       AS Gross_NMV,
        ROUND(SUM(o.`Total Discount`),0)                              AS Discount,
        SUM(o.`Delivered Units`)                                         AS Qty,
        COUNT(DISTINCT o.`Invoice Number`)                                 AS Bills,
        COUNT(DISTINCT o.`Store Code`)                                 AS Outlets
    FROM ordered_vs_delivered_rows o
    LEFT JOIN sku_master m ON m.sku_code = o.`SKU Code`
    WHERE o.`Distributor Code` = '{town_code}'
      AND o.`Delivery Date` BETWEEN '{start_date}' AND '{end_date}'
      AND o.`Delivered Units` > 0
    GROUP BY o.`SKU Code`, m.sku, m.brand, o.`category name`
    """
    try:
        return _sql(q)
    except Exception as e:
        raise RuntimeError(f"SKU core query failed for Distributor {town_code}: {e}") from e


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_sku_prev(start_date, end_date, town_code):
    """Same period previous cycle for delta computation."""
    rng_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    p_end   = (pd.to_datetime(start_date) - timedelta(days=1)).date()
    p_start = (pd.to_datetime(p_end)      - timedelta(days=rng_days-1)).date()
    q = f"""
    SELECT
        o.`SKU Code`                                                   AS SKU_Code,
        ROUND(SUM(o.`Delivered Amount`),0)                            AS Prev_NMV,
        COUNT(DISTINCT o.`Invoice Number`) AS Prev_Bills,
        SUM(o.`Delivered Units`)                                         AS Prev_Qty
    FROM ordered_vs_delivered_rows o
    WHERE o.`Distributor Code` = '{town_code}'
      AND o.`Delivery Date` BETWEEN '{p_start}' AND '{p_end}'
      AND o.`Delivered Units` > 0
    GROUP BY o.`SKU Code`
    """
    try:
        return _sql(q)
    except Exception as e:
        raise RuntimeError(f"SKU previous-period query failed for Distributor {town_code}: {e}") from e


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_bill_level(start_date, end_date, town_code):
    """Invoice-level SKU list for basket analysis."""
    q = f"""
    SELECT
        o.`Invoice Number`   AS Invoice_No,
        o.`SKU Code`     AS SKU_Code,
        o.`Store Code`   AS Store_Code,
        o.`Order Booker Name`         AS Booker
    FROM ordered_vs_delivered_rows o
    WHERE o.`Distributor Code` = '{town_code}'
      AND o.`Delivery Date` BETWEEN '{start_date}' AND '{end_date}'
      AND o.`Delivered Units` > 0
    LIMIT 200000
    """
    try:
        return _sql(q)
    except Exception as e:
        raise RuntimeError(f"SKU bill-level query failed for Distributor {town_code}: {e}") from e


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_monthly_sku(start_date, end_date, town_code, sku_codes: tuple):
    """Monthly NMV trend for selected SKUs (cannibalization view)."""
    if not sku_codes:
        return pd.DataFrame()
    codes_sql = ", ".join([f"'{c}'" for c in sku_codes[:10]])
    q = f"""
    SELECT
        o.`SKU Code`                                                   AS SKU_Code,
        COALESCE(m.sku, o.`SKU Code`)                         AS SKU_Name,
        DATE_FORMAT(o.`Delivery Date`,'%%Y-%%m-01')                   AS Month,
        ROUND(SUM(o.`Delivered Amount`+o.`Total Discount`),0)         AS NMV,
        SUM(o.`Delivered Units`)                                         AS Qty
    FROM ordered_vs_delivered_rows o
    LEFT JOIN sku_master m ON m.sku_code = o.`SKU Code`
    WHERE o.`Distributor Code` = '{town_code}'
      AND o.`Delivery Date` BETWEEN '{start_date}' AND '{end_date}'
      AND o.`SKU Code` IN ({codes_sql})
    GROUP BY o.`SKU Code`, m.sku, DATE_FORMAT(o.`Delivery Date`,'%%Y-%%m-01')
    ORDER BY Month
    """
    try:
        return _sql(q)
    except Exception as e:
        raise RuntimeError(f"SKU monthly trend query failed for Distributor {town_code}: {e}") from e


# ════════════════════════════════════════════════════════════════════════
#  COMPUTED HELPERS
# ════════════════════════════════════════════════════════════════════════

def _build_master(start_date, end_date, town_code):
    """Return enriched SKU master DataFrame (current + prev merged)."""
    curr = _fetch_sku_core(start_date, end_date, town_code).copy()
    prev = _fetch_sku_prev(start_date, end_date, town_code).copy()

    for col in ["NMV","Gross_NMV","Discount","Qty","Bills","Outlets"]:
        curr[col] = pd.to_numeric(curr.get(col, 0), errors="coerce").fillna(0)
    
    merged = curr.merge(prev, on="SKU_Code", how="left")
    for col in ["Prev_NMV", "Prev_Bills", "Prev_Qty"]:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    # derived
    total_bills          = merged["Bills"].sum()
    total_nmv            = merged["NMV"].sum()
    merged["SPB"]        = (merged["Bills"] / merged["Bills"].replace(0, np.nan)).fillna(0)   # share of bills
    merged["NMV_Share"]  = merged["NMV"] / total_nmv * 100
    merged["Margin_Pct"] = ((merged["NMV"] - merged["Discount"]) /
                             merged["Gross_NMV"].replace(0, np.nan) * 100).fillna(0).clip(0, 100)

    # velocity: daily sales rate vs median
    merged["Daily_NMV"]    = merged["NMV"] / max((pd.to_datetime(end_date) -
                                                   pd.to_datetime(start_date)).days, 1)
    med_daily              = merged["Daily_NMV"].median()
    conditions = [
        merged["Daily_NMV"] == 0,
        merged["Daily_NMV"] <  med_daily * 0.25,
        merged["Daily_NMV"] <  med_daily * 0.75,
        merged["Daily_NMV"] <  med_daily * 1.50,
        merged["Daily_NMV"] >= med_daily * 1.50,
    ]
    choices = ["Dead", "Slow", "Average", "Fast", "⭐ Star"]
    merged["Velocity"] = np.select(conditions, choices, default="Average")

    # growth
    merged["NMV_Growth"]  = ((merged["NMV"] - merged["Prev_NMV"]) /
                              merged["Prev_NMV"].replace(0, np.nan) * 100).fillna(0).round(1)
    merged["Bill_Growth"] = ((merged["Bills"] - merged["Prev_Bills"]) /
                              merged["Prev_Bills"].replace(0, np.nan) * 100).fillna(0).round(1)

    # cumulative share for Pareto
    merged = merged.sort_values("NMV", ascending=False).reset_index(drop=True)
    merged["Cum_NMV_Pct"] = merged["NMV"].cumsum() / total_nmv * 100

    return merged


def _compute_basket_pairs(bill_df, min_support=0.01, top_n=50):
    """
    Apriori-lite: count co-occurrences of SKU pairs across invoices.
    Returns DataFrame with [SKU_A, SKU_B, Co_Bills, Support, Confidence, Lift].
    """
    if bill_df is None or bill_df.empty:
        return pd.DataFrame()

    # group SKUs per invoice
    basket = bill_df.groupby("Invoice_No")["SKU_Code"].apply(list).to_dict()
    total_inv  = len(basket)

    # individual SKU frequency
    sku_freq = defaultdict(int)
    for items in basket.values():
        for s in set(items):
            sku_freq[s] += 1

    # pair co-occurrence
    pair_freq = defaultdict(int)
    for items in basket.values():
        unique_items = list(set(items))
        if len(unique_items) < 2:
            continue
        for a, b in combinations(sorted(unique_items), 2):
            pair_freq[(a, b)] += 1

    rows = []
    for (a, b), co_count in pair_freq.items():
        sup  = co_count / total_inv
        if sup < min_support:
            continue
        conf_ab = co_count / sku_freq[a] if sku_freq[a] else 0
        conf_ba = co_count / sku_freq[b] if sku_freq[b] else 0
        lift    = co_count * total_inv / (sku_freq[a] * sku_freq[b]) if (sku_freq[a] * sku_freq[b]) else 0
        rows.append({
            "SKU_A": a, "SKU_B": b,
            "Co_Bills":   co_count,
            "Support":    round(sup,  4),
            "Conf_A→B":   round(conf_ab, 3),
            "Conf_B→A":   round(conf_ba, 3),
            "Lift":       round(lift, 2),
        })

    if not rows:
        return pd.DataFrame()

    return (pd.DataFrame(rows)
              .sort_values("Lift", ascending=False)
              .head(top_n)
              .reset_index(drop=True))


def _velocity_badge(v):
    cls = {"⭐ Star":"vel-star","Fast":"vel-fast","Average":"vel-avg",
           "Slow":"vel-slow","Dead":"vel-dead"}.get(v,"vel-avg")
    return f"<span class='vel-badge {cls}'>{v}</span>"


def _html_table(df, col_defs, max_rows=100, height=380):
    """
    Render a styled HTML table inside .sku-table-card.
    col_defs = list of (col_name_in_df, display_label, alignment, formatter_fn_or_None)
    """
    header = "".join(
        f"<th class='{'num' if al=='right' else ''}'>{lbl}</th>"
        for _, lbl, al, _ in col_defs
    )
    rows_html = []
    for _, row in df.head(max_rows).iterrows():
        cells = []
        for col, _, al, fmt in col_defs:
            raw = row.get(col, "")
            val = fmt(raw) if fmt else escape(str(raw))
            cls = "num" if al == "right" else ""
            cells.append(f"<td class='{cls}'>{val}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    table_html = (
        "<div class='sku-table-card'>"
        f"<div style='max-height:{height}px;overflow:auto;'>"
        "<table>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table></div></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
#  SECTION 1 — SKU PER BILL ANALYSIS
# ════════════════════════════════════════════════════════════════════════

def _section_sku_per_bill(master_df, bill_df, start_date, end_date, town_code):
    _pill("🧾", "SKU per Bill Analysis")

    total_bills  = master_df["Bills"].sum()
    total_skus   = master_df["SKU_Code"].nunique()
    # weighted average SPB: total unique SKU-invoice combinations / total invoices
    avg_spb      = 0.0
    if bill_df is not None and not bill_df.empty:
        spb_per_inv  = bill_df.groupby("Invoice_No")["SKU_Code"].nunique()
        avg_spb      = spb_per_inv.mean()
        median_spb   = spb_per_inv.median()
        max_spb      = spb_per_inv.max()
        spb_series   = spb_per_inv
    else:
        avg_spb = median_spb = max_spb = 0.0
        spb_series = pd.Series([], dtype=float)

    k1,k2,k3,k4,k5 = st.columns(5)
    _card(k1, "Avg SKU / Bill",    f"{avg_spb:.2f}",
          gradient="linear-gradient(90deg,#5B5F97,#B8B8D1)")
    _card(k2, "Median SKU / Bill", f"{median_spb:.1f}",
          gradient="linear-gradient(90deg,#2563EB,#60A5FA)")
    _card(k3, "Max SKU / Bill",    f"{int(max_spb)}",
          gradient="linear-gradient(90deg,#D97706,#FCD34D)")
    _card(k4, "Total Active SKUs", f"{int(total_skus):,}",
          gradient="linear-gradient(90deg,#16A34A,#4ADE80)")
    _card(k5, "Total Bills",       f"{int(total_bills):,}",
          gradient="linear-gradient(90deg,#FF6B6C,#F87171)")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1])

    # Histogram of SPB distribution
    with c1:
        if not spb_series.empty:
            counts, edges = np.histogram(spb_series.clip(upper=20), bins=range(1, 22))
            fig = go.Figure(go.Bar(
                x=list(range(1, 21)),
                y=counts,
                marker=dict(
                    color=counts,
                    colorscale=[[0,"#B8B8D1"],[0.5,"#5B5F97"],[1,"#FFC145"]],
                    showscale=False,
                ),
                text=counts,
                textposition="outside",
                hovertemplate="SKU/Bill = %{x}<br>Bills = %{y:,}<extra></extra>",
            ))
            fig.add_vline(x=avg_spb, line_dash="dash", line_color="#EF4444",
                          annotation_text=f"Avg {avg_spb:.1f}",
                          annotation_position="top right")
            _layout(fig, h=310, title="Distribution: SKU Count per Bill")
            fig.update_xaxes(title="# SKUs in Bill", dtick=1)
            fig.update_yaxes(title="# Bills")
            st.plotly_chart(fig, use_container_width=True)
        else:
            _no_data()

    # Booker-level SPB
    with c2:
        if bill_df is not None and not bill_df.empty and "Booker" in bill_df.columns:
            booker_spb = (bill_df.groupby(["Invoice_No","Booker"])["SKU_Code"]
                          .nunique().reset_index()
                          .groupby("Booker")["SKU_Code"].mean()
                          .reset_index().rename(columns={"SKU_Code":"Avg_SPB"})
                          .sort_values("Avg_SPB"))
            bar_cols = ["#EF4444" if v < avg_spb * 0.8 else
                        "#FFC145" if v < avg_spb else "#16A34A"
                        for v in booker_spb["Avg_SPB"]]
            fig2 = go.Figure(go.Bar(
                y=booker_spb["Booker"], x=booker_spb["Avg_SPB"],
                orientation="h", marker_color=bar_cols,
                text=booker_spb["Avg_SPB"].round(2), textposition="outside",
                hovertemplate="<b>%{y}</b><br>Avg SKU/Bill: %{x:.2f}<extra></extra>",
            ))
            fig2.add_vline(x=avg_spb, line_dash="dash", line_color="#5B5F97",
                           annotation_text="Overall avg", annotation_position="top")
            _layout(fig2, h=310, title="Booker-wise Avg SKU per Bill")
            fig2.update_xaxes(title="Avg SKU / Bill")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            _no_data("Booker column not available in bill data.")

    # Top SKUs by Bill penetration
    top_pen = master_df.nlargest(20, "Bills")[["SKU_Code","SKU_Name","Brand","Bills","Outlets","NMV"]].copy()
    top_pen["Bill_Pen_%"] = (top_pen["Bills"] / total_bills * 100).round(2)
    st.markdown("**Top 20 SKUs by Bill Penetration**")
    _html_table(top_pen, [
        ("SKU_Code",  "SKU Code",    "left",  None),
        ("SKU_Name",  "Name",        "left",  lambda v: f"<span title='{escape(str(v))}'>{escape(str(v)[:28])}</span>"),
        ("Brand",     "Brand",       "left",  None),
        ("Bills",     "Bills",       "right", lambda v: f"{int(v):,}"),
        ("Bill_Pen_%","Bill Pen %",  "right", lambda v: f"{v:.2f}%"),
        ("Outlets",   "Outlets",     "right", lambda v: f"{int(v):,}"),
        ("NMV",       "NMV (Rs)",    "right", lambda v: f"{int(v):,}"),
    ], height=320)

    csv = top_pen.to_csv(index=False).encode()
    st.download_button("⬇️ Export Bill Penetration CSV", csv,
                       f"sku_per_bill_{town_code}.csv", "text/csv",
                       key="spb_export")


# ════════════════════════════════════════════════════════════════════════
#  SECTION 2 — BASKET ANALYSIS
# ════════════════════════════════════════════════════════════════════════

def _section_basket(master_df, bill_df, town_code):
    _pill("🛒", "Basket Analysis — SKU Co-Purchase Affinity")

    with st.spinner("Computing basket pairs…"):
        pair_df = _compute_basket_pairs(bill_df, min_support=0.005, top_n=60)

    if pair_df.empty:
        _no_data("Not enough co-purchase data to compute basket rules.")
        return

    # merge SKU names
    name_map = master_df.set_index("SKU_Code")["SKU_Name"].to_dict()
    pair_df["Name_A"] = pair_df["SKU_A"].map(name_map).fillna(pair_df["SKU_A"])
    pair_df["Name_B"] = pair_df["SKU_B"].map(name_map).fillna(pair_df["SKU_B"])
    pair_df["Pair"]   = pair_df["Name_A"].str[:18] + " × " + pair_df["Name_B"].str[:18]

    total_inv = bill_df["Invoice_No"].nunique() if bill_df is not None else 1
    top_lift  = pair_df["Lift"].max()
    avg_lift  = pair_df["Lift"].mean()
    strong_rules = (pair_df["Lift"] >= 1.5).sum()

    k1,k2,k3,k4 = st.columns(4)
    _card(k1, "Unique Pairs Found", f"{len(pair_df):,}",
          gradient="linear-gradient(90deg,#5B5F97,#B8B8D1)")
    _card(k2, "Max Lift Score",     f"{top_lift:.2f}",
          gradient="linear-gradient(90deg,#FFC145,#FBBF24)",
          tooltip="Lift > 1 = products bought together more than chance")
    _card(k3, "Avg Lift",           f"{avg_lift:.2f}",
          gradient="linear-gradient(90deg,#2563EB,#60A5FA)")
    _card(k4, "Strong Rules (Lift≥1.5)", f"{strong_rules:,}",
          gradient="linear-gradient(90deg,#16A34A,#4ADE80)")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.3, 1])

    # Top pairs — horizontal bar
    with c1:
        top30 = pair_df.head(25).copy()
        bar_c = ["#16A34A" if l >= 2 else "#FFC145" if l >= 1.5 else "#5B5F97"
                 for l in top30["Lift"]]
        fig = go.Figure(go.Bar(
            y=top30["Pair"][::-1], x=top30["Lift"][::-1],
            orientation="h", marker_color=bar_c[::-1],
            text=top30["Lift"][::-1].round(2), textposition="outside",
            customdata=np.column_stack([top30["Co_Bills"][::-1], top30["Support"][::-1]]),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Lift: %{x:.2f}<br>"
                "Co-Bills: %{customdata[0]:,}<br>"
                "Support: %{customdata[1]:.3f}<extra></extra>"
            ),
        ))
        fig.add_vline(x=1.5, line_dash="dash", line_color="#EF4444",
                      annotation_text="Lift 1.5 threshold")
        _layout(fig, h=520, title="Top 25 SKU Pairs by Lift Score")
        fig.update_xaxes(title="Lift Score")
        st.plotly_chart(fig, use_container_width=True)

    # Support vs Lift scatter bubble
    with c2:
        fig2 = px.scatter(
            pair_df, x="Support", y="Lift",
            size="Co_Bills", color="Lift",
            color_continuous_scale=["#B8B8D1","#5B5F97","#FFC145","#16A34A"],
            hover_data={"Pair": True, "Conf_A→B": True, "Co_Bills": True},
            labels={"Support":"Support (% of Bills)","Lift":"Lift Score"},
            size_max=30,
        )
        fig2.add_hline(y=1.0, line_dash="dot", line_color="#94A3B8",
                       annotation_text="Lift = 1 (random)")
        fig2.add_hline(y=1.5, line_dash="dash", line_color="#EF4444",
                       annotation_text="Strong rule")
        _layout(fig2, h=520, title="Support vs Lift Bubble Chart")
        fig2.update_coloraxes(showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Table
    st.markdown("**Top Basket Rules Detail**")
    tbl = pair_df[["Name_A","Name_B","Co_Bills","Support","Conf_A→B","Conf_B→A","Lift"]].copy()
    _html_table(tbl, [
        ("Name_A",   "SKU A",       "left",  lambda v: escape(str(v)[:26])),
        ("Name_B",   "SKU B",       "left",  lambda v: escape(str(v)[:26])),
        ("Co_Bills", "Co-Bills",    "right", lambda v: f"{int(v):,}"),
        ("Support",  "Support",     "right", lambda v: f"{float(v)*100:.2f}%"),
        ("Conf_A→B", "Conf A→B",   "right", lambda v: f"{float(v)*100:.1f}%"),
        ("Conf_B→A", "Conf B→A",   "right", lambda v: f"{float(v)*100:.1f}%"),
        ("Lift",     "Lift",        "right", lambda v: f"{float(v):.2f}"),
    ], height=360)

    csv = pair_df.to_csv(index=False).encode()
    st.download_button("⬇️ Export Basket Rules CSV", csv,
                       f"basket_analysis_{town_code}.csv", "text/csv",
                       key="basket_export")


# ════════════════════════════════════════════════════════════════════════
#  SECTION 3 — SKU-WISE PROFITABILITY
# ════════════════════════════════════════════════════════════════════════

def _section_profitability(master_df, town_code):
    _pill("💰", "SKU-wise Profitability")

    df = master_df.copy()
    total_nmv    = df["NMV"].sum()
    total_gross  = df["Gross_NMV"].sum()
    total_disc   = df["Discount"].sum()
    avg_margin   = df["Margin_Pct"].mean()
    top_contrib  = df.iloc[0]["SKU_Name"] if not df.empty else "-"

    k1,k2,k3,k4,k5 = st.columns(5)
    _card(k1, "Total NMV",          f"Rs {total_nmv/1e6:.2f}M",
          gradient="linear-gradient(90deg,#5B5F97,#B8B8D1)")
    _card(k2, "Total Gross Revenue", f"Rs {total_gross/1e6:.2f}M",
          gradient="linear-gradient(90deg,#2563EB,#60A5FA)")
    _card(k3, "Total Discount",      f"Rs {total_disc/1e3:.1f}K",
          gradient="linear-gradient(90deg,#EF4444,#F87171)")
    _card(k4, "Avg Margin %",        f"{avg_margin:.1f}%",
          gradient="linear-gradient(90deg,#16A34A,#4ADE80)",
          tooltip="(NMV - Discount) / Gross NMV")
    _card(k5, "#1 SKU by NMV",       str(top_contrib)[:18],
          gradient="linear-gradient(90deg,#FFC145,#FBBF24)")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    # Pareto NMV
    with c1:
        prt = df.head(40).copy()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=prt["SKU_Code"], y=prt["NMV"],
            name="NMV", marker_color=C["primary"], yaxis="y",
            hovertemplate="<b>%{x}</b><br>NMV: Rs %{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=prt["SKU_Code"], y=prt["Cum_NMV_Pct"],
            name="Cumulative %", mode="lines+markers",
            line=dict(color=C["accent"], width=2.5), yaxis="y2",
        ))
        fig.add_hline(y=80, yref="y2", line_dash="dash", line_color=C["danger"],
                      annotation_text="80% NMV", annotation_position="right")
        fig.update_layout(
            yaxis2=dict(overlaying="y", side="right", range=[0,110],
                        title="Cumulative NMV %", showgrid=False),
        )
        _layout(fig, h=340, title="SKU Pareto — NMV Contribution")
        fig.update_xaxes(showticklabels=False, title="SKUs (ranked by NMV)")
        st.plotly_chart(fig, use_container_width=True)

    # Margin % vs NMV bubble
    with c2:
        bubble = df[df["NMV"] > 0].copy()
        bubble["NMV_Share"] = bubble["NMV"] / total_nmv * 100
        fig2 = px.scatter(
            bubble.head(60),
            x="NMV", y="Margin_Pct",
            size="Bills", color="Brand",
            hover_data={"SKU_Name":True,"NMV":True,"Margin_Pct":True},
            size_max=35, opacity=0.8,
            labels={"NMV":"Net NMV (Rs)","Margin_Pct":"Margin %"},
        )
        fig2.add_hline(y=avg_margin, line_dash="dash", line_color=C["neutral"],
                       annotation_text=f"Avg {avg_margin:.1f}%",
                       annotation_position="right")
        _layout(fig2, h=340, title="NMV vs Margin % Bubble (size = Bills)")
        st.plotly_chart(fig2, use_container_width=True)

    # Profitability quadrant
    st.markdown("**Profitability Quadrant — High/Low NMV × High/Low Margin**")
    med_nmv = df["NMV"].median()
    med_mrg = df["Margin_Pct"].median()
    df["Quadrant"] = np.select(
        [
            (df["NMV"] >= med_nmv) & (df["Margin_Pct"] >= med_mrg),
            (df["NMV"] >= med_nmv) & (df["Margin_Pct"] <  med_mrg),
            (df["NMV"] <  med_nmv) & (df["Margin_Pct"] >= med_mrg),
            (df["NMV"] <  med_nmv) & (df["Margin_Pct"] <  med_mrg),
        ],
        ["⭐ Stars", "🔧 Fix Margin", "💎 Niche Gems", "⚠️ Review"],
        default="Review",
    )
    fig3 = px.scatter(
        df, x="NMV", y="Margin_Pct",
        color="Quadrant",
        color_discrete_map={
            "⭐ Stars":      "#16A34A",
            "🔧 Fix Margin": "#2563EB",
            "💎 Niche Gems": "#FFC145",
            "⚠️ Review":     "#EF4444",
        },
        size="Bills",
        hover_data={"SKU_Name":True,"Brand":True,"NMV":True,"Margin_Pct":True},
        size_max=25, opacity=0.8,
        labels={"NMV":"Net NMV (Rs)","Margin_Pct":"Margin %"},
    )
    fig3.add_vline(x=med_nmv, line_dash="dot", line_color="#94A3B8",
                   annotation_text="Median NMV")
    fig3.add_hline(y=med_mrg, line_dash="dot", line_color="#94A3B8",
                   annotation_text="Median Margin")
    _layout(fig3, h=420)
    st.plotly_chart(fig3, use_container_width=True)

    # Summary table
    prof_tbl = df[["SKU_Code","SKU_Name","Brand","NMV","Gross_NMV","Discount","Margin_Pct","NMV_Share","Quadrant"]].copy()
    prof_tbl = prof_tbl.sort_values("NMV", ascending=False)
    _html_table(prof_tbl, [
        ("SKU_Code",   "SKU Code",     "left",  None),
        ("SKU_Name",   "Name",         "left",  lambda v: f"<span title='{escape(str(v))}'>{escape(str(v)[:24])}</span>"),
        ("Brand",      "Brand",        "left",  None),
        ("NMV",        "NMV (Rs)",     "right", lambda v: f"{int(v):,}"),
        ("Gross_NMV",  "Gross (Rs)",   "right", lambda v: f"{int(v):,}"),
        ("Discount",   "Discount",     "right", lambda v: f"{int(v):,}"),
        ("Margin_Pct", "Margin %",     "right", lambda v: f"{float(v):.1f}%"),
        ("NMV_Share",  "Share %",      "right", lambda v: f"{float(v):.2f}%"),
        ("Quadrant",   "Quadrant",     "left",  None),
    ], height=340)

    csv = prof_tbl.to_csv(index=False).encode()
    st.download_button("⬇️ Export Profitability CSV", csv,
                       f"sku_profitability_{town_code}.csv", "text/csv",
                       key="profit_export")


# ════════════════════════════════════════════════════════════════════════
#  SECTION 4 — SLOW / FAST MOVING SKU
# ════════════════════════════════════════════════════════════════════════

def _section_velocity(master_df, town_code):
    _pill("⚡", "Slow / Fast Moving SKU Velocity Analysis")

    df = master_df.copy()
    vel_counts = df["Velocity"].value_counts()
    order      = ["⭐ Star","Fast","Average","Slow","Dead"]

    k_cols = st.columns(5)
    grad_map = {
        "⭐ Star": "linear-gradient(90deg,#854D0E,#FEF9C3)",
        "Fast":    "linear-gradient(90deg,#16A34A,#4ADE80)",
        "Average": "linear-gradient(90deg,#2563EB,#60A5FA)",
        "Slow":    "linear-gradient(90deg,#D97706,#FCD34D)",
        "Dead":    "linear-gradient(90deg,#64748B,#CBD5E1)",
    }
    for i, seg in enumerate(order):
        cnt = int(vel_counts.get(seg, 0))
        _card(k_cols[i], f"{seg} SKUs", str(cnt), gradient=grad_map[seg])

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    # Velocity donut
    with c1:
        vc = df.groupby("Velocity").agg(Count=("SKU_Code","count"), NMV=("NMV","sum")).reset_index()
        vc["Velocity"] = pd.Categorical(vc["Velocity"], categories=order, ordered=True)
        vc = vc.sort_values("Velocity")
        color_seq = ["#854D0E","#16A34A","#2563EB","#D97706","#94A3B8"]
        fig = go.Figure(go.Pie(
            labels=vc["Velocity"], values=vc["Count"],
            hole=0.48,
            marker=dict(colors=color_seq),
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        ))
        _layout(fig, h=320, title="SKU Count by Velocity Segment")
        st.plotly_chart(fig, use_container_width=True)

    # NMV contribution by velocity
    with c2:
        fig2 = go.Figure()
        for seg, col in zip(order, color_seq):
            sub = df[df["Velocity"] == seg]
            fig2.add_trace(go.Bar(
                name=seg, x=[seg],
                y=[sub["NMV"].sum()],
                marker_color=col,
                text=[f"Rs {sub['NMV'].sum()/1e3:.0f}K"],
                textposition="outside",
            ))
        _layout(fig2, h=320, title="NMV Contribution by Velocity Segment")
        fig2.update_yaxes(title="Total NMV (Rs)")
        st.plotly_chart(fig2, use_container_width=True)

    # Daily NMV boxplot per velocity
    fig3 = px.box(
        df, x="Velocity", y="Daily_NMV",
        color="Velocity",
        category_orders={"Velocity": order},
        color_discrete_sequence=color_seq,
        points="outliers",
        labels={"Daily_NMV":"Daily NMV (Rs)","Velocity":"Segment"},
    )
    _layout(fig3, h=320, title="Daily NMV Distribution per Velocity Segment")
    st.plotly_chart(fig3, use_container_width=True)

    # Filter & browse by segment
    seg_filter = st.selectbox(
        "Browse SKUs in segment",
        options=["All"] + order,
        key="vel_seg_filter",
    )
    view_df = df if seg_filter == "All" else df[df["Velocity"] == seg_filter]
    view_df = view_df[["SKU_Code","SKU_Name","Brand","Category","Velocity",
                        "NMV","Daily_NMV","Bills","Outlets","NMV_Growth"]].copy()
    view_df = view_df.sort_values("NMV", ascending=False)

    _html_table(view_df, [
        ("SKU_Code",  "SKU Code",     "left",  None),
        ("SKU_Name",  "Name",         "left",  lambda v: f"<span title='{escape(str(v))}'>{escape(str(v)[:26])}</span>"),
        ("Brand",     "Brand",        "left",  None),
        ("Category",  "Category",     "left",  None),
        ("Velocity",  "Velocity",     "left",  _velocity_badge),
        ("NMV",       "NMV (Rs)",     "right", lambda v: f"{int(v):,}"),
        ("Daily_NMV", "Daily NMV",    "right", lambda v: f"{int(v):,}"),
        ("Bills",     "Bills",        "right", lambda v: f"{int(v):,}"),
        ("NMV_Growth","Growth %",     "right",
         lambda v: (f"<span style='color:#16A34A;font-weight:700'>▲ {float(v):.1f}%</span>"
                    if float(v) >= 0 else
                    f"<span style='color:#EF4444;font-weight:700'>▼ {abs(float(v)):.1f}%</span>")),
    ], height=360)

    csv = view_df.to_csv(index=False).encode()
    st.download_button("⬇️ Export Velocity CSV", csv,
                       f"sku_velocity_{town_code}.csv", "text/csv",
                       key="vel_export")


# ════════════════════════════════════════════════════════════════════════
#  SECTION 5 — SKU CANNIBALIZATION EFFECT
# ════════════════════════════════════════════════════════════════════════

def _section_cannibalization(master_df, bill_df, start_date, end_date, town_code):
    _pill("🔄", "SKU Cannibalization Effect")

    st.markdown(
        "<p style='color:#64748B;font-size:13px;margin-bottom:12px'>"
        "Cannibalization = when a rising SKU suppresses sales of a sibling SKU within "
        "the same brand/category. Detected via negative correlation in monthly volume trends "
        "and overlap in outlet penetration."
        "</p>",
        unsafe_allow_html=True,
    )

    df = master_df.copy()

    # For cannibalization: focus on brands with ≥ 2 SKUs
    brand_sku_counts = df.groupby("Brand")["SKU_Code"].count()
    multi_brand_skus = brand_sku_counts[brand_sku_counts >= 2].index.tolist()
    multi_df = df[df["Brand"].isin(multi_brand_skus)].copy()

    if multi_df.empty:
        _no_data("No brand has ≥ 2 active SKUs — cannibalization analysis not applicable.")
        return

    # ── KPIs ─────────────────────────────────────────────────────────────
    # Herfindahl index per brand (higher = more concentration = less cannibalization risk)
    hhi_per_brand = (
        multi_df.groupby("Brand")
        .apply(lambda g: ((g["NMV"] / g["NMV"].sum()) ** 2).sum())
        .reset_index().rename(columns={0: "HHI"})
    )
    avg_hhi  = hhi_per_brand["HHI"].mean()
    low_hhi  = (hhi_per_brand["HHI"] < 0.5).sum()   # fragmented — higher cannibal risk

    k1,k2,k3,k4 = st.columns(4)
    _card(k1, "Brands Analysed",        str(len(multi_brand_skus)),
          gradient="linear-gradient(90deg,#5B5F97,#B8B8D1)")
    _card(k2, "SKUs in Multi-SKU Brands", f"{len(multi_df):,}",
          gradient="linear-gradient(90deg,#2563EB,#60A5FA)")
    _card(k3, "Avg Brand HHI",           f"{avg_hhi:.3f}",
          gradient="linear-gradient(90deg,#FFC145,#FBBF24)",
          tooltip="HHI < 0.5 → fragmented brand portfolio → higher cannibalization risk. HHI near 1 → dominated by one SKU.")
    _card(k4, "High-Risk Brands",         str(int(low_hhi)),
          gradient="linear-gradient(90deg,#EF4444,#F87171)",
          tooltip="Brands with HHI < 0.5 (fragmented = cannibalisation risk)")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Brand HHI bar ────────────────────────────────────────────────────
    hhi_per_brand = hhi_per_brand.sort_values("HHI")
    hhi_colors    = ["#EF4444" if h < 0.5 else "#FFC145" if h < 0.75 else "#16A34A"
                     for h in hhi_per_brand["HHI"]]
    fig_hhi = go.Figure(go.Bar(
        x=hhi_per_brand["Brand"], y=hhi_per_brand["HHI"],
        marker_color=hhi_colors,
        text=hhi_per_brand["HHI"].round(3), textposition="outside",
        hovertemplate="<b>%{x}</b><br>HHI: %{y:.3f}<extra></extra>",
    ))
    fig_hhi.add_hline(y=0.5, line_dash="dash", line_color="#EF4444",
                      annotation_text="Risk threshold 0.5")
    _layout(fig_hhi, h=280, title="Brand Concentration (HHI) — Lower = More Cannibalization Risk")
    st.plotly_chart(fig_hhi, use_container_width=True)

    # ── Monthly trend correlation — user picks brand ─────────────────────
    st.markdown("#### Monthly SKU Trend Correlation within Brand")
    brand_choice = st.selectbox(
        "Select Brand to analyse",
        options=multi_brand_skus,
        key="cannibal_brand_sel",
    )
    brand_skus = multi_df[multi_df["Brand"] == brand_choice]["SKU_Code"].tolist()
    monthly_df = _fetch_monthly_sku(start_date, end_date, town_code, tuple(brand_skus))

    if monthly_df.empty:
        _no_data("Monthly data not available.")
    else:
        monthly_df["Month"] = pd.to_datetime(monthly_df["Month"], errors="coerce")
        monthly_df["NMV"]   = pd.to_numeric(monthly_df["NMV"], errors="coerce").fillna(0)
        monthly_df["SKU_Name"] = monthly_df["SKU_Code"].map(
            master_df.set_index("SKU_Code")["SKU_Name"].to_dict()
        ).fillna(monthly_df["SKU_Code"])

        c1, c2 = st.columns([1.5, 1])

        # Trend lines
        with c1:
            fig_t = px.line(
                monthly_df.dropna(subset=["Month"]),
                x="Month", y="NMV", color="SKU_Name",
                markers=True,
                labels={"NMV":"NMV (Rs)","SKU_Name":"SKU"},
            )
            _layout(fig_t, h=360, title=f"Monthly NMV Trend — {brand_choice}")
            st.plotly_chart(fig_t, use_container_width=True)

        # Correlation heatmap
        with c2:
            pivot = monthly_df.pivot_table(index="Month", columns="SKU_Name",
                                            values="NMV", aggfunc="sum", fill_value=0)
            if pivot.shape[1] >= 2:
                corr = pivot.corr().round(2)
                fig_corr = px.imshow(
                    corr, text_auto=True, aspect="auto",
                    color_continuous_scale=["#EF4444","#FFFFFF","#16A34A"],
                    zmin=-1, zmax=1,
                    labels=dict(color="Correlation"),
                )
                _layout(fig_corr, h=360, title="SKU Pair Correlation Matrix")
                fig_corr.update_xaxes(tickangle=-35, tickfont_size=10)
                fig_corr.update_yaxes(tickfont_size=10)
                st.plotly_chart(fig_corr, use_container_width=True)

                # Flag negative correlations
                neg_pairs = []
                for i, sk_a in enumerate(corr.columns):
                    for j, sk_b in enumerate(corr.columns):
                        if j <= i:
                            continue
                        c_val = float(corr.loc[sk_a, sk_b])
                        if c_val <= -0.4:
                            neg_pairs.append({
                                "SKU A": sk_a[:24],
                                "SKU B": sk_b[:24],
                                "Correlation": f"{c_val:.2f}",
                                "Risk Level": "🔴 High" if c_val < -0.65 else "🟡 Medium",
                            })
                if neg_pairs:
                    st.warning(
                        f"⚠️ {len(neg_pairs)} potential cannibalization pair(s) detected "
                        f"(negative correlation ≤ -0.40).",
                        icon="🔄"
                    )
                    neg_df = pd.DataFrame(neg_pairs)
                    _html_table(neg_df, [
                        ("SKU A",       "SKU A",       "left",  None),
                        ("SKU B",       "SKU B",       "left",  None),
                        ("Correlation", "Correlation", "right", lambda v: f"<span style='color:#EF4444;font-weight:700'>{v}</span>"),
                        ("Risk Level",  "Risk Level",  "left",  None),
                    ], height=200)
                else:
                    st.success("No strong negative correlations found — low cannibalization risk in this brand.", icon="✅")
            else:
                st.info("Need ≥ 2 SKUs with monthly data to show correlation matrix.")

    # ── Outlet Overlap matrix ─────────────────────────────────────────────
    st.markdown("#### Outlet Overlap — Shared Distribution Points")
    if bill_df is not None and not bill_df.empty:
        brand_inv = bill_df[bill_df["SKU_Code"].isin(brand_skus)].copy()
        sku_outlet = brand_inv.groupby("SKU_Code")["Store_Code"].apply(set).to_dict()
        sku_list   = [s for s in brand_skus if s in sku_outlet][:8]
        name_map   = master_df.set_index("SKU_Code")["SKU_Name"].to_dict()

        if len(sku_list) >= 2:
            overlap_mat = pd.DataFrame(index=sku_list, columns=sku_list, dtype=float)
            for a in sku_list:
                for b in sku_list:
                    set_a = sku_outlet.get(a, set())
                    set_b = sku_outlet.get(b, set())
                    overlap_mat.loc[a, b] = (
                        len(set_a & set_b) / len(set_a | set_b)
                        if set_a | set_b else 0
                    )
            overlap_mat.index   = [name_map.get(s, s)[:20] for s in overlap_mat.index]
            overlap_mat.columns = [name_map.get(s, s)[:20] for s in overlap_mat.columns]
            overlap_mat = overlap_mat.astype(float).round(2)

            fig_ov = px.imshow(
                overlap_mat, text_auto=True, aspect="auto",
                color_continuous_scale=["#FFFFFF","#B8B8D1","#5B5F97"],
                zmin=0, zmax=1,
                labels=dict(color="Jaccard Overlap"),
            )
            _layout(fig_ov, h=360,
                    title="Outlet Overlap Jaccard Index (1.0 = same outlets = max cannibalization)")
            fig_ov.update_xaxes(tickangle=-35, tickfont_size=9)
            fig_ov.update_yaxes(tickfont_size=9)
            st.plotly_chart(fig_ov, use_container_width=True)
        else:
            _no_data("Not enough SKUs with outlet data for overlap matrix.")
    else:
        _no_data("Bill-level data not available for outlet overlap analysis.")


# ════════════════════════════════════════════════════════════════════════
#  MASTER ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

def render_sku_analytics_advanced_tab(start_date, end_date, town_code):
    """
    Main render function. Call inside:
        with tab_sku_adv:
            render_sku_analytics_advanced_tab(start_date, end_date, town_code)
    """
    _inject_css()
    _resolve_host_bindings()

    st.markdown(
        "<h2 style='margin:0 0 4px 0;font-size:22px;color:#0F172A;font-weight:800;'>"
        "📦 SKU Analytics — Advanced Dashboard</h2>"
        "<p style='color:#64748B;font-size:13px;margin-bottom:16px;'>"
        f"Period: {start_date} → {end_date} &nbsp;|&nbsp; Location: {town_code}"
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Load data (shared across all sections) ───────────────────────────
    try:
        with st.spinner("Loading SKU data…"):
            master_df = _build_master(start_date, end_date, town_code)
            bill_df   = _fetch_bill_level(start_date, end_date, town_code)
    except Exception as e:
        st.error(f"Live SKU data load failed: {e}", icon="🚨")
        return

    if master_df.empty:
        _no_data("No SKU data found for the selected date range and location.")
        return

    # ── Optional filters ─────────────────────────────────────────────────
    with st.expander("🔽 Filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        with f1:
            brands_avail = sorted(master_df["Brand"].dropna().unique().tolist())
            sel_brands   = st.multiselect("Brand", brands_avail, key="sku_adv_brand_filter")
        with f2:
            cats_avail   = sorted(master_df["Category"].dropna().unique().tolist())
            sel_cats     = st.multiselect("Category", cats_avail, key="sku_adv_cat_filter")
        with f3:
            bookers_avail = (sorted(bill_df["Booker"].dropna().unique().tolist())
                             if bill_df is not None and "Booker" in bill_df.columns else [])
            sel_bookers   = st.multiselect("Booker", bookers_avail, key="sku_adv_booker_filter")

    # Apply filters
    flt_master = master_df.copy()
    flt_bill   = bill_df.copy() if bill_df is not None else pd.DataFrame()
    if sel_brands:
        flt_master = flt_master[flt_master["Brand"].isin(sel_brands)]
        if not flt_bill.empty and "SKU_Code" in flt_bill.columns:
            flt_bill = flt_bill[flt_bill["SKU_Code"].isin(flt_master["SKU_Code"].unique())]
    if sel_cats:
        flt_master = flt_master[flt_master["Category"].isin(sel_cats)]
    if sel_bookers and not flt_bill.empty and "Booker" in flt_bill.columns:
        flt_bill = flt_bill[flt_bill["Booker"].isin(sel_bookers)]

    if flt_master.empty:
        _no_data("No data matching selected filters.")
        return

    # ── 5 Sub-sections ───────────────────────────────────────────────────
    (t_spb, t_basket, t_profit, t_vel, t_cannibal) = st.tabs([
        "🧾 SKU per Bill",
        "🛒 Basket Analysis",
        "💰 Profitability",
        "⚡ Slow / Fast Movers",
        "🔄 Cannibalization",
    ])

    with t_spb:
        _section_sku_per_bill(flt_master, flt_bill, start_date, end_date, town_code)

    with t_basket:
        _section_basket(flt_master, flt_bill, town_code)

    with t_profit:
        _section_profitability(flt_master, town_code)

    with t_vel:
        _section_velocity(flt_master, town_code)

    with t_cannibal:
        _section_cannibalization(flt_master, flt_bill, start_date, end_date, town_code)