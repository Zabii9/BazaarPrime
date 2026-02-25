"""
Bazaar Prime Analytics Dashboard - Streamlit Version
=====================================================

Converted from Dash to Streamlit while maintaining all charts and data functionality.
"""

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from functools import lru_cache
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# st.experimental_fragment
# Page configuration
st.set_page_config(
    page_title="Bazaar Prime Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",

)

# # Custom CSS
# st.markdown("""
# <style>
#     .main {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     }
#     .stMetric {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 20px;
#         border-radius: 10px;
#         box-shadow: 0 4px 12px rgba(0,0,0,0.15);
#     }
#     .stMetric label {
#         color: #ffffff !important;
#         font-weight: 600;
#     }
#     .stMetric [data-testid="stMetricValue"] {
#         color: #ffffff !important;
#         font-size: 28px !important;
#         font-weight: 700;
#     }
#     h1, h2, h3, h4 {
#         color: #ffffff;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
#     }
#     .stMarkdown {
#         color: #ffffff;
#     }
# </style>
# """, unsafe_allow_html=True)

# ======================
# 🔐 LOGIN CONFIG
# ======================
VALID_USERS = {
    "admin": "admin123",
    "viewer": "viewer123",
}

def check_authentication():
    """Check if user is authenticated"""
    # Initialize session state variables
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    
    if not st.session_state.authenticated:
        st.title("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username in VALID_USERS and VALID_USERS[username] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid credentials")
        st.stop()

# ======================
# 🛢 DATABASE CONFIG
# ======================
DB_CONFIG = {
    "username": "db42280",
    "password": "admin2233",
    "host": "db42280.public.databaseasp.net",
    "port": "3306",
    "database": "db42280",
}


@st.cache_resource
def get_engine():
    """Get SQLAlchemy engine for the specified town/database."""
    connection_string = f"mysql+pymysql://{DB_CONFIG['username']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(
        connection_string,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=10,
        max_overflow=20,
    )

@st.cache_data(ttl=3600)
def read_sql_cached(query, db_name="db42280"):
    """Execute SQL query with caching"""
    eng = get_engine()
    return pd.read_sql(query, eng)

# ======================
# 📊 DATA FETCHING FUNCTIONS
# ======================

@st.cache_data(ttl=3600)
def fetch_booker_less_ctn_data(months_back=3, town="db42280"):
    """Fetch booker less than half carton data"""
    eng = get_engine()
    
    booker_less_ctn_base_cte = f"""
WITH ContinuousDeliveries AS (
    SELECT
        m.brand,
        o.`Store Code` AS StoreCode,
        o.`Store Name` AS StoreName,
        o.`Order Booker Code` AS Booker_Code,
        o.`Order Booker Name` AS Booker_Name,
        o.`SKU Code` AS SKUCode,
        o.`Delivered Units` AS Del_Units,
        (o.`Delivered Units` / m.`UOM`) AS Deli_Ctn,
        m.`UOM`,
        ROW_NUMBER() OVER (PARTITION BY o.`Store Code`, m.brand ORDER BY o.`Delivery Date` ASC) AS RowNum,
        COUNT(*) OVER (PARTITION BY o.`Store Code`, m.brand) AS TotalDeliveries,
        CASE WHEN (o.`Delivered Units` / m.`UOM`) < 0.5 THEN 1 ELSE 0 END AS LessThanHalfCtn
    FROM
        (SELECT * FROM ordervsdelivered WHERE `Delivered Units` > 0) o
    LEFT JOIN
        sku_master m ON m.`Sku_Code` = o.`SKU Code`
    WHERE
        o.`Delivery Date` >= DATE_SUB(CURDATE(), INTERVAL {months_back} MONTH)
),
final AS (
    SELECT
        brand,
        StoreCode,
        StoreName,
        Booker_Code,
        Booker_Name,
        MAX(RowNum) AS Total_Deliveries,
        SUM(LessThanHalfCtn) AS HalfCtnDel
    FROM
        ContinuousDeliveries
    GROUP BY
        Booker_Code, Booker_Name, StoreCode, StoreName, brand
) select *,(SUM(HalfCtnDel) / SUM(Total_Deliveries)) AS age from final
"""
    
    detail_df = pd.read_sql(booker_less_ctn_base_cte, eng)
    
    if detail_df.empty:
        return pd.DataFrame(), detail_df
    
    pivot_df = detail_df.pivot_table(
        index='Booker_Name',
        columns='brand',
        values='age',
        aggfunc='mean',
        fill_value=0
    )
    
    pivot_df = pivot_df.reset_index()
    pivot_df = pivot_df.sort_values('Booker_Name')
    
    return pivot_df, detail_df

@st.cache_data(ttl=3600)
def fetch_kpi_data(start_date, end_date, town_code):
    """Fetch KPI metrics with YoY and MoM growth"""
    distributor_condition = ""
    if town_code and town_code != "all":
        distributor_condition = f"AND o.`Distributor Code` = '{town_code}'"

    query = f"""
    WITH raw AS (
    SELECT
        u.`Channel Type` AS Channel,
        SUM(o.`Delivered Amount` + o.`Total Discount`) AS Amount,
        o.`Delivery Date` AS D_Date,
				SUM(o.`Delivered (Litres)`) AS Ltr,
				Case when o.`Distributor Code`='D70002202' then 'Karachi'
				when o.`Distributor Code`='D70002246' then 'Lahore' else 'CBL' end Town,
				`Invoice Number` as Orders
    FROM
        ordervsdelivered o
    LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
    WHERE
        1=1
        {distributor_condition}
    GROUP BY
        u.`Channel Type`,
        o.`Delivery Date`,o.`Distributor Code`,`Invoice Number`
)

, selected_period AS (
    SELECT
		Town,
        raw.Channel,
        SUM(raw.Amount) AS NMV,
				SUM(ltr) as Ltr,
				count(DISTINCT orders) as Orders
    FROM
        raw
    WHERE
        raw.D_Date BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY
        raw.Channel,
				Town
)
, last_year_period AS (
    SELECT
		town,
        raw.Channel,
        SUM(raw.Amount) AS NMV,
				sum(ltr) as Ltr,
				count(DISTINCT orders) as Orders
    FROM
        raw
    WHERE
        raw.D_Date BETWEEN DATE_SUB('{start_date}', INTERVAL 1 YEAR) AND DATE_SUB('{end_date}', INTERVAL 1 YEAR)
    GROUP BY
        raw.Channel,town
),
last_month AS (
    SELECT
		town,
        raw.Channel,
        SUM(raw.Amount) AS NMV,
				sum(ltr) as Ltr,
				count(DISTINCT orders) as Orders
    FROM
        raw
    WHERE
        raw.D_Date BETWEEN DATE_SUB('{start_date}', INTERVAL 1 month) AND DATE_SUB('{end_date}', INTERVAL 1 month)
    GROUP BY
        raw.Channel,town
)

SELECT
		sp.town,
    sp.Channel,
    sp.NMV AS Current_Period_Sales,
    ly.NMV AS Last_Year_Sales,
		lm.NMV AS Last_Month_Sales,
		sp.ltr as Current_Period_Ltr,
		ly.ltr as Last_Year_Ltr,
		lm.ltr as Last_Month_Ltr,
        ROUND(((sp.nmv/ly.nmv)-1)*100) as Sales_Growth_LY,
		Round(((sp.nmv/lm.nmv)-1)*100) as Sales_Growth_LM,
		Round(((sp.ltr/ly.ltr)-1)*100) as Ltr_Growth_LY,
		Round(((sp.ltr/lm.ltr)-1)*100) as Ltr_Growth_LM,
		sp.orders as Current_Orders,
		ly.orders as Last_Year_Orders,
		lm.orders as Last_Month_Orders
FROM
    selected_period sp
LEFT JOIN
    last_year_period ly ON sp.Channel = ly.Channel
LEFT JOIN
    last_month lm ON sp.Channel = lm.Channel
WHERE 1=1

		
ORDER BY
    sp.Channel
    """
    
    return read_sql_cached(query, "db42280")

@st.cache_data(ttl=3600)
def fetch_channel_treemap():
    """Fetch 6-month channel treemap data"""
    
    
    query = f"""
    SELECT 
        CONCAT(YEAR(`Delivery Date`), '-', LPAD(MONTH(`Delivery Date`), 2, '0')) AS period,
        u.`Channel Type` AS channel,
        SUM(o.`Delivered Amount` + o.`Total Discount`) AS nmv
    FROM ordervsdelivered o
    INNER JOIN universe u ON u.`Store Code` = o.`Store Code`
    WHERE o.`Delivery Date` >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    AND o.`Delivery Date` IS NOT NULL
    GROUP BY period, u.`Channel Type`
    ORDER BY o.`Delivery Date` DESC
    """
    
    return read_sql_cached(query)

@st.cache_data(ttl=3600)
def fetch_sunburst_data():
    """Fetch channel-DM sunburst data"""
    db_name = "db42280"
    
    query = f"""
    SELECT 
        u.`Channel Type` AS channel,
        o.`Deliveryman Name` AS dm,
        SUM(o.`Delivered Amount` + o.`Total Discount`) AS nmv
    FROM ordervsdelivered o
    INNER JOIN universe u ON u.`Store Code` = o.`Store Code`
    WHERE o.`Delivery Date` >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    AND o.`Delivery Date` IS NOT NULL
    GROUP BY u.`Channel Type`, o.`Deliveryman Name`
    """
    
    return read_sql_cached(query, db_name)

# ======================
# 📈 VISUALIZATION FUNCTIONS
# ======================

def create_treemap(df, selected_channels=None):
    """Create channel treemap with period hierarchy"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for the selected period",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    if selected_channels:
        df = df[df['channel'].isin(selected_channels)]
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data for selected channels",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
    
    df['nmv_millions'] = df['nmv'] / 1_000_000
    period_totals = df.groupby('period')['nmv_millions'].sum().to_dict()
    
    # Build hierarchical structure
    labels = ["Total"]
    parents = [""]
    values = [df['nmv_millions'].sum()]
    ids = ["total"]
    text_labels = ["Total Sales"]
    
    period_order = sorted(df['period'].unique(), key=lambda x: datetime.strptime(x, '%Y-%m'))
    
    for period in period_order:
        labels.append(period)
        parents.append("Total")
        period_total = period_totals.get(period, 0)
        values.append(period_total)
        ids.append(f"period_{period}")
        text_labels.append(f"{period} | {period_total:.2f}M")
    
    for period in period_order:
        period_data = df[df['period'] == period]
        for _, row in period_data.iterrows():
            labels.append(row['channel'])
            parents.append(period)
            values.append(row['nmv_millions'])
            ids.append(f"{period}_{row['channel']}")
            text_labels.append(f"{row['channel']}<br>{row['nmv_millions']:.2f}M")
    
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        ids=ids,
        text=text_labels,
        textposition="middle center",
        textfont=dict(size=14, color='white'),
        marker=dict(
            colorscale='viridis',
            line=dict(width=2, color='white')
        ),
        hovertemplate='<b>%{label}</b><br>NMV: %{value:.2f}M<extra></extra>'
    ))
    
    fig.update_layout(
        title="🗺️ 6-Month Channel Performance (Period-wise Breakdown)",
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_sunburst(df, selected_dms=None):
    """Create channel-DM sunburst chart"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    if selected_dms:
        df = df[df['dm'].isin(selected_dms)]
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data for selected deliverymen",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
    
    df['nmv_millions'] = df['nmv'] / 1_000_000
    
    fig = px.sunburst(
        df,
        path=['channel', 'dm'],
        values='nmv_millions',
        title="☀️ Channel & Deliveryman Hierarchy"
    )
    
    fig.update_traces(
        textinfo="label+percent entry",
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{label}</b><br>NMV: %{value:.2f}M<extra></extra>'
    )
    
    fig.update_layout(
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig


def fetch_bazaarprime_dashboard(start_date, end_date, town, selected_dms=None, selected_channels=None):
    """Reuse BazaarPrime.py dashboard logic and return all figures/data for Streamlit rendering."""
    from BazaarPrime import update_dashboard

    selected_dms = selected_dms or []
    selected_channels = selected_channels or []

    return update_dashboard(
        str(start_date),
        str(end_date),
        town,
        selected_dms,
        selected_channels,
    )

# ======================
# 🎯 MAIN APP
# ======================

def main():
    # Check authentication
    check_authentication()

    # Sidebar
    st.sidebar.title("📊 Bazaar Prime Analytics")

    # Display username if available
    if st.session_state.get("username"):
        st.sidebar.markdown(f"**User:** {st.session_state.username}")

    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

    st.sidebar.markdown("---")

    # Date range picker
    st.sidebar.subheader("📅 Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.today() - timedelta(days=30),
            max_value=datetime.today()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.today(),
            max_value=datetime.today()
        )

    # Location selector
    st.sidebar.subheader("📍 Location")
    # passed the value and backend will handle the condition to filter data based on all or specific town
    town_code = st.sidebar.selectbox(
        "Select Location",
        options=["D70002202", "D70002246"],
        format_func=lambda x: "Karachi" if x == "D70002202" else "Lahore"
    )
    # Town mapping corrected
    town = {
    "D70002202": "Karachi",
    "D70002246": "Lahore",
    }.get(town_code, "Not Available")  # Default case in case of undefined town_code



    # Main content
    st.title(f"📊 Bazaar Prime Analytics Dashboard - {town}")
    st.markdown("---")

    # KPIs
    st.subheader("📈 Key Performance Indicators")
    kpi_data = fetch_kpi_data(start_date, end_date, town_code)

    if not kpi_data.empty:
        def col_sum(df, column_name):
            if column_name not in df.columns:
                return 0
            return pd.to_numeric(df[column_name], errors='coerce').fillna(0).sum()

        # Calculate totals from all rows/channels
        current_revenue = col_sum(kpi_data, 'Current_Period_Sales')
        ly_revenue = col_sum(kpi_data, 'Last_Year_Sales')
        lm_revenue = col_sum(kpi_data, 'Last_Month_Sales')

        current_orders = col_sum(kpi_data, 'Current_Orders')
        ly_orders = col_sum(kpi_data, 'Last_Year_Orders')
        lm_orders = col_sum(kpi_data, 'Last_Month_Orders')

        current_ltr = col_sum(kpi_data, 'Current_Period_Ltr')
        ly_ltr = col_sum(kpi_data, 'Last_Year_Ltr')
        lm_ltr = col_sum(kpi_data, 'Last_Month_Ltr')

        # Calculate AOV
        aov_current = current_revenue / current_orders if current_orders > 0 else 0
        aov_ly = ly_revenue / ly_orders if ly_orders > 0 else 0
        aov_lm = lm_revenue / lm_orders if lm_orders > 0 else 0

        # Growth percentages
        revenue_growth_ly = ((current_revenue - ly_revenue) / ly_revenue * 100) if ly_revenue > 0 else 0
        revenue_growth_lm = ((current_revenue - lm_revenue) / lm_revenue * 100) if lm_revenue > 0 else 0

        orders_growth_ly = ((current_orders - ly_orders) / ly_orders * 100) if ly_orders > 0 else 0
        orders_growth_lm = ((current_orders - lm_orders) / lm_orders * 100) if lm_orders > 0 else 0

        ltr_growth_ly = ((current_ltr - ly_ltr) / ly_ltr * 100) if ly_ltr > 0 else 0
        ltr_growth_lm = ((current_ltr - lm_ltr) / lm_ltr * 100) if lm_ltr > 0 else 0

        aov_growth_ly = ((aov_current - aov_ly) / aov_ly * 100) if aov_ly > 0 else 0
        aov_growth_lm = ((aov_current - aov_lm) / aov_lm * 100) if aov_lm > 0 else 0

        # Display KPIs in centered container with equal widths
        empty1, col1, col2, col3, col4, empty2 = st.columns([0.5, 2, 2, 2, 2, 0.5])

        with col1:
            st.markdown(f"""
            <div style='text-align: center ;box-shadow: 0 4px 12px rgba(0,0,0,0.15);border-radius: 10px;padding: 15px;background-color: #f0f0f0;'>
                <h6 style='color: black; margin-bottom: 5px;'>💰 Total Revenue</h6>
                <h3 style='color: black; margin: 1px 0;'>Rs {current_revenue / 1_000_000:.2f}M</h3>
                <p style='color: {"#39A039" if revenue_growth_ly >= 0 else '#FF6B6B'}; font-size: 14px;font-weight: bold; margin: 3px 0;'>
                    {'▲' if revenue_growth_ly >= 0 else '▼'} {abs(revenue_growth_ly):.2f}% vs Last Year
                </p>
                <p style='color: {"#39A039" if revenue_growth_lm >= 0 else '#FF6B6B'}; font-size: 12px;font-weight: bold; margin: 3px 0;'>
                    {'▲' if revenue_growth_lm >= 0 else '▼'} {abs(revenue_growth_lm):.2f}% vs Last Month
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style='text-align: center;box-shadow: 0 4px 12px rgba(0,0,0,0.15);border-radius: 10px;padding: 15px;background-color: #f0f0f0;'>
                <h6 style='color: black; margin-bottom: 5px;'>🛢 Total Litres</h6>
                <h3 style='color: black; margin: 1px 0;'>{current_ltr:,.0f} Ltr</h3>
                <p style='color: {"#39A039" if ltr_growth_ly >= 0 else '#FF6B6B'}; font-size: 14px;font-weight: bold; margin: 3px 0;'>
                    {'▲' if ltr_growth_ly >= 0 else '▼'} {abs(ltr_growth_ly):.2f}% vs Last Year
                </p>
                <p style='color: {"#39A039" if ltr_growth_lm >= 0 else '#FF6B6B'}; font-size: 12px;font-weight: bold; margin: 3px 0;'>
                    {'▲' if ltr_growth_lm >= 0 else '▼'} {abs(ltr_growth_lm):.2f}% vs Last Month
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style='text-align: center;box-shadow: 0 4px 12px rgba(0,0,0,0.15);border-radius: 10px;padding: 15px;background-color: #f0f0f0;'>
                <h6 style='color: black; margin-bottom: 5px;'>🧾 Total Orders</h6>
                <h3 style='color: black; margin: 1px 0;'>{int(current_orders):,}</h3>
                <p style='color: {"#39A039" if orders_growth_ly >= 0 else '#FF6B6B'}; font-size: 14px;font-weight: bold; margin: 3px 0;'>
                    {'▲' if orders_growth_ly >= 0 else '▼'} {abs(orders_growth_ly):.2f}% vs Last Year
                </p>
                <p style='color: {"#39A039" if orders_growth_lm >= 0 else '#FF6B6B'}; font-size: 12px;font-weight: bold; margin: 3px 0;'>
                    {'▲' if orders_growth_lm >= 0 else '▼'} {abs(orders_growth_lm):.2f}% vs Last Month
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div style='text-align: center;box-shadow: 0 4px 12px rgba(0,0,0,0.15);border-radius: 10px;padding: 15px;background-color: #f0f0f0;'>
                <h6 style='color: black; margin-bottom: 5px;'>📦 Avg Order Value</h6>
                <h3 style='color: black; margin: 1px 0;'>Rs {aov_current / 1000:.1f}K</h3>
                <p style='color: {"#39A039" if aov_growth_ly >= 0 else '#FF6B6B'}; font-size: 14px;font-weight: bold; margin: 3px 0;'>
                    {'▲' if aov_growth_ly >= 0 else '▼'} {abs(aov_growth_ly):.2f}% vs Last Year
                </p>
                <p style='color: {"#39A039" if aov_growth_lm >= 0 else '#FF6B6B'}; font-size: 12px;font-weight: bold; margin: 3px 0;'>
                    {'▲' if aov_growth_lm >= 0 else '▼'} {abs(aov_growth_lm):.2f}% vs Last Month
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Channel-wise AOV Cards Grid
        #
        required_cols = {
            'Channel', 'Current_Period_Sales', 'Last_Year_Sales', 'Last_Month_Sales',
            'Current_Orders', 'Last_Year_Orders', 'Last_Month_Orders'
        }
        if required_cols.issubset(kpi_data.columns):
            channel_df = kpi_data[[
                'Channel', 'Current_Period_Sales', 'Last_Year_Sales', 'Last_Month_Sales',
                'Current_Orders', 'Last_Year_Orders', 'Last_Month_Orders'
            ]].copy()

            numeric_cols = [
                'Current_Period_Sales', 'Last_Year_Sales', 'Last_Month_Sales',
                'Current_Orders', 'Last_Year_Orders', 'Last_Month_Orders'
            ]
            for col in numeric_cols:
                channel_df[col] = pd.to_numeric(channel_df[col], errors='coerce').fillna(0)

            channel_df['AOV_Current'] = np.where(
                channel_df['Current_Orders'] > 0,
                channel_df['Current_Period_Sales'] / channel_df['Current_Orders'],
                0
            )
            channel_df['AOV_LY'] = np.where(
                channel_df['Last_Year_Orders'] > 0,
                channel_df['Last_Year_Sales'] / channel_df['Last_Year_Orders'],
                0
            )
            channel_df['AOV_LM'] = np.where(
                channel_df['Last_Month_Orders'] > 0,
                channel_df['Last_Month_Sales'] / channel_df['Last_Month_Orders'],
                0
            )

            channel_df['Growth_LY'] = np.where(
                channel_df['AOV_LY'] > 0,
                ((channel_df['AOV_Current'] - channel_df['AOV_LY']) / channel_df['AOV_LY']) * 100,
                0
            )
            channel_df['Growth_LM'] = np.where(
                channel_df['AOV_LM'] > 0,
                ((channel_df['AOV_Current'] - channel_df['AOV_LM']) / channel_df['AOV_LM']) * 100,
                0
            )
            channel_df = channel_df.sort_values('AOV_Current', ascending=False)
            st.markdown("---")
            st.text("📦 Channel-wise Average Order Value (AOV)")
            # Display channels in columns grid inside the 5th column
            # instead of add  manually rows get the count of channels from table to get the count of rows need to loop

            cols_per_row = len(channel_df)
            for i in range(0, len(channel_df), cols_per_row):
                row_channels = channel_df.iloc[i:i+cols_per_row]
                cols = st.columns(len(row_channels))

                for col, (_, r) in zip(cols, row_channels.iterrows()):
                    aov_val = r['AOV_Current']
                    growth_ly = r['Growth_LY']
                    growth_lm = r['Growth_LM']
                    growth_ly_color = "#5DB35D" if growth_ly >= 0 else '#FF6B6B'
                    growth_lm_color = "#5DB35D" if growth_lm >= 0 else '#FF6B6B'
                    growth_ly_arrow = '▲' if growth_ly >= 0 else '▼'
                    growth_lm_arrow = '▲' if growth_lm >= 0 else '▼'
                    if aov_val >= 1_000_000_000:
                        aov_display = f"Rs {aov_val / 1_000_000_000:.1f}B"
                    elif aov_val >= 1_000_000:
                        aov_display = f"Rs {aov_val / 1_000_000:.1f}M"
                    elif aov_val >= 1_000:
                        aov_display = f"Rs {aov_val / 1_000:.1f}K"
                    else:
                        aov_display = f"Rs {aov_val:.2f}"
                    with col:
                        st.markdown(
                            f"""
                            <div style='text-align:center; padding:8px; border-radius:10px; background:#f0f0f0; box-shadow:0 4px 12px rgba(0,0,0,0.15);'>
                            <div style='font-size:13px; font-weight:600; margin:0;'>
                            🛍️ {str(r['Channel'])}
                            </div>
                                <div style='font-size:28px; font-weight:700; margin:2px 0; line-height:1;'>
                                {aov_display}
                            </div>

                            <div style='color:{growth_ly_color}; font-size:12px; font-weight:bold; margin:3px 0;'>
                                {growth_ly_arrow} {abs(growth_ly):.2f}% vs Last Year
                            </div>

                            <div style='color:{growth_lm_color}; font-size:12px; font-weight:bold; margin:3px 0;'>
                                {growth_lm_arrow} {abs(growth_lm):.2f}% vs Last Month
                            </div>
                        """,
                            unsafe_allow_html=True,
                        )

    st.markdown("---")

    # BazaarPrime Charts (all dashboard charts from BazaarPrime.py)
    if "selected_dms" not in st.session_state:
        st.session_state.selected_dms = []
    if "selected_channels" not in st.session_state:
        st.session_state.selected_channels = []

    try:
        (
            _kpis,
            fig_daily,
            fig_type,
            fig_sales_growth,
            fig_brand_comparison,
            fig_brand_growth_bar,
            brand_datatable,
            fig_dm_comparison,
            fig_tgt_vs_ach_ytd,
            fig_sku_heatmap,
            fig_channel_sunburst,
            dm_options,
            fig_brand_productivity,
            fig_tgt_vs_ach_booker,
            fig_channel_nmv_ytd,
            channel_options,
            fig_channel_treemap,
        ) = fetch_bazaarprime_dashboard(
            start_date,
            end_date,
            town_code,
            st.session_state.selected_dms,
            st.session_state.selected_channels,
        )
    except Exception as exc:
        st.error(f"Failed to load BazaarPrime charts: {exc}")
        st.stop()

    dm_values = [item["value"] for item in (dm_options or [])]
    channel_values = [item["value"] for item in (channel_options or [])]

    st.subheader("📊 Sales Overview")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(fig_daily, use_container_width=True)
    with c2:
        st.plotly_chart(fig_type, use_container_width=True)
    st.plotly_chart(fig_sales_growth, use_container_width=True)

    st.markdown("---")
    st.subheader("🏢 Brand Performance")
    st.plotly_chart(fig_brand_comparison, use_container_width=True)
    st.plotly_chart(fig_brand_growth_bar, use_container_width=True)
    st.dataframe(pd.DataFrame(brand_datatable), use_container_width=True, height=420)

    st.markdown("---")
    st.subheader("🚚 Delivery & Operations Analytics")
    st.plotly_chart(fig_dm_comparison, use_container_width=True)
    st.plotly_chart(fig_tgt_vs_ach_ytd, use_container_width=True)
    st.plotly_chart(fig_sku_heatmap, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Channel & Productivity Analysis")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        selected_dms = st.multiselect(
        "Filter Deliverymen",
        options=dm_values,
        default=[dm for dm in st.session_state.selected_dms if dm in dm_values],
        )

    if selected_dms != st.session_state.selected_dms :
        st.session_state.selected_dms = selected_dms
        st.rerun()
    
    c3, c4 = st.columns([1.5, 1])
    with c3:
        st.plotly_chart(fig_channel_sunburst, use_container_width=True)
    with c4:
        st.plotly_chart(fig_brand_productivity, use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Target vs Achievement")
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(fig_tgt_vs_ach_booker, use_container_width=True)
    with c6:
        st.plotly_chart(fig_channel_nmv_ytd, use_container_width=True)

    st.markdown("---")
    # st.subheader("🌳 Channel Performance - 6 Month Breakdown")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        selected_channels = st.multiselect(
        "Filter Channels",
        options=channel_values,
        default=[ch for ch in st.session_state.selected_channels if ch in channel_values],
    )
    if selected_channels != st.session_state.selected_channels:
        st.session_state.selected_channels = selected_channels
        st.rerun()
    st.plotly_chart(fig_channel_treemap, use_container_width=True)

    st.markdown("---")

    # Booker Analysis Section
    st.subheader("📋 Booker Less-Than-Half-Carton Analysis")

    months_back = st.selectbox(
        "Select Time Period",
        options=[1, 2, 3, 4],
        format_func=lambda x: f"Last {x} Month{'s' if x > 1 else ''}",
        index=2
    )

    pivot_df, detail_df = fetch_booker_less_ctn_data(months_back, town)

    if not pivot_df.empty:
        # Format percentages
        for col in pivot_df.columns:
            if col != 'Booker_Name':
                pivot_df[col] = (pivot_df[col] * 100).round(2).astype(str) + "%"

        st.dataframe(pivot_df, use_container_width=True, height=400)

        # Show detail on row selection
        if st.checkbox("Show Details"):
            selected_booker = st.selectbox(
                "Select Booker",
                options=pivot_df['Booker_Name'].tolist()
            )

            if selected_booker:
                drill_df = detail_df[detail_df['Booker_Name'] == selected_booker].copy()
                drill_df['age'] = (drill_df['age'] * 100).round(2).astype(str) + "%"

                st.write(f"**Details for: {selected_booker}**")
                st.dataframe(
                    drill_df[['brand', 'StoreCode', 'StoreName', 
                             'Total_Deliveries', 'HalfCtnDel', 'age']],
                    use_container_width=True
                )

        # Footer
        

    st.markdown("---")
    st.markdown("© 2026 Bazaar Prime Analytics Dashboard | Powered by Streamlit")

if __name__ == "__main__":
    main()
