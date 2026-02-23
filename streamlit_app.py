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

# Page configuration
st.set_page_config(
    page_title="Bazaar Prime Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .stMetric label {
        color: #ffffff !important;
        font-weight: 600;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 28px !important;
        font-weight: 700;
    }
    h1, h2, h3 {
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stMarkdown {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

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
}

TOWN_TO_DB = {
    "all": "db42280",
    "prime": "db42280",
    "primelhr": "db42280",
}

@st.cache_resource
def get_engine(town="prime"):
    """Get SQLAlchemy engine for the specified town/database."""
    db_name = TOWN_TO_DB.get(town, "prime")
    connection_string = f"mysql+pymysql://{DB_CONFIG['username']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{db_name}"
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
    eng = get_engine(db_name)
    return pd.read_sql(query, eng)

# ======================
# 📊 DATA FETCHING FUNCTIONS
# ======================

@st.cache_data(ttl=3600)
def fetch_booker_less_ctn_data(months_back=3, town="prime"):
    """Fetch booker less than half carton data"""
    db_name = TOWN_TO_DB.get(town, "prime")
    eng = get_engine(db_name)
    
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
def fetch_kpi_data(start_date, end_date, town="prime"):
    """Fetch KPI metrics"""
    db_name = TOWN_TO_DB.get(town, "prime")
    
    query = f"""
    SELECT 
        COUNT(DISTINCT o.`Invoice Number`) AS total_orders,
        SUM(o.`Delivered Amount` + o.`Total Discount`) AS total_revenue,
        COUNT(DISTINCT o.`Distributor Name`) AS active_customers,
        AVG(o.`Delivered Amount` + o.`Total Discount`) AS avg_order_value
    FROM ordervsdelivered o
    WHERE o.`Delivery Date` >= '{start_date}' AND o.`Delivery Date` <= '{end_date}'
    AND o.`Delivery Date` IS NOT NULL
    """
    
    return read_sql_cached(query, db_name)

@st.cache_data(ttl=3600)
def fetch_channel_treemap(town="prime"):
    """Fetch 6-month channel treemap data"""
    db_name = TOWN_TO_DB.get(town, "prime")
    
    query = f"""
    SELECT 
        DATE_FORMAT(o.`Delivery Date`, '%%b-%%y') AS period,
        u.`Channel Type` AS channel,
        SUM(o.`Delivered Amount` + o.`Total Discount`) AS nmv
    FROM ordervsdelivered o
    INNER JOIN universe u ON u.`Store Code` = o.`Store Code`
    WHERE o.`Delivery Date` >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    AND o.`Delivery Date` IS NOT NULL
    GROUP BY DATE_FORMAT(o.`Delivery Date`, '%%b-%%y'), u.`Channel Type`
    ORDER BY o.`Delivery Date` DESC
    """
    
    return read_sql_cached(query, db_name)

@st.cache_data(ttl=3600)
def fetch_sunburst_data(town="prime"):
    """Fetch channel-DM sunburst data"""
    db_name = TOWN_TO_DB.get(town, "prime")
    
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
    
    period_order = sorted(df['period'].unique(), key=lambda x: datetime.strptime(x, '%b-%y'))
    
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
    town = st.sidebar.selectbox(
        "Select Location",
        options=["db42280", "db42280"],
        format_func=lambda x: "🏙️ Karachi" if x == "prime" else "🏢 Lahore"
    )
    
    # Main content
    st.title("📊 Bazaar Prime Analytics Dashboard")
    st.markdown("---")
    
    # KPIs
    st.subheader("📈 Key Performance Indicators")
    kpi_data = fetch_kpi_data(start_date, end_date, town)
    
    if not kpi_data.empty:
        row = kpi_data.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📦 Total Orders",
                f"{int(row.get('total_orders', 0)):,}"
            )
        
        with col2:
            st.metric(
                "💰 Total Revenue",
                f"{row.get('total_revenue', 0) / 1_000_000:.2f}M"
            )
        
        with col3:
            st.metric(
                "👥 Active Customers",
                f"{int(row.get('active_customers', 0)):,}"
            )
        
        with col4:
            st.metric(
                "📊 Avg Order Value",
                f"{row.get('avg_order_value', 0) / 1000:.1f}K"
            )
    
    st.markdown("---")
    
    # Treemap Section
    st.subheader("🗺️ 6-Month Channel Performance")
    treemap_data = fetch_channel_treemap(town)
    
    if not treemap_data.empty:
        st.success(f"✅ Loaded {len(treemap_data)} records from database")
        channels = treemap_data['channel'].unique().tolist()
        selected_channels = st.multiselect(
            "Filter Channels",
            options=channels,
            default=channels
        )
        
        treemap_fig = create_treemap(treemap_data, selected_channels)
        st.plotly_chart(treemap_fig, use_container_width=True)
    else:
        st.warning("⚠️ No data available for treemap. Please check database connection.")
    
    st.markdown("---")
    
    # Sunburst Section
    st.subheader("☀️ Channel & Deliveryman Hierarchy")
    sunburst_data = fetch_sunburst_data(town)
    
    if not sunburst_data.empty:
        st.success(f"✅ Loaded {len(sunburst_data)} records from database")
        dms = sunburst_data['dm'].unique().tolist()
        selected_dms = st.multiselect(
            "Filter Deliverymen",
            options=dms,
            default=dms
        )
        
        sunburst_fig = create_sunburst(sunburst_data, selected_dms)
        st.plotly_chart(sunburst_fig, use_container_width=True)
    else:
        st.warning("⚠️ No data available for sunburst chart. Please check database connection.")
    
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
