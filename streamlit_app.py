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
import re
import sys
import subprocess
import time
import signal
from pathlib import Path
from dotenv import load_dotenv
from html import escape

try:
    import tomllib  # Python 3.11+
except Exception:
    tomllib = None

# Load environment variables
load_dotenv()
# st.experimental_fragment
# Page configuration
st.set_page_config(
    page_title="Bazaar Prime Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",

)


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
def _decrypt_secret_if_needed(raw_value, secret_key):
    """Decrypt base64-encoded secret using Fernet key when available."""
    if not raw_value:
        return raw_value
    if not secret_key:
        return raw_value
    try:
        Fernet = __import__("cryptography.fernet", fromlist=["Fernet"]).Fernet
    except Exception:
        return raw_value
    try:
        fernet = Fernet(secret_key.encode("utf-8"))
        decrypted = fernet.decrypt(raw_value.encode("utf-8"))
        return decrypted.decode("utf-8")
    except Exception:
        return raw_value


def load_db_config():
    """Load DB config from Streamlit secrets first, then environment variables."""
    secrets_db = {}
    if "database" in st.secrets:
        secrets_db = st.secrets["database"]

    secret_key = (
        st.secrets.get("DB_SECRET_KEY", "")
        or os.getenv("DB_SECRET_KEY", "")
    )

    username = secrets_db.get("username", os.getenv("DB_USER", ""))
    host = secrets_db.get("host", os.getenv("DB_HOST", ""))
    port = str(secrets_db.get("port", os.getenv("DB_PORT", "3306")))
    database = secrets_db.get("database", os.getenv("DB_NAME", ""))

    password_plain = secrets_db.get("password", os.getenv("DB_PASSWORD", ""))
    password_encrypted = secrets_db.get("password_encrypted", os.getenv("DB_PASSWORD_ENCRYPTED", ""))

    if password_plain:
        password = password_plain
    else:
        password = _decrypt_secret_if_needed(password_encrypted, secret_key)

    return {
        "username": username,
        "password": password,
        "host": host,
        "port": port,
        "database": database,
    }


DB_CONFIG = load_db_config()


@st.cache_resource
def get_engine():
    """Get SQLAlchemy engine for the specified town/database."""
    if not all([DB_CONFIG["username"], DB_CONFIG["password"], DB_CONFIG["host"], DB_CONFIG["database"]]):
        st.error("Database credentials are missing. Configure them in .streamlit/secrets.toml or environment variables.")
        st.stop()
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


@st.cache_data(ttl=900)
def fetch_account_last_update_data():
    """Fetch latest Order Date by distributor account from ordered_vs_delivered_rows."""
    query = """
    SELECT
        `Distributor Code` AS Distributor_Code,
        MAX(`Order Date`) AS Last_Update_At
    FROM ordered_vs_delivered_rows
    WHERE `Distributor Code` IN ('D70002202', 'D70002246')
      AND `Order Date` IS NOT NULL
    GROUP BY `Distributor Code`
    """
    return pd.read_sql(query, get_engine())


@st.cache_data(ttl=1800)
def fetch_revenue_sparkline_series(end_date, town_code):
    """Fetch monthly revenue trend from same month last year to selected end month."""
    end_ts = pd.to_datetime(end_date, errors='coerce')
    if pd.isna(end_ts):
        return []

    end_month_start = pd.Timestamp(end_ts).replace(day=1)
    start_month_start = end_month_start - pd.DateOffset(years=1)

    query = f"""
    SELECT
        DATE_FORMAT(`Delivery Date`, '%%Y-%%m-01') AS Month_Start,
        SUM(`Delivered Amount` + `Total Discount`) AS Revenue
    FROM ordered_vs_delivered_rows
    WHERE `Distributor Code` = '{town_code}'
      AND `Delivery Date` >= '{start_month_start.date()}'
      AND `Delivery Date` < DATE_ADD('{end_month_start.date()}', INTERVAL 1 MONTH)
    GROUP BY DATE_FORMAT(`Delivery Date`, '%%Y-%%m-01')
    ORDER BY Month_Start
    """

    spark_df = read_sql_cached(query, "db42280")
    if spark_df is None or spark_df.empty:
        return []

    spark_df = spark_df.copy()
    spark_df['Month_Start'] = pd.to_datetime(spark_df['Month_Start'], errors='coerce')
    spark_df['Revenue'] = pd.to_numeric(spark_df.get('Revenue', 0), errors='coerce').fillna(0)
    spark_df = spark_df.dropna(subset=['Month_Start'])

    month_range = pd.date_range(start=start_month_start, end=end_month_start, freq='MS')
    month_df = pd.DataFrame({'Month_Start': month_range})
    spark_df = month_df.merge(
        spark_df[['Month_Start', 'Revenue']],
        on='Month_Start',
        how='left'
    )
    spark_df['Revenue'] = pd.to_numeric(spark_df['Revenue'], errors='coerce').fillna(0)
    return spark_df['Revenue'].tolist()


@st.cache_data(ttl=1800)
def fetch_kpi_sparkline_series(end_date, town_code):
    """Fetch monthly sparkline series (Revenue, Litres, Orders, AOV) from same month last year to selected end month."""
    end_ts = pd.to_datetime(end_date, errors='coerce')
    if pd.isna(end_ts):
        return {"revenue": [], "litres": [], "orders": [], "aov": []}

    end_month_start = pd.Timestamp(end_ts).replace(day=1)
    start_month_start = end_month_start - pd.DateOffset(years=1)

    query = f"""
    SELECT
        DATE_FORMAT(`Delivery Date`, '%%Y-%%m-01') AS Month_Start,
        SUM(`Delivered Amount` + `Total Discount`) AS Revenue,
        SUM(CASE WHEN COALESCE(`Delivered (Litres)`, 0) = 0 THEN COALESCE(`Delivered (Kg)`, 0) ELSE COALESCE(`Delivered (Litres)`, 0) END) AS Litres,
        COUNT(DISTINCT `Invoice Number`) AS Orders
    FROM ordered_vs_delivered_rows
    WHERE `Distributor Code` = '{town_code}'
      AND `Delivery Date` >= '{start_month_start.date()}'
      AND `Delivery Date` < DATE_ADD('{end_month_start.date()}', INTERVAL 1 MONTH)
    GROUP BY DATE_FORMAT(`Delivery Date`, '%%Y-%%m-01')
    ORDER BY Month_Start
    """

    spark_df = read_sql_cached(query, "db42280")
    month_range = pd.date_range(start=start_month_start, end=end_month_start, freq='MS')
    month_df = pd.DataFrame({'Month_Start': month_range})

    if spark_df is None or spark_df.empty:
        empty_vals = [0.0] * len(month_df)
        return {"revenue": empty_vals, "litres": empty_vals, "orders": empty_vals, "aov": empty_vals}

    spark_df = spark_df.copy()
    spark_df['Month_Start'] = pd.to_datetime(spark_df['Month_Start'], errors='coerce')
    for col in ['Revenue', 'Litres', 'Orders']:
        spark_df[col] = pd.to_numeric(spark_df.get(col, 0), errors='coerce').fillna(0)
    spark_df = spark_df.dropna(subset=['Month_Start'])

    spark_df = month_df.merge(
        spark_df[['Month_Start', 'Revenue', 'Litres', 'Orders']],
        on='Month_Start',
        how='left'
    )
    for col in ['Revenue', 'Litres', 'Orders']:
        spark_df[col] = pd.to_numeric(spark_df[col], errors='coerce').fillna(0)

    spark_df['AOV'] = np.where(
        spark_df['Orders'] > 0,
        spark_df['Revenue'] / spark_df['Orders'],
        0,
    )

    return {
        "revenue": spark_df['Revenue'].tolist(),
        "litres": spark_df['Litres'].tolist(),
        "orders": spark_df['Orders'].tolist(),
        "aov": spark_df['AOV'].tolist(),
    }


@st.cache_data(ttl=1800)
def fetch_inventory_kpi_data(end_date, town_code):
    """Fetch inventory KPIs using end_stock_summary and sales from ordered_vs_delivered_rows."""
    end_ts = pd.to_datetime(end_date, errors='coerce')
    if pd.isna(end_ts):
        return None

    end_date_str = str(pd.Timestamp(end_ts).date())

    dates_query = f"""
    SELECT
        MAX(CASE WHEN report_date <= '{end_date_str}' THEN report_date END) AS latest_report_date,
        MAX(CASE
                WHEN report_date < (
                    SELECT MAX(report_date)
                    FROM end_stock_summary
                    WHERE distributor_code = '{town_code}'
                      AND report_date <= '{end_date_str}'
                ) THEN report_date
            END) AS prev_report_date
    FROM end_stock_summary
    WHERE distributor_code = '{town_code}'
    """
    dates_df = read_sql_cached(dates_query, "db42280")
    if dates_df is None or dates_df.empty or pd.isna(dates_df.loc[0, 'latest_report_date']):
        return None

    latest_report_date = pd.to_datetime(dates_df.loc[0, 'latest_report_date'], errors='coerce')
    prev_report_date = pd.to_datetime(dates_df.loc[0, 'prev_report_date'], errors='coerce')
    if pd.isna(latest_report_date):
        return None

    def _fetch_stock_snapshot(report_date_value):
        if pd.isna(report_date_value):
            return pd.DataFrame(columns=['sku_code', 'stock_value'])
        report_date_str = str(pd.Timestamp(report_date_value).date())
        q = f"""
        SELECT
            sku_code,
            ROUND(SUM(COALESCE(value, 0)), 4) AS stock_value
        FROM end_stock_summary
        WHERE distributor_code = '{town_code}'
          AND report_date = '{report_date_str}'
          AND LOWER(COALESCE(unit, '')) = 'value'
        GROUP BY sku_code
        """
        snapshot_df = read_sql_cached(q, "db42280")
        if snapshot_df is None:
            return pd.DataFrame(columns=['sku_code', 'stock_value'])
        snapshot_df = snapshot_df.copy()
        snapshot_df['sku_code'] = snapshot_df.get('sku_code', '').astype(str)
        snapshot_df['stock_value'] = pd.to_numeric(snapshot_df.get('stock_value', 0), errors='coerce').fillna(0)
        return snapshot_df

    def _fetch_sales_window(end_window_date, window_days):
        if pd.isna(end_window_date):
            return pd.DataFrame(columns=['sku_code', 'sales_value'])
        end_window_ts = pd.Timestamp(end_window_date).date()
        start_window_ts = (pd.Timestamp(end_window_date) - pd.Timedelta(days=window_days - 1)).date()
        q = f"""
        SELECT
            `SKU Code` AS sku_code,
            ROUND(SUM(COALESCE(`Delivered Amount`, 0) + COALESCE(`Total Discount`, 0)), 4) AS sales_value
        FROM ordered_vs_delivered_rows
        WHERE `Distributor Code` = '{town_code}'
          AND `Delivery Date` BETWEEN '{start_window_ts}' AND '{end_window_ts}'
        GROUP BY `SKU Code`
        """
        sales_df = read_sql_cached(q, "db42280")
        if sales_df is None:
            return pd.DataFrame(columns=['sku_code', 'sales_value'])
        sales_df = sales_df.copy()
        sales_df['sku_code'] = sales_df.get('sku_code', '').astype(str)
        sales_df['sales_value'] = pd.to_numeric(sales_df.get('sales_value', 0), errors='coerce').fillna(0)
        return sales_df

    stock_curr = _fetch_stock_snapshot(latest_report_date)
    stock_prev = _fetch_stock_snapshot(prev_report_date)

    sales_90_curr = _fetch_sales_window(latest_report_date, 90).rename(columns={'sales_value': 'sales_90'})
    sales_90_prev = _fetch_sales_window(prev_report_date, 90).rename(columns={'sales_value': 'sales_90'}) if pd.notna(prev_report_date) else pd.DataFrame(columns=['sku_code', 'sales_90'])
    sales_30_curr = _fetch_sales_window(latest_report_date, 30).rename(columns={'sales_value': 'sales_30'})
    sales_30_prev = _fetch_sales_window(prev_report_date, 30).rename(columns={'sales_value': 'sales_30'}) if pd.notna(prev_report_date) else pd.DataFrame(columns=['sku_code', 'sales_30'])

    def _compute_metrics(stock_df, sales_90_df, sales_30_df):
        if stock_df is None or stock_df.empty:
            return {
                'total_skus': 0,
                'current_inventory_value': 0.0,
                'in_stock_rate': 0.0,
                'oos_skus': 0,
                'avg_days_cover': 0.0,
                'slow_movers': 0,
            }

        merged = stock_df.copy()
        merged = merged.merge(sales_90_df, on='sku_code', how='left') if sales_90_df is not None and not sales_90_df.empty else merged
        merged = merged.merge(sales_30_df, on='sku_code', how='left') if sales_30_df is not None and not sales_30_df.empty else merged
        if 'sales_90' not in merged.columns:
            merged['sales_90'] = 0
        if 'sales_30' not in merged.columns:
            merged['sales_30'] = 0
        merged['sales_90'] = pd.to_numeric(merged['sales_90'], errors='coerce').fillna(0)
        merged['sales_30'] = pd.to_numeric(merged['sales_30'], errors='coerce').fillna(0)
        merged['stock_value'] = pd.to_numeric(merged['stock_value'], errors='coerce').fillna(0)

        total_skus = int(merged['sku_code'].nunique())
        current_inventory_value = float(merged['stock_value'].sum())
        in_stock_skus = int((merged['stock_value'] > 0).sum())
        oos_skus = int((merged['stock_value'] <= 0).sum())
        in_stock_rate = (in_stock_skus / total_skus * 100) if total_skus > 0 else 0.0

        merged['daily_sales_90'] = merged['sales_90'] / 90.0
        merged['days_cover'] = np.where(
            (merged['stock_value'] > 0) & (merged['daily_sales_90'] > 0),
            merged['stock_value'] / merged['daily_sales_90'],
            np.nan,
        )
        valid_cover_df = merged.loc[
            (pd.to_numeric(merged['stock_value'], errors='coerce').fillna(0) > 0)
            & (pd.to_numeric(merged['sales_90'], errors='coerce').fillna(0) > 0),
            ['sku_code', 'sales_90', 'days_cover']
        ].copy()

        if valid_cover_df.empty:
            avg_days_cover = 0.0
        else:
            valid_cover_df['sales_90'] = pd.to_numeric(valid_cover_df['sales_90'], errors='coerce').fillna(0)
            valid_cover_df['days_cover'] = pd.to_numeric(valid_cover_df['days_cover'], errors='coerce').fillna(0)
            total_sales_90 = float(valid_cover_df['sales_90'].sum())
            if total_sales_90 > 0:
                valid_cover_df['sku_contribution'] = valid_cover_df['sales_90'] / total_sales_90
                avg_days_cover = float((valid_cover_df['days_cover'] * valid_cover_df['sku_contribution']).sum())
            else:
                avg_days_cover = 0.0

        slow_movers = int(((merged['stock_value'] > 0) & (merged['sales_30'] <= 0)).sum())

        return {
            'total_skus': total_skus,
            'current_inventory_value': float(current_inventory_value),
            'in_stock_rate': float(in_stock_rate),
            'oos_skus': oos_skus,
            'avg_days_cover': float(avg_days_cover),
            'slow_movers': slow_movers,
        }

    current_metrics = _compute_metrics(stock_curr, sales_90_curr, sales_30_curr)
    prev_metrics = _compute_metrics(stock_prev, sales_90_prev, sales_30_prev) if pd.notna(prev_report_date) else None

    return {
        'latest_report_date': pd.Timestamp(latest_report_date).date(),
        'prev_report_date': (pd.Timestamp(prev_report_date).date() if pd.notna(prev_report_date) else None),
        'current': current_metrics,
        'previous': prev_metrics,
    }


@st.cache_data(ttl=1800)
def fetch_inventory_ytd_month_end_sparkline(end_date, town_code):
    """Fetch YTD month-end inventory values for sparkline in Inventory tab."""
    end_ts = pd.to_datetime(end_date, errors='coerce')
    if pd.isna(end_ts):
        return []

    end_day = pd.Timestamp(end_ts).date()
    year_start = pd.Timestamp(end_ts).replace(month=1, day=1).date()
    end_month_start = pd.Timestamp(end_ts).replace(day=1)

    query = f"""
    SELECT
        m.month_start AS Month_Start,
        ROUND(SUM(COALESCE(s.value, 0)), 4) AS Inventory_Value
    FROM (
        SELECT
            DATE_FORMAT(report_date, '%%Y-%%m-01') AS month_start,
            MAX(report_date) AS month_last_date
        FROM end_stock_summary
        WHERE distributor_code = '{town_code}'
          AND LOWER(COALESCE(unit, '')) = 'value'
          AND report_date BETWEEN '{year_start}' AND '{end_day}'
        GROUP BY DATE_FORMAT(report_date, '%%Y-%%m-01')
    ) m
    INNER JOIN end_stock_summary s
        ON s.report_date = m.month_last_date
       AND s.distributor_code = '{town_code}'
       AND LOWER(COALESCE(s.unit, '')) = 'value'
    GROUP BY m.month_start
    ORDER BY m.month_start
    """

    spark_df = read_sql_cached(query, "db42280")
    month_range = pd.date_range(start=pd.Timestamp(year_start), end=end_month_start, freq='MS')
    month_df = pd.DataFrame({'Month_Start': month_range})

    if spark_df is None or spark_df.empty:
        return [0.0] * len(month_df)

    spark_df = spark_df.copy()
    spark_df['Month_Start'] = pd.to_datetime(spark_df.get('Month_Start'), errors='coerce')
    spark_df['Inventory_Value'] = pd.to_numeric(spark_df.get('Inventory_Value', 0), errors='coerce').fillna(0)
    spark_df = spark_df.dropna(subset=['Month_Start'])

    spark_df = month_df.merge(
        spark_df[['Month_Start', 'Inventory_Value']],
        on='Month_Start',
        how='left'
    )
    spark_df['Inventory_Value'] = pd.to_numeric(spark_df['Inventory_Value'], errors='coerce').fillna(0)
    return spark_df['Inventory_Value'].tolist()


@st.cache_data(ttl=1800)
def fetch_inventory_chart_data(end_date, town_code):
    """Fetch chart data for Inventory tab: stock movement trend and days cover by category."""
    end_ts = pd.to_datetime(end_date, errors='coerce')
    if pd.isna(end_ts):
        return {'trend': pd.DataFrame(), 'days_cover': pd.DataFrame()}

    end_day = pd.Timestamp(end_ts).date()
    start_30 = (pd.Timestamp(end_ts) - pd.Timedelta(days=29)).date()
    start_31 = (pd.Timestamp(end_ts) - pd.Timedelta(days=30)).date()

    stock_daily_q = f"""
    SELECT
        report_date AS Report_Date,
        ROUND(SUM(COALESCE(value, 0)), 4) AS Stock_Value
    FROM end_stock_summary
    WHERE distributor_code = '{town_code}'
      AND LOWER(COALESCE(unit, '')) = 'value'
      AND report_date BETWEEN '{start_31}' AND '{end_day}'
    GROUP BY report_date
    ORDER BY report_date
    """
    stock_daily_df = read_sql_cached(stock_daily_q, "db42280")

    sales_daily_q = f"""
    SELECT
        `Delivery Date` AS Report_Date,
        ROUND(SUM(COALESCE(`Delivered Amount`, 0) + COALESCE(`Total Discount`, 0)), 4) AS Outflow_Value
    FROM ordered_vs_delivered_rows
    WHERE `Distributor Code` = '{town_code}'
      AND `Delivery Date` BETWEEN '{start_30}' AND '{end_day}'
    GROUP BY `Delivery Date`
    ORDER BY `Delivery Date`
    """
    sales_daily_df = read_sql_cached(sales_daily_q, "db42280")

    stock_daily_df = stock_daily_df if stock_daily_df is not None else pd.DataFrame(columns=['Report_Date', 'Stock_Value'])
    sales_daily_df = sales_daily_df if sales_daily_df is not None else pd.DataFrame(columns=['Report_Date', 'Outflow_Value'])

    stock_daily_df = stock_daily_df.copy()
    stock_daily_df['Report_Date'] = pd.to_datetime(stock_daily_df.get('Report_Date'), errors='coerce')
    stock_daily_df['Stock_Value'] = pd.to_numeric(stock_daily_df.get('Stock_Value', 0), errors='coerce').fillna(0)
    stock_daily_df = stock_daily_df.dropna(subset=['Report_Date'])

    full_days_31 = pd.date_range(start=pd.Timestamp(start_31), end=pd.Timestamp(end_day), freq='D')
    stock_daily_df = pd.DataFrame({'Report_Date': full_days_31}).merge(stock_daily_df, on='Report_Date', how='left')
    stock_daily_df['Stock_Value'] = pd.to_numeric(stock_daily_df['Stock_Value'], errors='coerce').fillna(0)
    stock_daily_df['Prev_Stock'] = stock_daily_df['Stock_Value'].shift(1)
    stock_daily_df['Prev_Stock'] = pd.to_numeric(stock_daily_df['Prev_Stock'], errors='coerce').fillna(stock_daily_df['Stock_Value'])
    stock_daily_df['Inflow_Value'] = (stock_daily_df['Stock_Value'] - stock_daily_df['Prev_Stock']).clip(lower=0)
    trend_df = stock_daily_df[stock_daily_df['Report_Date'] >= pd.Timestamp(start_30)][['Report_Date', 'Inflow_Value']].copy()

    sales_daily_df = sales_daily_df.copy()
    sales_daily_df['Report_Date'] = pd.to_datetime(sales_daily_df.get('Report_Date'), errors='coerce')
    sales_daily_df['Outflow_Value'] = pd.to_numeric(sales_daily_df.get('Outflow_Value', 0), errors='coerce').fillna(0)
    sales_daily_df = sales_daily_df.dropna(subset=['Report_Date'])
    full_days_30 = pd.date_range(start=pd.Timestamp(start_30), end=pd.Timestamp(end_day), freq='D')
    sales_daily_df = pd.DataFrame({'Report_Date': full_days_30}).merge(sales_daily_df, on='Report_Date', how='left')
    sales_daily_df['Outflow_Value'] = pd.to_numeric(sales_daily_df['Outflow_Value'], errors='coerce').fillna(0)

    trend_df = trend_df.merge(sales_daily_df[['Report_Date', 'Outflow_Value']], on='Report_Date', how='left')
    trend_df['Outflow_Value'] = pd.to_numeric(trend_df['Outflow_Value'], errors='coerce').fillna(0)

    latest_stock_date_q = f"""
    SELECT MAX(report_date) AS latest_report_date
    FROM end_stock_summary
    WHERE distributor_code = '{town_code}'
      AND report_date <= '{end_day}'
    """
    latest_stock_date_df = read_sql_cached(latest_stock_date_q, "db42280")
    latest_stock_date = None
    if latest_stock_date_df is not None and not latest_stock_date_df.empty:
        latest_stock_date = pd.to_datetime(latest_stock_date_df.loc[0, 'latest_report_date'], errors='coerce')

    if pd.isna(latest_stock_date):
        return {'trend': trend_df, 'days_cover': pd.DataFrame()}

    latest_stock_str = str(pd.Timestamp(latest_stock_date).date())
    days_cover_q = f"""
    WITH stock_sku AS (
        SELECT
            sku_code,
            ROUND(SUM(COALESCE(value, 0)), 4) AS stock_value
        FROM end_stock_summary
        WHERE distributor_code = '{town_code}'
          AND report_date = '{latest_stock_str}'
          AND LOWER(COALESCE(unit, '')) = 'value'
        GROUP BY sku_code
    ),
    sales_30_sku AS (
        SELECT
            `SKU Code` AS sku_code,
            ROUND(SUM(COALESCE(`Delivered Amount`, 0) + COALESCE(`Total Discount`, 0)), 4) AS sales_30
        FROM ordered_vs_delivered_rows
        WHERE `Distributor Code` = '{town_code}'
          AND `Delivery Date` BETWEEN '{start_30}' AND '{end_day}'
        GROUP BY `SKU Code`
    ),
    latest_sku_date AS (
        SELECT
            `SKU Code` AS sku_code,
            MAX(`Delivery Date`) AS max_delivery_date
        FROM ordered_vs_delivered_rows
        WHERE `Distributor Code` = '{town_code}'
        GROUP BY `SKU Code`
    ),
    sku_category AS (
        SELECT
            o.`SKU Code` AS sku_code,
            COALESCE(NULLIF(TRIM(o.`Category Name`), ''), 'Unknown') AS category_name
        FROM ordered_vs_delivered_rows o
        INNER JOIN latest_sku_date d
            ON d.sku_code = o.`SKU Code`
           AND d.max_delivery_date = o.`Delivery Date`
        WHERE o.`Distributor Code` = '{town_code}'
        GROUP BY o.`SKU Code`, COALESCE(NULLIF(TRIM(o.`Category Name`), ''), 'Unknown')
    )
    SELECT
        COALESCE(c.category_name, 'Unknown') AS Category,
        ROUND(SUM(COALESCE(s.stock_value, 0)), 4) AS Stock_Value,
        ROUND(SUM(COALESCE(x.sales_30, 0)), 4) AS Sales_30,
        ROUND(
            CASE
                WHEN SUM(COALESCE(x.sales_30, 0)) > 0 THEN SUM(COALESCE(s.stock_value, 0)) / (SUM(COALESCE(x.sales_30, 0)) / 30.0)
                ELSE NULL
            END,
            2
        ) AS Days_Cover
    FROM stock_sku s
    LEFT JOIN sales_30_sku x ON x.sku_code = s.sku_code
    LEFT JOIN sku_category c ON c.sku_code = s.sku_code
    GROUP BY COALESCE(c.category_name, 'Unknown')
    ORDER BY Days_Cover DESC
    """
    days_cover_df = read_sql_cached(days_cover_q, "db42280")
    if days_cover_df is None:
        days_cover_df = pd.DataFrame(columns=['Category', 'Stock_Value', 'Sales_30', 'Days_Cover'])

    days_cover_df = days_cover_df.copy()
    days_cover_df['Category'] = days_cover_df.get('Category', '').astype(str)
    days_cover_df['Days_Cover'] = pd.to_numeric(days_cover_df.get('Days_Cover', 0), errors='coerce').fillna(0)

    return {'trend': trend_df, 'days_cover': days_cover_df}


@st.cache_data(ttl=1800)
def fetch_inventory_cover_bucket_matrix(end_date, town_code, bucket_thresholds=(7, 15, 30, 45, 60)):
    """Build Salesflo-style stock cover day bucket matrix for inventory value."""
    end_ts = pd.to_datetime(end_date, errors='coerce')
    if pd.isna(end_ts):
        return {'table': pd.DataFrame(), 'percentages': {}}

    end_day = pd.Timestamp(end_ts).date()
    start_90 = (pd.Timestamp(end_ts) - pd.Timedelta(days=89)).date()

    distributor_filter = ""
    if town_code and str(town_code).lower() != 'all':
        distributor_filter = f" AND distributor_code = '{town_code}' "

    sales_distributor_filter = ""
    if town_code and str(town_code).lower() != 'all':
        sales_distributor_filter = f" AND `Distributor Code` = '{town_code}' "

    query = f"""
    WITH latest_date AS (
        SELECT distributor_code, MAX(report_date) AS latest_report_date
        FROM end_stock_summary
        WHERE report_date <= '{end_day}'
          {distributor_filter}
        GROUP BY distributor_code
    ),
    stock_sku AS (
        SELECT
            s.distributor_code,
            s.sku_code,
            ROUND(SUM(COALESCE(s.value, 0)), 4) AS stock_value
        FROM end_stock_summary s
        INNER JOIN latest_date d
            ON d.distributor_code = s.distributor_code
           AND d.latest_report_date = s.report_date
        WHERE LOWER(COALESCE(s.unit, '')) = 'value'
        GROUP BY s.distributor_code, s.sku_code
    ),
    sales_90_sku AS (
        SELECT
            `Distributor Code` AS distributor_code,
            `SKU Code` AS sku_code,
            ROUND(SUM(COALESCE(`Delivered Amount`, 0) + COALESCE(`Total Discount`, 0)), 4) AS sales_90
        FROM ordered_vs_delivered_rows
        WHERE `Delivery Date` BETWEEN '{start_90}' AND '{end_day}'
          {sales_distributor_filter}
        GROUP BY `Distributor Code`, `SKU Code`
    )
    SELECT
        s.distributor_code,
        s.sku_code,
        s.stock_value,
        COALESCE(x.sales_90, 0) AS sales_90
    FROM stock_sku s
    LEFT JOIN sales_90_sku x
      ON x.distributor_code = s.distributor_code
     AND x.sku_code = s.sku_code
    """

    base_df = read_sql_cached(query, "db42280")
    if base_df is None or base_df.empty:
        return {'table': pd.DataFrame(), 'percentages': {}}

    df = base_df.copy()
    df['distributor_code'] = df.get('distributor_code', '').astype(str)
    df['stock_value'] = pd.to_numeric(df.get('stock_value', 0), errors='coerce').fillna(0)
    df['sales_90'] = pd.to_numeric(df.get('sales_90', 0), errors='coerce').fillna(0)

    df = df[df['stock_value'] > 0].copy()
    if df.empty:
        return {'table': pd.DataFrame(), 'percentages': {}}

    df['cover_days'] = np.where(df['sales_90'] > 0, df['stock_value'] / (df['sales_90'] / 90.0), np.inf)

    account_map = {
        'D70002202': 'Karachi',
        'D70002246': 'Lahore',
    }
    df['Account'] = df['distributor_code'].map(account_map).fillna(df['distributor_code'])

    try:
        t1, t2, t3, t4, t5 = [float(x) for x in bucket_thresholds]
    except Exception:
        t1, t2, t3, t4, t5 = 7.0, 15.0, 30.0, 45.0, 60.0
    thresholds_sorted = sorted([t1, t2, t3, t4, t5])
    t1, t2, t3, t4, t5 = thresholds_sorted

    bucket_columns = [
        f'Inventory <{int(t1)}',
        f'Inventory {int(t1)}-{int(t2)}',
        f'Inventory {int(t2)}-{int(t3)}',
        f'Inventory {int(t3)}-{int(t4)}',
        f'Inventory {int(t4)}-{int(t5)}',
        f'Inventory {int(t5)}+',
        'Zero Sale',
    ]

    for col in bucket_columns:
        df[col] = 0.0

    df.loc[(df['sales_90'] > 0) & (df['cover_days'] < t1), bucket_columns[0]] = df['stock_value']
    df.loc[(df['sales_90'] > 0) & (df['cover_days'] >= t1) & (df['cover_days'] < t2), bucket_columns[1]] = df['stock_value']
    df.loc[(df['sales_90'] > 0) & (df['cover_days'] >= t2) & (df['cover_days'] < t3), bucket_columns[2]] = df['stock_value']
    df.loc[(df['sales_90'] > 0) & (df['cover_days'] >= t3) & (df['cover_days'] < t4), bucket_columns[3]] = df['stock_value']
    df.loc[(df['sales_90'] > 0) & (df['cover_days'] >= t4) & (df['cover_days'] < t5), bucket_columns[4]] = df['stock_value']
    df.loc[(df['sales_90'] > 0) & (df['cover_days'] >= t5), bucket_columns[5]] = df['stock_value']
    df.loc[df['sales_90'] <= 0, 'Zero Sale'] = df['stock_value']

    agg_cols = ['stock_value'] + bucket_columns
    grouped = df.groupby('Account', as_index=False)[agg_cols].sum()
    grouped = grouped.rename(columns={'stock_value': 'Total Inventory'})

    table_df = grouped.copy()

    grand_total = float(grouped['Total Inventory'].sum()) if not grouped.empty else 0.0
    percentages = {}
    for col in bucket_columns:
        col_total = float(grouped[col].sum()) if not grouped.empty else 0.0
        percentages[col] = (col_total / grand_total * 100.0) if grand_total > 0 else 0.0

    return {'table': table_df, 'percentages': percentages, 'bucket_columns': bucket_columns}


@st.cache_data(ttl=1800)
def fetch_inventory_health_data(end_date, town_code):
    """Compute inventory health metrics for Inventory tab."""
    end_ts = pd.to_datetime(end_date, errors='coerce')
    if pd.isna(end_ts):
        return None

    end_day = pd.Timestamp(end_ts).date()
    start_30 = (pd.Timestamp(end_ts) - pd.Timedelta(days=29)).date()
    start_90 = (pd.Timestamp(end_ts) - pd.Timedelta(days=89)).date()

    latest_date_q = f"""
    SELECT MAX(report_date) AS latest_report_date
    FROM end_stock_summary
    WHERE distributor_code = '{town_code}'
      AND report_date <= '{end_day}'
    """
    latest_date_df = read_sql_cached(latest_date_q, "db42280")
    if latest_date_df is None or latest_date_df.empty or pd.isna(latest_date_df.loc[0, 'latest_report_date']):
        return None

    latest_report_date = pd.to_datetime(latest_date_df.loc[0, 'latest_report_date'], errors='coerce')
    if pd.isna(latest_report_date):
        return None
    latest_report_day = pd.Timestamp(latest_report_date).date()

    stock_daily_q = f"""
    SELECT
        report_date AS Report_Date,
        ROUND(SUM(COALESCE(value, 0)), 4) AS Stock_Value
    FROM end_stock_summary
    WHERE distributor_code = '{town_code}'
      AND LOWER(COALESCE(unit, '')) = 'value'
      AND report_date BETWEEN '{start_90}' AND '{latest_report_day}'
    GROUP BY report_date
    ORDER BY report_date
    """
    stock_daily_df = read_sql_cached(stock_daily_q, "db42280")
    if stock_daily_df is None:
        stock_daily_df = pd.DataFrame(columns=['Report_Date', 'Stock_Value'])

    stock_daily_df = stock_daily_df.copy()
    stock_daily_df['Report_Date'] = pd.to_datetime(stock_daily_df.get('Report_Date'), errors='coerce')
    stock_daily_df['Stock_Value'] = pd.to_numeric(stock_daily_df.get('Stock_Value', 0), errors='coerce').fillna(0)
    stock_daily_df = stock_daily_df.dropna(subset=['Report_Date']).sort_values('Report_Date')

    avg_stock_90 = float(stock_daily_df['Stock_Value'].mean()) if not stock_daily_df.empty else 0.0
    avg_stock_30 = float(
        stock_daily_df.loc[stock_daily_df['Report_Date'] >= pd.Timestamp(start_30), 'Stock_Value'].mean()
    ) if not stock_daily_df.empty else 0.0

    sales_q = f"""
    SELECT
        ROUND(SUM(CASE WHEN `Delivery Date` BETWEEN '{start_90}' AND '{end_day}'
            THEN COALESCE(`Delivered Amount`, 0) + COALESCE(`Total Discount`, 0) ELSE 0 END), 4) AS Sales_90,
        ROUND(SUM(CASE WHEN `Delivery Date` BETWEEN '{start_30}' AND '{end_day}'
            THEN COALESCE(`Delivered Amount`, 0) + COALESCE(`Total Discount`, 0) ELSE 0 END), 4) AS Sales_30
    FROM ordered_vs_delivered_rows
    WHERE `Distributor Code` = '{town_code}'
    """
    sales_df = read_sql_cached(sales_q, "db42280")
    sales_90 = 0.0
    sales_30 = 0.0
    if sales_df is not None and not sales_df.empty:
        sales_90 = float(pd.to_numeric(sales_df.loc[0, 'Sales_90'], errors='coerce') or 0.0)
        sales_30 = float(pd.to_numeric(sales_df.loc[0, 'Sales_30'], errors='coerce') or 0.0)

    sku_stock_q = f"""
    SELECT
        sku_code,
        ROUND(SUM(COALESCE(value, 0)), 4) AS stock_value
    FROM end_stock_summary
    WHERE distributor_code = '{town_code}'
      AND report_date = '{latest_report_day}'
      AND LOWER(COALESCE(unit, '')) = 'value'
    GROUP BY sku_code
    """
    sku_sales_90_q = f"""
    SELECT
        `SKU Code` AS sku_code,
        ROUND(SUM(COALESCE(`Delivered Amount`, 0) + COALESCE(`Total Discount`, 0)), 4) AS sales_90
    FROM ordered_vs_delivered_rows
    WHERE `Distributor Code` = '{town_code}'
      AND `Delivery Date` BETWEEN '{start_90}' AND '{end_day}'
    GROUP BY `SKU Code`
    """
    sku_sales_30_q = f"""
    SELECT
        `SKU Code` AS sku_code,
        ROUND(SUM(COALESCE(`Delivered Amount`, 0) + COALESCE(`Total Discount`, 0)), 4) AS sales_30
    FROM ordered_vs_delivered_rows
    WHERE `Distributor Code` = '{town_code}'
      AND `Delivery Date` BETWEEN '{start_30}' AND '{end_day}'
    GROUP BY `SKU Code`
    """
    sku_name_map_q = f"""
    SELECT
        `SKU Code` AS sku_code,
        MAX(`SKU Name`) AS sku_name
    FROM ordered_vs_delivered_rows
    WHERE `Distributor Code` = '{town_code}'
      AND `Delivery Date` BETWEEN '{start_90}' AND '{end_day}'
      AND `SKU Name` IS NOT NULL
      AND TRIM(`SKU Name`) <> ''
    GROUP BY `SKU Code`
    """

    sku_stock_df = read_sql_cached(sku_stock_q, "db42280")
    sku_sales_90_df = read_sql_cached(sku_sales_90_q, "db42280")
    sku_sales_30_df = read_sql_cached(sku_sales_30_q, "db42280")
    try:
        sku_name_map_df = read_sql_cached(sku_name_map_q, "db42280")
    except Exception:
        sku_name_map_df = pd.DataFrame(columns=['sku_code', 'sku_name'])

    if sku_stock_df is None:
        sku_stock_df = pd.DataFrame(columns=['sku_code', 'stock_value'])
    if sku_sales_90_df is None:
        sku_sales_90_df = pd.DataFrame(columns=['sku_code', 'sales_90'])
    if sku_sales_30_df is None:
        sku_sales_30_df = pd.DataFrame(columns=['sku_code', 'sales_30'])
    if sku_name_map_df is None:
        sku_name_map_df = pd.DataFrame(columns=['sku_code', 'sku_name'])

    sku_stock_df = sku_stock_df.copy()
    sku_stock_df['sku_code'] = sku_stock_df.get('sku_code', '').astype(str)
    sku_stock_df['stock_value'] = pd.to_numeric(sku_stock_df.get('stock_value', 0), errors='coerce').fillna(0)

    sku_sales_90_df = sku_sales_90_df.copy()
    sku_sales_90_df['sku_code'] = sku_sales_90_df.get('sku_code', '').astype(str)
    sku_sales_90_df['sales_90'] = pd.to_numeric(sku_sales_90_df.get('sales_90', 0), errors='coerce').fillna(0)

    sku_sales_30_df = sku_sales_30_df.copy()
    sku_sales_30_df['sku_code'] = sku_sales_30_df.get('sku_code', '').astype(str)
    sku_sales_30_df['sales_30'] = pd.to_numeric(sku_sales_30_df.get('sales_30', 0), errors='coerce').fillna(0)

    sku_name_map_df = sku_name_map_df.copy()
    sku_name_map_df['sku_code'] = sku_name_map_df.get('sku_code', '').astype(str)
    sku_name_map_df['sku_name'] = sku_name_map_df.get('sku_name', '').astype(str)
    sku_name_map_df['sku_name'] = sku_name_map_df['sku_name'].str.strip()
    sku_name_map_df = sku_name_map_df[sku_name_map_df['sku_name'] != '']
    sku_name_map_df = sku_name_map_df.drop_duplicates(subset=['sku_code'], keep='first')

    merged = sku_stock_df.merge(sku_sales_90_df, on='sku_code', how='left').merge(sku_sales_30_df, on='sku_code', how='left')
    merged['sales_90'] = pd.to_numeric(merged.get('sales_90', 0), errors='coerce').fillna(0)
    merged['sales_30'] = pd.to_numeric(merged.get('sales_30', 0), errors='coerce').fillna(0)

    total_stock_value = float(merged['stock_value'].sum()) if not merged.empty else 0.0
    total_sales_90_for_contribution = float(merged['sales_90'].sum()) if not merged.empty else 0.0

    par_target_cover_days = 14.0
    if merged.empty:
        par_level_inventory_value = 0.0
    else:
        merged['daily_sales_90'] = pd.to_numeric(merged['sales_90'], errors='coerce').fillna(0) / 90.0
        par_level_inventory_value = float((merged['daily_sales_90'] * par_target_cover_days).sum())

    if merged.empty:
        dead_stock_value = 0.0
        dead_sku_count = 0
        dead_sku_details = pd.DataFrame(columns=['SKU Name', 'Stock Value', 'Sales 90D', 'Sales Contribution %', 'Cover Days'])
    else:
        if total_sales_90_for_contribution > 0:
            merged['sales_contribution'] = merged['sales_90'] / total_sales_90_for_contribution
        else:
            merged['sales_contribution'] = 0.0

        merged['cover_days_90'] = np.where(
            (merged['stock_value'] > 0) & (merged['sales_90'] > 0),
            merged['stock_value'] / (merged['sales_90'] / 90.0),
            np.where(merged['stock_value'] > 0, np.inf, np.nan)
        )

        dead_mask = (
            (merged['stock_value'] > 0)
            & (merged['sales_contribution'] < 0.02)
            & (merged['cover_days_90'] > 25)
        )
        dead_stock_value = float(merged.loc[dead_mask, 'stock_value'].sum())
        dead_sku_count = int(merged.loc[dead_mask, 'sku_code'].nunique())
        dead_sku_details = merged.loc[dead_mask, ['sku_code', 'stock_value', 'sales_90', 'sales_contribution', 'cover_days_90']].copy()
        dead_sku_details = dead_sku_details.merge(sku_name_map_df[['sku_code', 'sku_name']], on='sku_code', how='left')
        dead_sku_details['sku_name'] = dead_sku_details['sku_name'].fillna(dead_sku_details['sku_code'])
        dead_sku_details = dead_sku_details.rename(columns={
            'sku_name': 'SKU Name',
            'stock_value': 'Stock Value',
            'sales_90': 'Sales 90D',
            'sales_contribution': 'Sales Contribution %',
            'cover_days_90': 'Cover Days',
        })
        if 'sku_code' in dead_sku_details.columns:
            dead_sku_details = dead_sku_details.drop(columns=['sku_code'])
        dead_sku_details['Sales Contribution %'] = pd.to_numeric(dead_sku_details['Sales Contribution %'], errors='coerce').fillna(0) * 100.0
        dead_sku_details['Cover Days'] = pd.to_numeric(dead_sku_details['Cover Days'], errors='coerce')
        dead_sku_details['Cover Days'] = np.where(np.isinf(dead_sku_details['Cover Days']), np.nan, dead_sku_details['Cover Days'])
        dead_sku_details = dead_sku_details.sort_values(['Stock Value', 'Cover Days'], ascending=[False, False])

    wastage_rate = (dead_stock_value / total_stock_value * 100.0) if total_stock_value > 0 else 0.0

    inventory_turnover = (sales_90 / avg_stock_90) if avg_stock_90 > 0 else 0.0
    gmroii = (sales_30 / avg_stock_30) if avg_stock_30 > 0 else 0.0
    safety_stock_coverage = (total_stock_value / (sales_30 / 30.0)) if sales_30 > 0 else 0.0

    inflow_cycle_days = 0.0
    if not stock_daily_df.empty:
        stock_daily_df['Prev_Stock'] = stock_daily_df['Stock_Value'].shift(1)
        stock_daily_df['Prev_Stock'] = pd.to_numeric(stock_daily_df['Prev_Stock'], errors='coerce').fillna(stock_daily_df['Stock_Value'])
        stock_daily_df['Inflow'] = (stock_daily_df['Stock_Value'] - stock_daily_df['Prev_Stock']).clip(lower=0)
        inflow_days = stock_daily_df.loc[stock_daily_df['Inflow'] > 0, 'Report_Date'].dt.normalize().drop_duplicates().sort_values()
        if len(inflow_days) >= 2:
            inflow_cycle_days = float(np.diff(inflow_days.values).astype('timedelta64[D]').astype(int).mean())

    return {
        'current_inventory_value': float(total_stock_value),
        'par_level_inventory_value': float(par_level_inventory_value),
        'par_target_cover_days': float(par_target_cover_days),
        'inventory_turnover': float(inventory_turnover),
        'dead_stock_value': float(dead_stock_value),
        'dead_sku_count': int(dead_sku_count),
        'dead_sku_details': dead_sku_details,
        'avg_replenishment_cycle': float(inflow_cycle_days),
        'wastage_rate': float(wastage_rate),
        'gmroii': float(gmroii),
        'safety_stock_coverage': float(safety_stock_coverage),
    }


@st.cache_data(ttl=3600)
def fetch_table_structure_data():
    """Fetch table and column metadata for current database."""
    db_name = DB_CONFIG.get("database", "db42280")
    safe_db_name = str(db_name).replace("'", "''")
    schema_query = f"""
    SELECT
        TABLE_NAME AS Table_Name,
        COLUMN_NAME AS Column_Name,
        DATA_TYPE AS Data_Type
    FROM information_schema.columns
    WHERE table_schema = '{safe_db_name}'
    ORDER BY TABLE_NAME, ORDINAL_POSITION
    """
    return pd.read_sql(schema_query, get_engine())

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
        (SELECT * FROM ordered_vs_delivered_rows WHERE `Delivered Units` > 0) o
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
)"""
    booker_less_ctn_query = f"""
{booker_less_ctn_base_cte}
SELECT
    brand,
    Booker_Name,
    SUM(Total_Deliveries) AS T_Del,
    SUM(HalfCtnDel) AS T_H_C_Del,
    (SUM(HalfCtnDel) / SUM(Total_Deliveries)) AS age
FROM final
GROUP BY Booker_Name, brand
HAVING T_H_C_Del > 0
ORDER BY Booker_Name, brand, (SUM(HalfCtnDel) / SUM(Total_Deliveries)) DESC
"""

    booker_less_ctn_df = pd.read_sql(booker_less_ctn_query, eng)
    booker_less_ctn_detail_query = f"""
{booker_less_ctn_base_cte}
SELECT
    brand,
    StoreCode,
    StoreName,
    Booker_Name,
    SUM(Total_Deliveries) AS Total_Deliveries,
    SUM(HalfCtnDel) AS HalfCtnDel,
    (SUM(HalfCtnDel) / SUM(Total_Deliveries)) AS age
FROM final
WHERE HalfCtnDel > 0
GROUP BY brand, StoreCode, StoreName, Booker_Name
ORDER BY Booker_Name, brand, age DESC
"""

    detail_df = pd.read_sql(booker_less_ctn_detail_query, eng)
    
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
        Case when u.`Channel Type` is null then "Channel Empty"  else u.`Channel Type` end Channel,
        SUM(o.`Delivered Amount` + o.`Total Discount`) AS Amount,
        o.`Delivery Date` AS D_Date,
				SUM(o.`Delivered (Litres)`) AS Ltr,
				Case when o.`Distributor Code`='D70002202' then 'Karachi'
				when o.`Distributor Code`='D70002246' then 'Lahore' else 'CBL' end Town,
				`Invoice Number` as Orders
    FROM
        ordered_vs_delivered_rows o
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
        Case when u.`Channel Type` is null then "Channel Empty"  else u.`Channel Type` end channel,
        SUM(o.`Delivered Amount` + o.`Total Discount`) AS nmv
    FROM ordered_vs_delivered_rows o
    INNER JOIN universe u ON u.`Store Code` = o.`Store Code`
    WHERE o.`Delivery Date` >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    AND o.`Delivery Date` IS NOT NULL
    GROUP BY period, channel
    ORDER BY o.`Delivery Date` DESC
    """
    
    return read_sql_cached(query)

@st.cache_data(ttl=3600)
def fetch_Channel_dm_sunburst_data(start, end, town_code):
    
    """Fetch channel-DM sunburst data"""
    db_name = "db42280"
    
    query = f"""
    SELECT
    Case when u.`Channel Type` is null then "Channel Empty"  else u.`Channel Type` end Channel,
		s.Brand,
		o.DM,
		count(DISTINCT o.`Store Code`) as StoreCount,
        town	
FROM
	(SELECT DISTINCT `Deliveryman Name` as DM ,`Store Code`,`SKU Code`,`Delivery Date`,Case when `Distributor Code`='D70002202' then 'Karachi'
				when `Distributor Code`='D70002246' then 'Lahore' else 'CBL' end Town from ordered_vs_delivered_rows where `Distributor Code` = '{town_code}') o
	LEFT JOIN sku_master s ON s.Sku_Code = o.`SKU Code` 
	LEFT JOIN universe u on u.`Store Code`=o.`Store Code`
	where `Delivery Date` BETWEEN '{start}' AND '{end}'
	GROUP BY Channel,s.Brand,o.dm,o.town"""
    
    return read_sql_cached(query, db_name)

@st.cache_data(ttl=3600)
def Channelwise_performance_data(start, end, town_code):
    """Fetch channel-wise performance data for comparison charts"""
    db_name = "db42280"
    
    query = f"""
    WITH raw AS (
    SELECT
        Case when u.`Channel Type` is null then "Channel Empty"  else u.`Channel Type` end Channel,
        SUM(o.`Delivered Amount` + o.`Total Discount`) AS Amount,
        o.`Delivery Date` AS D_Date,
				SUM(o.`Delivered (Litres)`) AS Ltr,
				Case when o.`Distributor Code`='D70002202' then 'Karachi'
				when o.`Distributor Code`='D70002246' then 'Lahore' else 'CBL' end Town,
				`Invoice Number` as Orders
    FROM
        ordered_vs_delivered_rows o
    LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
    where o.`Distributor Code`= '{town_code}'
    GROUP BY
        Channel,
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
        raw.D_Date BETWEEN '{start}' AND '{end}'
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
        raw.D_Date BETWEEN DATE_SUB('{start}', INTERVAL 1 YEAR) AND DATE_SUB('{end}', INTERVAL 1 YEAR)
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
        raw.D_Date BETWEEN DATE_SUB('{start}', INTERVAL 1 month) AND DATE_SUB('{end}', INTERVAL 1 month)
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

		
ORDER BY
    sp.Channel;
    """
    
    return read_sql_cached(query, db_name)

@st.cache_data(ttl=3600)
def fetch_brand_growth_channel_options(start, end, town_code):
    """Fetch available channel options for Brand-wise Growth section."""
    query = f"""
    SELECT DISTINCT CASE WHEN u.`Channel Type` IS NULL THEN 'Channel Empty' ELSE u.`Channel Type` END AS Channel
    FROM ordered_vs_delivered_rows o
    LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
    WHERE o.`Distributor Code` = '{town_code}'
      AND o.`Delivery Date` BETWEEN '{start}' AND '{end}'
      AND u.`Channel Type` IS NOT NULL
      AND TRIM(u.`Channel Type`) <> ''
    ORDER BY Channel
    """
    return read_sql_cached(query, "db42280")


@st.cache_data(ttl=3600)
def Brand_wise_performance_growth_data(start, end, town_code, selected_channel='All'):
    """Fetch brand-wise performance data for comparison charts"""
    db_name = "db42280"

    channel_condition = ""
    if selected_channel and selected_channel != 'All':
        selected_channel_safe = str(selected_channel).replace("'", "''")
        channel_condition = f"AND u.`Channel Type` = '{selected_channel_safe}'"
    
    query = f"""
 WITH raw AS (
    SELECT
                CASE WHEN u.`Channel Type` IS NULL OR TRIM(u.`Channel Type`) = '' THEN 'Channel Empty' ELSE u.`Channel Type` END AS Channel,
        s.brand AS Brand,
        SUM(o.`Delivered Amount` + o.`Total Discount`) AS Amount,
        o.`Delivery Date` AS D_Date,
				case when SUM(o.`Delivered (Litres)`)=0 then SUM(o.`Delivered (Kg)`) else SUM(o.`Delivered (Litres)`) END Ltr,
				Case when o.`Distributor Code`='D70002202' then 'Karachi' else 'Lahore' end Town
    FROM
        ordered_vs_delivered_rows o
    LEFT JOIN sku_master s ON s.sku_code= o.`sku code`
    left join universe u on u.`Store Code`=o.`Store Code`
    where o.`Distributor Code`= '{town_code}'
    {channel_condition}
    GROUP BY
        Channel,
        s.brand,
        o.`Delivery Date`,o.`Distributor Code`
)

, selected_period AS (
    SELECT
		Town,
                raw.Channel,
        raw.brand,
        SUM(raw.Amount) AS NMV,
				SUM(ltr) as Ltr
    FROM
        raw
    WHERE
        raw.D_Date BETWEEN '{start}' AND '{end}'
    GROUP BY
                raw.Channel,
        raw.brand,
				Town
)
, last_year_period AS (
    SELECT
		town,
                raw.Channel,
        raw.brand,
        SUM(raw.Amount) AS NMV,
				sum(ltr) as Ltr
    FROM
        raw
    WHERE
        raw.D_Date BETWEEN DATE_SUB('{start}', INTERVAL 1 YEAR) AND DATE_SUB('{end}', INTERVAL 1 YEAR)
    GROUP BY
                raw.Channel,raw.brand,town
),
last_month AS (
    SELECT
		town,
                raw.Channel,
        raw.brand,
        SUM(raw.Amount) AS NMV,
				sum(ltr) as Ltr
    FROM
        raw
    WHERE
        raw.D_Date BETWEEN DATE_SUB('{start}', INTERVAL 1 month) AND DATE_SUB('{end}', INTERVAL 1 month)
    GROUP BY
                raw.Channel,raw.brand,town
)

SELECT
		sp.town,
        sp.Channel,
    sp.brand,
    sp.NMV AS Current_Period_Sales,
    ly.NMV AS Last_Year_Sales,
		lm.NMV AS Last_Month_Sales,
		sp.ltr as Current_Period_Ltr,
		ly.ltr as Last_Year_Ltr,
		lm.ltr as Last_Month_Ltr,
        ROUND(((sp.nmv/ly.nmv)-1)*100) as Sales_Growth_LY,
		Round(((sp.nmv/lm.nmv)-1)*100) as Sales_Growth_LM,
		Round(((sp.ltr/ly.ltr)-1)*100) as Ltr_Growth_LY,
		Round(((sp.ltr/lm.ltr)-1)*100) as Ltr_Growth_LM
FROM
    selected_period sp
LEFT JOIN
    last_year_period ly ON sp.brand = ly.brand AND sp.Channel = ly.Channel AND sp.town = ly.town
LEFT JOIN
    last_month lm ON sp.brand = lm.brand AND sp.Channel = lm.Channel AND sp.town = lm.town

		
ORDER BY
    sp.nmv DESC;

"""

    return read_sql_cached(query, db_name)

@st.cache_data(ttl=3600)
def dm_wise_performance_growth_data(start, end, town_code):
    """Fetch DM-wise performance data for comparison charts"""
    db_name = "db42280"
    
    query = f"""
    WITH raw AS (
    SELECT
        o.`Deliveryman Name` AS DeliveryMan,
        SUM(o.`Delivered Amount` + o.`Total Discount`) AS Amount,
        o.`Delivery Date` AS D_Date,
				case when SUM(o.`Delivered (Litres)`)=0 then SUM(o.`Delivered (Kg)`) else SUM(o.`Delivered (Litres)`) END Ltr,
                Case when o.`Distributor Code`='D70002202' then 'Karachi'
				when o.`Distributor Code`='D70002246' then 'Lahore' else 'CBL' end Town
    FROM
        ordered_vs_delivered_rows o
    LEFT JOIN sku_master s ON s.sku_code= o.`sku code`
    where o.`Distributor Code`= '{town_code}'
    GROUP BY
        o.`Deliveryman Name`,
        o.`Delivery Date`,o.`Distributor Code`
)

, selected_period AS (
    SELECT
		Town,
        raw.DeliveryMan,
        SUM(raw.Amount) AS NMV,
				SUM(ltr) as Ltr
    FROM
        raw
    WHERE
        raw.D_Date BETWEEN '{start}' AND '{end}'
    GROUP BY
        raw.DeliveryMan,
				Town
)
, last_year_period AS (
    SELECT
		town,
        raw.DeliveryMan,
        SUM(raw.Amount) AS NMV,
				sum(ltr) as Ltr
    FROM
        raw
    WHERE
        raw.D_Date BETWEEN DATE_SUB('{start}', INTERVAL 1 YEAR) AND DATE_SUB('{end}', INTERVAL 1 YEAR)
    GROUP BY
        raw.DeliveryMan,town
),
last_month AS (
    SELECT
		town,
        raw.DeliveryMan,
        SUM(raw.Amount) AS NMV,
				sum(ltr) as Ltr
    FROM
        raw
    WHERE
        raw.D_Date BETWEEN DATE_SUB('{start}', INTERVAL 1 month) AND DATE_SUB('{end}',INTERVAL 1 month)
    GROUP BY
        raw.DeliveryMan,town
)

SELECT
		sp.town,
    sp.DeliveryMan,
    sp.NMV AS Current_Period_Sales,
    ly.NMV AS Last_Year_Sales,
		lm.NMV AS Last_Month_Sales,
		sp.ltr as Current_Period_Ltr,
		ly.ltr as Last_Year_Ltr,
		lm.ltr as Last_Month_Ltr,
		ROUND(((sp.nmv/ly.nmv)-1)*100) as Sales_Growth_LY,
		Round(((sp.nmv/lm.nmv)-1)*100) as Sales_Growth_LM,
		Round(((sp.ltr/ly.ltr)-1)*100) as Ltr_Growth_LY,
		Round(((sp.ltr/lm.ltr)-1)*100) as Ltr_Growth_LM
FROM
    selected_period sp
LEFT JOIN
    last_year_period ly ON sp.DeliveryMan = ly.DeliveryMan
LEFT JOIN
    last_month lm ON sp.DeliveryMan = lm.DeliveryMan
		
ORDER BY
    sp.DeliveryMan;
"""

    return read_sql_cached(query, db_name)

@st.cache_data(ttl=3600)
def tgtvsach_YTD_data(town_code):
    """Fetch target vs achievement YTD data for comparison charts"""
    db_name = "db42280"
    
    query = f"""
SELECT 
concat(MONTH(`Delivery Date`),"-",YEAR(`Delivery Date`)) as period,
round(t.Target_In_Value) as Target_Value,
round(sum(`Delivered Amount`+`Total Discount`)) as NMV,
Round((sum(`Delivered Amount`+`Total Discount`)/t.Target_In_Value)*100) as Value_Ach,
round(t.Target_In_Volume) as Target_Ltr,
round(sum(`Delivered (Litres)`+`Delivered (KG)`)) as Ltr,
Round((sum(`Delivered (Litres)`+`Delivered (KG)`)/t.Target_In_Volume)*100) as Ltr_Ach,
Case when o.`Distributor Code`='D70002202' then 'Karachi'
				when o.`Distributor Code`='D70002246' then 'Lahore' else 'CBL' end Town



 from ordered_vs_delivered_rows o
 LEFT JOIN (SELECT month,year,sum(Target_In_Value) as Target_In_Value,sum(Target_In_Volume) as Target_In_Volume,Distributor_Code  from targets group by year,month,Distributor_Code) t on t.month= month(o.`Delivery Date`) and t.year=YEAR(o.`Delivery Date`) and t.Distributor_Code = o.`Distributor Code`
where o.`Distributor Code` = '{town_code}'
 GROUP BY MONTH(`Delivery Date`),YEAR(`Delivery Date`),Town
order by YEAR(`Delivery Date`) desc,MONTH(`Delivery Date`) desc
 limit 14


"""

    return read_sql_cached(query, db_name)

@st.cache_data(ttl=3600)
def tgtvsach_YTD_heatmap_data(town_code):
    """Fetch target vs achievement YTD data for heatmap visualization"""
    db_name = "db42280"
    
    query = f"""
    SELECT 
o.`Order Booker Name` as Booker,
concat(MONTH(`Delivery Date`),"-",YEAR(`Delivery Date`)) as period,
round(t.Target_In_Value) as Target_Value,
round(sum(`Delivered Amount`+`Total Discount`)) as NMV,
Round((sum(`Delivered Amount`+`Total Discount`)/t.Target_In_Value)*100) as Value_Ach,
round(t.Target_In_Volume) as Target_Ltr,
round(sum(`Delivered (Litres)`+`Delivered (KG)`)) as Ltr,
Round((sum(`Delivered (Litres)`+`Delivered (KG)`)/t.Target_In_Volume)*100) as Ltr_Ach,
Case when o.`Distributor Code`='D70002202' then 'Karachi'
				when o.`Distributor Code`='D70002246' then 'Lahore' else 'CBL' end Town



 from ordered_vs_delivered_rows o
 LEFT JOIN (SELECT month,year,sum(Target_In_Value) as Target_In_Value,sum(Target_In_Volume) as Target_In_Volume,Distributor_Code,Order_Booker_Code  from targets group by year,month,Distributor_Code,Order_Booker_Code) t on t.month= month(o.`Delivery Date`) and t.year=YEAR(o.`Delivery Date`) and t.Distributor_Code = o.`Distributor Code` and t.Order_Booker_Code=o.`Order Booker Code`
where o.`Distributor Code` = '{town_code}'
 GROUP BY MONTH(`Delivery Date`),YEAR(`Delivery Date`),Town,o.`Order Booker Name`
order by YEAR(`Delivery Date`) desc,MONTH(`Delivery Date`) desc
--
    """
    return read_sql_cached(query, db_name)

@st.cache_data(ttl=3600)
def tgtvsach_channelwise_heatmap(town_code):
    """Fetch target vs achievement YTD data for heatmap visualization"""
    db_name = "db42280"
    
    query = f"""
    SELECT
    CONCAT(YEAR(`Delivery Date`), '-', LPAD(MONTH(`Delivery Date`), 2, '0')) AS period,
    CASE WHEN u.`Channel Type` IS NULL THEN "Channel Empty" ELSE u.`Channel Type` END Channel,
    ROUND(SUM(`Delivered Amount` + `Total Discount`)) AS NMV,
    ROUND(SUM(`Delivered (Litres)` + `Delivered (KG)`)) AS Ltr
FROM 
    ordered_vs_delivered_rows o
INNER JOIN 
    universe u ON u.`Store Code` = o.`Store Code`
WHERE 
    `Delivery Date` >= DATE_SUB(CURDATE(), INTERVAL 7 MONTH)  -- Get data from 7 months ago
    AND (YEAR(`Delivery Date`) < YEAR(CURDATE()) OR (YEAR(`Delivery Date`) = YEAR(CURDATE()) AND MONTH(`Delivery Date`) < MONTH(CURDATE())))  -- Exclude current month
    AND o.`Distributor Code` = '{town_code}'
    GROUP BY 
    period, Channel
ORDER BY 
    period, Channel
    """
    return read_sql_cached(query, db_name)

@st.cache_data(ttl=3600)
def tgtvsach_brand_level(town_code, selected_period, selected_channel='All'):

    channel_condition = ""
    if selected_channel and selected_channel != 'All':
        selected_channel_safe = str(selected_channel).replace("'", "''")
        channel_condition = f"AND u.`Channel Type` = '{selected_channel_safe}'"

    query = f"""
    WITH sales_agg AS (
        SELECT
            o.`Distributor Code` AS Distributor_Code,
            o.`Order Booker Code` AS Booker_Code,
            o.`Order Booker Name` AS Booker,
            s.Brand AS brand,
            MONTH(o.`Delivery Date`) AS month_no,
            YEAR(o.`Delivery Date`) AS year_no,
            ROUND(SUM(o.`Delivered Amount` + o.`Total Discount`)) AS NMV,
            CASE
                WHEN o.`Distributor Code`='D70002202' THEN 'Karachi'
                WHEN o.`Distributor Code`='D70002246' THEN 'Lahore'
                ELSE 'CBL'
            END AS Town
        FROM ordered_vs_delivered_rows o
        LEFT JOIN sku_master s ON s.Sku_Code = o.`SKU Code`
        LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
        WHERE o.`Distributor Code` = '{town_code}'
                    AND o.`Delivery Date` BETWEEN STR_TO_DATE(CONCAT('{selected_period}', '-01'), '%%Y-%%m-%%d')
                                                                    AND LAST_DAY(STR_TO_DATE(CONCAT('{selected_period}', '-01'), '%%Y-%%m-%%d'))
                    {channel_condition}
        GROUP BY
            o.`Distributor Code`,
            o.`Order Booker Code`,
            o.`Order Booker Name`,
            s.Brand,
            MONTH(o.`Delivery Date`),
            YEAR(o.`Delivery Date`)
    ),
    target_agg AS (
        SELECT
            t.Distributor_Code,
            t.`AppUser Code` AS Booker_Code,
            t.Brand,
            t.month AS month_no,
            t.year AS year_no,
            ROUND(SUM(COALESCE(t.Target, 0))) AS Target_Value
        FROM targets_new t
        WHERE t.KPI = 'Value'
          AND t.Distributor_Code = '{town_code}'
        GROUP BY
            t.Distributor_Code,
            t.`AppUser Code`,
            t.Brand,
            t.month,
            t.year
    )
    SELECT
        s.Booker,
        s.brand,
        CONCAT(s.month_no, '-', s.year_no) AS period,
        COALESCE(t.Target_Value, 0) AS Target_Value,
        s.NMV,
        ROUND(
            CASE
                WHEN COALESCE(t.Target_Value, 0) > 0 THEN (s.NMV / t.Target_Value) * 100
                ELSE 0
            END
        ) AS Value_Ach,
        s.Town
    FROM sales_agg s
    LEFT JOIN target_agg t
        ON t.Distributor_Code = s.Distributor_Code
       AND t.Booker_Code = s.Booker_Code
       AND t.Brand = s.brand
       AND t.month_no = s.month_no
       AND t.year_no = s.year_no
    ORDER BY s.year_no DESC, s.month_no DESC, s.Booker, s.brand
    """
    return read_sql_cached(query, "db42280")

@st.cache_data(ttl=3600)
def fetch_treemap_channel_options(town_code, selected_period):
        """Fetch available channel options for selected treemap period."""
        query = f"""
        SELECT DISTINCT CASE WHEN u.`Channel Type` IS NULL THEN "Channel Empty" ELSE u.`Channel Type` END Channel
        FROM ordered_vs_delivered_rows o
        LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
        WHERE o.`Distributor Code` = '{town_code}'
            AND o.`Delivery Date` BETWEEN STR_TO_DATE(CONCAT('{selected_period}', '-01'), '%%Y-%%m-%%d')
                                                            AND LAST_DAY(STR_TO_DATE(CONCAT('{selected_period}', '-01'), '%%Y-%%m-%%d'))
            AND u.`Channel Type` IS NOT NULL
            AND TRIM(u.`Channel Type`) <> ''
        ORDER BY Channel
        """
        return read_sql_cached(query, "db42280")

@st.cache_data(ttl=3600)
def fetch_treemap_period_options(town_code):
        """Fetch available monthly periods for Booker treemap."""
        query = f"""
        select distinct concat(`Year`,"-",`Month`) as period from targets_new
WHERE Distributor_Code ='{town_code}'
        ORDER BY period DESC
        """
        return read_sql_cached(query, "db42280")

@st.cache_data(ttl=3600)
def AOV_MOPU_data(town_code, months_back): 
    """Fetch data for Average Order Value (AOV) and MOPU calculations."""
    query = f"""
    WITH last_6_months AS (
    SELECT *,`Delivered Units`/UOM as Del_Ctn
    FROM ordered_vs_delivered_rows
		left join sku_master m on m.Sku_Code= `SKU Code`
    WHERE `Order Date` >= DATE_SUB(CURDATE(), INTERVAL {months_back} MONTH)
		and `Distributor Code`='{town_code}'
)
SELECT 
    DATE_FORMAT(`Order Date`, '%%Y-%%m') AS `Month`,
    COUNT(DISTINCT `Order Number`) AS `Total_Orders`,                     -- total orders by booker
    SUM(`Delivered Units`)/COUNT(DISTINCT `shopcode`) AS `Drop_Size`,                                 -- total units delivered
    AVG(`SKU Count Per Order`) AS `SKU_Per_Bill`,                           -- average SKU per order
    COUNT(DISTINCT `Order Number`) / COUNT(DISTINCT `shopcode`) AS `MOPU`                            -- monthly orders per user (simplified)
FROM (
    SELECT 
        `Order Number`,
        `Order Date`,
        COUNT(DISTINCT `SKU Code`) AS `SKU Count Per Order`,
        SUM(Del_Ctn) AS `Delivered Units`,
				`store code` as shopcode
    FROM last_6_months
		
    GROUP BY `Order Number`, `Order Date`,`store code`
) AS sub
GROUP BY `Month`
ORDER BY `Month` DESC
"""
    return read_sql_cached(query, "db42280")

@st.cache_data(ttl=3600)
def ON_Wise_Visit_freq_data(town_code):
    """Fetch data for visit frequency analysis."""
    query = f"""
    SELECT 
    DATE_FORMAT(`Visit Date`, '%Y-%m') AS VisitMonth,
    TRIM(SUBSTRING_INDEX(`App User`, '[', 1)) AS OB_Name,
    COUNT(*) AS Total_Visits,
    SUM(CASE WHEN `Visit Complete` = 'Yes' THEN 1 ELSE 0 END) AS Completed_Visits,
    SUM(CASE WHEN `Non Productive` = 'Yes' THEN 1 ELSE 0 END) AS Non_Productive_Visits
FROM visits_summary_rows
WHERE `Visit Date` >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
and TRIM(SUBSTRING_INDEX(SUBSTRING_INDEX(`Distributor`, '[', -1),']',1)) = '{town_code}'
GROUP BY VisitMonth, OB_Name
ORDER BY VisitMonth DESC, OB_Name;    
"""

@st.cache_data(ttl=3600)
def GMV_OB_calendar_heatmap_data(town_code, start, end):
    """Fetch data for GMV by Order Booker calendar heatmap."""
    query = f"""
    SELECT
        o.`Order Date` AS Order_Date,
        o.`Order Booker Name`,
        CASE WHEN u.`Channel Type` IS NULL THEN "Channel Empty" ELSE u.`Channel Type` END Channel,
        ROUND(SUM(o.`Order Amount`), 0) AS GMV,
        COUNT(DISTINCT o.`Order Number`) AS Orders
    FROM ordered_vs_delivered_rows o
    LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
    WHERE o.`Distributor Code`='{town_code}'
      AND o.`Order Date` BETWEEN '{start}' AND '{end}'
    GROUP BY o.`Order Booker Name`, o.`Order Date`, u.`Channel Type`
    """
    return read_sql_cached(query, "db42280")

@st.cache_data(ttl=3600)
def fetch_booker_fieldforce_deep_data(start_date, end_date, town_code):
    """Fetch deep analysis dataset for Booker and Field Force performance."""
    query = f"""
    SELECT
        o.`Order Booker Name` AS Booker,
        o.`Deliveryman Name` AS Deliveryman,
                Case when u.`Channel Type` is null then "Channel Empty"  else u.`Channel Type` end Channel,
        ROUND(SUM(o.`Delivered Amount` + o.`Total Discount`), 0) AS NMV,
        COUNT(DISTINCT o.`Invoice Number`) AS Orders,
        COUNT(DISTINCT o.`Store Code`) AS Stores,
        ROUND(
            SUM(o.`Delivered Amount` + o.`Total Discount`) /
            NULLIF(COUNT(DISTINCT o.`Invoice Number`), 0),
            0
        ) AS AOV,
        ROUND(SUM(o.`Delivered (Litres)` + o.`Delivered (KG)`), 0) AS Volume
    FROM ordered_vs_delivered_rows o
    LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
    WHERE o.`Distributor Code` = '{town_code}'
      AND o.`Delivery Date` BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY o.`Order Booker Name`, o.`Deliveryman Name`, u.`Channel Type`
    ORDER BY NMV DESC
    """
    return read_sql_cached(query, "db42280")

@st.cache_data(ttl=3600)
def fetch_routewise_ob_achievement(start_date, end_date, town_code, selected_channels=()):
    """Fetch OB-wise achieved vs target performance for selected period."""
    channel_condition = ""
    if selected_channels:
        safe_channels = [str(channel).replace("'", "''") for channel in selected_channels]
        channel_values = "', '".join(safe_channels)
        channel_condition = f"AND u.`Channel Type` IN ('{channel_values}')"

    query = f"""
    WITH achieved_brand AS (
        SELECT
            o.`Order Booker Code` AS Booker_Code,
            o.`Order Booker Name` AS Booker,
            s.Brand AS Brand,
            ROUND(SUM(o.`Delivered Amount` + o.`Total Discount`), 0) AS Achieved_Value
        FROM ordered_vs_delivered_rows o
        LEFT JOIN sku_master s ON s.Sku_Code = o.`SKU Code`
        LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
        WHERE o.`Distributor Code` = '{town_code}'
          AND o.`Delivery Date` BETWEEN '{start_date}' AND '{end_date}'
          {channel_condition}
        GROUP BY o.`Order Booker Code`, o.`Order Booker Name`, s.Brand
    ),
    target_brand AS (
        SELECT
            t.`AppUser Code` AS Booker_Code,
            t.Brand,
            ROUND(SUM(COALESCE(t.Target, 0)), 0) AS Target_Value
        FROM targets_new t
        WHERE t.Distributor_Code = '{town_code}'
          AND t.KPI = 'Value'
          AND STR_TO_DATE(CONCAT(t.year, '-', LPAD(t.month, 2, '0'), '-01'), '%%Y-%%m-%%d')
              BETWEEN DATE_FORMAT('{start_date}', '%%Y-%%m-01') AND DATE_FORMAT('{end_date}', '%%Y-%%m-01')
        GROUP BY t.`AppUser Code`, t.Brand
    ),
    ob_rollup AS (
        SELECT
            a.Booker,
            ROUND(SUM(a.Achieved_Value), 0) AS Achieved_Value,
            ROUND(SUM(COALESCE(t.Target_Value, 0)), 0) AS Target_Value
        FROM achieved_brand a
        LEFT JOIN target_brand t
            ON t.Booker_Code = a.Booker_Code
           AND t.Brand = a.Brand
        GROUP BY a.Booker
    )
    SELECT
        o.Booker,
        o.Target_Value,
        o.Achieved_Value,
        ROUND(
            CASE
                WHEN o.Target_Value > 0 THEN (o.Achieved_Value / o.Target_Value) * 100
                ELSE 0
            END,
            1
        ) AS Achieved_Pct
    FROM ob_rollup o
    ORDER BY Achieved_Pct DESC, o.Achieved_Value DESC
    """
    return read_sql_cached(query, "db42280")

@st.cache_data(ttl=3600)
def fetch_daily_calls_trend_data(start_date, end_date, town_code):
    """Fetch daily planned vs executed calls from visits table."""
    query = f"""
    SELECT
        v.`Visit Date` AS Call_Date,
        TRIM(SUBSTRING_INDEX(v.`App User`, '[', 1)) AS Booker,
        COUNT(*) AS Planned_Calls,
        SUM(CASE WHEN v.`Visit Complete` = 'Yes' THEN 1 ELSE 0 END) AS Executed_Calls,
        SUM(
            CASE
                WHEN v.`Visit Complete` = 'Yes'
                     AND COALESCE(v.`Non Productive w.r.t Order`, 'No') <> 'Yes'
                THEN 1
                ELSE 0
            END
        ) AS Productive_Calls
    FROM visits_summary_rows v
    WHERE v.`Visit Date` BETWEEN '{start_date}' AND '{end_date}'
      AND v.`Visit Date` <> '0000-00-00'
      AND TRIM(SUBSTRING_INDEX(SUBSTRING_INDEX(v.`Distributor`, '[', -1),']',1)) = '{town_code}'
    GROUP BY v.`Visit Date`, TRIM(SUBSTRING_INDEX(v.`App User`, '[', 1))
    ORDER BY v.`Visit Date`
    """
    return read_sql_cached(query, "db42280")

@st.cache_data(ttl=3600)
def fetch_booker_leaderboard_data(start_date, end_date, town_code, selected_channels=()):
    """Fetch leaderboard metrics per Booker for selected period."""
    channel_condition_sales = ""
    if selected_channels:
        safe_channels = [str(channel).replace("'", "''") for channel in selected_channels]
        channel_values = "', '".join(safe_channels)
        channel_condition_sales = f"AND u.`Channel Type` IN ('{channel_values}')"

    query = f"""
    WITH sales AS (
        SELECT
            o.`Order Booker Code` AS Booker_Code,
            o.`Order Booker Name` AS Booker,
            ROUND(SUM(o.`Delivered Amount` + o.`Total Discount`), 0) AS Revenue,
            COUNT(DISTINCT o.`Invoice Number`) AS Orders,
            COUNT(DISTINCT o.`Store Code`) AS Active_Outlets
        FROM ordered_vs_delivered_rows o
        LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
        WHERE o.`Distributor Code` = '{town_code}'
          AND o.`Delivery Date` BETWEEN '{start_date}' AND '{end_date}'
          {channel_condition_sales}
        GROUP BY o.`Order Booker Code`, o.`Order Booker Name`
    ),
    channel_rank AS (
        SELECT
            o.`Order Booker Code` AS Booker_Code,
            o.`Order Booker Name` AS Booker,
                    Case when u.`Channel Type` is null then "Channel Empty"  else u.`Channel Type` end  Region,
            COUNT(*) AS Channel_Orders,
            ROW_NUMBER() OVER (
                PARTITION BY o.`Order Booker Code`
                ORDER BY COUNT(*) DESC
            ) AS rn
        FROM ordered_vs_delivered_rows o
        LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
        WHERE o.`Distributor Code` = '{town_code}'
          AND o.`Delivery Date` BETWEEN '{start_date}' AND '{end_date}'
          {channel_condition_sales}
        GROUP BY o.`Order Booker Code`, o.`Order Booker Name`, u.`Channel Type`
    ),
    calls AS (
        SELECT
            TRIM(SUBSTRING_INDEX(v.`App User`, '[', 1)) AS Booker,
            COUNT(*) AS Planned_Calls,
            SUM(CASE WHEN v.`Visit Complete` = 'Yes' THEN 1 ELSE 0 END) AS Executed_Calls,
            COUNT(DISTINCT v.`Visit Date`) AS Call_Days
        FROM visits_summary_rows v
        WHERE v.`Visit Date` BETWEEN '{start_date}' AND '{end_date}'
          AND v.`Visit Date` <> '0000-00-00'
          AND TRIM(SUBSTRING_INDEX(SUBSTRING_INDEX(v.`Distributor`, '[', -1),']',1)) = '{town_code}'
        GROUP BY TRIM(SUBSTRING_INDEX(v.`App User`, '[', 1))
    ),
    first_outlet AS (
        SELECT
            o.`Order Booker Code` AS Booker_Code,
            o.`Store Code` AS Store_Code,
            MIN(o.`Delivery Date`) AS First_Date
        FROM ordered_vs_delivered_rows o
        LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
        WHERE o.`Distributor Code` = '{town_code}'
          {channel_condition_sales}
        GROUP BY o.`Order Booker Code`, o.`Store Code`
    ),
    new_outlets AS (
        SELECT
            f.Booker_Code,
            COUNT(DISTINCT f.Store_Code) AS New_Outlets
        FROM first_outlet f
        WHERE f.First_Date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY f.Booker_Code
    ),
    sku_bill AS (
        SELECT
            o.`Order Booker Code` AS Booker_Code,
            ROUND(
                COUNT(DISTINCT CONCAT(o.`Invoice Number`, '||', o.`SKU Code`))
                / NULLIF(COUNT(DISTINCT o.`Invoice Number`), 0),
                2
            ) AS SKU_Per_Bill
        FROM ordered_vs_delivered_rows o
        LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
        WHERE o.`Distributor Code` = '{town_code}'
          AND o.`Delivery Date` BETWEEN '{start_date}' AND '{end_date}'
          {channel_condition_sales}
        GROUP BY o.`Order Booker Code`
    ),
    target_agg AS (
        SELECT
            t.`AppUser Code` AS Booker_Code,
            ROUND(SUM(COALESCE(t.Target, 0)), 0) AS Target_Value
        FROM targets_new t
        WHERE t.Distributor_Code = '{town_code}'
          AND t.KPI = 'Value'
          AND STR_TO_DATE(CONCAT(t.year, '-', LPAD(t.month, 2, '0'), '-01'), '%%Y-%%m-%%d')
              BETWEEN DATE_FORMAT('{start_date}', '%%Y-%%m-01') AND DATE_FORMAT('{end_date}', '%%Y-%%m-01')
        GROUP BY t.`AppUser Code`
    )
    SELECT
        s.Booker,
        COALESCE(cr.Region, 'Unknown') AS Region,
        COALESCE(t.Target_Value, 0) AS Target_Value,
        s.Revenue,
        s.Orders,
        s.Active_Outlets,
        ROUND(s.Revenue / NULLIF(s.Orders, 0), 0) AS Avg_Order_Val,
        COALESCE(sb.SKU_Per_Bill, 0) AS SKU_Per_Bill,
        ROUND(
            CASE
                WHEN COALESCE(t.Target_Value, 0) > 0 THEN (s.Revenue / t.Target_Value) * 100
                ELSE 0
            END,
            1
        ) AS Achievement_Pct,
        COALESCE(c.Planned_Calls, 0) AS Planned_Calls,
        COALESCE(c.Executed_Calls, 0) AS Executed_Calls,
        COALESCE(c.Call_Days, 0) AS Call_Days,
        COALESCE(n.New_Outlets, 0) AS New_Outlets
    FROM sales s
    LEFT JOIN (
        SELECT Booker_Code, Booker, Region
        FROM channel_rank
        WHERE rn = 1
    ) cr ON cr.Booker_Code = s.Booker_Code
    LEFT JOIN calls c ON c.Booker = s.Booker
    LEFT JOIN new_outlets n ON n.Booker_Code = s.Booker_Code
    LEFT JOIN sku_bill sb ON sb.Booker_Code = s.Booker_Code
    LEFT JOIN target_agg t ON t.Booker_Code = s.Booker_Code
    ORDER BY s.Revenue DESC
    """
    return read_sql_cached(query, "db42280")

@st.cache_data(ttl=3600)
def fetch_activity_segmentation_data(start_date, end_date, town_code, selected_channels=(), selected_bookers=()):
    """Fetch per-store order frequency for activity segmentation."""
    channel_condition = ""
    if selected_channels:
        safe_channels = [str(channel).replace("'", "''") for channel in selected_channels]
        channel_values = "', '".join(safe_channels)
        channel_condition = f"AND u.`Channel Type` IN ('{channel_values}')"

    booker_condition = ""
    if selected_bookers:
        safe_bookers = [str(booker).replace("'", "''") for booker in selected_bookers]
        booker_values = "', '".join(safe_bookers)
        booker_condition = f"AND o.`Order Booker Name` IN ('{booker_values}')"

    query = f"""
    WITH scoped_stores AS (
        SELECT DISTINCT o.`Store Code` AS Store_Code
        FROM ordered_vs_delivered_rows o
        LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
        WHERE o.`Distributor Code` = '{town_code}'
        AND u.status='Active'
          {channel_condition}
          {booker_condition}
    ),
    period_orders AS (
        SELECT
            o.`Store Code` AS Store_Code,
            COUNT(DISTINCT o.`Invoice Number`) AS Orders_In_Period
        FROM ordered_vs_delivered_rows o
        LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
        WHERE o.`Distributor Code` = '{town_code}'
          AND o.`Delivery Date` BETWEEN '{start_date}' AND '{end_date}'
          {channel_condition}
          {booker_condition}
        GROUP BY o.`Store Code`
    )
    SELECT
        s.Store_Code,
        COALESCE(p.Orders_In_Period, 0) AS Orders_In_Period
    FROM scoped_stores s
    LEFT JOIN period_orders p ON p.Store_Code = s.Store_Code
    """
    return read_sql_cached(query, "db42280")

@st.cache_data(ttl=3600)
def fetch_activity_segmentation_booker_data(start_date, end_date, town_code, selected_channels=(), selected_bookers=()):
    """Fetch Booker + store level order frequency for booker-wise activity segmentation."""
    channel_condition = ""
    if selected_channels:
        safe_channels = [str(channel).replace("'", "''") for channel in selected_channels]
        channel_values = "', '".join(safe_channels)
        channel_condition = f"AND u.`Channel Type` IN ('{channel_values}')"

    booker_condition = ""
    if selected_bookers:
        safe_bookers = [str(booker).replace("'", "''") for booker in selected_bookers]
        booker_values = "', '".join(safe_bookers)
        booker_condition = f"AND o.`Order Booker Name` IN ('{booker_values}')"

    query = f"""
    WITH scoped_pairs AS (
        SELECT
            o.`Order Booker Name` AS Booker,
            o.`Store Code` AS Store_Code,
            MAX(o.`Store Name`) AS Store_Name,
            MAX(o.`Delivery Date`) AS Last_Order_Date
        FROM ordered_vs_delivered_rows o
        LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
        WHERE o.`Distributor Code` = '{town_code}'
        AND u.status='Active'
          {channel_condition}
          {booker_condition}
        GROUP BY o.`Order Booker Name`, o.`Store Code`
    ),
    period_orders AS (
        SELECT
            o.`Order Booker Name` AS Booker,
            o.`Store Code` AS Store_Code,
            COUNT(DISTINCT o.`Invoice Number`) AS Orders_In_Period
        FROM ordered_vs_delivered_rows o
        LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
        WHERE o.`Distributor Code` = '{town_code}'
          AND o.`Delivery Date` BETWEEN '{start_date}' AND '{end_date}'
          {channel_condition}
          {booker_condition}
        GROUP BY o.`Order Booker Name`, o.`Store Code`
    )
    SELECT
        s.Booker,
        s.Store_Code,
        s.Store_Name,
        s.Last_Order_Date,
        COALESCE(p.Orders_In_Period, 0) AS Orders_In_Period
    FROM scoped_pairs s
    LEFT JOIN period_orders p
        ON p.Booker = s.Booker
       AND p.Store_Code = s.Store_Code
    """
    return read_sql_cached(query, "db42280")

@st.cache_data(ttl=3600)
def fetch_weekly_cohort_orders(start_date, end_date, town_code, selected_channels=(), selected_bookers=()):
    """Fetch store-level order dates for weekly cohort retention."""
    channel_condition = ""
    if selected_channels:
        safe_channels = [str(channel).replace("'", "''") for channel in selected_channels]
        channel_values = "', '".join(safe_channels)
        channel_condition = f"AND u.`Channel Type` IN ('{channel_values}')"

    booker_condition = ""
    if selected_bookers:
        safe_bookers = [str(booker).replace("'", "''") for booker in selected_bookers]
        booker_values = "', '".join(safe_bookers)
        booker_condition = f"AND o.`Order Booker Name` IN ('{booker_values}')"

    query = f"""
    SELECT
        o.`Store Code` AS Store_Code,
        DATE(o.`Delivery Date`) AS Order_Date
    FROM ordered_vs_delivered_rows o
    LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
    WHERE o.`Distributor Code` = '{town_code}'
      AND o.`Delivery Date` BETWEEN '{start_date}' AND '{end_date}'
      {channel_condition}
      {booker_condition}
    GROUP BY o.`Store Code`, DATE(o.`Delivery Date`)
    ORDER BY DATE(o.`Delivery Date`)
    """
    return read_sql_cached(query, "db42280")

@st.cache_data(ttl=3600)
def fetch_sku_per_bill_metric(start_date, end_date, town_code, selected_channels=(), selected_bookers=()):
    """Fetch SKU per bill metric for selected period and filters."""
    channel_condition = ""
    if selected_channels:
        safe_channels = [str(channel).replace("'", "''") for channel in selected_channels]
        channel_values = "', '".join(safe_channels)
        channel_condition = f"AND u.`Channel Type` IN ('{channel_values}')"

    booker_condition = ""
    if selected_bookers:
        safe_bookers = [str(booker).replace("'", "''") for booker in selected_bookers]
        booker_values = "', '".join(safe_bookers)
        booker_condition = f"AND o.`Order Booker Name` IN ('{booker_values}')"

    query = f"""
    SELECT
        COUNT(DISTINCT o.`Invoice Number`) AS Total_Orders,
        COUNT(DISTINCT CONCAT(o.`Invoice Number`, '::', o.`SKU Code`)) AS Invoice_SKU_Count,
        ROUND(
            COUNT(DISTINCT CONCAT(o.`Invoice Number`, '::', o.`SKU Code`))
            / NULLIF(COUNT(DISTINCT o.`Invoice Number`), 0),
            2
        ) AS SKU_Per_Bill
    FROM ordered_vs_delivered_rows o
    LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
    WHERE o.`Distributor Code` = '{town_code}'
      AND o.`Delivery Date` BETWEEN '{start_date}' AND '{end_date}'
      {channel_condition}
      {booker_condition}
    """
    return read_sql_cached(query, "db42280")

@st.cache_data(ttl=3600)
def fetch_booker_brand_scoring_data(start_date, end_date, town_code, selected_channels=(), selected_bookers=()):
    """Fetch brand-wise sales by booker for scoring analysis."""
    channel_condition = ""
    if selected_channels:
        safe_channels = [str(channel).replace("'", "''") for channel in selected_channels]
        channel_values = "', '".join(safe_channels)
        channel_condition = f"AND u.`Channel Type` IN ('{channel_values}')"

    booker_condition = ""
    if selected_bookers:
        safe_bookers = [str(booker).replace("'", "''") for booker in selected_bookers]
        booker_values = "', '".join(safe_bookers)
        booker_condition = f"AND o.`Order Booker Name` IN ('{booker_values}')"

    query = f"""
    SELECT
        o.`Order Booker Name` AS Booker,
        COALESCE(s.Brand, 'Unknown') AS Brand,
        ROUND(SUM(o.`Delivered Amount` + o.`Total Discount`), 0) AS NMV,
        COUNT(DISTINCT o.`Invoice Number`) AS Orders
    FROM ordered_vs_delivered_rows o
    LEFT JOIN universe u ON u.`Store Code` = o.`Store Code`
    LEFT JOIN sku_master s ON s.Sku_Code = o.`SKU Code`
    WHERE o.`Distributor Code` = '{town_code}'
      AND o.`Delivery Date` BETWEEN '{start_date}' AND '{end_date}'
      {channel_condition}
      {booker_condition}
    GROUP BY o.`Order Booker Name`, COALESCE(s.Brand, 'Unknown')
    HAVING SUM(o.`Delivered Amount` + o.`Total Discount`) > 0
    ORDER BY Booker, NMV DESC
    """
    return read_sql_cached(query, "db42280")

@st.cache_data(ttl=3600)
def fetch_latest_visit_date(town_code):
        """Fetch latest valid visit date for selected distributor."""
        query = f"""
        SELECT MAX(v.`Visit Date`) AS Latest_Visit_Date
        FROM visits_summary_rows v
        WHERE v.`Visit Date` <> '0000-00-00'
            AND TRIM(SUBSTRING_INDEX(SUBSTRING_INDEX(v.`Distributor`, '[', -1),']',1)) = '{town_code}'
        """
        return read_sql_cached(query, "db42280")

# ======================
# 📈 VISUALIZATION FUNCTIONS
# ======================

def create_Channel_dm_sunburst(df, selected_dms=None, selected_channels=None):
    """Create Channel-Brand-DM sunburst chart with optional DM filter."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    if selected_dms:
        df = df[df['DM'].isin(selected_dms)]
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data for selected deliverymen",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
    
    if selected_channels:
        df = df[df['Channel'].isin(selected_channels)]
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data for selected channels",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig

    df_plot = df.copy()
    df_plot['StoreCount'] = pd.to_numeric(df_plot['StoreCount'], errors='coerce').fillna(0)
    my_palette = [
    "#FFC145",  # blue
    "#5B5F97",  # red
    "#B8B8D1",  # green
    "#FFFFFB",  # orange
    "#FF6B6C"   # purple
]
    fig = px.sunburst(
        df_plot,
        path=['Channel', 'Brand', 'DM'],
        values='StoreCount',
        title="☀️ Channel, Brand & Deliveryman Hierarchy"
    )
    
    fig.update_traces(
        textinfo="label+percent entry",
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{label}</b><br>Stores: %{value:.0f}<extra></extra>',
        marker=dict(colorscale=my_palette)
    )
    
    fig.update_layout(
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def create_Channel_performance_chart(df, metric_type='Value'):
    """
    Create channel performance comparison chart
    metric_type: 'Value' for Sales or 'Ltr' for Litres
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Select metric columns based on type
    if metric_type == 'Value':
        current_col = 'Current_Period_Sales'
        last_year_col = 'Last_Year_Sales'
        last_month_col = 'Last_Month_Sales'
        growth_ly_col = 'Sales_Growth_LY'
        growth_lm_col = 'Sales_Growth_LM'
        unit_label = 'M'
        divisor = 1_000_000
    else:  # Ltr
        current_col = 'Current_Period_Ltr'
        last_year_col = 'Last_Year_Ltr'
        last_month_col = 'Last_Month_Ltr'
        growth_ly_col = 'Ltr_Growth_LY'
        growth_lm_col = 'Ltr_Growth_LM'
        unit_label = 'T'
        divisor = 1000
    
    # Prepare data
    df_processed = df[[
        'Channel', current_col, last_year_col, last_month_col, 
        growth_ly_col, growth_lm_col
    ]].copy()
    
    # Convert to appropriate units
    for col in [current_col, last_year_col, last_month_col]:
        df_processed[col] = df_processed[col] / divisor
    
    df_processed[last_year_col] = pd.to_numeric(df_processed[last_year_col],errors="coerce")
    df_processed[last_month_col] = pd.to_numeric(df_processed[last_month_col],errors="coerce")
    df_processed[growth_ly_col] = pd.to_numeric(df_processed[growth_ly_col],errors="coerce")
    df_processed[growth_lm_col] = pd.to_numeric(df_processed[growth_lm_col],errors="coerce")

    # Format values for display
    if metric_type == 'Value':
        current_vals = df_processed[current_col].round(1)
        last_year_vals = df_processed[last_year_col].round(1)
        last_month_vals = df_processed[last_month_col].round(1)
    else:
        current_vals = df_processed[current_col].round(0)
        last_year_vals = df_processed[last_year_col].round(0)
        last_month_vals = df_processed[last_month_col].round(0)
    
    # Create hover text with growth percentages
    hover_current = [f"<b>Current</b><br>{metric_type}: {val}{unit_label}" 
                     for ch, val in zip(df_processed['Channel'], current_vals)]
    
    hover_last_year = [f"<b>Last Year</b><br>{metric_type}: {val}{unit_label}<br>Growth vs Current: {growth:.1f}%" 
                       for ch, val, growth in zip(df_processed['Channel'], last_year_vals, df_processed[growth_ly_col])]
    
    hover_last_month = [f"<b>Last Month</b><br>{metric_type}: {val}{unit_label}<br>Growth vs Current: {growth:.1f}%" 
                        for ch, val, growth in zip(df_processed['Channel'], last_month_vals, df_processed[growth_lm_col])]
    
    # Create grouped bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=df_processed['Channel'],
            y=df_processed[current_col],
            name='Current Period',
            text=current_vals,
            textposition='outside',
            texttemplate='%{text}' + unit_label,
            hovertext=hover_current,
            hoverinfo='text',
            marker=dict(color='#5B5F97')
        ),
        go.Bar(
            x=df_processed['Channel'],
            y=df_processed[last_year_col],
            name='Last Year',
            text=last_year_vals,
            textposition='outside',
            texttemplate='%{text}' + unit_label,
            hovertext=hover_last_year,
            hoverinfo='text',
            marker=dict(color='#B8B8D1')
        ),
        go.Bar(
            x=df_processed['Channel'],
            y=df_processed[last_month_col],
            name='Last Month',
            text=last_month_vals,
            textposition='outside',
            texttemplate='%{text}' + unit_label,
            hovertext=hover_last_month,
            hoverinfo='text',
            marker=dict(color='#FFC145')
        )
    ])
    
    y_axis_title = 'Sales (in Millions)' if metric_type == 'Value' else 'Litres'
    
    fig.update_layout(
        title=f"📊 Channel Performance Comparison - {metric_type}",
        yaxis=dict(title=y_axis_title),
        xaxis=dict(title='Channel'),
        # height=600,
        barmode='group',
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, weight='bold'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )

    return apply_theme_aware_bar_labels(fig)

def create_channel_wise_growth_chart(df, metric_type='Value'):
    """Create channel-wise growth percentage chart with value/ltr hover details."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    if metric_type == 'Value':
        current_col = 'Current_Period_Sales'
        last_year_col = 'Last_Year_Sales'
        last_month_col = 'Last_Month_Sales'
        growth_ly_col = 'Sales_Growth_LY'
        growth_lm_col = 'Sales_Growth_LM'
        divisor = 1_000_000
        unit_label = 'M'
        metric_label = 'Value'
    else:
        current_col = 'Current_Period_Ltr'
        last_year_col = 'Last_Year_Ltr'
        last_month_col = 'Last_Month_Ltr'
        growth_ly_col = 'Ltr_Growth_LY'
        growth_lm_col = 'Ltr_Growth_LM'
        divisor = 1000
        unit_label = 'T'
        metric_label = 'Volume'

    df_processed = df[[
        'Channel', current_col, last_year_col, last_month_col,
        growth_ly_col, growth_lm_col
    ]].copy()

    numeric_cols = [current_col, last_year_col, last_month_col, growth_ly_col, growth_lm_col]
    for column in numeric_cols:
        df_processed[column] = pd.to_numeric(df_processed[column], errors='coerce').fillna(0)

    df_processed[current_col] = df_processed[current_col] / divisor
    df_processed[last_year_col] = df_processed[last_year_col] / divisor
    df_processed[last_month_col] = df_processed[last_month_col] / divisor

    current_vals = df_processed[current_col].round(2)
    last_year_vals = df_processed[last_year_col].round(2)
    last_month_vals = df_processed[last_month_col].round(2)

    customdata = np.column_stack([
        current_vals,
        last_year_vals,
        last_month_vals,
    ])

    ly_colors = ['#B8B8D1' if value >= 0 else '#FF6B6C' for value in df_processed[growth_ly_col]]
    lm_colors = ['#FFC145' if value >= 0 else '#FF6B6C' for value in df_processed[growth_lm_col]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_processed['Channel'],
            y=df_processed[growth_ly_col],
            name='Growth vs Last Year',
            marker=dict(color=ly_colors),
            text=df_processed[growth_ly_col].apply(lambda x: f"{x:.1f}%"),
            textposition='outside',
            customdata=customdata,
            hovertemplate=(
                '<b>%{x}</b>'
                '<br>Growth vs Last Year: %{y:.1f}%'
                '<br>Current ' + metric_label + ': %{customdata[0]:.2f}' + unit_label +
                '<br>Last Year ' + metric_label + ': %{customdata[1]:.2f}' + unit_label +
                '<br>Last Month ' + metric_label + ': %{customdata[2]:.2f}' + unit_label +
                '<extra></extra>'
            )
        )
    )
    fig.add_trace(
        go.Bar(
            x=df_processed['Channel'],
            y=df_processed[growth_lm_col],
            name='Growth vs Last Month',
            marker=dict(color=lm_colors),
            text=df_processed[growth_lm_col].apply(lambda x: f"{x:.1f}%"),
            textposition='outside',
            customdata=customdata,
            hovertemplate=(
                '<b>%{x}</b>'
                '<br>Growth vs Last Month: %{y:.1f}%'
                '<br>Current ' + metric_label + ': %{customdata[0]:.2f}' + unit_label +
                '<br>Last Year ' + metric_label + ': %{customdata[1]:.2f}' + unit_label +
                '<br>Last Month ' + metric_label + ': %{customdata[2]:.2f}' + unit_label +
                '<extra></extra>'
            )
        )
    )

    fig.update_layout(
        title=f"📈 Channel-wise Growth Percentage - {metric_type}",
        yaxis=dict(title='Growth %'),
        xaxis=dict(title='Channel'),
        # height=500,
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, weight='bold'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )

    return apply_theme_aware_bar_labels(fig)

def create_brand_wise_growth_chart(df, metric_type='Value'):
    """Create brand-wise growth percentage bar chart with value/ltr hover details."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    if metric_type == 'Value':
        current_col = 'Current_Period_Sales'
        last_year_col = 'Last_Year_Sales'
        last_month_col = 'Last_Month_Sales'
        growth_ly_col = 'Sales_Growth_LY'
        growth_lm_col = 'Sales_Growth_LM'
        divisor = 1_000_000
        unit_label = 'M'
        metric_label = 'Sales'
    else:
        current_col = 'Current_Period_Ltr'
        last_year_col = 'Last_Year_Ltr'
        last_month_col = 'Last_Month_Ltr'
        growth_ly_col = 'Ltr_Growth_LY'
        growth_lm_col = 'Ltr_Growth_LM'
        divisor = 1000
        unit_label = 'T'
        metric_label = 'Litres'

    required_cols = ['brand', current_col, last_year_col, last_month_col]
    missing_cols = [column for column in required_cols if column not in df.columns]
    if missing_cols:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing columns for chart: {', '.join(missing_cols)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        return fig

    def _ellipsis_label(text, max_len=14):
        text = str(text or '')
        return text if len(text) <= max_len else text[:max_len] + '...'

    df_processed = df[required_cols].copy()
    numeric_cols = [current_col, last_year_col, last_month_col]
    for column in numeric_cols:
        df_processed[column] = pd.to_numeric(df_processed[column], errors='coerce').fillna(0)

    df_processed['brand'] = df_processed['brand'].fillna('Unknown Brand').astype(str)
    df_processed = (
        df_processed
        .groupby('brand', as_index=False)
        .agg(
            **{
                current_col: (current_col, 'sum'),
                last_year_col: (last_year_col, 'sum'),
                last_month_col: (last_month_col, 'sum'),
            }
        )
    )

    df_processed[growth_ly_col] = np.where(
        pd.to_numeric(df_processed[last_year_col], errors='coerce').fillna(0) > 0,
        ((pd.to_numeric(df_processed[current_col], errors='coerce').fillna(0)
          / pd.to_numeric(df_processed[last_year_col], errors='coerce').fillna(0)) - 1) * 100,
        0,
    )
    df_processed[growth_lm_col] = np.where(
        pd.to_numeric(df_processed[last_month_col], errors='coerce').fillna(0) > 0,
        ((pd.to_numeric(df_processed[current_col], errors='coerce').fillna(0)
          / pd.to_numeric(df_processed[last_month_col], errors='coerce').fillna(0)) - 1) * 100,
        0,
    )

    df_processed[growth_ly_col] = pd.to_numeric(df_processed[growth_ly_col], errors='coerce').fillna(0).round(1)
    df_processed[growth_lm_col] = pd.to_numeric(df_processed[growth_lm_col], errors='coerce').fillna(0).round(1)

    df_processed['brand_short'] = df_processed['brand'].apply(lambda value: _ellipsis_label(value, 14))
    df_processed[current_col] = df_processed[current_col] / divisor
    df_processed[last_year_col] = df_processed[last_year_col] / divisor
    df_processed[last_month_col] = df_processed[last_month_col] / divisor

    customdata = np.column_stack([
        df_processed[current_col].round(2),
        df_processed[last_year_col].round(2),
        df_processed[last_month_col].round(2),
        df_processed['brand'].astype(str),
    ])

    ly_colors = ['#B8B8D1' if value >= 0 else '#FF6B6C' for value in df_processed[growth_ly_col]]
    lm_colors = ['#FFC145' if value >= 0 else '#FF6B6C' for value in df_processed[growth_lm_col]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_processed['brand_short'],
            y=df_processed[growth_ly_col],
            name='Growth vs Last Year',
            marker=dict(color=ly_colors),
            text=df_processed[growth_ly_col].apply(lambda value: f"{value:.1f}%"),
            textposition='outside',
            customdata=customdata,
            hovertemplate=(
                '<b>%{customdata[3]}</b>'
                '<br>Growth vs Last Year: %{y:.1f}%'
                '<br>Current ' + metric_label + ': %{customdata[0]:.2f}' + unit_label +
                '<br>Last Year ' + metric_label + ': %{customdata[1]:.2f}' + unit_label +
                '<br>Last Month ' + metric_label + ': %{customdata[2]:.2f}' + unit_label +
                '<extra></extra>'
            )
        )
    )
    fig.add_trace(
        go.Bar(
            x=df_processed['brand_short'],
            y=df_processed[growth_lm_col],
            name='Growth vs Last Month',
            marker=dict(color=lm_colors),
            text=df_processed[growth_lm_col].apply(lambda value: f"{value:.1f}%"),
            textposition='outside',
            customdata=customdata,
            hovertemplate=(
                '<b>%{customdata[3]}</b>'
                '<br>Growth vs Last Month: %{y:.1f}%'
                '<br>Current ' + metric_label + ': %{customdata[0]:.2f}' + unit_label +
                '<br>Last Year ' + metric_label + ': %{customdata[1]:.2f}' + unit_label +
                '<br>Last Month ' + metric_label + ': %{customdata[2]:.2f}' + unit_label +
                '<extra></extra>'
            )
        )
    )

    fig.update_layout(
        title=f"📈 Brand-wise Growth Percentage - {metric_type}",
        yaxis=dict(title='Growth %'),
        xaxis=dict(title='Brand'),
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, weight='bold'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    return apply_theme_aware_bar_labels(fig)


def create_brand_wise_growth_segmented_chart(df, metric_type='Value'):
    """Create channel-brand segmented sunburst chart for growth/degrowth."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    if metric_type == 'Value':
        current_col = 'Current_Period_Sales'
        last_year_col = 'Last_Year_Sales'
        last_month_col = 'Last_Month_Sales'
        growth_ly_col = 'Sales_Growth_LY'
        growth_lm_col = 'Sales_Growth_LM'
        divisor = 1_000_000
        unit_label = 'M'
        metric_label = 'Sales'
    else:
        current_col = 'Current_Period_Ltr'
        last_year_col = 'Last_Year_Ltr'
        last_month_col = 'Last_Month_Ltr'
        growth_ly_col = 'Ltr_Growth_LY'
        growth_lm_col = 'Ltr_Growth_LM'
        divisor = 1000
        unit_label = 'T'
        metric_label = 'Litres'

    growth_basis = st.session_state.get('brand_growth_basis_filter', 'Last Year')
    growth_col = growth_ly_col if growth_basis == 'Last Year' else growth_lm_col
    basis_label = 'Last Year' if growth_basis == 'Last Year' else 'Last Month'

    required_cols = ['Channel', 'brand', current_col, last_year_col, last_month_col, growth_col]
    missing_cols = [column for column in required_cols if column not in df.columns]
    if missing_cols:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing columns for chart: {', '.join(missing_cols)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        return fig

    def _ellipsis_label(text, max_len=14):
        text = str(text or '')
        return text if len(text) <= max_len else text[:max_len] + '...'

    df_processed = df[required_cols].copy()

    numeric_cols = [current_col, last_year_col, last_month_col, growth_col]
    for column in numeric_cols:
        df_processed[column] = pd.to_numeric(df_processed[column], errors='coerce').fillna(0)

    df_processed['Channel'] = df_processed['Channel'].fillna('Channel Empty').astype(str)
    df_processed['brand'] = df_processed['brand'].fillna('Unknown Brand').astype(str)

    df_processed['SegmentSize'] = df_processed[growth_col].abs().clip(lower=0)
    if df_processed['SegmentSize'].sum() <= 0:
        df_processed['SegmentSize'] = (df_processed[current_col].abs() / divisor).clip(lower=0)

    if df_processed['SegmentSize'].sum() <= 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No growth values available for selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    channel_summary = (
        df_processed
        .groupby('Channel', as_index=False)
        .agg(
            SegmentSize=('SegmentSize', 'sum'),
            Growth_Value=(growth_col, 'mean'),
            Current=(current_col, 'sum'),
            Last_Year=(last_year_col, 'sum'),
            Last_Month=(last_month_col, 'sum')
        )
    )

    def _growth_band(value):
        value = float(value)
        if value <= -30:
            return '<= -30%', '#FF6B6C'
        if value <= -20:
            return '-30% to -20%', '#F98080'
        if value <= -10:
            return '-20% to -10%', '#F8A3A3'
        if value < 0:
            return '-10% to 0%', '#FBCACA'
        if value == 0:
            return '0%', '#94A3B8'
        if value <= 10:
            return '0% to 10%', '#FFE7A3'
        if value <= 20:
            return '10% to 20%', '#FFD86B'
        if value <= 30:
            return '20% to 30%', '#FFC145'
        if value <= 50:
            return '30% to 50%', '#A7E3A2'
        if value <= 70:
            return '50% to 70%', '#6CCD7A'
        return '> 70%', '#16A34A'

    labels = []
    ids = []
    parents = []
    values = []
    colors = []
    hover_text = []
    text_values = []

    for _, row in channel_summary.iterrows():
        channel_name = str(row['Channel'])
        channel_label = _ellipsis_label(channel_name, 14)
        channel_id = f"channel::{channel_name}"
        labels.append(channel_label)
        ids.append(channel_id)
        parents.append("")
        values.append(float(row['SegmentSize']) if pd.notna(row['SegmentSize']) else 0.0)
        channel_growth = float(row['Growth_Value']) if pd.notna(row['Growth_Value']) else 0.0
        channel_band_label, channel_color = _growth_band(channel_growth)
        colors.append(channel_color)
        text_values.append(f"{channel_growth:.1f}%")
        hover_text.append(
            f"<b>{channel_name}</b><br>{basis_label} Growth (avg): {channel_growth:.1f}%"
            f"<br>Band: {channel_band_label}"
            f"<br>Current {metric_label}: {float(row['Current'])/divisor:.2f}{unit_label}"
            f"<br>Last Year {metric_label}: {float(row['Last_Year'])/divisor:.2f}{unit_label}"
            f"<br>Last Month {metric_label}: {float(row['Last_Month'])/divisor:.2f}{unit_label}<extra></extra>"
        )

    for _, row in df_processed.iterrows():
        channel_name = str(row['Channel'])
        brand_name = str(row['brand'])
        brand_label = _ellipsis_label(brand_name, 14)
        channel_id = f"channel::{channel_name}"
        brand_id = f"brand::{channel_name}::{brand_name}"

        labels.append(brand_label)
        ids.append(brand_id)
        parents.append(channel_id)
        values.append(float(row['SegmentSize']) if pd.notna(row['SegmentSize']) else 0.0)
        brand_growth = float(row[growth_col]) if pd.notna(row[growth_col]) else 0.0
        brand_band_label, brand_color = _growth_band(brand_growth)
        colors.append(brand_color)
        text_values.append(f"{brand_growth:.1f}%")
        hover_text.append(
            f"<b>{brand_name}</b><br>Channel: {channel_name}<br>{basis_label} Growth: {brand_growth:.1f}%"
            f"<br>Band: {brand_band_label}"
            f"<br>Current {metric_label}: {float(row[current_col])/divisor:.2f}{unit_label}"
            f"<br>Last Year {metric_label}: {float(row[last_year_col])/divisor:.2f}{unit_label}"
            f"<br>Last Month {metric_label}: {float(row[last_month_col])/divisor:.2f}{unit_label}<extra></extra>"
        )

    fig = go.Figure(
        go.Sunburst(
            labels=labels,
            ids=ids,
            parents=parents,
            values=values,
            branchvalues='total',
            marker=dict(
                colors=colors,
                line=dict(color='rgba(255,255,255,0.35)', width=1),
            ),
            hovertemplate='%{customdata}',
            customdata=hover_text,
            text=text_values,
            insidetextorientation='radial',
            textinfo='label+text',
            maxdepth=2,
        )
    )

    fig.update_layout(
        title=f"📈 Brand-wise Growth by Channel (Segmented) - {metric_type} | Basis: {basis_label}",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, weight='bold'),
        margin=dict(t=70, l=10, r=10, b=10)
    )
    return fig

def create_dm_wise_growth_chart(df, metric_type='Value'):
    """Create DM-wise growth percentage chart with value/ltr hover details."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    if metric_type == 'Value':
        current_col = 'Current_Period_Sales'
        last_year_col = 'Last_Year_Sales'
        last_month_col = 'Last_Month_Sales'
        growth_ly_col = 'Sales_Growth_LY'
        growth_lm_col = 'Sales_Growth_LM'
        divisor = 1_000_000
        unit_label = 'M'
        metric_label = 'Sales'
    else:
        current_col = 'Current_Period_Ltr'
        last_year_col = 'Last_Year_Ltr'
        last_month_col = 'Last_Month_Ltr'
        growth_ly_col = 'Ltr_Growth_LY'
        growth_lm_col = 'Ltr_Growth_LM'
        divisor = 1000
        unit_label = 'T'
        metric_label = 'Litres'

    df_processed = df[[
        'DeliveryMan', current_col, last_year_col, last_month_col,
        growth_ly_col, growth_lm_col
    ]].copy()

    numeric_cols = [current_col, last_year_col, last_month_col, growth_ly_col, growth_lm_col]
    for column in numeric_cols:
        df_processed[column] = pd.to_numeric(df_processed[column], errors='coerce').fillna(0)

    df_processed[current_col] = df_processed[current_col] / divisor
    df_processed[last_year_col] = df_processed[last_year_col] / divisor
    df_processed[last_month_col] = df_processed[last_month_col] / divisor

    current_vals = df_processed[current_col].round(2)
    last_year_vals = df_processed[last_year_col].round(2)
    last_month_vals = df_processed[last_month_col].round(2)

    customdata = np.column_stack([
        current_vals,
        last_year_vals,
        last_month_vals,
    ])

    ly_colors = ['#B8B8D1' if value >= 0 else '#FF6B6C' for value in df_processed[growth_ly_col]]
    lm_colors = ['#FFC145' if value >= 0 else '#FF6B6C' for value in df_processed[growth_lm_col]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_processed['DeliveryMan'],
            y=df_processed[growth_ly_col],
            name='Growth vs Last Year',
            marker=dict(color=ly_colors),
            text=df_processed[growth_ly_col].apply(lambda x: f"{x:.1f}%"),
            textposition='outside',
            customdata=customdata,
            hovertemplate=(
                '<b>%{x}</b>'
                '<br>Growth vs Last Year: %{y:.1f}%'
                '<br>Current ' + metric_label + ': %{customdata[0]:.2f}' + unit_label +
                '<br>Last Year ' + metric_label + ': %{customdata[1]:.2f}' + unit_label +
                '<br>Last Month ' + metric_label + ': %{customdata[2]:.2f}' + unit_label +
                '<extra></extra>'
            )
        )
    )
    fig.add_trace(
        go.Bar(
            x=df_processed['DeliveryMan'],
            y=df_processed[growth_lm_col],
            name='Growth vs Last Month',
            marker=dict(color=lm_colors),
            text=df_processed[growth_lm_col].apply(lambda x: f"{x:.1f}%"),
            textposition='outside',
            customdata=customdata,
            hovertemplate=(
                '<b>%{x}</b>'
                '<br>Growth vs Last Month: %{y:.1f}%'
                '<br>Current ' + metric_label + ': %{customdata[0]:.2f}' + unit_label +
                '<br>Last Year ' + metric_label + ': %{customdata[1]:.2f}' + unit_label +
                '<br>Last Month ' + metric_label + ': %{customdata[2]:.2f}' + unit_label +
                '<extra></extra>'
            )
        )
    )
    fig.update_layout(
        title=f"📈 Deliveryman-wise Growth Percentage - {metric_type}",
        yaxis=dict(title='Growth %'),
        xaxis=dict(title='Deliveryman'),
        # height=600,
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, weight='bold'),
        legend=dict( orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    return apply_theme_aware_bar_labels(fig)

def create_target_achievement_chart(df, metric_type='Value'):

    """Create target vs achievement comparison chart"""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
        
    if metric_type == 'Value':
        df = df.rename(columns={'Target_Value': 'Target', 'NMV': 'Achievement'})
        ach_percent_col = np.where(
            pd.to_numeric(df['Target'], errors='coerce').fillna(0) > 0,
            (pd.to_numeric(df['Achievement'], errors='coerce').fillna(0) / pd.to_numeric(df['Target'], errors='coerce').fillna(0)) * 100,
            0,
        )
        ach_percent_col = pd.Series(ach_percent_col).round(1)
        divisor = 1_000_000
        unit_label = 'M'
        metric_label = 'Sales'
        y_axis_title = 'Sales (in Millions)'
    else:
        df = df.rename(columns={'Target_Ltr': 'Target', 'Ltr': 'Achievement'})
        ach_percent_col = np.where(
            pd.to_numeric(df['Target'], errors='coerce').fillna(0) > 0,
            (pd.to_numeric(df['Achievement'], errors='coerce').fillna(0) / pd.to_numeric(df['Target'], errors='coerce').fillna(0)) * 100,
            0,
        )
        ach_percent_col = pd.Series(ach_percent_col).round(1)
        divisor = 1_000
        unit_label = 'T'
        metric_label = 'Volume'
        y_axis_title = 'Volume (in Thousands)'

    
    df['period'] = pd.to_datetime(df['period'], format='%m-%Y')
    df.sort_values('period', inplace=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df['period'],
            y=df['Target'],
            name='Target',
            marker=dict(color='#5B5F97'),
            text =(df['Target'] / divisor).round(2).astype(str) + unit_label,
            textposition='inside',
            hovertemplate=f'<b>%{{x|%b %Y}}</b><br>Target: %{{y/divisor:.2f}}{unit_label}<extra></extra>'
        )
    )
    fig.add_trace(
        go.Bar(
            x=df['period'],
            y=df['Achievement'],
            name='Achievement',
            marker=dict(color='#FFC145'),
            text = (
                    (df['Achievement'] / divisor).round(2).astype(str) + unit_label
                        + " | "
                    + ach_percent_col.astype(str)
                        + "%"),
            textposition='inside',
            hovertemplate=f'<b>%{{x|%b %Y}}</b><br>Achievement: %{{y/divisor:.2f}}{unit_label}<extra></extra>'
        )
    )
    fig.update_layout(
        title=f"🎯 Target vs Achievement Comparison - {metric_label}",
        yaxis=dict(title=y_axis_title),
        xaxis=dict(title='Period'),
        # height=600,
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        # font=dict(color='white')
    )
    return apply_theme_aware_bar_labels(fig)

def brand_wise_productivity_chart(df,selected_dms=None, selected_channels=None):
    """Create brand-wise productivity chart with optional DM and channel filters."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    if selected_dms:
        df = df[df['DM'].isin(selected_dms)]
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data for selected deliverymen",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
    
    if selected_channels:
        df = df[df['Channel'].isin(selected_channels)]
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data for selected channels",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig

    df_plot = df.copy()
    brand_col = 'Brand' if 'Brand' in df_plot.columns else 'brand'
    df_plot['StoreCount'] = pd.to_numeric(df_plot['StoreCount'], errors='coerce').fillna(0)
    brand_prod = df_plot.groupby(brand_col)['StoreCount'].sum().reset_index()  # Aggregate productivity by brand
    brand_prod = brand_prod.sort_values('StoreCount', ascending=False)  # Sort by productivity

    fig = px.bar(
        brand_prod,
        y=brand_col,
        x='StoreCount',
        title="📊 Brand-wise Productivity",
        labels={brand_col: 'Brand', 'StoreCount': 'Productivity (Units per Store)'},
        color_discrete_sequence=['#5B5F97'] * len(brand_prod),
        orientation='h',
    )
    
    fig.update_traces(
        text=brand_prod['StoreCount'].round(1).astype(str),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Productivity: %{x:.1f} Units/Store<extra></extra>'
    )
    
    fig.update_layout(
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        # font=dict(color='white'),
        coloraxis_showscale=False
    )
    
    return apply_theme_aware_bar_labels(fig)

# Create heatmap with Booker on the y-axis and periods on the x-axis from tgtvsach_YTD_heatmap_data
def create_booker_period_heatmap(df, metric_type='Value'):
    """Create heatmap of target vs achievement percentages by Booker and Period."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    df_plot = df.copy()

    if metric_type == 'Value':
        metric_col = 'NMV'
        ach_col = 'Value_Ach'
        divisor = 1_000_000
        unit_label = ''
        metric_label = 'Value'
    else:
        metric_col = 'Ltr'
        ach_col = 'Ltr_Ach'
        divisor = 1_000
        unit_label = ''
        metric_label = 'Ltr'

    required_cols = {'Booker', 'period', metric_col, ach_col}
    if not required_cols.issubset(df_plot.columns):
        fig = go.Figure()
        fig.add_annotation(
            text="Required heatmap columns not found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    df_plot['period'] = pd.to_datetime(df_plot['period'], format='%m-%Y', errors='coerce')
    df_plot[metric_col] = pd.to_numeric(df_plot[metric_col], errors='coerce')
    df_plot[ach_col] = pd.to_numeric(df_plot[ach_col], errors='coerce')
    df_plot = df_plot.dropna(subset=['Booker', 'period', metric_col, ach_col])

    if df_plot.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    df_plot.sort_values('period', inplace=True)
    df_plot['metric_label'] = (df_plot[metric_col] / divisor).round(1).astype(str) + unit_label

    heatmap_data = df_plot.pivot_table(index='Booker', columns='period', values=ach_col, aggfunc='mean')
    label_data = df_plot.pivot_table(index='Booker', columns='period', values=metric_col, aggfunc='sum')
    label_data = (label_data / divisor).round(1)
    label_data = label_data.reindex(index=heatmap_data.index, columns=heatmap_data.columns)

    x_labels = heatmap_data.columns.strftime('%b %Y')
    label_text = label_data.applymap(lambda value: f"{value:.1f}{unit_label}" if pd.notna(value) else "")

    # Tableau-like discrete color bins for achievement %
    zmin, zmax = 0, 130
    discrete_bins = [0, 70, 85, 100, 115, 130]
    discrete_colors = ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']
    discrete_colorscale = []
    for idx, color in enumerate(discrete_colors):
        start = (discrete_bins[idx] - zmin) / (zmax - zmin)
        end = (discrete_bins[idx + 1] - zmin) / (zmax - zmin)
        discrete_colorscale.append([start, color])
        discrete_colorscale.append([end, color])

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=x_labels,
        y=heatmap_data.index,
        text=label_text.values,
        customdata=label_text.values,
        colorscale=discrete_colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(
            title='Achievement %',
            tickvals=[35, 77.5, 92.5, 107.5, 122.5],
            ticktext=['0-70%', '70-85%', '85-100%', '100-115%', '115-130%']
        ),
        hovertemplate=(
            '<b>%{y}</b><br>%{x}'
            '<br>Achievement: %{z:.1f}%'
            '<br>' + metric_label + ': %{customdata}'
            '<extra></extra>'
        )
    ))

    annotations = []
    for row_idx, booker in enumerate(heatmap_data.index):
        for col_idx, period_label in enumerate(x_labels):
            text_value = label_text.iat[row_idx, col_idx]
            if text_value:
                annotations.append(
                    dict(
                        x=period_label,
                        y=booker,
                        text=text_value,
                        showarrow=False,
                        font=dict(size=10, color='white')
                    )
                )

    fig.update_layout(
        title=f"🔥 Booker Target Achievement Heatmap - {metric_label}",
        # xaxis_title="Period",
        yaxis_title="Booker",
        height=600,
        annotations=annotations,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        
    )

    return fig

def create_channel_heatmap_YTD(df, metric_type='Value'):
    """Create heatmap of channel performance by month for YTD data."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    df_plot = df.copy()

    if metric_type == 'Value':
        metric_col = 'NMV'
        divisor = 1_000_000
        unit_label = 'M'
        metric_label = 'Value'
    else:
        metric_col = 'Ltr'
        divisor = 1_000
        unit_label = 'T'
        metric_label = 'Volume'

    required_cols = {'Channel', 'period', metric_col}
    if not required_cols.issubset(df_plot.columns):
        fig = go.Figure()
        fig.add_annotation(
            text="Required heatmap columns not found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    df_plot['period'] = pd.to_datetime(df_plot['period'], format='%Y-%m', errors='coerce')
    if df_plot['period'].isna().all():
        df_plot['period'] = pd.to_datetime(df_plot['period'], format='%m-%Y', errors='coerce')
    df_plot[metric_col] = pd.to_numeric(df_plot[metric_col], errors='coerce')
    df_plot = df_plot.dropna(subset=['Channel', 'period', metric_col])

    if df_plot.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    df_plot.sort_values('period', inplace=True)
    df_plot['metric_label'] = (df_plot[metric_col] / divisor).round(1).astype(str) + unit_label

    heatmap_data = df_plot.pivot_table(index='Channel', columns='period', values=metric_col, aggfunc='sum')
    label_data = (heatmap_data / divisor).round(1)

    x_labels = heatmap_data.columns.strftime('%b %Y')
    label_text = label_data.applymap(lambda value: f"{value:.1f}{unit_label}" if pd.notna(value) else "")
    
    discrete_colors = ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=x_labels,
        y=heatmap_data.index,
        text=label_text.values,
        customdata=label_text.values,
        colorscale=discrete_colors,
        hovertemplate=(
            '<b>%{y}</b><br>%{x}'
            '<br>' + metric_label + ': %{customdata}'
            '<extra></extra>'
        )
    ))
    annotations = []
    for row_idx, channel in enumerate(heatmap_data.index):
        for col_idx, period_label in enumerate(x_labels):
            text_value = label_text.iat[row_idx, col_idx]
            if text_value:
                annotations.append(
                    dict(
                        x=period_label,
                        y=channel,
                        text=text_value,
                        showarrow=False,
                        font=dict(size=10, color='white')
                    )
                )
    fig.update_layout(
        title=f"📊 Channel Performance Heatmap - {metric_label}",
        # xaxis_title="Period",
        yaxis_title="Channel",
        # height=700,
        annotations=annotations,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        # font=dict(color='white')
    )
    return fig


def create_gmv_ob_calendar_heatmap(
    df,
    selected_bookers=None,
):
    """Create GitHub-style calendar heatmap (weekday x week) using GMV."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    required_cols = {'Order_Date', 'Order Booker Name', 'GMV', 'Orders'}
    if not required_cols.issubset(df.columns):
        fig = go.Figure()
        fig.add_annotation(
            text="Required columns for calendar heatmap not found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    df_plot = df.copy()
    df_plot['Order_Date'] = pd.to_datetime(df_plot['Order_Date'], errors='coerce')
    df_plot['GMV'] = pd.to_numeric(df_plot['GMV'], errors='coerce').fillna(0)
    df_plot['Orders'] = pd.to_numeric(df_plot['Orders'], errors='coerce').fillna(0)
    df_plot = df_plot.dropna(subset=['Order_Date', 'Order Booker Name'])

    if selected_bookers:
        df_plot = df_plot[df_plot['Order Booker Name'].astype(str).isin([str(booker) for booker in selected_bookers])]

    if df_plot.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    daily_totals = (
        df_plot
        .groupby('Order_Date', as_index=False)
        .agg({'GMV': 'sum', 'Orders': 'sum'})
        .sort_values('Order_Date')
    )

    min_date = daily_totals['Order_Date'].min().normalize()
    max_date = daily_totals['Order_Date'].max().normalize()
    all_dates = pd.DataFrame({'Order_Date': pd.date_range(min_date, max_date, freq='D')})
    calendar_df = all_dates.merge(daily_totals, on='Order_Date', how='left')
    calendar_df['GMV'] = calendar_df['GMV'].fillna(0)
    calendar_df['Orders'] = calendar_df['Orders'].fillna(0)

    calendar_df['weekday_idx'] = (calendar_df['Order_Date'].dt.dayofweek + 1) % 7
    calendar_df['week_start'] = calendar_df['Order_Date'] - pd.to_timedelta(calendar_df['weekday_idx'], unit='D')
    min_week_start = calendar_df['week_start'].min()
    calendar_df['week_idx'] = ((calendar_df['week_start'] - min_week_start).dt.days // 7).astype(int)

    calendar_df['band'] = np.select(
        [
            calendar_df['GMV'] <= 40000,
            (calendar_df['GMV'] > 40000) & (calendar_df['GMV'] <= 50000),
            (calendar_df['GMV'] > 50000) & (calendar_df['GMV'] <= 100000),
            (calendar_df['GMV'] > 100000) & (calendar_df['GMV'] <= 500000),
            calendar_df['GMV'] > 500000,
        ],
        [0, 1, 2, 3, 4],
        default=0,
    ).astype(float)

    discrete_colors = ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']
    discrete_colorscale = []
    for idx, color in enumerate(discrete_colors):
        start = idx / len(discrete_colors)
        end = (idx + 1) / len(discrete_colors)
        discrete_colorscale.append([start, color])
        discrete_colorscale.append([end, color])

    month_starts = pd.date_range(min_date.replace(day=1), max_date, freq='MS')
    month_tickvals = []
    month_ticktext = []
    for month_start in month_starts:
        weekday_idx = (month_start.dayofweek + 1) % 7
        month_week_start = month_start - pd.Timedelta(days=weekday_idx)
        week_idx = int((month_week_start - min_week_start).days // 7)
        if week_idx not in month_tickvals:
            month_tickvals.append(week_idx)
            month_ticktext.append(month_start.strftime('%b'))

    fig = go.Figure(
        data=go.Scatter(
            x=calendar_df['week_idx'],
            y=calendar_df['weekday_idx'],
            mode='markers',
            customdata=np.column_stack([
                calendar_df['Order_Date'].dt.strftime('%d-%b-%Y'),
                calendar_df['GMV'],
                calendar_df['Orders']
            ]),
            marker=dict(
                symbol='circle',
                size=14,
                color=calendar_df['band'],
                colorscale=discrete_colorscale,
                cmin=0,
                cmax=4,
                line=dict(color='white', width=1),
                colorbar=dict(
                    title='GMV Bands',
                    tickvals=[0, 1, 2, 3, 4],
                    ticktext=['0-40K', '40K-50K', '50K-100K', '100K-500K', '500K+'],
                    len=0.65,
                    thickness=14,
                ),
            ),
            hovertemplate=(
                '<b>Date:</b> %{customdata[0]}'
                '<br><b>GMV:</b> %{customdata[1]:,.0f}'
                '<br><b>Visits(Orders):</b> %{customdata[2]:,.0f}'
                '<extra></extra>'
            ),
        )
    )

    fig.update_layout(
        title='📅 GMV Calendar Heatmap',
        xaxis=dict(
            tickmode='array',
            tickvals=month_tickvals,
            ticktext=month_ticktext,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            ticktext=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
            autorange='reversed',
            showgrid=False,
            zeroline=False,
        ),
        height=260,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=8, r=8, t=42, b=8),
        font=dict(color=get_theme_text_color() or '#111827')
    )
    return fig

def create_daily_sales_trend_orders_gmv(df, selected_bookers=None, selected_channels=None, title_suffix=""):
    """Create daily trend chart for Orders and GMV."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No daily trend data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    required_cols = {'Order_Date', 'Order Booker Name', 'GMV', 'Orders'}
    if not required_cols.issubset(df.columns):
        fig = go.Figure()
        fig.add_annotation(
            text="Required columns for daily trend not found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    df_plot = df.copy()
    df_plot['Order_Date'] = pd.to_datetime(df_plot['Order_Date'], errors='coerce')
    df_plot['GMV'] = pd.to_numeric(df_plot['GMV'], errors='coerce').fillna(0)
    df_plot['Orders'] = pd.to_numeric(df_plot['Orders'], errors='coerce').fillna(0)
    df_plot = df_plot.dropna(subset=['Order_Date'])

    if selected_bookers:
        df_plot = df_plot[df_plot['Order Booker Name'].astype(str).isin([str(booker) for booker in selected_bookers])]
    if selected_channels and 'Channel' in df_plot.columns:
        df_plot = df_plot[df_plot['Channel'].astype(str).isin([str(channel) for channel in selected_channels])]

    if df_plot.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No daily trend data after filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    trend_df = (
        df_plot
        .groupby('Order_Date', as_index=False)
        .agg({'GMV': 'sum', 'Orders': 'sum'})
        .sort_values('Order_Date')
    )
    trend_df['GMV_M'] = trend_df['GMV'] / 1_000_000

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=trend_df['Order_Date'],
            y=trend_df['Orders'],
            name='Orders',
            marker_color='#5B5F97',
            hovertemplate='<b>%{x|%d-%b-%Y}</b><br>Orders: %{y:,.0f}<extra></extra>',
            yaxis='y'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend_df['Order_Date'],
            y=trend_df['GMV_M'],
            mode='lines+markers',
            name='GMV',
            line=dict(color='#FFC145', width=2),
            marker=dict(size=6, color='#FFC145'),
            hovertemplate='<b>%{x|%d-%b-%Y}</b><br>GMV: %{y:.2f}M<extra></extra>',
            yaxis='y2'
        )
    )

    fig.update_layout(
        title=f'📈 Daily Sales Trend (Orders & GMV){title_suffix}',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Orders', side='left', showgrid=True),
        yaxis2=dict(title='GMV (Millions)', side='right', overlaying='y', showgrid=False),
        barmode='group',
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=get_theme_text_color() or '#111827'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=8, r=8, t=42, b=8)
    )
    return fig

def create_top_booker_deep_chart(booker_df):
    """Create top Booker chart for deep analysis tab."""
    if booker_df is None or booker_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No booker data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    df_plot = booker_df.sort_values('NMV', ascending=False).head(12).copy()
    df_plot['NMV_M'] = pd.to_numeric(df_plot['NMV'], errors='coerce').fillna(0) / 1_000_000

    fig = px.bar(
        df_plot,
        x='Booker',
        y='NMV_M',
        color='Orders',
        color_continuous_scale='Blues',
        text=df_plot['AOV'].apply(lambda value: f"AOV {value/1000:.1f}K" if pd.notna(value) and value >= 1000 else f"AOV {value:,.0f}"),
        title='🏆 Top Booker by NMV'
    )
    fig.update_traces(
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>NMV: %{y:.2f}M<br>Orders: %{marker.color:,.0f}<extra></extra>'
    )
    fig.update_layout(
        # xaxis_title='Booker',
        yaxis_title='NMV (Millions)',
        xaxis_tickangle=-30,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=8, r=8, t=42, b=8)
    )
    return fig

def create_fieldforce_efficiency_chart(dm_df):
    """Create field force efficiency bubble chart."""
    if dm_df is None or dm_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No field force data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    df_plot = dm_df.copy()
    df_plot['NMV_M'] = pd.to_numeric(df_plot['NMV'], errors='coerce').fillna(0) / 1_000_000

    fig = px.scatter(
        df_plot,
        x='Orders',
        y='NMV_M',
        size='Stores',
        color='AOV',
        text='Deliveryman',
        color_continuous_scale='Viridis',
        title='🚚 Field Force Efficiency (DM)'
    )
    fig.update_traces(
        textposition='top center',
        marker=dict(opacity=0.85, line=dict(width=1, color='white')),
        hovertemplate='<b>%{text}</b><br>Orders: %{x:,.0f}<br>NMV: %{y:.2f}M<br>Stores: %{marker.size:,.0f}<extra></extra>'
    )
    fig.update_layout(
        # xaxis_title='Orders',
        yaxis_title='NMV (Millions)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=8, r=8, t=42, b=8)
    )
    return fig

def create_routewise_sales_performance_chart(df, title_suffix=""):
    """Create route-wise (OB-wise) achieved vs target percentage chart."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No route-wise performance data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    df_plot = df.copy()
    df_plot['Achieved_Value'] = pd.to_numeric(df_plot.get('Achieved_Value', 0), errors='coerce').fillna(0)
    df_plot['Target_Value'] = pd.to_numeric(df_plot.get('Target_Value', 0), errors='coerce').fillna(0)
    df_plot['Achieved_Pct'] = pd.to_numeric(df_plot.get('Achieved_Pct', 0), errors='coerce').fillna(0)
    df_plot['Booker'] = df_plot.get('Booker', '').astype(str)
    df_plot = df_plot.sort_values('Booker', ascending=True)
    df_plot['Target_Pct'] = 100

    def _achieved_band_color(pct):
        pct = float(pct or 0)
        if pct < 50:
            return '#EF4444'
        if pct < 60:
            return '#F97316'
        if pct <= 70:
            return '#FACC15'
        return '#22C55E'

    def _achieved_band_label(pct):
        pct = float(pct or 0)
        if pct < 50:
            return 'Red (<50%)'
        if pct < 60:
            return 'Orange (50-60%)'
        if pct <= 70:
            return 'Yellow (60-70%)'
        return 'Green (>70%)'

    achieved_colors = df_plot['Achieved_Pct'].apply(_achieved_band_color).tolist()
    achieved_bands = df_plot['Achieved_Pct'].apply(_achieved_band_label).tolist()

    def _short_label(text, max_len=12):
        text = str(text)
        return text if len(text) <= max_len else f"{text[:max_len-1]}…"

    achieved_customdata = np.column_stack([df_plot['Booker'], df_plot['Achieved_Value'], df_plot['Target_Value'], achieved_bands])
    target_customdata = np.column_stack([df_plot['Booker'], df_plot['Target_Value'], df_plot['Achieved_Value']])

    x_values = df_plot['Booker'].astype(str).tolist()
    x_tick_text = [ _short_label(booker, max_len=14) for booker in x_values ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_values,
            y=df_plot['Achieved_Pct'],
            name='Achieved',
            marker_color=achieved_colors,
            text=df_plot['Achieved_Pct'].apply(lambda value: f"{value:.1f}%"),
            textposition='outside',
            customdata=achieved_customdata,
            hovertemplate=(
                '<b>%{customdata[0]}</b>'
                '<br>Achieved %: %{y:.1f}%'
                '<br>Achieved Value: Rs %{customdata[1]:,.0f}'
                '<br>Target Value: Rs %{customdata[2]:,.0f}'
                '<br>Band: %{customdata[3]}'
                '<extra></extra>'
            )
        )
    )
    fig.add_trace(
        go.Bar(
            x=x_values,
            y=df_plot['Target_Pct'],
            name='Target',
            marker_color='#9CA3AF',
            text=df_plot['Target_Value'].apply(lambda value: f"{value/1e6:.0f}M"),
            textposition='outside',
            customdata=target_customdata,
            # hovertemplate=(
            #     '<b>%{customdata[0]}</b>'
            #     '<br>Target %: %{y:.0f}%'
            #     '<br>Target Value: Rs %{customdata[1]:,.0f}'
            #     '<br>Achieved Value: Rs %{customdata[2]:,.0f}'
            #     '<extra></extra>'
            # )
        )
    )

    fig.update_layout(
        title=f'📊 Route-wise Sales Performance (Achieved vs Target per OB){title_suffix}',
        yaxis=dict(title='Achievement %'),
        xaxis=dict(
            tickmode='array',
            tickvals=x_values,
            ticktext=x_tick_text,
            tickangle=-25,
        ),
        barmode='group',
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=get_theme_text_color() or '#111827'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=8, r=8, t=42, b=8)
    )
    return apply_theme_aware_bar_labels(fig)

def create_daily_calls_trend_chart(df, selected_bookers=None, title_suffix=""):
    """Create daily calls trend chart with planned vs executed calls."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No daily calls data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    required_cols = {'Call_Date', 'Booker', 'Planned_Calls', 'Executed_Calls'}
    if not required_cols.issubset(df.columns):
        fig = go.Figure()
        fig.add_annotation(
            text="Required columns for calls trend not found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    df_plot = df.copy()
    df_plot['Call_Date'] = pd.to_datetime(df_plot['Call_Date'], errors='coerce')
    df_plot['Planned_Calls'] = pd.to_numeric(df_plot['Planned_Calls'], errors='coerce').fillna(0)
    df_plot['Executed_Calls'] = pd.to_numeric(df_plot['Executed_Calls'], errors='coerce').fillna(0)
    df_plot = df_plot.dropna(subset=['Call_Date'])

    if selected_bookers:
        df_plot = df_plot[df_plot['Booker'].astype(str).isin([str(booker) for booker in selected_bookers])]

    if df_plot.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No daily calls data after filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    trend_df = (
        df_plot
        .groupby('Call_Date', as_index=False)
        .agg({'Planned_Calls': 'sum', 'Executed_Calls': 'sum'})
        .sort_values('Call_Date')
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend_df['Call_Date'],
            y=trend_df['Executed_Calls'],
            name='Executed Calls',
            mode='lines',
            line=dict(color='#FFC145', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(255,193,69,0.20)',
            hovertemplate='<b>%{x|%d-%b-%Y}</b><br>Executed Calls: %{y:,.0f}<extra></extra>'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend_df['Call_Date'],
            y=trend_df['Planned_Calls'],
            name='Planned Calls',
            mode='lines',
            line=dict(color='#5B5F97', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(91,95,151,0.20)',
            hovertemplate='<b>%{x|%d-%b-%Y}</b><br>Planned Calls: %{y:,.0f}<extra></extra>'
        )
    )
    fig.update_layout(
        title=f'📞 Daily Calls Trend (Planned vs Executed){title_suffix}',
        # xaxis=dict(title='Date'),
        yaxis=dict(title='Calls'),
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=get_theme_text_color() or '#111827'),
        legend=dict(orientation='h', yanchor='top', y=-0.08, xanchor='center', x=0.5),
        
        margin=dict(l=8, r=8, t=42, b=8)
    )
    return fig

def render_booker_leaderboard_table(df, title, table_key):
    """Render styled leaderboard table using HTML."""
    if df is None or df.empty:
        st.info(f"No data available for {title}.")
        return

    rows = []
    for idx, row in df.reset_index(drop=True).iterrows():
        rank = idx + 1
        perf_score = int(round(float(row.get('Perf_Score', 0))))
        strike_rate = float(row.get('Strike_Rate', 0))
        calls_per_day = float(row.get('Calls_Per_Day', 0))
        target_value = float(row.get('Target_Value', 0))
        revenue = float(row.get('Revenue', 0))
        new_outlets = int(round(float(row.get('New_Outlets', 0))))
        avg_order_val = float(row.get('Avg_Order_Val', 0))
        sku_per_bill = float(row.get('SKU_Per_Bill', 0))
        achievement_pct = float(row.get('Achievement_Pct', 0))
        booker = str(row.get('Booker', ''))
        region = str(row.get('Region', 'Unknown'))

        medal_bg = '#FFB347' if rank == 1 else ('#AAB7C4' if rank == 2 else ('#FF7A59' if rank == 3 else 'transparent'))
        medal_style = (
            f"background:{medal_bg};color:#0f172a;border-radius:50%;display:inline-flex;"
            "align-items:center;justify-content:center;width:30px;height:30px;font-weight:700;"
        ) if rank <= 3 else "color:#60A5FA;font-weight:700;"

        progress_pct = max(0, min(perf_score, 100))
        achievement_progress = max(0, min(achievement_pct, 100))

        if achievement_pct < 50:
            ach_color = '#EF4444'
        elif achievement_pct >= 70:
            ach_color = '#22C55E'
        else:
            ach_color = '#F59E0B'

        rows.append(
            "<tr style='border-bottom:1px solid rgba(148,163,184,0.18);'>"
            f"<td style='padding:10px 8px;'><span style=\"{medal_style}\">{rank}</span></td>"
            f"<td style='padding:10px 8px;color:#F8FAFC;font-weight:600;'>{booker}</td>"
            f"<td style='padding:10px 8px;color:#60A5FA;'>{region}</td>"
            f"<td style='padding:10px 8px;color:#00FF85;font-weight:700;'>{strike_rate:.1f}%</td>"
            f"<td style='padding:10px 8px;color:#93C5FD;'>{calls_per_day:.1f}</td>"
            f"<td style='padding:10px 8px;color:#93C5FD;'>Rs {target_value/1_000_000:.1f}M</td>"
            f"<td style='padding:10px 8px;color:#00E5FF;font-weight:700;'>Rs {revenue/1_000_000:.1f}M</td>"
            f"<td style='padding:10px 8px;color:#93C5FD;'>{sku_per_bill:.2f}</td>"
            f"<td style='padding:10px 8px;color:#93C5FD;'>{new_outlets}</td>"
            f"<td style='padding:10px 8px;color:#93C5FD;'>Rs {avg_order_val:,.0f}</td>"
            "<td style='padding:10px 8px;'>"
            "<div style='display:flex;align-items:center;gap:10px;'>"
            "<div style='background:#1E3A5F;height:6px;width:120px;border-radius:999px;overflow:hidden;'>"
            f"<div style='background:{ach_color};height:100%;width:{achievement_progress}%;'></div>"
            "</div>"
            f"<span style='color:{ach_color};font-weight:700;'>{achievement_pct:.1f}%</span>"
            "</div>"
            "</td>"
            "<td style='padding:10px 8px;'>"
            "<div style='display:flex;align-items:center;gap:10px;'>"
            "<div style='background:#1E3A5F;height:6px;width:120px;border-radius:999px;overflow:hidden;'>"
            f"<div style='background:#00FF85;height:100%;width:{progress_pct}%;'></div>"
            "</div>"
            f"<span style='color:#00FF85;font-weight:700;'>{perf_score}</span>"
            "</div>"
            "</td>"
            "</tr>"
        )

    rows_html = "".join(rows)
    achievement_legend_html = (
        "<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:0 0 8px 0;'>"
        "<span style='font-size:11px;letter-spacing:1px;color:#93C5FD;text-transform:uppercase;'>Achievement Bands:</span>"
        "<span style='display:inline-flex;align-items:center;gap:6px;font-size:11px;color:#E2E8F0;'>"
        "<span style='width:10px;height:10px;border-radius:2px;background:#EF4444;display:inline-block;'></span>&lt; 50%"
        "</span>"
        "<span style='display:inline-flex;align-items:center;gap:6px;font-size:11px;color:#E2E8F0;'>"
        "<span style='width:10px;height:10px;border-radius:2px;background:#F59E0B;display:inline-block;'></span>50% - 69.9%"
        "</span>"
        "<span style='display:inline-flex;align-items:center;gap:6px;font-size:11px;color:#E2E8F0;'>"
        "<span style='width:10px;height:10px;border-radius:2px;background:#22C55E;display:inline-block;'></span>≥ 70%"
        "</span>"
        "</div>"
    )

    table_html = (
        f"<div id='{table_key}' style='background:#071A2E;border:1px solid rgba(59,130,246,0.25);border-radius:12px;padding:12px 12px 8px;'>"
        f"{achievement_legend_html}"
        f"<div style='font-size:20px;font-weight:700;color:#F8FAFC;margin-bottom:6px;'>{title}</div>"
        "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        "<thead>"
        "<tr style='text-transform:uppercase;letter-spacing:2px;font-size:11px;color:#60A5FA;'>"
        "<th style='text-align:left;padding:8px;'>Rank</th>"
        "<th style='text-align:left;padding:8px;'>Booker</th>"
        "<th style='text-align:left;padding:8px;'>Region</th>"
        "<th style='text-align:left;padding:8px;'>Strike Rate</th>"
        "<th style='text-align:left;padding:8px;'>Calls/Day</th>"
        "<th style='text-align:left;padding:8px;'>Target</th>"
        "<th style='text-align:left;padding:8px;'>Revenue</th>"
        "<th style='text-align:left;padding:8px;'>SKU/Bill</th>"
        "<th style='text-align:left;padding:8px;'>New Outlets</th>"
        "<th style='text-align:left;padding:8px;'>Avg Order Val</th>"
        "<th style='text-align:left;padding:8px;'>Achievement %</th>"
        "<th style='text-align:left;padding:8px;'>Perf Score</th>"
        "</tr>"
        "</thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
        "</div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)
def render_top_bottom_brand_table(df, height_px=420):
    """Render Top/Bottom Brand table with sticky Booker column."""
    if df is None or df.empty:
        st.info("No Top/Bottom brand data available.")
        return

    table_df = df.copy()
    expected_cols = [
        'Booker', 'Top Brand', 'Top Brand NMV', 'Top Brand Score %',
        'Bottom Brand', 'Bottom Brand NMV', 'Bottom Brand Score %'
    ]
    for column in expected_cols:
        if column not in table_df.columns:
            table_df[column] = ''
    table_df = table_df[expected_cols]

    def _fmt_num(value):
        numeric = pd.to_numeric(value, errors='coerce')
        if pd.isna(numeric):
            return '-'
        return f"{numeric:,.0f}"

    def _fmt_pct(value):
        numeric = pd.to_numeric(value, errors='coerce')
        if pd.isna(numeric):
            return '-'
        return f"{numeric:.1f}%"

    rows_html = []
    for _, row in table_df.iterrows():
        rows_html.append(
            "<tr style='border-bottom:1px solid #EEF2F7;'>"
            f"<td style='position:sticky;left:0;background:#FFFFFF;z-index:1;padding:8px 10px;color:#0F172A;font-size:12px;font-weight:600;white-space:nowrap;'>{escape(str(row.get('Booker', '')))}</td>"
            f"<td style='padding:8px 10px;color:#334155;font-size:12px;white-space:nowrap;'>{escape(str(row.get('Top Brand', '-')))}</td>"
            f"<td style='padding:8px 10px;color:#334155;font-size:12px;text-align:right;white-space:nowrap;'>Rs {_fmt_num(row.get('Top Brand NMV'))}</td>"
            f"<td style='padding:8px 10px;color:#334155;font-size:12px;text-align:right;white-space:nowrap;'>{_fmt_pct(row.get('Top Brand Score %'))}</td>"
            f"<td style='padding:8px 10px;color:#334155;font-size:12px;white-space:nowrap;'>{escape(str(row.get('Bottom Brand', '-')))}</td>"
            f"<td style='padding:8px 10px;color:#334155;font-size:12px;text-align:right;white-space:nowrap;'>Rs {_fmt_num(row.get('Bottom Brand NMV'))}</td>"
            f"<td style='padding:8px 10px;color:#334155;font-size:12px;text-align:right;white-space:nowrap;'>{_fmt_pct(row.get('Bottom Brand Score %'))}</td>"
            "</tr>"
        )

    table_html = (
        "<div style='border:1px solid #D9E3EF;border-radius:12px;background:#FFFFFF;overflow:hidden;'>"
        f"<div style='max-height:{int(height_px)}px;overflow:auto;'>"
        "<table style='width:100%;border-collapse:separate;border-spacing:0;'>"
        "<thead><tr>"
        "<th style='position:sticky;top:0;left:0;z-index:3;background:#F8FAFC;border-bottom:1px solid #E2E8F0;padding:8px 10px;text-align:left;color:#334155;font-size:12px;white-space:nowrap;'>Booker</th>"
        "<th style='position:sticky;top:0;z-index:2;background:#F8FAFC;border-bottom:1px solid #E2E8F0;padding:8px 10px;text-align:left;color:#334155;font-size:12px;white-space:nowrap;'>Top Brand</th>"
        "<th style='position:sticky;top:0;z-index:2;background:#F8FAFC;border-bottom:1px solid #E2E8F0;padding:8px 10px;text-align:right;color:#334155;font-size:12px;white-space:nowrap;'>Top Brand NMV</th>"
        "<th style='position:sticky;top:0;z-index:2;background:#F8FAFC;border-bottom:1px solid #E2E8F0;padding:8px 10px;text-align:right;color:#334155;font-size:12px;white-space:nowrap;'>Top Brand Score %</th>"
        "<th style='position:sticky;top:0;z-index:2;background:#F8FAFC;border-bottom:1px solid #E2E8F0;padding:8px 10px;text-align:left;color:#334155;font-size:12px;white-space:nowrap;'>Bottom Brand</th>"
        "<th style='position:sticky;top:0;z-index:2;background:#F8FAFC;border-bottom:1px solid #E2E8F0;padding:8px 10px;text-align:right;color:#334155;font-size:12px;white-space:nowrap;'>Bottom Brand NMV</th>"
        "<th style='position:sticky;top:0;z-index:2;background:#F8FAFC;border-bottom:1px solid #E2E8F0;padding:8px 10px;text-align:right;color:#334155;font-size:12px;white-space:nowrap;'>Bottom Brand Score %</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table></div></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


ACHIEVEMENT_BAND_COLORS = {
    "Below 50%": "#d62728",
    "50-59%": "#ff7f0e",
    "60-69%": "#bcbd22",
    "70%+": "#2ca02c",
}

def get_achievement_band(value):
    if value < 50:
        return "Below 50%"
    if value < 60:
        return "50-59%"
    if value < 70:
        return "60-69%"
    return "70%+"

def render_achievement_band_legend():
    legend_items = "".join(
        [
            (
                "<div style='display:flex;align-items:center;gap:8px;padding:6px 10px;"
                "background:#F8FAFC;border:1px solid #E2E8F0;"
                "border-radius:8px;'>"
                f"<span style='width:14px;height:14px;border-radius:3px;background:{color};display:inline-block;'></span>"
                f"<span style='font-size:12px;color:#1E293B;font-weight:600;'>{label}</span>"
                "</div>"
            )
            for label, color in ACHIEVEMENT_BAND_COLORS.items()
        ]
    )

    st.markdown(
        (
            "<div style='margin:6px 0 12px 0;'>"
            "<div style='font-size:12px;font-weight:700;margin-bottom:6px;'>Achievement Color Bands</div>"
            "<div style='display:flex;flex-wrap:wrap;gap:8px;'>"
            f"{legend_items}"
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )

def get_theme_text_color():
    """Return readable text color for current Streamlit theme."""
    theme_text_color = st.get_option("theme.textColor")
    if theme_text_color:
        return theme_text_color

    theme_bg = st.get_option("theme.backgroundColor")
    if isinstance(theme_bg, str) and theme_bg.startswith("#") and len(theme_bg) == 7:
        red = int(theme_bg[1:3], 16)
        green = int(theme_bg[3:5], 16)
        blue = int(theme_bg[5:7], 16)
        luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255
        return "#111827" if luminance > 0.5 else "#F8FAFC"

    base_theme = (st.get_option("theme.base") or "").strip().lower()
    if base_theme == "light":
        return "#111827"
    if base_theme == "dark":
        return "#F8FAFC"
    return None

def apply_theme_aware_bar_labels(fig):
    """Apply theme-aware text color on all bar-trace data labels."""
    label_color = get_theme_text_color()
    if label_color:
        fig.update_traces(selector=dict(type="bar"), textfont=dict(color=label_color))
    return fig

def render_unified_kpi_card(
    label,
    value,
    line_gradient='linear-gradient(90deg, #06B6D4, #38BDF8)',
    tooltip=None,
    delta_primary=None,
    delta_primary_color='#10B981',
    delta_secondary=None,
    delta_secondary_color='#10B981',
    sparkline_points=None,
    sparkline_color='#38BDF8',
):
    """Render a consistent KPI card style used across tabs."""
    label_safe = escape(str(label))
    value_safe = escape(str(value))
    tooltip_safe = escape(str(tooltip), quote=True) if tooltip else None

    tooltip_html = (
        f" <span title='{tooltip_safe}' style='cursor:help;color:#94A3B8;'>?</span>"
        if tooltip else ""
    )
    primary_html = (
        f"<div style='margin-top:8px;font-size:18px;font-weight:700;color:{delta_primary_color};'>{escape(str(delta_primary))}</div>"
        if delta_primary else ""
    )
    secondary_html = (
        f"<div style='margin-top:3px;font-size:13px;font-weight:700;color:{delta_secondary_color};'>{escape(str(delta_secondary))}</div>"
        if delta_secondary else ""
    )

    sparkline_html = ""
    if sparkline_points is not None:
        try:
            points = [float(point) for point in list(sparkline_points)]
            if len(points) >= 2:
                width, height = 220, 40
                min_val = min(points)
                max_val = max(points)
                value_range = (max_val - min_val) if (max_val - min_val) != 0 else 1.0

                x_step = width / (len(points) - 1)
                polyline_points = []
                for idx, point in enumerate(points):
                    x = idx * x_step
                    y = height - ((point - min_val) / value_range) * (height - 4) - 2
                    polyline_points.append(f"{x:.1f},{y:.1f}")

                points_attr = " ".join(polyline_points)
                gradient_id = f"sparkFill{abs(hash(label_safe)) % 100000}"
                area_points = f"0,{height - 2} {points_attr} {width},{height - 2}"
                sparkline_html = (
                    "<div style='margin-top:10px;background:rgba(59,130,246,0.05);border-radius:8px;padding:2px 2px 0 2px;'>"
                    f"<svg width='100%' height='{height}' viewBox='0 0 {width} {height}' preserveAspectRatio='none' role='img' aria-label='Trend'>"
                    f"<defs><linearGradient id='{gradient_id}' x1='0' y1='0' x2='0' y2='1'>"
                    f"<stop offset='0%' stop-color='{escape(str(sparkline_color), quote=True)}' stop-opacity='0.35'></stop>"
                    f"<stop offset='100%' stop-color='{escape(str(sparkline_color), quote=True)}' stop-opacity='0.03'></stop>"
                    "</linearGradient></defs>"
                    f"<polygon points='{area_points}' fill='url(#{gradient_id})' stroke='none'></polygon>"
                    f"<polyline fill='none' stroke='{escape(str(sparkline_color), quote=True)}' stroke-width='2.6' points='{points_attr}' stroke-linecap='round' stroke-linejoin='round'></polyline>"
                    "</svg></div>"
                )
        except Exception:
            sparkline_html = ""

    card_html = (
        f"<div style='background:#FFFFFF;border:1px solid #D9E3EF;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(15,23,42,0.08);'>"
        f"<div style='height:4px;background:{line_gradient};'></div>"
        "<div style='padding:14px 16px 14px 16px;text-align:left;'>"
        f"<div style='font-size:12px;letter-spacing:3px;text-transform:uppercase;color:#64748B;margin-bottom:8px;font-weight:500;'>{label_safe}{tooltip_html}</div>"
        f"<div style='font-size:30px;font-weight:700;color:#0F172A;line-height:1.05;'>{value_safe}</div>"
        f"{sparkline_html}"
        f"{primary_html}{secondary_html}"
        "</div></div>"
    )

    st.markdown(
        card_html,
        unsafe_allow_html=True,
    )

def render_booker_segmentation_table(df, height_px=350):
    """Render segmentation table with sticky Shop Name column and fixed height."""
    if df is None or df.empty:
        st.info("No data available for table with current filters.")
        return

    display_cols = ['Shop Name', 'Segment', 'Orders In Period', 'Last Order Date']
    table_df = df.copy()
    for column in display_cols:
        if column not in table_df.columns:
            table_df[column] = ''
    table_df = table_df[display_cols].fillna('')

    header_html = "".join([
        "<th style='position:sticky;top:0;z-index:3;left:0;background:#F8FAFC;border-bottom:1px solid #E2E8F0;padding:8px 10px;text-align:left;color:#334155;font-size:12px;white-space:nowrap;'>Shop Name</th>",
        "<th style='position:sticky;top:0;z-index:2;background:#F8FAFC;border-bottom:1px solid #E2E8F0;padding:8px 10px;text-align:left;color:#334155;font-size:12px;white-space:nowrap;'>Segment</th>",
        "<th style='position:sticky;top:0;z-index:2;background:#F8FAFC;border-bottom:1px solid #E2E8F0;padding:8px 10px;text-align:right;color:#334155;font-size:12px;white-space:nowrap;'>Orders In Period</th>",
        "<th style='position:sticky;top:0;z-index:2;background:#F8FAFC;border-bottom:1px solid #E2E8F0;padding:8px 10px;text-align:left;color:#334155;font-size:12px;white-space:nowrap;'>Last Order Date</th>",
    ])

    rows = []
    for _, row in table_df.iterrows():
        shop_name = escape(str(row.get('Shop Name', '')))
        segment = escape(str(row.get('Segment', '')))
        orders = escape(str(row.get('Orders In Period', '')))
        last_order = escape(str(row.get('Last Order Date', '')))
        rows.append(
            "<tr style='border-bottom:1px solid #EEF2F7;'>"
            f"<td title='{shop_name}' style='position:sticky;left:0;background:#FFFFFF;z-index:1;padding:8px 10px;color:#0F172A;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:180px;'>{shop_name}</td>"
            f"<td title='{segment}' style='padding:8px 10px;color:#334155;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:140px;'>{segment}</td>"
            f"<td style='padding:8px 10px;color:#334155;font-size:12px;text-align:right;white-space:nowrap;'>{orders}</td>"
            f"<td style='padding:8px 10px;color:#334155;font-size:12px;white-space:nowrap;'>{last_order}</td>"
            "</tr>"
        )

    table_html = (
        "<div style='border:1px solid #D9E3EF;border-radius:12px;background:#FFFFFF;overflow:hidden;'>"
        f"<div style='max-height:{int(height_px)}px;overflow:auto;'>"
        "<table style='width:100%;border-collapse:separate;border-spacing:0;table-layout:fixed;'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)

def create_activity_segmentation_donut(df, start_date, end_date, title_suffix=""):
    """Create donut chart for activity segmentation (Power/Regular/Occasional/Dormant)."""
    segment_order = ['Power Users (>4x/mo)', 'Regular (2–4x/mo)', 'Occasional (1x/mo)', 'Dormant (0 orders)']
    segment_colors = {
        'Power Users (>4x/mo)': '#B8B8D1',
        'Regular (2–4x/mo)': '#FFC145',
        'Occasional (1x/mo)': '#5B5F97',
        'Dormant (0 orders)': '#FF6B6C',
    }

    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No activity segmentation data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#64748B")
        )
        fig.update_layout(
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF',
            font=dict(color='#0F172A'),
            margin=dict(t=90, r=20, b=80, l=20),
            height=560,
        )
        return fig

    data_map = {row['Segment_Label']: row['Outlet_Count'] for _, row in df.iterrows()}
    values = [int(data_map.get(segment, 0)) for segment in segment_order]
    total_outlets = sum(values)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=segment_order,
                values=values,
                hole=0.58,
                sort=False,
                marker=dict(
                    colors=[segment_colors[label] for label in segment_order],
                    line=dict(color='#FFFFFF', width=3),
                ),
                texttemplate='%{value:,}<br>(%{percent})',
                textinfo='text',
                textposition='outside',
                textfont=dict(size=11, color='#334155'),
                hovertemplate=(
                    '<b>%{label}</b><br>'
                    'Outlets: %{value:,}<br>'
                    'Share: %{percent}<extra></extra>'
                ),
            )
        ]
    )

    fig.add_annotation(
        text=f"Outlets<br><b>{total_outlets:,}</b>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color='#1E293B'),
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Segmentation by Activity</b>{title_suffix}",
            x=0.02,
            font=dict(size=20, color='#111827'),
            
        ),
        annotations=[
            dict(
                x=0.02,
                y=1.08,
                xref='paper',
                yref='paper',
                text='Power   /   Regular   /   Occasional   /   Dormant',
                showarrow=False,
                font=dict(size=14, color='#64748B'),
                align='left',
            ),
            dict(
                x=0.5,
                y=0.5,
                xref='paper',
                yref='paper',
                text=f"Outlets<br><b>{total_outlets:,}</b>",
                showarrow=False,
                font=dict(size=16, color='#1E293B'),
            ),
        ],
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.12,
            xanchor='center',
            x=0.5,
            font=dict(size=12, color='#334155'),
            itemwidth=120,
        ),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(color='#0F172A'),
        margin=dict(t=110, r=20, b=90, l=20),
        height=620,
    )
    return fig

def create_weekly_cohort_chart(df, title_suffix=""):
    """Create week-wise cohort retention heatmap."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No cohort data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#64748B")
        )
        fig.update_layout(
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF',
            font=dict(color='#0F172A'),
            margin=dict(t=70, r=20, b=40, l=20),
            height=620,
        )
        return fig

    cohort_df = df.copy()
    cohort_df['Order_Date'] = pd.to_datetime(cohort_df.get('Order_Date'), errors='coerce')
    cohort_df = cohort_df.dropna(subset=['Store_Code', 'Order_Date'])

    if cohort_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid order dates for cohort chart",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#64748B")
        )
        fig.update_layout(
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF',
            font=dict(color='#0F172A'),
            margin=dict(t=70, r=20, b=40, l=20),
            height=620,
        )
        return fig

    cohort_df['Order_Week'] = cohort_df['Order_Date'].dt.to_period('W-MON').apply(lambda period: period.start_time.date())
    first_week = cohort_df.groupby('Store_Code', as_index=False)['Order_Week'].min().rename(columns={'Order_Week': 'Cohort_Week'})
    cohort_df = cohort_df.merge(first_week, on='Store_Code', how='left')
    cohort_df['Week_Number'] = (
        (pd.to_datetime(cohort_df['Order_Week']) - pd.to_datetime(cohort_df['Cohort_Week'])).dt.days // 7
    ).astype(int)

    cohort_counts = (
        cohort_df
        .groupby(['Cohort_Week', 'Week_Number'], as_index=False)
        .agg(Outlets=('Store_Code', 'nunique'))
    )

    cohort_sizes = (
        cohort_counts[cohort_counts['Week_Number'] == 0][['Cohort_Week', 'Outlets']]
        .rename(columns={'Outlets': 'Cohort_Size'})
    )
    cohort_counts = cohort_counts.merge(cohort_sizes, on='Cohort_Week', how='left')
    cohort_counts['Retention_Pct'] = np.where(
        pd.to_numeric(cohort_counts['Cohort_Size'], errors='coerce').fillna(0) > 0,
        (pd.to_numeric(cohort_counts['Outlets'], errors='coerce').fillna(0)
         / pd.to_numeric(cohort_counts['Cohort_Size'], errors='coerce').fillna(0)) * 100,
        0,
    )

    heatmap = cohort_counts.pivot_table(
        index='Cohort_Week',
        columns='Week_Number',
        values='Retention_Pct',
        aggfunc='mean',
    ).fillna(0)

    heatmap = heatmap.sort_index()
    heatmap.index = [pd.to_datetime(week).strftime('%d %b %Y') for week in heatmap.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap.values,
            x=[f"W+{int(col)}" for col in heatmap.columns],
            y=heatmap.index.tolist(),
            colorscale=[
                [0.00, '#F1F5F9'],
                [0.30, '#B8B8D1'],
                [0.60, "#7A7AB5"],
                [1.00, "#5B5F97"],
            ],
            zmin=0,
            zmax=100,
            text=np.vectorize(lambda value: f"{value:.0f}%")(heatmap.values),
            texttemplate='%{text}',
            textfont=dict(color='#0F172A', size=10),
            colorbar=dict(title='Retention %', tickfont=dict(color='#334155')),
            hovertemplate='Cohort Week: %{y}<br>Relative Week: %{x}<br>Retention: %{z:.1f}%<extra></extra>',
        )
    )

    fig.update_layout(
        title=dict(text=f"<b>Weekly Cohort Retention</b>{title_suffix}", x=0.02, font=dict(size=20, color='#111827')),
        # xaxis_title='Weeks Since First Order',
        yaxis_title='Cohort Week',
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(color='#0F172A'),
        xaxis=dict(gridcolor='#E2E8F0', zerolinecolor='#CBD5E1'),
        yaxis=dict(gridcolor='#E2E8F0', zerolinecolor='#CBD5E1'),
        margin=dict(t=70, r=20, b=60, l=20),
        height=620,
    )
    return fig

def create_booker_wise_activity_segmentation_chart(df, title_suffix=""):
    """Create stacked bar chart for booker-wise activity segmentation using same activity logic."""
    segment_order = ['Power Users (>4x/mo)', 'Regular (2–4x/mo)', 'Occasional (1x/mo)', 'Dormant (0 orders)']
    segment_colors = {
        'Power Users (>4x/mo)': '#B8B8D1',
        'Regular (2–4x/mo)': '#FFC145',
        'Occasional (1x/mo)': '#5B5F97',
        'Dormant (0 orders)': '#FF6B6C',
    }

    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No Booker-wise segmentation data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#64748B")
        )
        fig.update_layout(
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF',
            font=dict(color='#0F172A'),
            margin=dict(t=70, r=20, b=40, l=20),
            height=460,
        )
        return fig

    plot_df = df.copy()
    plot_df['Outlet_Count'] = pd.to_numeric(plot_df.get('Outlet_Count', 0), errors='coerce').fillna(0)

    pivot_df = (
        plot_df
        .pivot_table(index='Booker', columns='Segment_Label', values='Outlet_Count', aggfunc='sum', fill_value=0)
        .reset_index()
    )

    for segment in segment_order:
        if segment not in pivot_df.columns:
            pivot_df[segment] = 0

    pivot_df['Total_Outlets'] = pivot_df[segment_order].sum(axis=1)
    pivot_df = pivot_df.sort_values('Total_Outlets', ascending=False).head(15)

    def _short_label(text, max_len=12):
        text = str(text)
        return text if len(text) <= max_len else f"{text[:max_len-1]}…"

    pivot_df['Booker_Display'] = pivot_df['Booker'].astype(str).apply(_short_label)

    fig = go.Figure()
    for segment in segment_order:
        fig.add_trace(
            go.Bar(
                x=pivot_df['Booker_Display'],
                y=pivot_df[segment],
                name=segment,
                marker_color=segment_colors[segment],
                customdata=pivot_df[['Booker']],
                hovertemplate='<b>%{customdata[0]}</b><br>' + segment + ': %{y:,.0f}<extra></extra>'
            )
        )

    fig.update_layout(
        title=dict(
            text=f"<b>Booker-wise Segmentation by Activity</b>{title_suffix}",
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            font=dict(size=18, color='#111827')
        ),
        # xaxis_title='',
        yaxis_title='Outlets',
        barmode='stack',
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(color='#0F172A'),
        xaxis=dict(tickangle=-25, gridcolor='#E2E8F0'),
        yaxis=dict(gridcolor='#E2E8F0'),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.22,
            xanchor='center',
            x=0.5,
            font=dict(size=12, color='#334155')
        ),
        margin=dict(t=84, r=20, b=120, l=20),
        height=480,
    )
    return fig

def create_tgtach_brand_maptree(df, achievement_below=None, selected_brands=None):
    """Create Booker -> Brand treemap with achievement percentage labels."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    required_cols = {'Booker', 'brand', 'Target_Value', 'NMV', 'Value_Ach'}
    if not required_cols.issubset(df.columns):
        fig = go.Figure()
        fig.add_annotation(
            text="Required columns for treemap not found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    df_plot = df.copy()
    df_plot['Target_Value'] = pd.to_numeric(df_plot['Target_Value'], errors='coerce').fillna(0)
    df_plot['NMV'] = pd.to_numeric(df_plot['NMV'], errors='coerce').fillna(0)
    df_plot['Value_Ach'] = pd.to_numeric(df_plot['Value_Ach'], errors='coerce').fillna(0)

    if achievement_below is not None:
        df_plot = df_plot[df_plot['Value_Ach'] < achievement_below]
        if df_plot.empty:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data below {achievement_below}% achievement",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig

    if selected_brands:
        df_plot = df_plot[df_plot['brand'].astype(str).isin([str(brand) for brand in selected_brands])]
        if df_plot.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data for selected brand filter",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig

    brand_level = (
        df_plot
        .groupby(['Booker', 'brand'], as_index=False)
        .agg({'Target_Value': 'sum', 'NMV': 'sum', 'Value_Ach': 'mean'})
    )
    brand_level['Value_Ach'] = brand_level['Value_Ach'].round(1)
    brand_level['Treemap_Value'] = np.where(
        brand_level['Target_Value'] > 0,
        brand_level['Target_Value'],
        brand_level['NMV']
    )
    brand_level['Treemap_Value'] = brand_level['Treemap_Value'].clip(lower=0)

    booker_level = (
        brand_level
        .groupby('Booker', as_index=False)
        .agg({'Target_Value': 'sum', 'NMV': 'sum', 'Treemap_Value': 'sum'})
    )
    booker_level['Value_Ach'] = np.where(
        booker_level['Target_Value'] > 0,
        (booker_level['NMV'] / booker_level['Target_Value']) * 100,
        0,
    ).round(1)
    booker_level = booker_level.sort_values('Treemap_Value', ascending=False)
    brand_level = brand_level.sort_values('Treemap_Value', ascending=False)

    root_target = booker_level['Target_Value'].sum()
    root_nmv = booker_level['NMV'].sum()
    root_treemap_value = booker_level['Treemap_Value'].sum()
    root_ach = round((root_nmv / root_target) * 100, 1) if root_target > 0 else 0

    labels = ['All Bookers']
    parents = ['']
    values = [root_treemap_value]
    ids = ['root']
    ach_values = [root_ach]
    nmv_values = [root_nmv]
    node_levels = ['root']

    for _, row in booker_level.iterrows():
        booker_id = f"booker::{row['Booker']}"
        labels.append(str(row['Booker']))
        parents.append('root')
        values.append(row['Treemap_Value'])
        ids.append(booker_id)
        ach_values.append(row['Value_Ach'])
        nmv_values.append(row['NMV'])
        node_levels.append('booker')

    for _, row in brand_level.iterrows():
        booker_id = f"booker::{row['Booker']}"
        labels.append(str(row['brand']))
        parents.append(booker_id)
        values.append(row['Treemap_Value'])
        ids.append(f"brand::{row['Booker']}::{row['brand']}")
        ach_values.append(row['Value_Ach'])
        nmv_values.append(row['NMV'])
        node_levels.append('brand')

    if sum(values) <= 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No measurable Target/NMV values available for selected period",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    def lighten_hex(hex_color, factor=0.22):
        hex_color = hex_color.lstrip('#')
        red = int(hex_color[0:2], 16)
        green = int(hex_color[2:4], 16)
        blue = int(hex_color[4:6], 16)
        red = int(red + (255 - red) * factor)
        green = int(green + (255 - green) * factor)
        blue = int(blue + (255 - blue) * factor)
        return f"#{red:02x}{green:02x}{blue:02x}"

    ach_bands = [get_achievement_band(v) for v in ach_values]
    node_colors = []
    for band, level in zip(ach_bands, node_levels):
        if level == 'root':
            node_colors.append('#94A3B8')
            continue
        base_color = ACHIEVEMENT_BAND_COLORS[band]
        if level == 'booker':
            node_colors.append(base_color)
        else:
            node_colors.append(lighten_hex(base_color))

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            ids=ids,
            branchvalues='total',
            customdata=np.column_stack([ach_values, nmv_values, ach_bands, node_levels]),
            marker=dict(
                colors=node_colors,
                line=dict(color='rgba(255,255,255,0.35)', width=1)
            ),
            tiling=dict(pad=2),
            texttemplate='%{label}<br>%{customdata[0]:.0f}%',
            textfont=dict(size=12, color='white'),
            hovertemplate=(
                '<b>%{label}</b>'
                '<br>Target: %{value:,.0f}'
                '<br>Achievement (NMV): %{customdata[1]:,.0f}'
                '<br>Achievement %: %{customdata[0]:.1f}%'
                '<br>Band: %{customdata[2]}'
                '<br>Level: %{customdata[3]}'
                '<extra></extra>'
            )
        )
    )

    fig.update_layout(
        title='🎯 Booker-wise Brand Achievement Treemap',
        margin=dict(l=8, r=8, t=42, b=8),
        uniformtext=dict(minsize=10, mode='hide'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def create_tgtach_brand_booker_maptree(df, achievement_below=None, selected_brands=None):
    """Create Brand -> Booker treemap with same logic/format as Booker -> Brand treemap."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    required_cols = {'Booker', 'brand', 'Target_Value', 'NMV', 'Value_Ach'}
    if not required_cols.issubset(df.columns):
        fig = go.Figure()
        fig.add_annotation(
            text="Required columns for treemap not found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    df_plot = df.copy()
    df_plot['Target_Value'] = pd.to_numeric(df_plot['Target_Value'], errors='coerce').fillna(0)
    df_plot['NMV'] = pd.to_numeric(df_plot['NMV'], errors='coerce').fillna(0)
    df_plot['Value_Ach'] = pd.to_numeric(df_plot['Value_Ach'], errors='coerce').fillna(0)

    if achievement_below is not None:
        df_plot = df_plot[df_plot['Value_Ach'] < achievement_below]
        if df_plot.empty:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data below {achievement_below}% achievement",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig

    if selected_brands:
        df_plot = df_plot[df_plot['brand'].astype(str).isin([str(brand) for brand in selected_brands])]
        if df_plot.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data for selected brand filter",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig

    brand_booker_level = (
        df_plot
        .groupby(['brand', 'Booker'], as_index=False)
        .agg({'Target_Value': 'sum', 'NMV': 'sum', 'Value_Ach': 'mean'})
    )
    brand_booker_level['Value_Ach'] = brand_booker_level['Value_Ach'].round(1)
    brand_booker_level['Treemap_Value'] = np.where(
        brand_booker_level['Target_Value'] > 0,
        brand_booker_level['Target_Value'],
        brand_booker_level['NMV']
    )
    brand_booker_level['Treemap_Value'] = brand_booker_level['Treemap_Value'].clip(lower=0)

    brand_level = (
        brand_booker_level
        .groupby('brand', as_index=False)
        .agg({'Target_Value': 'sum', 'NMV': 'sum', 'Treemap_Value': 'sum'})
    )
    brand_level['Value_Ach'] = np.where(
        brand_level['Target_Value'] > 0,
        (brand_level['NMV'] / brand_level['Target_Value']) * 100,
        0,
    ).round(1)
    brand_level = brand_level.sort_values('Treemap_Value', ascending=False)
    brand_booker_level = brand_booker_level.sort_values('Treemap_Value', ascending=False)

    root_target = brand_level['Target_Value'].sum()
    root_nmv = brand_level['NMV'].sum()
    root_treemap_value = brand_level['Treemap_Value'].sum()
    root_ach = round((root_nmv / root_target) * 100, 1) if root_target > 0 else 0

    labels = ['All Brands']
    parents = ['']
    values = [root_treemap_value]
    ids = ['root']
    ach_values = [root_ach]
    nmv_values = [root_nmv]
    node_levels = ['root']

    for _, row in brand_level.iterrows():
        brand_id = f"brand::{row['brand']}"
        labels.append(str(row['brand']))
        parents.append('root')
        values.append(row['Treemap_Value'])
        ids.append(brand_id)
        ach_values.append(row['Value_Ach'])
        nmv_values.append(row['NMV'])
        node_levels.append('brand')

    for _, row in brand_booker_level.iterrows():
        brand_id = f"brand::{row['brand']}"
        labels.append(str(row['Booker']))
        parents.append(brand_id)
        values.append(row['Treemap_Value'])
        ids.append(f"booker::{row['brand']}::{row['Booker']}")
        ach_values.append(row['Value_Ach'])
        nmv_values.append(row['NMV'])
        node_levels.append('booker')

    if sum(values) <= 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No measurable Target/NMV values available for selected period",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    def lighten_hex(hex_color, factor=0.22):
        hex_color = hex_color.lstrip('#')
        red = int(hex_color[0:2], 16)
        green = int(hex_color[2:4], 16)
        blue = int(hex_color[4:6], 16)
        red = int(red + (255 - red) * factor)
        green = int(green + (255 - green) * factor)
        blue = int(blue + (255 - blue) * factor)
        return f"#{red:02x}{green:02x}{blue:02x}"

    ach_bands = [get_achievement_band(v) for v in ach_values]
    node_colors = []
    for band, level in zip(ach_bands, node_levels):
        if level == 'root':
            node_colors.append('#94A3B8')
            continue
        base_color = ACHIEVEMENT_BAND_COLORS[band]
        if level == 'brand':
            node_colors.append(base_color)
        else:
            node_colors.append(lighten_hex(base_color))

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            ids=ids,
            branchvalues='total',
            customdata=np.column_stack([ach_values, nmv_values, ach_bands, node_levels]),
            marker=dict(
                colors=node_colors,
                line=dict(color='rgba(255,255,255,0.35)', width=1)
            ),
            tiling=dict(pad=2),
            texttemplate='%{label}<br>%{customdata[0]:.0f}%',
            textfont=dict(size=12, color='white'),
            hovertemplate=(
                '<b>%{label}</b>'
                '<br>Target: %{value:,.0f}'
                '<br>Achievement (NMV): %{customdata[1]:,.0f}'
                '<br>Achievement %: %{customdata[0]:.1f}%'
                '<br>Band: %{customdata[2]}'
                '<br>Level: %{customdata[3]}'
                '<extra></extra>'
            )
        )
    )

    fig.update_layout(
        title='🎯 Brand-wise Booker Achievement Treemap',
        margin=dict(l=8, r=8, t=42, b=8),
        uniformtext=dict(minsize=10, mode='hide'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def create_ob_brand_nmv_sankey(
    df,
    top_n=10,
    bottom_n=5,
    label_max_len=20,
    split_source_sides=True,
    force_all_source_left=False,
    flow_direction='OB_TO_BRAND',
):
    """Create Sankey chart for OB <-> Brand flow using NMV values with source-side limiter."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    required_cols = {'Booker', 'brand', 'NMV'}
    if not required_cols.issubset(df.columns):
        fig = go.Figure()
        fig.add_annotation(
            text="Required columns for Sankey chart not found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    df_plot = df.copy()
    df_plot['NMV'] = pd.to_numeric(df_plot['NMV'], errors='coerce').fillna(0)
    df_plot = df_plot.dropna(subset=['Booker', 'brand'])
    df_plot = df_plot[df_plot['NMV'] > 0]

    if df_plot.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No positive NMV values available for Sankey chart",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    link_df = (
        df_plot
        .groupby(['Booker', 'brand'], as_index=False)['NMV']
        .sum()
        .sort_values('NMV', ascending=False)
    )

    flow_mode = str(flow_direction or 'OB_TO_BRAND').upper()
    if flow_mode == 'BRAND_TO_OB':
        source_col = 'brand'
        target_col = 'Booker'
        source_label = 'Brand'
        target_label = 'OB'
    else:
        source_col = 'Booker'
        target_col = 'brand'
        source_label = 'OB'
        target_label = 'Brand'

    source_totals = (
        link_df
        .groupby(source_col, as_index=False)['NMV']
        .sum()
        .sort_values('NMV', ascending=False)
    )

    top_n = max(0, int(top_n or 0))
    bottom_n = max(0, int(bottom_n or 0))
    if top_n > 0 or bottom_n > 0:
        selected_source = []
        if top_n > 0:
            selected_source.extend(source_totals.head(top_n)[source_col].astype(str).tolist())
        if bottom_n > 0:
            selected_source.extend(source_totals.tail(bottom_n)[source_col].astype(str).tolist())
        selected_source = list(dict.fromkeys(selected_source))
        link_df = link_df[link_df[source_col].astype(str).isin(selected_source)]

    if link_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data after applying Top/Bottom limiter",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    source_nodes = sorted(link_df[source_col].astype(str).unique().tolist())
    target_nodes = sorted(link_df[target_col].astype(str).unique().tolist())
    all_full_labels = source_nodes + target_nodes

    def shorten_label(text_value, max_len=26):
        label = str(text_value).replace('_', ' ').strip()
        if len(label) <= max_len:
            return label
        return label[: max_len - 1] + '…'

    all_labels = [shorten_label(label, label_max_len) for label in all_full_labels]

    source_index = {name: idx for idx, name in enumerate(source_nodes)}
    target_index = {name: idx + len(source_nodes) for idx, name in enumerate(target_nodes)}

    source = link_df[source_col].astype(str).map(source_index).tolist()
    target = link_df[target_col].astype(str).map(target_index).tolist()
    values = link_df['NMV'].tolist()

    brand_palette = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.D3
        + px.colors.qualitative.Dark24
    )
    all_brands = sorted(link_df['brand'].astype(str).unique().tolist())
    brand_color_map = {
        brand: brand_palette[idx % len(brand_palette)]
        for idx, brand in enumerate(all_brands)
    }

    def hex_to_rgba(hex_color, alpha=0.35):
        color = str(hex_color).strip()
        if color.startswith('#') and len(color) == 7:
            red = int(color[1:3], 16)
            green = int(color[3:5], 16)
            blue = int(color[5:7], 16)
            return f'rgba({red},{green},{blue},{alpha})'
        return color

    ob_color = '#5B5F97'

    source_node_colors = [
        brand_color_map[node] if source_col == 'brand' else ob_color
        for node in source_nodes
    ]
    target_node_colors = [
        brand_color_map[node] if target_col == 'brand' else ob_color
        for node in target_nodes
    ]
    node_colors = source_node_colors + target_node_colors
    link_colors = [hex_to_rgba(brand_color_map[brand], 0.35) for brand in link_df['brand'].astype(str)]

    node_x = None
    node_y = None
    arrangement_mode = 'snap'

    if split_source_sides:
        arrangement_mode = 'fixed'
        if force_all_source_left:
            left_source_nodes = source_nodes
            right_source_nodes = []
        else:
            left_source_count = (len(source_nodes) + 1) // 2
            left_source_nodes = source_nodes[:left_source_count]
            right_source_nodes = source_nodes[left_source_count:]

        x_map = {}
        for item in left_source_nodes:
            x_map[item] = 0.02
        for item in right_source_nodes:
            x_map[item] = 0.98
        for item in target_nodes:
            x_map[item] = 0.50

        def spaced_positions(items):
            if len(items) <= 1:
                return {items[0]: 0.5} if items else {}
            return {item: idx / (len(items) - 1) for idx, item in enumerate(items)}

        y_map = {}
        y_map.update(spaced_positions(left_source_nodes))
        y_map.update(spaced_positions(right_source_nodes))
        y_map.update(spaced_positions(target_nodes))

        node_x = [x_map[label] for label in all_full_labels]
        node_y = [y_map[label] for label in all_full_labels]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement=arrangement_mode,
                textfont=dict(
                    size=11,
                    color=get_theme_text_color() or '#111827',
                    family='Arial'
                ),
                node=dict(
                    pad=18,
                    thickness=18,
                    line=dict(color='rgba(255,255,255,0.28)', width=1),
                    label=all_labels,
                    customdata=all_full_labels,
                    x=node_x,
                    y=node_y,
                    hovertemplate='<b>%{customdata}</b><extra></extra>',
                    color=node_colors
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=values,
                    customdata=np.column_stack([
                        link_df[source_col].astype(str),
                        link_df[target_col].astype(str)
                    ]),
                    color=link_colors,
                    hovertemplate=(
                        '<b>%{customdata[0]}</b> → <b>%{customdata[1]}</b>'
                        '<br>NMV: %{value:,.0f}'
                        '<extra></extra>'
                    )
                )
            )
        ]
    )

    fig.update_layout(
        title=f'🔀 {source_label} → {target_label} Flow by NMV',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=get_theme_text_color() or "#FFFFFF", size=12),
        margin=dict(l=8, r=8, t=42, b=8)
    )
    return fig
    
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #B8B8D1, #B8B8D1);
}

div[data-testid="stPlotlyChart"] {
    overflow-x: hidden !important;
    overflow-y: hidden !important;
}

div[data-testid="stPlotlyChart"] > div {
    overflow: hidden !important;
    max-width: 100% !important;
    max-height: 100% !important;
}

div[data-testid="stPlotlyChart"] .js-plotly-plot,
div[data-testid="stPlotlyChart"] .plotly,
div[data-testid="stPlotlyChart"] .plot-container {
    width: 100% !important;
    max-width: 100% !important;
    overflow: hidden !important;
}

div[data-testid="stPlotlyChart"] .svg-container,
div[data-testid="stPlotlyChart"] .main-svg,
div[data-testid="stPlotlyChart"] .cartesianlayer,
div[data-testid="stPlotlyChart"] .gl-container {
    overflow: hidden !important;
}

div[data-testid="stPlotlyChart"] [style*="overflow: auto"],
div[data-testid="stPlotlyChart"] [style*="overflow:auto"] {
    overflow: hidden !important;
}

.js-plotly-plot .sankey .node text {
    text-shadow: none !important;
    filter: none !important;
    stroke: none !important;
    stroke-width: 0 !important;
    paint-order: normal !important;
}

.js-plotly-plot .plotly text {
    text-shadow: none !important;
    filter: none !important;
    stroke: none !important;
    stroke-width: 0 !important;
}
</style>
""", unsafe_allow_html=True)

def AOV_MOPU_bar_chart(df):
    """Create AOV and MOPU bar chart with dynamic labels and hover info."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    required_cols = {'Month', 'Total_Orders', 'Drop_Size','SKU_Per_Bill','MOPU'}
    if not required_cols.issubset(df.columns):
        fig = go.Figure()
        fig.add_annotation(
            text="Required columns for AOV/MOPU chart not found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    df_plot = df.copy()
    df_plot['Total_Orders'] = pd.to_numeric(df_plot['Total_Orders'], errors='coerce')
    df_plot['Drop_Size'] = pd.to_numeric(df_plot['Drop_Size'], errors='coerce')
    df_plot['SKU_Per_Bill'] = pd.to_numeric(df_plot['SKU_Per_Bill'], errors='coerce')
    df_plot['MOPU'] = pd.to_numeric(df_plot['MOPU'], errors='coerce')

    df_plot['Month'] = pd.to_datetime(df_plot['Month'], format='%Y-%m', errors='coerce')
    if df_plot['Month'].isna().all():
        df_plot['Month'] = pd.to_datetime(df_plot['Month'], errors='coerce')

    df_plot = df_plot.dropna(subset=['Month', 'Total_Orders', 'Drop_Size','SKU_Per_Bill','MOPU'])
    if df_plot.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid AOV/MOPU data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    df_plot = df_plot.sort_values('Month')
    df_plot['Month_Label'] = df_plot['Month'].dt.strftime('%b-%y')

    fig = go.Figure(data=[
        go.Bar(
            name='Drop Size',
            x=df_plot['Month_Label'],
            y=df_plot['Drop_Size'],
            text=df_plot['Drop_Size'].apply(lambda x: f"{x:,.0f}"),
            textposition='auto',
            hovertemplate='<b>Drop Size</b><br>Value: %{y:,.0f}<extra></extra>',
            marker_color='#B8B8D1',
            yaxis='y'
        ),
        go.Scatter(
            name='SKU Per Bill',
            x=df_plot['Month_Label'],
            y=df_plot['SKU_Per_Bill'],
            mode='lines+markers+text',
            text=df_plot['SKU_Per_Bill'].apply(lambda x: f"{x:.2f}"),
            textposition='top center',
            hovertemplate='<b>SKU Per Bill</b><br>Value: %{y:.2f}<extra></extra>',
            line=dict(color='#FFC145', width=2),
            marker=dict(size=8),
            yaxis='y2'
        ),
        go.Scatter(
            name='MOPU',
            x=df_plot['Month_Label'],
            y=df_plot['MOPU'],
            mode='lines+markers+text',
            text=df_plot['MOPU'].apply(lambda x: f"{x:.2f}"),
            textposition='top center',
            hovertemplate='<b>MOPU</b><br>Value: %{y:.2f}<extra></extra>',
            line=dict(color='#FF6B6C', width=2),
            marker=dict(size=8),
            yaxis='y2'
        )
    ])
    fig.update_layout(
        title='📊 Total Orders, Drop Size, SKU Per Bill & MOPU by Period',
        # xaxis_title='Period',
        yaxis=dict(title='Total Orders', side='left', showgrid=True),
        yaxis2=dict(title='Drop Size / SKU Per Bill / MOPU', side='right', overlaying='y', showgrid=False),
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=get_theme_text_color() or '#111827'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    return fig


# ======================
# 🎯 MAIN APP
# ======================

def _get_bot_script_path():
    return os.path.join(os.path.dirname(__file__), "Scrapper", "bot.py")


def _get_bot_log_candidates():
    base_dir = os.path.dirname(__file__)
    return [
        os.path.join(base_dir, "salesflo_bot.log"),
        os.path.join(base_dir, "Scrapper", "salesflo_bot.log"),
    ]


def _get_primary_bot_log_path():
    return _get_bot_log_candidates()[0]


def _safe_file_size(path):
    try:
        if path and os.path.exists(path):
            return int(os.path.getsize(path))
    except Exception:
        return 0
    return 0


def _is_pid_running(pid):
    try:
        if pid is None:
            return False
        pid_int = int(pid)
    except Exception:
        return False

    try:
        if os.name == "nt":
            check = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid_int}"],
                capture_output=True,
                text=True,
                check=False,
            )
            return str(pid_int) in (check.stdout or "")
        os.kill(pid_int, 0)
        return True
    except Exception:
        return False


def _start_bot_process(extra_env=None):
    bot_script = _get_bot_script_path()
    if not os.path.exists(bot_script):
        raise FileNotFoundError(f"Bot script not found: {bot_script}")

    log_path = _get_primary_bot_log_path()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as log_file:
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process_env = os.environ.copy()
        if isinstance(extra_env, dict):
            for key, value in extra_env.items():
                if value is not None:
                    process_env[str(key)] = str(value)
        process = subprocess.Popen(
            [sys.executable, bot_script],
            cwd=os.path.dirname(__file__),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
            env=process_env,
        )
    return process.pid


def _get_bot_runtime_env():
    runtime_env = {}

    def _as_mapping(value):
        return value if hasattr(value, "get") else {}

    secrets_db = {}
    secrets_salesflo = {}
    try:
        if "database" in st.secrets:
            secrets_db = _as_mapping(st.secrets["database"])
    except Exception:
        secrets_db = {}

    try:
        if "salesflo" in st.secrets:
            secrets_salesflo = _as_mapping(st.secrets["salesflo"])
    except Exception:
        secrets_salesflo = {}

    # Prefer already-resolved DB config used by Streamlit app itself.
    db_user = str(DB_CONFIG.get("username", "") or "").strip()
    db_pass = str(DB_CONFIG.get("password", "") or "").strip()
    db_host = str(DB_CONFIG.get("host", "") or "").strip()
    db_port = str(DB_CONFIG.get("port", "") or "").strip()
    db_name = str(DB_CONFIG.get("database", "") or "").strip()

    # Fallback chain if DB_CONFIG is missing any field.
    if not db_user:
        db_user = str(secrets_db.get("username") or st.secrets.get("DB_USER", "") or st.secrets.get("DB_USERNAME", "") or os.getenv("DB_USER") or os.getenv("DB_USERNAME") or "").strip()
    if not db_pass:
        db_pass = str(secrets_db.get("password") or st.secrets.get("DB_PASSWORD", "") or st.secrets.get("DB_PASS", "") or os.getenv("DB_PASSWORD") or os.getenv("DB_PASS") or "").strip()
    if not db_host:
        db_host = str(secrets_db.get("host") or st.secrets.get("DB_HOST", "") or os.getenv("DB_HOST") or "").strip()
    if not db_port:
        db_port = str(secrets_db.get("port") or st.secrets.get("DB_PORT", "") or os.getenv("DB_PORT") or "3306").strip()
    if not db_name:
        db_name = str(secrets_db.get("database") or st.secrets.get("DB_NAME", "") or os.getenv("DB_NAME") or "").strip()

    if db_user:
        runtime_env["DB_USER"] = str(db_user)
        runtime_env["DB_USERNAME"] = str(db_user)
    if db_pass:
        runtime_env["DB_PASSWORD"] = str(db_pass)
        runtime_env["DB_PASS"] = str(db_pass)
    if db_host:
        runtime_env["DB_HOST"] = str(db_host)
    if db_port:
        runtime_env["DB_PORT"] = str(db_port)
    if db_name:
        runtime_env["DB_NAME"] = str(db_name)

    sf_user = (
        secrets_salesflo.get("SALESFLO_USERNAME")
        or secrets_salesflo.get("username")
        or st.secrets.get("SALESFLO_USERNAME", "")
        or os.getenv("SALESFLO_USERNAME")
        or ""
    )
    sf_pass = (
        secrets_salesflo.get("SALESFLO_PASSWORD")
        or secrets_salesflo.get("password")
        or st.secrets.get("SALESFLO_PASSWORD", "")
        or os.getenv("SALESFLO_PASSWORD")
        or ""
    )
    sf_user2 = (
        secrets_salesflo.get("SALESFLO_USERNAME2")
        or secrets_salesflo.get("SALESFLO_USERNAME_2")
        or secrets_salesflo.get("username2")
        or st.secrets.get("SALESFLO_USERNAME2", "")
        or st.secrets.get("SALESFLO_USERNAME_2", "")
        or os.getenv("SALESFLO_USERNAME2")
        or os.getenv("SALESFLO_USERNAME_2")
        or ""
    )
    sf_pass2 = (
        secrets_salesflo.get("SALESFLO_PASSWORD2")
        or secrets_salesflo.get("SALESFLO_PASSWORD_2")
        or secrets_salesflo.get("password2")
        or st.secrets.get("SALESFLO_PASSWORD2", "")
        or st.secrets.get("SALESFLO_PASSWORD_2", "")
        or os.getenv("SALESFLO_PASSWORD2")
        or os.getenv("SALESFLO_PASSWORD_2")
        or ""
    )

    if sf_user:
        runtime_env["SALESFLO_USERNAME"] = str(sf_user)
    if sf_pass:
        runtime_env["SALESFLO_PASSWORD"] = str(sf_pass)
    if sf_user2:
        runtime_env["SALESFLO_USERNAME2"] = str(sf_user2)
    if sf_pass2:
        runtime_env["SALESFLO_PASSWORD2"] = str(sf_pass2)

    return runtime_env


def _get_refresh_data_password():
    def _load_user_streamlit_secrets() -> dict:
        if tomllib is None:
            return {}

        configured_secret_path = (
            os.getenv("STREAMLIT_SECRETS_FILE")
            or os.getenv("SECRETS_FILE")
            or ""
        ).strip()

        candidate_paths = [
            Path(configured_secret_path) if configured_secret_path else None,
            Path.home() / ".streamlit" / "secrets.toml",
            Path(os.getenv("USERPROFILE", "")) / ".streamlit" / "secrets.toml" if os.getenv("USERPROFILE") else None,
        ]

        for candidate in [path for path in candidate_paths if path is not None]:
            try:
                if candidate.exists() and candidate.stat().st_size > 0:
                    with candidate.open("rb") as handle:
                        parsed = tomllib.load(handle)
                        if isinstance(parsed, dict):
                            return parsed
            except Exception:
                continue
        return {}

    secret_password = ""
    try:
        scheduler_section = st.secrets.get("scheduler", {})
        if isinstance(scheduler_section, dict):
            secret_password = (
                scheduler_section.get("REFRESH_PASSWORD")
                or scheduler_section.get("refresh_password")
                or ""
            )
    except Exception:
        secret_password = ""

    if not secret_password:
        try:
            secret_password = (
                st.secrets.get("REFRESH_PASSWORD", "")
                or st.secrets.get("refresh_password", "")
                or st.secrets.get("REFRESH_DATA_PASSWORD", "")
                or st.secrets.get("refresh_data_password", "")
                or ""
            )
        except Exception:
            secret_password = ""

    if not secret_password:
        file_secrets = _load_user_streamlit_secrets()
        if isinstance(file_secrets, dict):
            scheduler_section = file_secrets.get("scheduler", {})
            if isinstance(scheduler_section, dict):
                secret_password = (
                    scheduler_section.get("REFRESH_PASSWORD")
                    or scheduler_section.get("refresh_password")
                    or ""
                )

            if not secret_password:
                secret_password = (
                    file_secrets.get("REFRESH_PASSWORD")
                    or file_secrets.get("refresh_password")
                    or file_secrets.get("REFRESH_DATA_PASSWORD")
                    or file_secrets.get("refresh_data_password")
                    or ""
                )

    if secret_password:
        return str(secret_password).strip()

    return str(
        os.getenv("REFRESH_DATA_PASSWORD")
        or os.getenv("REFRESH_PASSWORD")
        or ""
    ).strip()


def _stop_pid(pid):
    try:
        if pid is None:
            return False
        pid_int = int(pid)
    except Exception:
        return False

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid_int), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            os.kill(pid_int, signal.SIGTERM)
        return not _is_pid_running(pid_int)
    except Exception:
        return False


def _read_bot_logs_tail(max_lines=200, start_pos=0, forced_log_path=None):
    chosen_log = None
    if forced_log_path and os.path.exists(forced_log_path):
        chosen_log = forced_log_path
    else:
        existing_logs = [path for path in _get_bot_log_candidates() if os.path.exists(path)]
        if not existing_logs:
            return "", None
        chosen_log = max(existing_logs, key=lambda p: os.path.getmtime(p))

    try:
        with open(chosen_log, "r", encoding="utf-8", errors="ignore") as file_obj:
            safe_start = max(0, int(start_pos or 0))
            file_obj.seek(safe_start)
            lines = file_obj.readlines()
        return "".join(lines[-max_lines:]), chosen_log
    except Exception as exc:
        return f"Could not read log file: {exc}", chosen_log


def _render_colored_log_html(logs_text):
    lines = list(reversed(str(logs_text or "").splitlines()))
    if not lines:
        return (
            "<div style='border:1px solid #D9E3EF;border-radius:10px;padding:12px;background:#FFFFFF;color:#64748B;'>"
            "No logs available."
            "</div>"
        )

    row_html = []
    for raw_line in lines:
        line_lower = raw_line.lower()
        if "error" in line_lower:
            bg, fg = "#7F1D1D", "#FECACA"
        elif "warning" in line_lower:
            bg, fg = "#FEE2E2", "#991B1B"
        elif "info" in line_lower:
            bg, fg = "#DCFCE7", "#166534"
        else:
            bg, fg = "#F8FAFC", "#334155"

        row_html.append(
            "<div style=\""
            f"background:{bg};color:{fg};"
            "padding:6px 8px;border-bottom:1px solid #E2E8F0;"
            "font-family:Consolas, 'Courier New', monospace;font-size:12px;white-space:pre-wrap;"
            "\">"
            f"{escape(str(raw_line))}"
            "</div>"
        )

    return (
        "<div style='border:1px solid #D9E3EF;border-radius:10px;overflow:hidden;background:#FFFFFF;'>"
        "<div style='max-height:480px;overflow:auto;'>"
        + "".join(row_html)
        + "</div></div>"
    )


@st.dialog("🤖 Bot Live Logs", width="large")
def show_bot_logs_dialog(max_lines):
    st.caption("Live logs while bot is running")

    dialog_col1, dialog_col2 = st.columns([1, 1])
    with dialog_col1:
        auto_refresh = st.toggle("Auto Refresh", value=True, key="bot_dialog_auto_refresh")
    with dialog_col2:
        if st.button("🔄 Refresh", key="bot_dialog_refresh_btn"):
            st.rerun()

    logs_text, log_path = _read_bot_logs_tail(
        max_lines=int(max_lines),
        start_pos=st.session_state.get("bot_runner_log_start_pos", 0),
        forced_log_path=st.session_state.get("bot_runner_log_path"),
    )
    if log_path:
        st.caption(f"Log file: {log_path}")

    st.markdown(_render_colored_log_html(logs_text), unsafe_allow_html=True)

    if auto_refresh and _is_pid_running(st.session_state.get("bot_runner_pid")):
        time.sleep(2)
        st.rerun()

def main():
    # Check authentication
    check_authentication()

    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.background_color = "#B8B8D1"

    # Display username if available
    if st.session_state.get("username"):
        st.sidebar.markdown(f"**User:** {st.session_state.username}")

    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()
        
    
    # Period selector
    # st.sidebar.subheader("📅 Period")
    period_option = st.sidebar.selectbox(
        "📅 Select Period",
        options=["Last 7 Days", "Last 30 Days", "This Month", "Last Month", "Last 3 Months", "YTD", "Custom"],
        index=2,
    )

    today = datetime.today().date()
    if period_option == "Last 7 Days":
        start_date = today - timedelta(days=6)
        end_date = today
    elif period_option == "Last 30 Days":
        start_date = today - timedelta(days=29)
        end_date = today
    elif period_option == "This Month":
        start_date = today.replace(day=1)
        end_date = today
    elif period_option == "Last Month":
        first_day_this_month = today.replace(day=1)
        end_date = first_day_this_month - timedelta(days=1)
        start_date = end_date.replace(day=1)
    elif period_option == "Last 3 Months":
        first_day_this_month = today.replace(day=1)
        end_date = today
        start_date = (first_day_this_month - timedelta(days=1)).replace(day=1) - timedelta(days=0)
        start_date = (start_date.replace(day=1) - timedelta(days=1)).replace(day=1)
    elif period_option == "YTD":
        start_date = today.replace(month=1, day=1)
        end_date = today
    else:  # Custom
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=today - timedelta(days=29),
                max_value=today,
                key="custom_start_date",
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=today,
                max_value=today,
                key="custom_end_date",
            )

        if start_date > end_date:
            st.sidebar.error("Start Date cannot be after End Date")
            start_date, end_date = end_date, start_date

    st.sidebar.caption(f"Range: {start_date} to {end_date}")

    # Location selector
    # st.sidebar.subheader("📍 Location")
    # passed the value and backend will handle the condition to filter data based on all or specific town
    town_code = st.sidebar.selectbox(
        "📍 Select Location",
        options=["D70002202", "D70002246"],
        format_func=lambda x: "Karachi" if x == "D70002202" else "Lahore"
    )
    # Town mapping corrected
    town = {
    "D70002202": "Karachi",
    "D70002246": "Lahore",
    }.get(town_code, "Not Available")  # Default case in case of undefined town_code

    st.sidebar.subheader("🕒 Account Last Update")
    try:
        last_update_df = fetch_account_last_update_data()
        if last_update_df is None or last_update_df.empty:
            st.sidebar.caption("No update information available.")
        else:
            account_labels = {
                "D70002202": "Karachi",
                "D70002246": "Lahore",
            }
            last_update_map = {
                str(row.get("Distributor_Code", "")): row.get("Last_Update_At")
                for _, row in last_update_df.iterrows()
            }
            for account_code in ["D70002202", "D70002246"]:
                raw_dt = last_update_map.get(account_code)
                formatted_dt = "N/A"
                if pd.notna(raw_dt):
                    parsed_dt = pd.to_datetime(raw_dt, errors='coerce')
                    if pd.notna(parsed_dt):
                        formatted_dt = parsed_dt.strftime("%d-%b-%Y %I:%M %p")
                selected_marker = "👉 " if account_code == town_code else ""
                st.sidebar.caption(f"{selected_marker}{account_labels.get(account_code, account_code)}: {formatted_dt}")
    except Exception as exc:
        st.sidebar.caption(f"Could not fetch last update info: {exc}")


    # st.balloons()
    # Main content
    st.title(f"📊 Bazaar Prime Analytics Dashboard - {town}")
    tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(["📈 Sales Growth Analysis","🎯 Booker Performance","🧭 Booker & Field Force Deep Analysis","📦 Inventory","🧪 Custom Query","🤖 Bot Runner"])
    with tab1:
    # KPIs
        st.subheader("📈 Key Performance Indicator")
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

            rev_ly_text = f"{'▲' if revenue_growth_ly >= 0 else '▼'} {abs(revenue_growth_ly):.2f}% vs Last Year"
            rev_lm_text = f"{'▲' if revenue_growth_lm >= 0 else '▼'} {abs(revenue_growth_lm):.2f}% vs Last Month"
            rev_ly_color = '#16A34A' if revenue_growth_ly >= 0 else '#DC2626'
            rev_lm_color = '#16A34A' if revenue_growth_lm >= 0 else '#DC2626'

            ltr_ly_text = f"{'▲' if ltr_growth_ly >= 0 else '▼'} {abs(ltr_growth_ly):.2f}% vs Last Year"
            ltr_lm_text = f"{'▲' if ltr_growth_lm >= 0 else '▼'} {abs(ltr_growth_lm):.2f}% vs Last Month"
            ltr_ly_color = '#16A34A' if ltr_growth_ly >= 0 else '#DC2626'
            ltr_lm_color = '#16A34A' if ltr_growth_lm >= 0 else '#DC2626'

            ord_ly_text = f"{'▲' if orders_growth_ly >= 0 else '▼'} {abs(orders_growth_ly):.2f}% vs Last Year"
            ord_lm_text = f"{'▲' if orders_growth_lm >= 0 else '▼'} {abs(orders_growth_lm):.2f}% vs Last Month"
            ord_ly_color = '#16A34A' if orders_growth_ly >= 0 else '#DC2626'
            ord_lm_color = '#16A34A' if orders_growth_lm >= 0 else '#DC2626'

            aov_ly_text = f"{'▲' if aov_growth_ly >= 0 else '▼'} {abs(aov_growth_ly):.2f}% vs Last Year"
            aov_lm_text = f"{'▲' if aov_growth_lm >= 0 else '▼'} {abs(aov_growth_lm):.2f}% vs Last Month"
            aov_ly_color = '#16A34A' if aov_growth_ly >= 0 else '#DC2626'
            aov_lm_color = '#16A34A' if aov_growth_lm >= 0 else '#DC2626'

            with col1:
                    kpi_sparkline_series = fetch_kpi_sparkline_series(end_date, town_code)
                    render_unified_kpi_card(
                        label='Total Revenue',
                        value=f"Rs {current_revenue / 1_000_000:.2f}M",
                        line_gradient='linear-gradient(90deg, #06B6D4, #38BDF8)',
                        sparkline_points=kpi_sparkline_series.get('revenue', []),
                        sparkline_color='#06B6D4',
                        delta_primary=rev_ly_text,
                        delta_primary_color=rev_ly_color,
                        delta_secondary=rev_lm_text,
                        delta_secondary_color=rev_lm_color,
                    )

            with col2:
                render_unified_kpi_card(
                    label='Total Litres',
                    value=f"{current_ltr:,.0f} Ltr",
                    line_gradient='linear-gradient(90deg, #10B981, #34D399)',
                    sparkline_points=kpi_sparkline_series.get('litres', []),
                    sparkline_color='#10B981',
                    delta_primary=ltr_ly_text,
                    delta_primary_color=ltr_ly_color,
                    delta_secondary=ltr_lm_text,
                    delta_secondary_color=ltr_lm_color,
                )

            with col3:
                render_unified_kpi_card(
                    label='Total Orders',
                    value=f"{int(current_orders):,}",
                    line_gradient='linear-gradient(90deg, #8B5CF6, #A78BFA)',
                    sparkline_points=kpi_sparkline_series.get('orders', []),
                    sparkline_color='#8B5CF6',
                    delta_primary=ord_ly_text,
                    delta_primary_color=ord_ly_color,
                    delta_secondary=ord_lm_text,
                    delta_secondary_color=ord_lm_color,
                )

            with col4:
                render_unified_kpi_card(
                    label='Avg Order Value',
                    value=f"Rs {aov_current / 1000:.1f}K",
                    line_gradient='linear-gradient(90deg, #F59E0B, #FBBF24)',
                    sparkline_points=kpi_sparkline_series.get('aov', []),
                    sparkline_color='#F59E0B',
                    delta_primary=aov_ly_text,
                    delta_primary_color=aov_ly_color,
                    delta_secondary=aov_lm_text,
                    delta_secondary_color=aov_lm_color,
                )

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
                # st.markdown("---")
                st.text("📦 Channel-wise Average Order Value (AOV)")
            # Display channels in columns grid inside the 5th column
            # instead of add  manually rows get the count of channels from table to get the count of rows need to loop

                cols_per_row = len(channel_df)
                for i in range(0, len(channel_df), cols_per_row):
                    row_channels = channel_df.iloc[i:i+cols_per_row]
                    cols = st.columns(len(row_channels))
                    channel_gradients = [
                        'linear-gradient(90deg, #06B6D4, #38BDF8)',
                        'linear-gradient(90deg, #10B981, #34D399)',
                        'linear-gradient(90deg, #8B5CF6, #A78BFA)',
                        'linear-gradient(90deg, #F59E0B, #FBBF24)',
                        'linear-gradient(90deg, #14B8A6, #22D3EE)',
                    ]

                    for idx, (col, (_, r)) in enumerate(zip(cols, row_channels.iterrows())):
                        aov_val = r['AOV_Current']
                        growth_ly = r['Growth_LY']
                        growth_lm = r['Growth_LM']
                        growth_ly_color = "#16A34A" if growth_ly >= 0 else '#DC2626'
                        growth_lm_color = "#16A34A" if growth_lm >= 0 else '#DC2626'
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
                                render_unified_kpi_card(
                                    label=f"🛍️ {str(r['Channel'])}",
                                    value=aov_display,
                                    line_gradient=channel_gradients[idx % len(channel_gradients)],
                                    delta_primary=f"{growth_ly_arrow} {abs(growth_ly):.2f}% vs Last Year",
                                    delta_primary_color=growth_ly_color,
                                    delta_secondary=f"{growth_lm_arrow} {abs(growth_lm):.2f}% vs Last Month",
                                    delta_secondary_color=growth_lm_color,
                                )
        # Booker Analysis Section
        #channel wise sales performance chart
        st.markdown("---")
        st.subheader(f"📊 Channel-wise Performance Comparison")
        
       
        #channel wise growth percentage chart
        leftcol,right_col = st.columns([1.5,1])
        with leftcol:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col3:
                    metric_filter = st.radio(
                        "Select Metric",
                        options=['Value', 'Ltr'],
                        horizontal=True,
                        help="Toggle between Sales Value and Litres comparison"
                    )
            
                channel_perf_df = Channelwise_performance_data(start_date, end_date, town_code)
                st.plotly_chart(create_Channel_performance_chart(channel_perf_df, metric_type=metric_filter), use_container_width=True, key="channel_performance_chart")

        with right_col:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                metric_filter = st.radio(
                    "Select Metric",
                    options=['Value', 'Ltr'],
                    horizontal=True,
                    help="Toggle between Sales Value and Litres growth comparison"
                )

            channel_growth_df = Channelwise_performance_data(start_date, end_date, town_code)
            st.plotly_chart(create_channel_wise_growth_chart(channel_growth_df, metric_type=metric_filter), use_container_width=True, key="channel_growth_chart")
            # brand wise growth percentage chart
        st.markdown("---")
        st.subheader(f"📈 Brand-wise Growth Percentage")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            brand_channel_options_df = fetch_brand_growth_channel_options(start_date, end_date, town_code)
            brand_channel_options = ["All"]
            if (
                brand_channel_options_df is not None
                and not brand_channel_options_df.empty
                and 'Channel' in brand_channel_options_df.columns
            ):
                brand_channel_options.extend(
                    brand_channel_options_df['Channel'].dropna().astype(str).tolist()
                )

            selected_brand_channel = st.selectbox(
                "Channel Filter for Brand Growth",
                options=brand_channel_options,
                index=0,
                key="brand_growth_channel_filter"
            )

        with col3:
            metric_filter = st.radio(
                "Select Metric for Brand Growth",
                options=['Value', 'Ltr'],
                horizontal=True,
                help="Toggle between Sales Value and Litres growth comparison"
            )

        with col1:
            st.radio(
                "Growth Basis",
                options=['Last Year', 'Last Month'],
                horizontal=True,
                key='brand_growth_basis_filter',
                help='Controls growth/degrowth used for segment sizing and colors.'
            )

        brand_growth_df = Brand_wise_performance_growth_data(
            start_date,
            end_date,
            town_code,
            selected_channel=selected_brand_channel
        )
        brand_growth_left, brand_growth_right = st.columns(2)
        with brand_growth_left:
            st.plotly_chart(
                create_brand_wise_growth_chart(brand_growth_df, metric_type=metric_filter),
                use_container_width=True,
                key="brand_growth_bar_chart"
            )
        with brand_growth_right:
            st.plotly_chart(
                create_brand_wise_growth_segmented_chart(brand_growth_df, metric_type=metric_filter),
                use_container_width=True,
                key="brand_growth_segmented_chart"
            )
        # DM wise growth percentage chart
        st.markdown("---")
        st.subheader(f"📈 Deliveryman-wise Growth Percentage")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col3:
            metric_filter = st.radio(
                "Select Metric for Deliveryman Growth",
                options=['Value', 'Ltr'],
                horizontal=True,
                help="Toggle between Sales Value and Litres growth comparison"
            )

        dm_growth_df = dm_wise_performance_growth_data(start_date, end_date, town_code)
        st.plotly_chart(create_dm_wise_growth_chart(dm_growth_df, metric_type=metric_filter), use_container_width=True, key="dm_growth_chart")
        #target vs achievement comparison chart
        st.markdown("---")
        
        st.subheader(f"🎯 Target vs Achievement Comparison")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col3:
            metric_filter = st.radio(
                "Select Metric for Target vs Achievement",
                options=['Value', 'Ltr'],
                horizontal=True,
                help="Toggle between Sales Value and Litres comparison"
            )
        tgt_vs_ach_df = tgtvsach_YTD_data(town_code)
        st.plotly_chart(create_target_achievement_chart(tgt_vs_ach_df, metric_type=metric_filter), use_container_width=True,key="target_achievement_chart")
        
        # Channel DM Sunburst chart
        st.markdown("---")
        st.subheader(f"🌐 Channel DM Sunburst Chart")

        sunburst_df = fetch_Channel_dm_sunburst_data(start_date, end_date, town_code)
        if sunburst_df is None:
            sunburst_df = pd.DataFrame()

        if not sunburst_df.empty:
            sunburst_df["Channel"] = sunburst_df["Channel"].fillna("Unknown Channel")
            sunburst_df["Brand"] = sunburst_df["Brand"].fillna("Unknown Brand")
            sunburst_df["DM"] = sunburst_df["DM"].fillna("Unknown DM")

        sunburst_dm_values = sorted(sunburst_df['DM'].dropna().astype(str).unique().tolist()) if not sunburst_df.empty and 'DM' in sunburst_df.columns else []
        sunbrust_channel_values= sorted(sunburst_df['Channel'].dropna().astype(str).unique().tolist()) if not sunburst_df.empty and 'Channel' in sunburst_df.columns else []
        
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col3:
            selected_sunburst_dms = st.multiselect(
                "Filter Deliverymen for Sunburst",
                options=sunburst_dm_values,
                default=[],
            )
        with col2:
            selected_sunburst_channels = st.multiselect(
                "Filter Channels for Sunburst",
                options=sunbrust_channel_values,
                default=[],
            )

        left_col, right_col = st.columns([1.5, 1])
        with left_col:
            st.plotly_chart(
                create_Channel_dm_sunburst(sunburst_df, selected_dms=selected_sunburst_dms, selected_channels=selected_sunburst_channels),
                use_container_width=True,
                key="channel_dm_sunburst_chart"
            )
        with right_col:
            st.plotly_chart(
                brand_wise_productivity_chart(sunburst_df, selected_dms=selected_sunburst_dms, selected_channels=selected_sunburst_channels),
                use_container_width=True,
                key="brand_productivity_chart"
            )
        st.markdown("---")
        st.subheader(f"📊 Target Achievement Heatmap by Booker and Period")
        left_col, right_col = st.columns([1.5, 1])
        with left_col:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col3:
                        metric_filter1 = st.radio(
                "Select Metric",
                options=['Value', 'Ltr'],
                horizontal=True,
                help="Toggle between Sales Value and Litres comparison",
                key="heatmap_metric_filter"
                )
                heatmap_df = tgtvsach_YTD_heatmap_data(town_code)
                st.plotly_chart(create_booker_period_heatmap(heatmap_df, metric_type=metric_filter1), use_container_width=True, key="booker_period_heatmap")
        with right_col:
                col1, col2, col3 = st.columns([1, 2, 1])
                col3.metric("", "")
                col3.markdown("<div style='text-align: center; font-size: 12px; color: gray;'>(Hover over cells for details)</div>", unsafe_allow_html=True)
            
                channel_heatmap_df = tgtvsach_channelwise_heatmap(town_code)
                st.plotly_chart(
                create_channel_heatmap_YTD(channel_heatmap_df, metric_type=metric_filter1),
                use_container_width=True,
                key="channel_performance_heatmap"
            )
        
        
        #booker analysis section
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

            st.dataframe(pivot_df, use_container_width=True, height=200)

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
            # MOPU,Total ORder, Drop Size Chart for Period
            st.markdown("---")
            st.subheader("📊 MOPU and Drop Size Analysis")
            mopu_df = AOV_MOPU_data(town_code, months_back)
            st.plotly_chart(AOV_MOPU_bar_chart(mopu_df), use_container_width=True, key="booker_mopu_chart")

    with tab2:
        st.subheader("🎯 Booker Performance Analysis")
        period_options_df = fetch_treemap_period_options(town_code)

        if period_options_df is None or period_options_df.empty:
            st.warning("No periods available for treemap data.")
            st.plotly_chart(create_tgtach_brand_maptree(pd.DataFrame()), use_container_width=True, key="booker_treemap_chart")
        else:
            treemap_period_options = period_options_df['period'].dropna().astype(str).tolist()
            col_period, col_channel, col_ach = st.columns([2, 1.5, 1])
            with col_period:
                selected_treemap_period = st.selectbox(
                    "Select Period for Booker Treemap",
                    options=treemap_period_options,
                    index=0,
                    key="booker_treemap_period",
                    help="Upload Brand Wise Target Data to get the available periods in this dropdown"
                )

            channel_options_df = fetch_treemap_channel_options(town_code, selected_treemap_period)
            channel_options = ["All"]
            if channel_options_df is not None and not channel_options_df.empty and 'Channel' in channel_options_df.columns:
                channel_options.extend(channel_options_df['Channel'].dropna().astype(str).tolist())

            with col_channel:
                selected_channel = st.selectbox(
                    "Channel Filter",
                    options=channel_options,
                    index=0,
                    key="booker_treemap_channel_filter"
                )
                
            with col_ach:
                achievement_filter = st.selectbox(
                    "Achievement Filter",
                    options=["All", "Below 50%", "Below 60%", "Below 70%"],
                    index=0,
                    key="booker_treemap_ach_filter"
                )

            threshold_map = {
                "All": None,
                "Below 50%": 50,
                "Below 60%": 60,
                "Below 70%": 70,
            }
            selected_threshold = threshold_map.get(achievement_filter)

            treemap_df = tgtvsach_brand_level(town_code, selected_treemap_period, selected_channel)
            left_col, right_col = st.columns([1.5, 1])
            with left_col:
                brand_options = []
                if treemap_df is not None and not treemap_df.empty and 'brand' in treemap_df.columns:
                    brand_options = sorted(treemap_df['brand'].dropna().astype(str).unique().tolist())

                selected_brands = st.segmented_control(
                    "Brand Filter",
                    options=brand_options,
                    selection_mode="multi",
                    default=[],
                    key="booker_treemap_brand_filter",
                    help="Leave empty to show all brands"
                )
                if treemap_df is not None and not treemap_df.empty:
                    debug_nmv = pd.to_numeric(treemap_df.get('NMV', 0), errors='coerce').fillna(0).sum()
                    debug_target = pd.to_numeric(treemap_df.get('Target_Value', 0), errors='coerce').fillna(0).sum()
                    st.caption(
                        f"Channel: {selected_channel} | Rows: {len(treemap_df)} | NMV: {debug_nmv:,.0f} | Target: {debug_target:,.0f}"
                    )
                else:
                    st.caption(f"Channel: {selected_channel} | Rows: 0 for selected period")
            with right_col:
                
                render_achievement_band_legend()
            st.plotly_chart(
                create_tgtach_brand_maptree(
                    treemap_df,
                    achievement_below=selected_threshold,
                    selected_brands=selected_brands
                ),
                use_container_width=True,
                key="booker_treemap_chart"
            )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.plotly_chart(
                create_tgtach_brand_booker_maptree(
                    treemap_df,
                    achievement_below=selected_threshold,
                    selected_brands=selected_brands
                ),
                use_container_width=True,
                key="brand_booker_treemap_chart"
            )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.subheader("🔀 OB / Brand NMV Sankey")

            sankey_df = treemap_df.copy() if treemap_df is not None else pd.DataFrame()
            if sankey_df is not None and not sankey_df.empty:
                if selected_threshold is not None and 'Value_Ach' in sankey_df.columns:
                    sankey_df = sankey_df[pd.to_numeric(sankey_df['Value_Ach'], errors='coerce') < selected_threshold]
                if selected_brands:
                    sankey_df = sankey_df[sankey_df['brand'].astype(str).isin([str(brand) for brand in selected_brands])]

            controls_col1, controls_col2, controls_col3, controls_col4, controls_col5 = st.columns([2, 1, 1, 1.6, 1.6])
            with controls_col1:
                flow_choice = st.radio(
                    "Flow Direction",
                    options=["OB → Brand", "Brand → OB"],
                    horizontal=True,
                    key="ob_brand_sankey_flow"
                )

            current_source_col = 'Booker' if flow_choice == "OB → Brand" else 'brand'
            current_target_label = 'Brand' if flow_choice == "OB → Brand" else 'OB'

            source_count = (
                sankey_df[current_source_col].nunique()
                if sankey_df is not None and not sankey_df.empty and current_source_col in sankey_df.columns
                else 0
            )

            sankey_filter_signature = (
                str(selected_treemap_period),
                str(selected_channel),
                str(achievement_filter),
                tuple(sorted([str(item) for item in selected_brands])) if selected_brands else tuple(),
                str(flow_choice),
            )
            current_source_max = max(0, int(source_count))
            previous_signature = st.session_state.get("ob_sankey_filter_signature")
            if previous_signature != sankey_filter_signature:
                st.session_state["ob_sankey_filter_signature"] = sankey_filter_signature
                st.session_state["ob_sankey_top_n"] = current_source_max
            elif st.session_state.get("ob_sankey_top_n", 0) > current_source_max:
                st.session_state["ob_sankey_top_n"] = current_source_max

            with controls_col2:
                top_n_limit = st.number_input(
                    f"Top N {current_source_col}",
                    min_value=0,
                    max_value=max(0, int(source_count)),
                    value=current_source_max,
                    step=1,
                    key="ob_sankey_top_n"
                )
            with controls_col3:
                bottom_n_limit = st.number_input(
                    f"Bottom N {current_source_col}",
                    min_value=0,
                    max_value=max(0, int(source_count)),
                    value=min(5, int(source_count)) if source_count > 0 else 0,
                    step=1,
                    key="ob_sankey_bottom_n"
                )
            with controls_col4:
                split_source_layout = st.toggle(
                    f"Split {current_source_col} Left/Right ({current_target_label} Center)",
                    value=True,
                    key="ob_sankey_split_layout"
                )

            with controls_col5:
                force_source_left_layout = st.toggle(
                    f"Force All {current_source_col} Left ({current_target_label} Center)",
                    value=False,
                    key="ob_sankey_force_left",
                    disabled=not bool(split_source_layout)
                )

            st.plotly_chart(
                create_ob_brand_nmv_sankey(
                    sankey_df,
                    top_n=int(top_n_limit),
                    bottom_n=int(bottom_n_limit),
                    split_source_sides=bool(split_source_layout),
                    force_all_source_left=bool(force_source_left_layout),
                    flow_direction='OB_TO_BRAND' if flow_choice == "OB → Brand" else 'BRAND_TO_OB'
                ),
                use_container_width=True,
                key="ob_brand_nmv_sankey_chart"
            )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.subheader("📅 Booker GMV Calendar Heatmap")

            cal_filter_col1, cal_filter_col2 = st.columns([1, 2])
            with cal_filter_col1:
                gmv_calendar_month_window = st.selectbox(
                    "Month Filter",
                    options=[3, 6, 12],
                    index=0,
                    format_func=lambda value: f"Last {value} Months",
                    key="gmv_calendar_month_filter"
                )

            gmv_calendar_end = datetime.today().date()
            gmv_calendar_start = (pd.Timestamp(gmv_calendar_end) - pd.DateOffset(months=int(gmv_calendar_month_window))).date()

            gmv_calendar_df = GMV_OB_calendar_heatmap_data(town_code, gmv_calendar_start, gmv_calendar_end)

            if gmv_calendar_df is None or gmv_calendar_df.empty:
                st.info("No GMV calendar data available for selected date range.")
            else:
                gmv_calendar_df['GMV'] = pd.to_numeric(gmv_calendar_df.get('GMV', 0), errors='coerce').fillna(0)
                gmv_calendar_df['Orders'] = pd.to_numeric(gmv_calendar_df.get('Orders', 0), errors='coerce').fillna(0)

                calendar_bookers = sorted(gmv_calendar_df['Order Booker Name'].dropna().astype(str).unique().tolist())

                with cal_filter_col2:
                    selected_calendar_bookers = st.multiselect(
                        "Booker Filter",
                        options=calendar_bookers,
                        default=[],
                        key="gmv_calendar_booker_filter",
                        help="Leave empty to show all bookers"
                    )

                st.caption(f"Range: {gmv_calendar_start} to {gmv_calendar_end}")

                st.plotly_chart(
                    create_gmv_ob_calendar_heatmap(
                        gmv_calendar_df,
                        selected_bookers=selected_calendar_bookers,
                    ),
                    use_container_width=True,
                    key="gmv_ob_calendar_heatmap_chart"
                )

    with tab3:
        st.subheader("🧭 Booker & Field Force Deep Analysis")

        deep_df = fetch_booker_fieldforce_deep_data(start_date, end_date, town_code)
        if deep_df is None or deep_df.empty:
            st.info("No deep analysis data available for selected date range.")
        else:
            for column in ['NMV', 'Orders', 'Stores', 'AOV', 'Volume']:
                deep_df[column] = pd.to_numeric(deep_df.get(column, 0), errors='coerce').fillna(0)

            all_bookers = sorted(deep_df['Booker'].dropna().astype(str).unique().tolist())
            all_channels = sorted(deep_df['Channel'].dropna().astype(str).unique().tolist())

            filter_col1, filter_col2 = st.columns([2, 2])
            with filter_col1:
                selected_deep_bookers = st.multiselect(
                    "Booker Filter",
                    options=all_bookers,
                    default=[],
                    key="deep_booker_filter",
                    help="Leave empty to include all bookers"
                )
            with filter_col2:
                selected_deep_channels = st.multiselect(
                    "Channel Filter",
                    options=all_channels,
                    default=[],
                    key="deep_channel_filter",
                    help="Leave empty to include all channels"
                )

            deep_plot_df = deep_df.copy()
            if selected_deep_bookers:
                deep_plot_df = deep_plot_df[deep_plot_df['Booker'].astype(str).isin([str(booker) for booker in selected_deep_bookers])]
            if selected_deep_channels:
                deep_plot_df = deep_plot_df[deep_plot_df['Channel'].astype(str).isin([str(channel) for channel in selected_deep_channels])]

            if deep_plot_df.empty:
                st.warning("No records after applying selected filters.")
            else:
                filter_parts = []
                if selected_deep_bookers:
                    filter_parts.append(f"Booker: {', '.join(selected_deep_bookers[:2])}{'...' if len(selected_deep_bookers) > 2 else ''}")
                if selected_deep_channels:
                    filter_parts.append(f"Channel: {', '.join(selected_deep_channels[:2])}{'...' if len(selected_deep_channels) > 2 else ''}")
                title_suffix = f" | {' | '.join(filter_parts)}" if filter_parts else ""

                top_calls_kpi_container = st.container()
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                top_summary_kpi_container = st.container()
                st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

                route_perf_df = fetch_routewise_ob_achievement(
                    start_date,
                    end_date,
                    town_code,
                    tuple(selected_deep_channels) if selected_deep_channels else tuple()
                )
                if selected_deep_bookers:
                    route_perf_df = route_perf_df[route_perf_df['Booker'].astype(str).isin([str(booker) for booker in selected_deep_bookers])]

                st.plotly_chart(
                    create_routewise_sales_performance_chart(route_perf_df, title_suffix=title_suffix),
                    use_container_width=True,
                    key="deep_routewise_sales_perf_chart"
                )

                daily_trend_df = GMV_OB_calendar_heatmap_data(town_code, start_date, end_date)
                st.plotly_chart(
                    create_daily_sales_trend_orders_gmv(
                        daily_trend_df,
                        selected_bookers=selected_deep_bookers,
                        selected_channels=selected_deep_channels,
                        title_suffix=title_suffix,
                    ),
                    use_container_width=True,
                    key="deep_daily_sales_trend_chart"
                )

                daily_calls_df = fetch_daily_calls_trend_data(start_date, end_date, town_code)
                calls_title_suffix = title_suffix
                if daily_calls_df is None or daily_calls_df.empty:
                    latest_visit_df = fetch_latest_visit_date(town_code)
                    if latest_visit_df is not None and not latest_visit_df.empty and pd.notna(latest_visit_df.loc[0, 'Latest_Visit_Date']):
                        latest_visit_date = pd.to_datetime(latest_visit_df.loc[0, 'Latest_Visit_Date']).date()
                        fallback_start = (pd.Timestamp(latest_visit_date) - pd.DateOffset(days=29)).date()
                        daily_calls_df = fetch_daily_calls_trend_data(fallback_start, latest_visit_date, town_code)
                        calls_title_suffix = f"{title_suffix} | Latest data: {fallback_start} to {latest_visit_date}"

                st.plotly_chart(
                    create_daily_calls_trend_chart(
                        daily_calls_df,
                        selected_bookers=selected_deep_bookers,
                        title_suffix=calls_title_suffix,
                    ),
                    use_container_width=True,
                    key="deep_daily_calls_trend_chart"
                )

                activity_df = fetch_activity_segmentation_data(
                    start_date,
                    end_date,
                    town_code,
                    tuple(selected_deep_channels) if selected_deep_channels else tuple(),
                    tuple(selected_deep_bookers) if selected_deep_bookers else tuple(),
                )

                activity_booker_df = fetch_activity_segmentation_booker_data(
                    start_date,
                    end_date,
                    town_code,
                    tuple(selected_deep_channels) if selected_deep_channels else tuple(),
                    tuple(selected_deep_bookers) if selected_deep_bookers else tuple(),
                )

                cohort_orders_df = fetch_weekly_cohort_orders(
                    start_date,
                    end_date,
                    town_code,
                    tuple(selected_deep_channels) if selected_deep_channels else tuple(),
                    tuple(selected_deep_bookers) if selected_deep_bookers else tuple(),
                )

                if activity_df is not None and not activity_df.empty:
                    activity_plot_df = activity_df.copy()
                    activity_plot_df['Orders_In_Period'] = pd.to_numeric(
                        activity_plot_df.get('Orders_In_Period', 0), errors='coerce'
                    ).fillna(0)

                    months_in_range = max(
                        (pd.to_datetime(end_date).year - pd.to_datetime(start_date).year) * 12
                        + (pd.to_datetime(end_date).month - pd.to_datetime(start_date).month)
                        + 1,
                        1,
                    )
                    activity_plot_df['Avg_Orders_Per_Month'] = (
                        activity_plot_df['Orders_In_Period'] / months_in_range
                    )

                    activity_plot_df['Segment_Label'] = np.select(
                        [
                            activity_plot_df['Orders_In_Period'] <= 0,
                            activity_plot_df['Avg_Orders_Per_Month'] > 4,
                            activity_plot_df['Avg_Orders_Per_Month'] >= 2,
                            activity_plot_df['Avg_Orders_Per_Month'] >= 1,
                        ],
                        [
                            'Dormant (0 orders)',
                            'Power Users (>4x/mo)',
                            'Regular (2–4x/mo)',
                            'Occasional (1x/mo)',
                        ],
                        default='Occasional (1x/mo)',
                    )

                    activity_summary = (
                        activity_plot_df
                        .groupby('Segment_Label', as_index=False)
                        .agg(Outlet_Count=('Store_Code', 'nunique'))
                    )

                    cohort_col, segmentation_col = st.columns(2)
                    with cohort_col:
                        st.markdown(
                            """
                            <div style='margin: 0 0 6px 2px; font-size: 13px; color: #334155; font-weight: 600;'>
                                Weekly Retention
                                <span title='Retention shows how many customers from a starting cohort week order again in later weeks. W+0 is the first week, W+1 is the next week, and so on.' style='cursor:help; margin-left:6px; color:#64748B;'>?</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.plotly_chart(
                            create_weekly_cohort_chart(cohort_orders_df, title_suffix=title_suffix),
                            use_container_width=True,
                            key="deep_weekly_cohort_chart"
                        )
                    with segmentation_col:
                        st.markdown(
                            """
                            <div style='margin: 0 0 6px 2px; font-size: 13px; color: #334155; font-weight: 600;'>
                                Segmentation by Activity
                                <span title='This chart groups customers by average monthly order activity: Power (>4x/mo), Regular (2–4x/mo), Occasional (1x/mo), and Dormant (0 orders in selected period).' style='cursor:help; margin-left:6px; color:#64748B;'>?</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.plotly_chart(
                            create_activity_segmentation_donut(activity_summary, start_date, end_date, title_suffix=title_suffix),
                            use_container_width=True,
                            key="deep_activity_segmentation_donut"
                        )

                    if activity_booker_df is not None and not activity_booker_df.empty:
                        activity_booker_plot_df = activity_booker_df.copy()
                        activity_booker_plot_df['Orders_In_Period'] = pd.to_numeric(
                            activity_booker_plot_df.get('Orders_In_Period', 0), errors='coerce'
                        ).fillna(0)
                        activity_booker_plot_df['Last_Order_Date'] = pd.to_datetime(
                            activity_booker_plot_df.get('Last_Order_Date'), errors='coerce'
                        )
                        activity_booker_plot_df['Avg_Orders_Per_Month'] = (
                            activity_booker_plot_df['Orders_In_Period'] / months_in_range
                        )
                        activity_booker_plot_df['Segment_Label'] = np.select(
                            [
                                activity_booker_plot_df['Orders_In_Period'] <= 0,
                                activity_booker_plot_df['Avg_Orders_Per_Month'] > 4,
                                activity_booker_plot_df['Avg_Orders_Per_Month'] >= 2,
                                activity_booker_plot_df['Avg_Orders_Per_Month'] >= 1,
                            ],
                            [
                                'Dormant (0 orders)',
                                'Power Users (>4x/mo)',
                                'Regular (2–4x/mo)',
                                'Occasional (1x/mo)',
                            ],
                            default='Occasional (1x/mo)',
                        )

                        activity_booker_summary = (
                            activity_booker_plot_df
                            .groupby(['Booker', 'Segment_Label'], as_index=False)
                            .agg(Outlet_Count=('Store_Code', 'nunique'))
                        )

                        booker_seg_left, booker_seg_right = st.columns([1.65, 1.05])
                        with booker_seg_left:
                            st.plotly_chart(
                                create_booker_wise_activity_segmentation_chart(activity_booker_summary, title_suffix=title_suffix),
                                use_container_width=True,
                                key="deep_activity_segmentation_booker_chart"
                            )

                        with booker_seg_right:
                            table_booker_options = sorted(activity_booker_plot_df['Booker'].dropna().astype(str).unique().tolist())
                            table_segment_options = [
                                'Power Users (>4x/mo)',
                                'Regular (2–4x/mo)',
                                'Occasional (1x/mo)',
                                'Dormant (0 orders)',
                            ]

                            table_filter_col1, table_filter_col2 = st.columns(2)
                            with table_filter_col1:
                                table_filter_bookers = st.multiselect(
                                    "Booker (Table Filter)",
                                    options=table_booker_options,
                                    default=[],
                                    key="deep_booker_seg_table_booker_filter",
                                    help="Applies only on this table"
                                )
                            with table_filter_col2:
                                table_filter_segments = st.multiselect(
                                    "Segment (Table Filter)",
                                    options=table_segment_options,
                                    default=[],
                                    key="deep_booker_seg_table_segment_filter",
                                    help="Applies only on this table"
                                )

                            table_df = activity_booker_plot_df.copy()
                            if table_filter_bookers:
                                table_df = table_df[
                                    table_df['Booker'].astype(str).isin([str(booker) for booker in table_filter_bookers])
                                ]
                            if table_filter_segments:
                                table_df = table_df[
                                    table_df['Segment_Label'].astype(str).isin([str(segment) for segment in table_filter_segments])
                                ]

                            table_df['Last_Order_Date'] = pd.to_datetime(table_df['Last_Order_Date'], errors='coerce').dt.strftime('%d-%b-%Y')
                            table_df['Last_Order_Date'] = table_df['Last_Order_Date'].fillna('-')
                            table_df = table_df.rename(columns={
                                'Booker': 'Booker',
                                'Store_Name': 'Shop Name',
                                'Segment_Label': 'Segment',
                                'Orders_In_Period': 'Orders In Period',
                                'Last_Order_Date': 'Last Order Date',
                            })

                            render_booker_segmentation_table(
                                table_df[['Shop Name', 'Segment', 'Orders In Period', 'Last Order Date']],
                                # height_px=480,
                                
                            )
                else:
                    cohort_col, segmentation_col = st.columns(2)
                    with cohort_col:
                        st.markdown(
                            """
                            <div style='margin: 0 0 6px 2px; font-size: 13px; color: #334155; font-weight: 600;'>
                                Weekly Retention
                                <span title='Retention shows how many customers from a starting cohort week order again in later weeks. W+0 is the first week, W+1 is the next week, and so on.' style='cursor:help; margin-left:6px; color:#64748B;'>?</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.plotly_chart(
                            create_weekly_cohort_chart(cohort_orders_df, title_suffix=title_suffix),
                            use_container_width=True,
                            key="deep_weekly_cohort_chart_empty"
                        )
                    with segmentation_col:
                        st.markdown(
                            """
                            <div style='margin: 0 0 6px 2px; font-size: 13px; color: #334155; font-weight: 600;'>
                                Segmentation by Activity
                                <span title='This chart groups customers by average monthly order activity: Power (>4x/mo), Regular (2–4x/mo), Occasional (1x/mo), and Dormant (0 orders in selected period).' style='cursor:help; margin-left:6px; color:#64748B;'>?</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.info("No activity segmentation data available for selected filters.")

                    booker_seg_left, booker_seg_right = st.columns([1.65, 1.05])
                    with booker_seg_left:
                        st.plotly_chart(
                            create_booker_wise_activity_segmentation_chart(pd.DataFrame(), title_suffix=title_suffix),
                            use_container_width=True,
                            key="deep_activity_segmentation_booker_chart_empty"
                        )
                    with booker_seg_right:
                        st.info("No data available for table with current filters.")

                st.markdown("---")
                st.markdown(
                    """
                    <div style='font-size: 24px; font-weight: 700; color:#0F172A; margin: 0 0 6px 0;'>
                        🏷️ Booker Brand-Level Scoring
                        <span title='Brand Score = (Brand NMV / Booker Total NMV) × 100. This section highlights top and low-focus brands for each booker.' style='cursor:help; margin-left:6px; color:#64748B;'>?</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                brand_score_df = fetch_booker_brand_scoring_data(
                    start_date,
                    end_date,
                    town_code,
                    tuple(selected_deep_channels) if selected_deep_channels else tuple(),
                    tuple(selected_deep_bookers) if selected_deep_bookers else tuple(),
                )

                if brand_score_df is None or brand_score_df.empty:
                    st.info("No brand-level sales data available for current filters.")
                else:
                    score_df = brand_score_df.copy()
                    score_df['NMV'] = pd.to_numeric(score_df.get('NMV', 0), errors='coerce').fillna(0)
                    score_df['Orders'] = pd.to_numeric(score_df.get('Orders', 0), errors='coerce').fillna(0)
                    score_df['Booker'] = score_df['Booker'].astype(str)
                    score_df['Brand'] = score_df['Brand'].astype(str)

                    score_df['Booker_Total_NMV'] = score_df.groupby('Booker')['NMV'].transform('sum')
                    score_df['Brand_Score'] = np.where(
                        score_df['Booker_Total_NMV'] > 0,
                        (score_df['NMV'] / score_df['Booker_Total_NMV']) * 100,
                        0,
                    )
                    score_df['Brand_Score'] = score_df['Brand_Score'].round(1)

                    score_df['Rank_Desc'] = score_df.groupby('Booker')['NMV'].rank(method='first', ascending=False)
                    score_df['Rank_Asc'] = score_df.groupby('Booker')['NMV'].rank(method='first', ascending=True)

                    top_brand_df = (
                        score_df[score_df['Rank_Desc'] == 1][['Booker', 'Brand', 'NMV', 'Brand_Score']]
                        .rename(columns={'Brand': 'Top Brand', 'NMV': 'Top Brand NMV', 'Brand_Score': 'Top Brand Score %'})
                    )
                    bottom_brand_df = (
                        score_df[score_df['Rank_Asc'] == 1][['Booker', 'Brand', 'NMV', 'Brand_Score']]
                        .rename(columns={'Brand': 'Bottom Brand', 'NMV': 'Bottom Brand NMV', 'Brand_Score': 'Bottom Brand Score %'})
                    )
                    top_bottom_df = top_brand_df.merge(bottom_brand_df, on='Booker', how='outer').sort_values('Booker')

                    score_left, score_right = st.columns([1.15, 1.35])
                    with score_left:
                        st.markdown(
                            """
                            <div style='font-size:14px;font-weight:700;color:#0F172A;'>
                                Top/Bottom Brand by Booker
                                <span title='Top Brand = highest NMV brand for booker. Bottom Brand = lowest NMV brand for booker in selected period.' style='cursor:help; margin-left:6px; color:#64748B;'>?</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        render_top_bottom_brand_table(top_bottom_df, height_px=420)

                    with score_right:
                        score_booker_options = sorted(score_df['Booker'].dropna().unique().tolist())
                        selected_score_booker = st.selectbox(
                            "Select Booker for Brand Score Detail",
                            options=score_booker_options,
                            index=0,
                            key="deep_brand_score_booker_select",
                            help="Shows brand contribution % for selected booker."
                        )

                        score_detail_df = score_df[score_df['Booker'] == selected_score_booker].sort_values('NMV', ascending=False)
                        detail_fig = go.Figure(
                            data=[
                                go.Bar(
                                    x=score_detail_df['Brand'],
                                    y=score_detail_df['Brand_Score'],
                                    marker_color='#5B5F97',
                                    text=score_detail_df['Brand_Score'].apply(lambda value: f"{value:.1f}%"),
                                    textposition='outside',
                                    customdata=np.column_stack([score_detail_df['NMV'], score_detail_df['Orders']]),
                                    hovertemplate=(
                                        '<b>%{x}</b>'
                                        '<br>Brand Score: %{y:.1f}%'
                                        '<br>NMV: Rs %{customdata[0]:,.0f}'
                                        '<br>Orders: %{customdata[1]:,.0f}'
                                        '<extra></extra>'
                                    )
                                )
                            ]
                        )
                        detail_fig.update_layout(
                            title=f"Brand Score Detail - {selected_score_booker}",
                            xaxis_title='Brand',
                            yaxis_title='Brand Score %',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color=get_theme_text_color() or '#111827'),
                            margin=dict(l=8, r=8, t=42, b=8),
                        )
                        st.plotly_chart(detail_fig, use_container_width=True, key="deep_booker_brand_score_detail_chart")

                    st.markdown(
                        """
                        <div style='font-size:14px;font-weight:700;color:#0F172A;'>
                            Booker-wise Low Focus Brand Summary
                            <span title='Low-focus brands are those with contribution <= threshold, plus zero-sale brands for that booker.' style='cursor:help; margin-left:6px; color:#64748B;'>?</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    low_focus_threshold = st.slider(
                        "Low Focus Threshold (%)",
                        min_value=0.0,
                        max_value=20.0,
                        value=5.0,
                        step=0.5,
                        key="deep_low_focus_brand_threshold",
                        help="Brand score <= threshold will be considered low focus; zero-sale brands are also included."
                    )

                    all_brands = sorted(score_df['Brand'].dropna().astype(str).unique().tolist())
                    summary_rows = []
                    for booker_name, grp in score_df.groupby('Booker', as_index=False):
                        grp = grp.copy()
                        sold_brands = set(grp['Brand'].astype(str).tolist())
                        grp['Brand_Score'] = pd.to_numeric(grp.get('Brand_Score', 0), errors='coerce').fillna(0)
                        brand_score_map = {
                            str(item['Brand']): float(item['Brand_Score'])
                            for _, item in grp[['Brand', 'Brand_Score']].iterrows()
                        }

                        low_score_brands = set(
                            grp.loc[grp['Brand_Score'] <= float(low_focus_threshold), 'Brand'].astype(str).tolist()
                        )
                        zero_sale_brands = set(all_brands) - sold_brands

                        no_focus_brand_items = []
                        for brand in low_score_brands:
                            no_focus_brand_items.append((str(brand), float(brand_score_map.get(str(brand), 0.0))))
                        for brand in zero_sale_brands:
                            no_focus_brand_items.append((str(brand), 0.0))

                        no_focus_brand_items = list(dict.fromkeys(no_focus_brand_items))
                        no_focus_brand_items = sorted(no_focus_brand_items, key=lambda item: (item[1], item[0]))

                        summary_rows.append({
                            'Booker': str(booker_name),
                            'Low Focus Brand Count': int(len(no_focus_brand_items)),
                            'Low Focus Brand List': no_focus_brand_items,
                        })

                    low_focus_summary_df = pd.DataFrame(summary_rows).sort_values(
                        ['Low Focus Brand Count', 'Booker'], ascending=[False, True]
                    )

                    st.caption("Chip Color Guide: Grey = 0% | Red = <=2% | Amber = <=5% | Green = >5% (if included by threshold)")

                    rows_html = []
                    def _chip_style(score_value):
                        if score_value <= 0:
                            return '#F1F5F9', '#475569', '#CBD5E1'
                        if score_value <= 2:
                            return '#FEE2E2', '#991B1B', '#FCA5A5'
                        if score_value <= 5:
                            return '#FEF3C7', '#92400E', '#FCD34D'
                        return '#DCFCE7', '#166534', '#86EFAC'

                    for _, row in low_focus_summary_df.iterrows():
                        booker_name = escape(str(row.get('Booker', '')))
                        brand_count = int(row.get('Low Focus Brand Count', 0))
                        brand_list = row.get('Low Focus Brand List', [])
                        if isinstance(brand_list, list) and brand_list:
                            chip_items = []
                            for brand_item in brand_list:
                                if isinstance(brand_item, tuple) and len(brand_item) == 2:
                                    brand_name, brand_score = brand_item
                                else:
                                    brand_name, brand_score = str(brand_item), 0.0
                                chip_bg, chip_text, chip_border = _chip_style(float(brand_score))
                                chip_items.append(
                                    f"<span title='Contribution: {float(brand_score):.1f}%' style='display:inline-block;background:{chip_bg};color:{chip_text};border:1px solid {chip_border};border-radius:999px;padding:3px 8px;font-size:11px;font-weight:600;line-height:1.2;'>{escape(str(brand_name))}</span>"
                                )
                            chips_html = "".join(chip_items)
                        else:
                            chips_html = "<span style='color:#94A3B8;'>-</span>"

                        rows_html.append(
                            "<tr style='border-bottom:1px solid #EEF2F7;'>"
                            f"<td style='padding:8px 10px;color:#0F172A;font-size:12px;font-weight:600;white-space:nowrap;'>{booker_name}</td>"
                            f"<td style='padding:8px 10px;color:#334155;font-size:12px;text-align:right;white-space:nowrap;'>{brand_count}</td>"
                            f"<td style='padding:8px 10px;'><div style='display:flex;flex-wrap:wrap;gap:6px;'>{chips_html}</div></td>"
                            "</tr>"
                        )

                    low_focus_table_html = (
                        "<div style='border:1px solid #D9E3EF;border-radius:12px;background:#FFFFFF;overflow:hidden;'>"
                        "<div style='max-height:260px;overflow:auto;'>"
                        "<table style='width:100%;border-collapse:separate;border-spacing:0;'>"
                        "<thead><tr>"
                        "<th title='Order Booker Name' style='position:sticky;top:0;z-index:2;background:#F8FAFC;border-bottom:1px solid #E2E8F0;padding:8px 10px;text-align:left;color:#334155;font-size:12px;'>Booker</th>"
                        "<th title='Number of brands considered low/no focus for this booker' style='position:sticky;top:0;z-index:2;background:#F8FAFC;border-bottom:1px solid #E2E8F0;padding:8px 10px;text-align:right;color:#334155;font-size:12px;'>Low Focus Brand Count</th>"
                        "<th title='Colored chips represent contribution level; hover a chip for exact contribution %' style='position:sticky;top:0;z-index:2;background:#F8FAFC;border-bottom:1px solid #E2E8F0;padding:8px 10px;text-align:left;color:#334155;font-size:12px;'>Low Focus Brands</th>"
                        "</tr></thead>"
                        f"<tbody>{''.join(rows_html)}</tbody>"
                        "</table></div></div>"
                    )
                    st.markdown(low_focus_table_html, unsafe_allow_html=True)

                calls_kpi_df = daily_calls_df.copy() if daily_calls_df is not None else pd.DataFrame()
                if calls_kpi_df is not None and not calls_kpi_df.empty and selected_deep_bookers:
                    calls_kpi_df = calls_kpi_df[calls_kpi_df['Booker'].astype(str).isin([str(booker) for booker in selected_deep_bookers])]

                def _compute_calls_metrics(frame):
                    if frame is None or frame.empty:
                        return {
                            'avg_strike_rate': 0.0,
                            'avg_calls_day': 0.0,
                            'productive_calls_pct': 0.0,
                        }
                    planned = pd.to_numeric(frame.get('Planned_Calls', 0), errors='coerce').fillna(0).sum()
                    executed = pd.to_numeric(frame.get('Executed_Calls', 0), errors='coerce').fillna(0).sum()
                    productive = pd.to_numeric(frame.get('Productive_Calls', 0), errors='coerce').fillna(0).sum()
                    call_days = pd.to_datetime(frame.get('Call_Date'), errors='coerce').dropna().nunique()

                    strike_rate = (executed / planned * 100) if planned > 0 else 0.0
                    calls_day = (planned / call_days) if call_days > 0 else 0.0
                    productive_pct = (productive / planned * 100) if planned > 0 else 0.0
                    return {
                        'avg_strike_rate': float(strike_rate),
                        'avg_calls_day': float(calls_day),
                        'productive_calls_pct': float(productive_pct),
                    }

                curr_calls_metrics = _compute_calls_metrics(calls_kpi_df)

                range_days = max((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1, 1)
                prev_end_date = (pd.to_datetime(start_date) - pd.Timedelta(days=1)).date()
                prev_start_date = (pd.to_datetime(prev_end_date) - pd.Timedelta(days=range_days - 1)).date()
                prev_calls_df = fetch_daily_calls_trend_data(prev_start_date, prev_end_date, town_code)
                if prev_calls_df is not None and not prev_calls_df.empty and selected_deep_bookers:
                    prev_calls_df = prev_calls_df[prev_calls_df['Booker'].astype(str).isin([str(booker) for booker in selected_deep_bookers])]
                prev_calls_metrics = _compute_calls_metrics(prev_calls_df)

                def _delta_text(current, previous, as_pct=False):
                    delta = float(current) - float(previous)
                    arrow = '▲' if delta >= 0 else '▼'
                    color = '#00FF85' if delta >= 0 else '#FF4D6D'
                    if as_pct:
                        text = f"{arrow} {abs(delta):.1f}%"
                    else:
                        text = f"{arrow} {abs(delta):.1f}"
                    return text, color

                strike_delta_text, strike_delta_color = _delta_text(
                    curr_calls_metrics['avg_strike_rate'],
                    prev_calls_metrics['avg_strike_rate'],
                    as_pct=True
                )
                calls_delta_text, calls_delta_color = _delta_text(
                    curr_calls_metrics['avg_calls_day'],
                    prev_calls_metrics['avg_calls_day'],
                    as_pct=False
                )
                productive_delta_text, productive_delta_color = _delta_text(
                    curr_calls_metrics['productive_calls_pct'],
                    prev_calls_metrics['productive_calls_pct'],
                    as_pct=True
                )

                curr_total_visits = 0.0
                if calls_kpi_df is not None and not calls_kpi_df.empty:
                    curr_total_visits = pd.to_numeric(
                        calls_kpi_df.get('Planned_Calls', 0), errors='coerce'
                    ).fillna(0).sum()

                prev_total_visits = 0.0
                if prev_calls_df is not None and not prev_calls_df.empty:
                    prev_total_visits = pd.to_numeric(
                        prev_calls_df.get('Planned_Calls', 0), errors='coerce'
                    ).fillna(0).sum()

                visits_delta_text, visits_delta_color = _delta_text(
                    curr_total_visits,
                    prev_total_visits,
                    as_pct=False
                )

                curr_sku_df = fetch_sku_per_bill_metric(
                    start_date,
                    end_date,
                    town_code,
                    tuple(selected_deep_channels) if selected_deep_channels else tuple(),
                    tuple(selected_deep_bookers) if selected_deep_bookers else tuple(),
                )
                prev_sku_df = fetch_sku_per_bill_metric(
                    prev_start_date,
                    prev_end_date,
                    town_code,
                    tuple(selected_deep_channels) if selected_deep_channels else tuple(),
                    tuple(selected_deep_bookers) if selected_deep_bookers else tuple(),
                )

                curr_sku_per_bill = 0.0
                if curr_sku_df is not None and not curr_sku_df.empty and 'SKU_Per_Bill' in curr_sku_df.columns:
                    curr_sku_per_bill = float(pd.to_numeric(curr_sku_df.loc[0, 'SKU_Per_Bill'], errors='coerce') or 0.0)

                prev_sku_per_bill = 0.0
                if prev_sku_df is not None and not prev_sku_df.empty and 'SKU_Per_Bill' in prev_sku_df.columns:
                    prev_sku_per_bill = float(pd.to_numeric(prev_sku_df.loc[0, 'SKU_Per_Bill'], errors='coerce') or 0.0)

                sku_delta_text, sku_delta_color = _delta_text(
                    curr_sku_per_bill,
                    prev_sku_per_bill,
                    as_pct=False
                )

                with top_calls_kpi_container:
                    call_kpi_col1, call_kpi_col2, call_kpi_col3, call_kpi_col4, call_kpi_col5 = st.columns(5)
                    with call_kpi_col1:
                        render_unified_kpi_card(
                            label='Avg Strike Rate',
                            value=f"{curr_calls_metrics['avg_strike_rate']:.1f}%",
                            delta_primary=strike_delta_text,
                            delta_primary_color=strike_delta_color,
                            tooltip='(Executed Calls / Planned Calls) × 100',
                            line_gradient='linear-gradient(90deg, #06B6D4, #38BDF8)'
                        )
                    with call_kpi_col2:
                        render_unified_kpi_card(
                            label='Avg Calls/Day',
                            value=f"{curr_calls_metrics['avg_calls_day']:.1f}",
                            delta_primary=calls_delta_text,
                            delta_primary_color=calls_delta_color,
                            tooltip='Planned Calls / Distinct Call Days',
                            line_gradient='linear-gradient(90deg, #10B981, #34D399)'
                        )
                    with call_kpi_col3:
                        render_unified_kpi_card(
                            label='Productive Calls',
                            value=f"{curr_calls_metrics['productive_calls_pct']:.1f}%",
                            delta_primary=productive_delta_text,
                            delta_primary_color=productive_delta_color,
                            tooltip='(Productive Calls / Planned Calls) × 100',
                            line_gradient='linear-gradient(90deg, #8B5CF6, #A78BFA)'
                        )
                    with call_kpi_col4:
                        render_unified_kpi_card(
                            label='SKU / Bill',
                            value=f"{curr_sku_per_bill:.2f}",
                            delta_primary=sku_delta_text,
                            delta_primary_color=sku_delta_color,
                            tooltip='Distinct (Invoice + SKU) / Distinct Invoices',
                            line_gradient='linear-gradient(90deg, #F59E0B, #FBBF24)'
                        )
                    with call_kpi_col5:
                        render_unified_kpi_card(
                            label='Total Visits',
                            value=f"{int(round(curr_total_visits)):,}",
                            delta_primary=visits_delta_text,
                            delta_primary_color=visits_delta_color,
                            tooltip='Total planned visits in selected period',
                            line_gradient='linear-gradient(90deg, #14B8A6, #22D3EE)'
                        )

                booker_agg = (
                    deep_plot_df
                    .groupby('Booker', as_index=False)
                    .agg({'NMV': 'sum', 'Orders': 'sum', 'Stores': 'sum'})
                )
                booker_agg['AOV'] = np.where(
                    booker_agg['Orders'] > 0,
                    booker_agg['NMV'] / booker_agg['Orders'],
                    0
                )

                total_nmv = booker_agg['NMV'].sum()
                total_orders = booker_agg['Orders'].sum()
                avg_aov = total_nmv / total_orders if total_orders > 0 else 0

                with top_summary_kpi_container:
                    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
                    with kpi_col1:
                        render_unified_kpi_card(
                            label='Active Bookers',
                            value=f"{booker_agg['Booker'].nunique():,}",
                            line_gradient='linear-gradient(90deg, #06B6D4, #38BDF8)',
                        )
        
                    with kpi_col2:
                        render_unified_kpi_card(
                            label='Total NMV',
                            value=f"Rs {total_nmv / 1_000_000:.2f}M",
                            line_gradient='linear-gradient(90deg, #8B5CF6, #A78BFA)',
                        )
                    with kpi_col3:
                        render_unified_kpi_card(
                            label='Total Orders',
                            value=f"{int(total_orders):,}",
                            line_gradient='linear-gradient(90deg, #F59E0B, #FBBF24)',
                        )
                    with kpi_col4:
                        render_unified_kpi_card(
                            label='Avg AOV',
                            value=f"Rs {avg_aov / 1000:.1f}K",
                            line_gradient='linear-gradient(90deg, #14B8A6, #22D3EE)',
                        )

                leaderboard_df = fetch_booker_leaderboard_data(
                    start_date,
                    end_date,
                    town_code,
                    tuple(selected_deep_channels) if selected_deep_channels else tuple()
                )
                if selected_deep_bookers:
                    leaderboard_df = leaderboard_df[
                        leaderboard_df['Booker'].astype(str).isin([str(booker) for booker in selected_deep_bookers])
                    ]

                if leaderboard_df is not None and not leaderboard_df.empty:
                    leaderboard_df['Strike_Rate'] = np.where(
                        pd.to_numeric(leaderboard_df['Planned_Calls'], errors='coerce').fillna(0) > 0,
                        (pd.to_numeric(leaderboard_df['Executed_Calls'], errors='coerce').fillna(0)
                         / pd.to_numeric(leaderboard_df['Planned_Calls'], errors='coerce').fillna(0)) * 100,
                        0,
                    )
                    leaderboard_df['Calls_Per_Day'] = np.where(
                        pd.to_numeric(leaderboard_df['Call_Days'], errors='coerce').fillna(0) > 0,
                        pd.to_numeric(leaderboard_df['Planned_Calls'], errors='coerce').fillna(0)
                        / pd.to_numeric(leaderboard_df['Call_Days'], errors='coerce').fillna(0),
                        0,
                    )

                    metric_columns = ['Revenue', 'Strike_Rate', 'Calls_Per_Day', 'New_Outlets', 'Avg_Order_Val']
                    for column in metric_columns:
                        leaderboard_df[column] = pd.to_numeric(leaderboard_df[column], errors='coerce').fillna(0)
                        col_min = leaderboard_df[column].min()
                        col_max = leaderboard_df[column].max()
                        if col_max > col_min:
                            leaderboard_df[f'{column}_Norm'] = (leaderboard_df[column] - col_min) / (col_max - col_min)
                        else:
                            leaderboard_df[f'{column}_Norm'] = 0.5

                    leaderboard_df['Perf_Score'] = (
                        leaderboard_df['Revenue_Norm'] * 0.35
                        + leaderboard_df['Strike_Rate_Norm'] * 0.25
                        + leaderboard_df['Calls_Per_Day_Norm'] * 0.15
                        + leaderboard_df['New_Outlets_Norm'] * 0.15
                        + leaderboard_df['Avg_Order_Val_Norm'] * 0.10
                    ) * 100

                    top_5_df = leaderboard_df.sort_values('Perf_Score', ascending=False).head(5)
                    bottom_5_df = leaderboard_df.sort_values('Perf_Score', ascending=True).head(5)

                    st.markdown("### 🏅 Booker Leaderboard")
                    # st.caption(
                    #     "Perf Score (0-100) = weighted performance index using Revenue (35%), Strike Rate (25%), "
                    #     "Calls/Day (15%), New Outlets (15%), and Avg Order Value (10%). Higher score means better overall performance."
                    # )
                    leaderboard_view = st.radio(
                        "Leaderboard View",
                        options=["Top 5", "Bottom 5", "All"],
                        horizontal=True,
                        key="booker_leaderboard_toggle",
                        help="Perf Score (0-100) = weighted performance index using Revenue (35%), Strike Rate (25%), "
                             "Calls/Day (15%), New Outlets (15%), and Avg Order Value (10%). Higher score means better overall performance."
                    )
                    if leaderboard_view == "Bottom 5":
                        render_booker_leaderboard_table(
                            bottom_5_df,
                            f"Bottom 5 Performers{title_suffix}",
                            "leaderboard_bottom_5"
                        )
                    elif leaderboard_view == "All":
                        all_ranked_df = leaderboard_df.sort_values('Perf_Score', ascending=False).copy()
                        render_booker_leaderboard_table(
                            all_ranked_df,
                            f"All Performers{title_suffix}",
                            "leaderboard_all"
                        )
                    else:
                        render_booker_leaderboard_table(
                            top_5_df,
                            f"Top 5 Performers{title_suffix}",
                            "leaderboard_top_5"
                        )

            



    with tab4:
        st.subheader("📦 Inventory")
        inventory_data = fetch_inventory_kpi_data(end_date, town_code)

        if inventory_data is None:
            st.info("No inventory data available for selected account.")
        else:
            curr = inventory_data.get('current', {}) or {}
            prev = inventory_data.get('previous', {}) or {}

            def _delta_text(curr_value, prev_value, unit_suffix="", invert_good=False):
                if prev_value is None:
                    return "-", "#64748B"
                delta = float(curr_value) - float(prev_value)
                arrow = '▲' if delta >= 0 else '▼'
                good_up = (delta >= 0 and not invert_good) or (delta < 0 and invert_good)
                color = '#16A34A' if good_up else '#DC2626'
                if unit_suffix == "%":
                    text = f"{arrow} {abs(delta):.1f}%"
                elif unit_suffix == "days":
                    text = f"{arrow} {abs(delta):.1f} days"
                else:
                    text = f"{arrow} {abs(delta):,.0f}{unit_suffix}"
                return text, color

            total_skus = int(curr.get('total_skus', 0) or 0)
            current_inventory_value = float(curr.get('current_inventory_value', 0.0) or 0.0)
            in_stock_rate = float(curr.get('in_stock_rate', 0.0) or 0.0)
            oos_skus = int(curr.get('oos_skus', 0) or 0)
            avg_days_cover = float(curr.get('avg_days_cover', 0.0) or 0.0)
            slow_movers = int(curr.get('slow_movers', 0) or 0)

            inventory_sparkline = fetch_inventory_ytd_month_end_sparkline(end_date, town_code)

            total_skus_delta, total_skus_color = _delta_text(total_skus, prev.get('total_skus'), unit_suffix="")
            current_inventory_delta, current_inventory_color = _delta_text(
                current_inventory_value,
                prev.get('current_inventory_value'),
                unit_suffix="",
            )
            in_stock_delta, in_stock_color = _delta_text(in_stock_rate, prev.get('in_stock_rate'), unit_suffix="%")
            oos_delta, oos_color = _delta_text(oos_skus, prev.get('oos_skus'), unit_suffix="", invert_good=True)
            cover_delta, cover_color = _delta_text(avg_days_cover, prev.get('avg_days_cover'), unit_suffix="days", invert_good=True)
            slow_delta, slow_color = _delta_text(slow_movers, prev.get('slow_movers'), unit_suffix="", invert_good=True)

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                render_unified_kpi_card(
                    label='Current Inventory',
                    value=f"Rs {current_inventory_value / 1_000_000:.2f}M",
                    line_gradient='linear-gradient(90deg, #14B8A6, #2DD4BF)',
                    delta_primary=current_inventory_delta,
                    delta_primary_color=current_inventory_color,
                    sparkline_points=inventory_sparkline,
                    sparkline_color='#14B8A6',
                    tooltip='YTD month-end inventory trend (last available stock day of each month).',
                )
                
            with c2:
                render_unified_kpi_card(
                    label='Total SKUs',
                    value=f"{total_skus:,}",
                    line_gradient='linear-gradient(90deg, #06B6D4, #38BDF8)',
                    delta_primary=total_skus_delta,
                    delta_primary_color=total_skus_color,
                )
            with c3:
                render_unified_kpi_card(
                    label='In-Stock Rate',
                    value=f"{in_stock_rate:.1f}%",
                    line_gradient='linear-gradient(90deg, #10B981, #34D399)',
                    delta_primary=in_stock_delta,
                    delta_primary_color=in_stock_color,
                )
            with c4:
                render_unified_kpi_card(
                    label='OOS SKUs',
                    value=f"{oos_skus:,}",
                    line_gradient='linear-gradient(90deg, #EF4444, #F87171)',
                    delta_primary=oos_delta,
                    delta_primary_color=oos_color,
                )
            with c5:
                render_unified_kpi_card(
                    label='Avg Days Cover',
                    value=f"{avg_days_cover:.1f} days",
                    line_gradient='linear-gradient(90deg, #F59E0B, #FBBF24)',
                    delta_primary=cover_delta,
                    delta_primary_color=cover_color,
                    tooltip='Weighted by SKU contribution from last 90 days sales: Σ[(Stock/(Sales90/90)) × (SKU Sales90 / Total Sales90)]',
                )
            with c6:
                render_unified_kpi_card(
                    label='Slow Movers',
                    value=f"{slow_movers:,} SKUs",
                    line_gradient='linear-gradient(90deg, #8B5CF6, #A78BFA)',
                    delta_primary=slow_delta,
                    delta_primary_color=slow_color,
                    tooltip="Slow Movers = SKUs with stock > 0 and no sales in last 30 days"
                )

            latest_dt = inventory_data.get('latest_report_date')
            prev_dt = inventory_data.get('prev_report_date')
            if latest_dt:
                st.caption(f"Stock snapshot date: {latest_dt} | Comparison base: {prev_dt if prev_dt else '-'}")

            chart_data = fetch_inventory_chart_data(end_date, town_code)
            trend_df = chart_data.get('trend', pd.DataFrame())
            days_cover_df = chart_data.get('days_cover', pd.DataFrame())

            st.markdown(
                """
                <style>
                div[data-testid="stPlotlyChart"] {
                    background: #FFFFFF;
                    border: 1px solid #D9E3EF;
                    border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                    padding: 0;
                    box-sizing: border-box;
                    overflow-x: hidden !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            ch1, ch2 = st.columns(2)

            with ch1:
                st.markdown("#### Stock Movement Trend")
                if trend_df is None or trend_df.empty:
                    st.info("No stock movement trend data available.")
                else:
                    trend_plot = trend_df.copy()
                    trend_plot['Report_Date'] = pd.to_datetime(trend_plot['Report_Date'], errors='coerce')
                    trend_plot = trend_plot.dropna(subset=['Report_Date']).sort_values('Report_Date')

                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=trend_plot['Report_Date'],
                        y=trend_plot['Inflow_Value'],
                        mode='lines',
                        name='Inflow',
                        line=dict(color='#22C55E', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(34, 197, 94, 0.15)'
                    ))
                    fig_trend.add_trace(go.Scatter(
                        x=trend_plot['Report_Date'],
                        y=trend_plot['Outflow_Value'],
                        mode='lines',
                        name='Outflow',
                        line=dict(color='#3B82F6', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(59, 130, 246, 0.12)'
                    ))
                    fig_trend.update_layout(
                        margin=dict(t=10, b=10, l=10, r=10),
                        height=360,
                        paper_bgcolor='#FFFFFF',
                        plot_bgcolor='#FFFFFF',
                        xaxis=dict(title=None, showgrid=False, showline=True, linecolor='#E2E8F0', linewidth=1),
                        yaxis=dict(title=None, gridcolor='rgba(148,163,184,0.15)', tickformat=',.0f', showline=True, linecolor='#E2E8F0', linewidth=1),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)

            with ch2:
                st.markdown("#### Days Cover by Category")
                if days_cover_df is None or days_cover_df.empty:
                    st.info("No category-level days cover data available.")
                else:
                    cover_plot = days_cover_df.copy()
                    cover_plot['Days_Cover'] = pd.to_numeric(cover_plot['Days_Cover'], errors='coerce').fillna(0)
                    cover_plot = cover_plot.sort_values('Days_Cover', ascending=False).head(12)

                    def _cover_color(v):
                        if v < 7:
                            return '#EF4444'
                        if v <= 14:
                            return '#F59E0B'
                        return '#22C55E'

                    cover_plot['Color'] = cover_plot['Days_Cover'].apply(_cover_color)
                    cover_plot['Category_Short'] = cover_plot['Category'].astype(str).apply(
                        lambda txt: txt if len(txt) <= 18 else f"{txt[:18]}..."
                    )

                    fig_cover = go.Figure(go.Bar(
                        x=cover_plot['Category_Short'],
                        y=cover_plot['Days_Cover'],
                        marker_color=cover_plot['Color'],
                        text=cover_plot['Days_Cover'].round(1),
                        textposition='outside',
                        customdata=cover_plot[['Category']],
                        hovertemplate='<b>%{customdata[0]}</b><br>Days Cover: %{y:.1f}<extra></extra>'
                    ))
                    fig_cover.add_hline(y=7, line_dash='dash', line_color='rgba(239,68,68,0.6)', line_width=1)
                    fig_cover.add_hline(y=14, line_dash='dash', line_color='rgba(245,158,11,0.6)', line_width=1)
                    fig_cover.update_layout(
                        margin=dict(t=10, b=10, l=10, r=10),
                        height=360,
                        paper_bgcolor='#FFFFFF',
                        plot_bgcolor='#FFFFFF',
                        xaxis=dict(title=None, tickangle=-30, showline=True, linecolor='#E2E8F0', linewidth=1),
                        yaxis=dict(title=None, gridcolor='rgba(148,163,184,0.15)', showline=True, linecolor='#E2E8F0', linewidth=1),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_cover, use_container_width=True)
                    st.caption("Thresholds: Red < 7 days | Amber 7-14 days | Green > 14 days")

            health_data = fetch_inventory_health_data(end_date, town_code)
            if health_data:
                with st.expander("Stock Cover Bucket Settings", expanded=False):
                    b1, b2, b3, b4, b5 = st.columns(5)
                    with b1:
                        bucket_1 = st.number_input("B1", min_value=1, value=7, step=1, key="inv_bucket_1")
                    with b2:
                        bucket_2 = st.number_input("B2", min_value=1, value=15, step=1, key="inv_bucket_2")
                    with b3:
                        bucket_3 = st.number_input("B3", min_value=1, value=30, step=1, key="inv_bucket_3")
                    with b4:
                        bucket_4 = st.number_input("B4", min_value=1, value=45, step=1, key="inv_bucket_4")
                    with b5:
                        bucket_5 = st.number_input("B5", min_value=1, value=60, step=1, key="inv_bucket_5")
                    st.caption("Buckets: <B1, B1-B2, B2-B3, B3-B4, B4-B5, B5+, Zero Sale")

                raw_thresholds = [int(bucket_1), int(bucket_2), int(bucket_3), int(bucket_4), int(bucket_5)]
                if raw_thresholds != sorted(raw_thresholds):
                    st.warning("Bucket limits were auto-sorted because they were not in ascending order.")
                thresholds = tuple(sorted(raw_thresholds))

                cover_matrix_data = fetch_inventory_cover_bucket_matrix(end_date, town_code, thresholds)
                cover_table = cover_matrix_data.get('table', pd.DataFrame())
                cover_pct = cover_matrix_data.get('percentages', {}) or {}
                bucket_order = cover_matrix_data.get('bucket_columns', []) or []

                st.markdown("### Stock Cover Days")

                def _fmt_short_rupee(v):
                    value = float(v or 0)
                    abs_value = abs(value)
                    if abs_value >= 1_000_000:
                        return f"{value / 1_000_000:.1f}m"
                    if abs_value >= 1_000:
                        return f"{value / 1_000:.1f}k"
                    return f"{value:.0f}"

                if cover_table is None or cover_table.empty:
                    st.info("No stock cover bucket data available.")
                else:
                    bucket_cell_styles = {
                        bucket_order[0]: ('#FEF2F2', '#B91C1C') if len(bucket_order) > 0 else ('#FFFFFF', '#0F172A'),
                        bucket_order[1]: ('#FFF7ED', '#C2410C') if len(bucket_order) > 1 else ('#FFFFFF', '#0F172A'),
                        bucket_order[2]: ('#FFFBEB', '#A16207') if len(bucket_order) > 2 else ('#FFFFFF', '#0F172A'),
                        bucket_order[3]: ('#F8FAFC', '#334155') if len(bucket_order) > 3 else ('#FFFFFF', '#0F172A'),
                        bucket_order[4]: ('#ECFDF5', '#047857') if len(bucket_order) > 4 else ('#FFFFFF', '#0F172A'),
                        bucket_order[5]: ('#DCFCE7', '#15803D') if len(bucket_order) > 5 else ('#FFFFFF', '#0F172A'),
                        bucket_order[6]: ('#FEE2E2', '#991B1B') if len(bucket_order) > 6 else ('#FFFFFF', '#0F172A'),
                    }

                    pct_cells = "".join([
                        f"<th style='padding:7px 8px;color:{bucket_cell_styles.get(col, ('#EEF2FF', '#0F172A'))[1]};background:{bucket_cell_styles.get(col, ('#EEF2FF', '#0F172A'))[0]};text-align:center;font-size:12px;font-weight:700;border-bottom:1px solid #E2E8F0;'>{cover_pct.get(col, 0.0):.1f}%</th>"
                        for col in bucket_order
                    ])

                    header_cells = "".join([
                        "<th style='padding:10px 8px;background:#F8FAFC;color:#334155;text-align:center;font-size:12px;letter-spacing:0.04em;text-transform:uppercase;border-bottom:1px solid #E2E8F0;'>Account</th>",
                        "<th style='padding:10px 8px;background:#F8FAFC;color:#334155;text-align:center;font-size:12px;letter-spacing:0.04em;text-transform:uppercase;border-bottom:1px solid #E2E8F0;'>Total Inventory</th>",
                    ] + [
                        f"<th style='padding:10px 8px;background:#F8FAFC;color:#334155;text-align:center;font-size:12px;letter-spacing:0.04em;text-transform:uppercase;border-bottom:1px solid #E2E8F0;'>{col}</th>"
                        for col in bucket_order
                    ])

                    row_html = ""
                    for idx, row in cover_table.iterrows():
                        font_w = '500'
                        bg = '#FFFFFF' if idx % 2 == 0 else '#FAFCFF'
                        row_border = "border-bottom:1px solid #EEF2F7;"
                        cells = [
                            f"<td style='padding:9px 10px;background:{bg};font-weight:{font_w};text-align:left;color:#0F172A;{row_border}'>{escape(str(row.get('Account', '')))}</td>",
                            f"<td style='padding:9px 8px;background:{bg};font-weight:{font_w};text-align:center;color:#0F172A;{row_border}'>{_fmt_short_rupee(row.get('Total Inventory', 0))}</td>",
                        ]
                        for col in bucket_order:
                            cell_bg, cell_text = bucket_cell_styles.get(col, ('#FFFFFF', '#0F172A'))
                            cells.append(
                                f"<td style='padding:9px 8px;background:{cell_bg};font-weight:600;text-align:center;color:{cell_text};{row_border}'>{_fmt_short_rupee(row.get(col, 0))}</td>"
                            )
                        row_html += f"<tr>{''.join(cells)}</tr>"

                    st.markdown(
                        f"""
                        <div style="border:1px solid #D9E3EF;border-radius:12px;overflow:hidden;background:#FFFFFF;box-shadow:0 2px 8px rgba(15,23,42,0.06);">
                            <div style="padding:12px 14px;background:#FFFFFF;border-bottom:1px solid #E2E8F0;">
                                <div style="font-size:15px;font-weight:700;color:#0F172A;">Stock Cover Days (as per Salesflo)</div>
                            </div>
                            <div style="overflow:auto;">
                            <table style="width:100%;border-collapse:collapse;min-width:1100px;">
                                <thead>
                                    <tr>
                                        <th colspan="2" style="background:#EEF2FF;padding:7px 8px;border-bottom:1px solid #E2E8F0;"></th>
                                        {pct_cells}
                                    </tr>
                                    <tr>
                                        {header_cells}
                                    </tr>
                                </thead>
                                <tbody>
                                    {row_html}
                                </tbody>
                            </table>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("### Inventory Health")

                def _health_color(metric_name, metric_value):
                    if metric_name == 'dead_stock_value':
                        return '#FF4D6D' if metric_value > 0 else '#00E676'
                    if metric_name == 'wastage_rate':
                        if metric_value >= 2:
                            return '#FF4D6D'
                        if metric_value >= 1:
                            return '#FFB020'
                        return '#00E676'
                    if metric_name in {'inventory_turnover', 'gmroii'}:
                        if metric_value >= 3:
                            return '#00E676'
                        if metric_value >= 1.5:
                            return '#FFB020'
                        return '#FF6B35'
                    if metric_name in {'avg_replenishment_cycle', 'safety_stock_coverage'}:
                        if metric_value <= 5:
                            return '#00E5FF'
                        if metric_value <= 10:
                            return '#FFB020'
                        return '#FF6B35'
                    return '#00E676'

                inventory_turnover = float(health_data.get('inventory_turnover', 0) or 0)
                dead_stock_value = float(health_data.get('dead_stock_value', 0) or 0)
                dead_sku_count = int(health_data.get('dead_sku_count', 0) or 0)
                avg_replenishment_cycle = float(health_data.get('avg_replenishment_cycle', 0) or 0)
                wastage_rate = float(health_data.get('wastage_rate', 0) or 0)
                gmroii = float(health_data.get('gmroii', 0) or 0)
                safety_stock_coverage = float(health_data.get('safety_stock_coverage', 0) or 0)

                metric_tooltips = {
                    'inventory_turnover': 'Total sales over last 90 days divided by average stock value over same period.',
                    'dead_stock_value': 'Current stock value of SKUs where last-90-day sales contribution is <2% and stock cover days is >25.',
                    'avg_replenishment_cycle': 'Average days between positive stock refill events based on daily stock movement.',
                    'wastage_rate': 'Dead stock value as a percentage of total current stock value.',
                    'gmroii': 'Gross margin return proxy: last 30-day sales value divided by average stock value for last 30 days.',
                    'safety_stock_coverage': 'Current stock value divided by average daily sales for last 30 days, shown in days.',
                }

                rows = [
                    ('Inventory Turnover', f"{inventory_turnover:.1f}x", _health_color('inventory_turnover', inventory_turnover), metric_tooltips['inventory_turnover']),
                    ('Dead Stock Value', f"Rs {dead_stock_value / 1_000_000:.1f}M ({dead_sku_count:,} SKUs)", _health_color('dead_stock_value', dead_stock_value), metric_tooltips['dead_stock_value']),
                    ('Avg Replenishment Cycle', f"{avg_replenishment_cycle:.1f} days", _health_color('avg_replenishment_cycle', avg_replenishment_cycle), metric_tooltips['avg_replenishment_cycle']),
                    ('Wastage Rate', f"{wastage_rate:.2f}%", _health_color('wastage_rate', wastage_rate), metric_tooltips['wastage_rate']),
                    ('GMROII', f"{gmroii:.1f}x", _health_color('gmroii', gmroii), metric_tooltips['gmroii']),
                    ('Safety Stock Coverage', f"{safety_stock_coverage:.1f} days", _health_color('safety_stock_coverage', safety_stock_coverage), metric_tooltips['safety_stock_coverage']),
                ]

                rows_html = ""
                for idx, (name, value, color, tooltip_text) in enumerate(rows):
                    border = "border-bottom: 1px solid rgba(71, 85, 105, 0.35);" if idx < len(rows) - 1 else ""
                    safe_tooltip = str(tooltip_text).replace("'", "&#39;").replace('"', '&quot;')
                    rows_html += (
                        f"<div style='display:flex; justify-content:space-between; align-items:center; padding: 12px 0; {border}'>"
                        f"<div style='display:flex; align-items:center; gap:10px;'>"
                        f"<div title='{safe_tooltip}' style='color:#93C5FD; font-size:24px; font-weight:500; letter-spacing:0.2px; cursor:help;'>{name}</div>"
                        f"<span title='{safe_tooltip}' style='display:inline-flex; align-items:center; justify-content:center; width:18px; height:18px; border-radius:999px; border:1px solid rgba(147,197,253,0.45); color:#93C5FD; font-size:12px; line-height:1; cursor:help;'>i</span>"
                        f"</div>"
                        f"<div style='color:{color}; font-size:32px; font-weight:700; letter-spacing:0.3px;'>{value}</div>"
                        "</div>"
                    )

                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(180deg, rgba(2,15,35,0.95) 0%, rgba(4,24,50,0.95) 100%);
                        border: 1px solid rgba(30, 64, 175, 0.25);
                        border-radius: 14px;
                        padding: 12px 18px 8px 18px;
                        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
                    ">
                        {rows_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                dead_sku_details = health_data.get('dead_sku_details', pd.DataFrame())
                with st.expander(f"Dead Stock SKU-wise Detail ({dead_sku_count:,} SKUs)", expanded=False):
                    if dead_sku_details is None or dead_sku_details.empty:
                        st.info("No dead stock SKUs found for the selected period and rules.")
                    else:
                        detail_df = dead_sku_details.copy()
                        detail_df['Stock Value'] = pd.to_numeric(detail_df['Stock Value'], errors='coerce').fillna(0)
                        detail_df['Sales 90D'] = pd.to_numeric(detail_df['Sales 90D'], errors='coerce').fillna(0)
                        detail_df['Sales Contribution %'] = pd.to_numeric(detail_df['Sales Contribution %'], errors='coerce').fillna(0)
                        detail_df['Cover Days'] = pd.to_numeric(detail_df['Cover Days'], errors='coerce')
                        st.dataframe(
                            detail_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                'SKU Name': st.column_config.TextColumn('SKU Name'),
                                'Stock Value': st.column_config.NumberColumn('Stock Value', format='%.2f'),
                                'Sales 90D': st.column_config.NumberColumn('Sales 90D', format='%.2f'),
                                'Sales Contribution %': st.column_config.NumberColumn('Sales Contribution %', format='%.2f%%'),
                                'Cover Days': st.column_config.NumberColumn('Cover Days', format='%.1f'),
                            },
                        )

    with tab5:
        st.subheader("🧪 Custom Query Runner")
        st.caption("Run read-only SQL queries on current database. Only single SELECT statements are allowed.")

        default_query = (
            "SELECT `Order Booker Name`, ROUND(SUM(`Delivered Amount` + `Total Discount`),0) AS NMV\n"
            "FROM ordered_vs_delivered_rows\n"
            f"WHERE `Distributor Code` = '{town_code}'\n"
            "GROUP BY `Order Booker Name`\n"
            "ORDER BY NMV DESC\n"
            "LIMIT 50"
        )

        sample_queries = {
            "Booker-wise NMV (Top 50)": default_query,
            "Brand-wise Sales (Top 20)": (
                "SELECT m.brand AS Brand, ROUND(SUM(o.`Delivered Amount` + o.`Total Discount`),0) AS NMV\n"
                "FROM ordered_vs_delivered_rows o\n"
                "LEFT JOIN sku_master m ON m.`Sku_Code` = o.`SKU Code`\n"
                f"WHERE o.`Distributor Code` = '{town_code}'\n"
                "GROUP BY m.brand\n"
                "ORDER BY NMV DESC\n"
                "LIMIT 20"
            ),
            "Daily Sales Trend (Last 30 days)": (
                "SELECT DATE(`Delivery Date`) AS Day, ROUND(SUM(`Delivered Amount` + `Total Discount`),0) AS NMV\n"
                "FROM ordered_vs_delivered_rows\n"
                f"WHERE `Distributor Code` = '{town_code}'\n"
                "  AND `Delivery Date` >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)\n"
                "GROUP BY DATE(`Delivery Date`)\n"
                "ORDER BY Day"
            ),
            "Top 25 Outlets by NMV": (
                "SELECT `Store Name` AS Outlet, ROUND(SUM(`Delivered Amount` + `Total Discount`),0) AS NMV\n"
                "FROM ordered_vs_delivered_rows\n"
                f"WHERE `Distributor Code` = '{town_code}'\n"
                "GROUP BY `Store Name`\n"
                "ORDER BY NMV DESC\n"
                "LIMIT 25"
            ),
        }

        if "custom_sql_query_text" not in st.session_state:
            st.session_state["custom_sql_query_text"] = default_query

        with st.expander("📚 Table & Column Structure", expanded=False):
            try:
                schema_df = fetch_table_structure_data()
                if schema_df is None or schema_df.empty:
                    st.info("No table metadata found for current database.")
                else:
                    column_search = st.text_input(
                        "Search table/column",
                        placeholder="e.g. store, booker, delivery",
                        key="custom_sql_column_search"
                    ).strip()

                    def _append_to_query(text_to_add):
                        current_query = str(st.session_state.get("custom_sql_query_text", "")).rstrip()
                        if current_query:
                            st.session_state["custom_sql_query_text"] = f"{current_query}\n{text_to_add}"
                        else:
                            st.session_state["custom_sql_query_text"] = text_to_add

                    if column_search:
                        match_mask = (
                            schema_df["Table_Name"].astype(str).str.contains(column_search, case=False, na=False)
                            | schema_df["Column_Name"].astype(str).str.contains(column_search, case=False, na=False)
                        )
                        matched_df = schema_df.loc[match_mask, ["Table_Name", "Column_Name", "Data_Type"]].copy()
                        st.caption(f"Search matches: {len(matched_df):,}")
                        st.caption("Tip: matched grid me multiple rows select karo, columns query me auto-add ho jayenge.")
                        match_event = st.dataframe(
                            matched_df,
                            use_container_width=True,
                            hide_index=True,
                            height=220,
                            on_select="rerun",
                            selection_mode="multi-row",
                            key="custom_sql_match_grid"
                        )

                        selected_match_rows = []
                        try:
                            selected_match_rows = match_event.selection.rows
                        except Exception:
                            try:
                                selected_match_rows = match_event.get("selection", {}).get("rows", [])
                            except Exception:
                                selected_match_rows = []

                        if selected_match_rows:
                            selected_match_cols = [
                                str(matched_df.iloc[idx]["Column_Name"])
                                for idx in selected_match_rows
                                if idx < len(matched_df)
                            ]
                            selected_match_expr = ", ".join([f"`{col}`" for col in selected_match_cols])
                            selected_match_token = f"search::{selected_match_expr}"
                            if st.session_state.get("custom_sql_last_inserted_token") != selected_match_token:
                                _append_to_query(selected_match_expr)
                                st.session_state["custom_sql_last_inserted_token"] = selected_match_token
                                st.rerun()

                    table_list = sorted(schema_df["Table_Name"].dropna().unique().tolist())
                    selected_table = st.selectbox(
                        "Select table",
                        options=table_list,
                        key="custom_sql_table_select"
                    )
                    table_columns_df = schema_df[schema_df["Table_Name"] == selected_table][["Column_Name", "Data_Type"]].copy()
                    st.caption("Tip: table columns me multi-select karte hi SELECT query template auto-generate ho jayega.")
                    table_cols_event = st.dataframe(
                        table_columns_df,
                        use_container_width=True,
                        hide_index=True,
                        height=280,
                        on_select="rerun",
                        selection_mode="multi-row",
                        key="custom_sql_table_cols_grid"
                    )
                    st.caption(f"Quick start: SELECT * FROM `{selected_table}` LIMIT 100")

                    selected_table_rows = []
                    try:
                        selected_table_rows = table_cols_event.selection.rows
                    except Exception:
                        try:
                            selected_table_rows = table_cols_event.get("selection", {}).get("rows", [])
                        except Exception:
                            selected_table_rows = []

                    if selected_table_rows:
                        selected_table_cols = [
                            str(table_columns_df.iloc[idx]["Column_Name"])
                            for idx in selected_table_rows
                            if idx < len(table_columns_df)
                        ]
                        selected_table_expr = ", ".join([f"`{col}`" for col in selected_table_cols])
                        selected_table_token = f"table::{selected_table}::{selected_table_expr}"
                        if st.session_state.get("custom_sql_last_inserted_token") != selected_table_token:
                            st.session_state["custom_sql_query_text"] = (
                                f"SELECT {selected_table_expr}\n"
                                f"FROM `{selected_table}`\n"
                                "LIMIT 100"
                            )
                            st.session_state["custom_sql_last_inserted_token"] = selected_table_token
                            st.rerun()

                    table_insert_col1, table_insert_col2 = st.columns(2)
                    with table_insert_col1:
                        if st.button("Insert Table Name", key="custom_sql_insert_table_btn"):
                            _append_to_query(f"`{selected_table}`")
                            st.rerun()
                    with table_insert_col2:
                        if st.button("Reset Last Column Insert", key="custom_sql_reset_last_insert_btn"):
                            st.session_state["custom_sql_last_inserted_token"] = None
                            st.rerun()
            except Exception as schema_exc:
                st.warning(f"Could not load table structure: {schema_exc}")

        sample_col, load_col = st.columns([4, 1])
        with sample_col:
            selected_sample = st.selectbox(
                "Saved sample queries",
                options=list(sample_queries.keys()),
                key="custom_sql_sample_select"
            )
        with load_col:
            st.markdown("<div style='height: 1.9rem;'></div>", unsafe_allow_html=True)
            if st.button("Load Sample", key="custom_sql_load_sample_btn"):
                st.session_state["custom_sql_query_text"] = sample_queries[selected_sample]

        query_text = st.text_area(
            "SQL Query",
            height=220,
            key="custom_sql_query_text",
            help="Allowed: SELECT (single statement)."
        )

        result_limit = st.number_input(
            "Preview rows",
            min_value=10,
            max_value=5000,
            value=500,
            step=10,
            key="custom_sql_preview_limit"
        )

        run_query = st.button("Run Query", key="custom_sql_run_btn", type="primary")

        if run_query:
            raw_query = str(query_text or "").strip()
            normalized = raw_query.lower().strip()
            no_block_comments = re.sub(r"/\*.*?\*/", " ", normalized, flags=re.DOTALL)
            normalized_no_comments = "\n".join([
                line.split("--")[0] for line in no_block_comments.splitlines()
            ]).strip()
            query_without_trailing_semicolon = normalized_no_comments.rstrip().rstrip(";").strip()

            blocked_keywords = [
                "insert", "update", "delete", "drop", "alter", "create", "truncate",
                "grant", "revoke", "replace", "merge", "call", "set", "use",
            ]

            starts_ok = query_without_trailing_semicolon.startswith("select")
            has_multi_stmt = ";" in query_without_trailing_semicolon
            has_blocked = any(
                re.search(rf"\\b{keyword}\\b", query_without_trailing_semicolon) is not None
                for keyword in blocked_keywords
            )

            if not raw_query:
                st.warning("Please enter a SQL query.")
            elif not starts_ok:
                st.error("Only SELECT queries are allowed.")
            elif has_multi_stmt:
                st.error("Only a single SQL statement is allowed.")
            elif has_blocked:
                st.error("Query contains blocked keywords. Only read-only queries are allowed.")
            else:
                try:
                    result_df = pd.read_sql(raw_query, get_engine())
                    if result_df is None or result_df.empty:
                        st.info("Query executed successfully but returned no rows.")
                    else:
                        preview_df = result_df.head(int(result_limit)).copy()
                        st.success(f"Query executed successfully. Returned {len(result_df):,} rows.")
                        st.dataframe(preview_df, use_container_width=True, hide_index=True, height=420)

                        csv_bytes = preview_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Preview CSV",
                            data=csv_bytes,
                            file_name="custom_query_preview.csv",
                            mime="text/csv",
                            key="custom_sql_download_btn"
                        )
                except Exception as exc:
                    st.error(f"Query failed: {exc}")

    with tab6:
        st.subheader("🤖 Salesflo Bot Runner")
        st.caption("Run Scrapper bot from dashboard and monitor logs.")

        if "bot_runner_pid" not in st.session_state:
            st.session_state["bot_runner_pid"] = None
        if "bot_open_live_popup" not in st.session_state:
            st.session_state["bot_open_live_popup"] = False
        if "bot_runner_log_path" not in st.session_state:
            st.session_state["bot_runner_log_path"] = _get_primary_bot_log_path()
        if "bot_runner_log_start_pos" not in st.session_state:
            st.session_state["bot_runner_log_start_pos"] = 0

        current_pid = st.session_state.get("bot_runner_pid")
        is_running = _is_pid_running(current_pid)
        if not is_running and current_pid is not None:
            st.session_state["bot_runner_pid"] = None

        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 3])
        with ctrl_col1:
            if st.button("▶️ Start Bot", key="bot_runner_start_btn", type="primary"):
                if is_running:
                    st.warning(f"Bot is already running (PID: {current_pid}).")
                    st.session_state["bot_open_live_popup"] = True
                else:
                    try:
                        current_log_path = _get_primary_bot_log_path()
                        current_log_start = _safe_file_size(current_log_path)
                        new_pid = _start_bot_process(extra_env=_get_bot_runtime_env())
                        st.session_state["bot_runner_pid"] = int(new_pid)
                        st.session_state["bot_runner_log_path"] = current_log_path
                        st.session_state["bot_runner_log_start_pos"] = int(current_log_start)
                        st.session_state["bot_open_live_popup"] = True
                        st.success(f"Bot started successfully (PID: {new_pid}).")
                    except Exception as bot_start_exc:
                        st.error(f"Could not start bot: {bot_start_exc}")

        with ctrl_col2:
            if st.button("⏹ Stop Bot", key="bot_runner_stop_btn"):
                active_pid = st.session_state.get("bot_runner_pid")
                if _is_pid_running(active_pid):
                    stopped = _stop_pid(active_pid)
                    if stopped:
                        st.session_state["bot_runner_pid"] = None
                        st.session_state["bot_open_live_popup"] = False
                        st.session_state["bot_runner_log_start_pos"] = 0
                        st.success(f"Bot stopped (PID: {active_pid}).")
                    else:
                        st.error(f"Could not stop bot PID: {active_pid}")
                else:
                    st.info("Bot is not running.")

        with ctrl_col3:
            if st.button("🔄 Refresh Logs", key="bot_runner_refresh_btn"):
                st.rerun()

        with ctrl_col4:
            if _is_pid_running(st.session_state.get("bot_runner_pid")):
                st.success(f"Status: Running | PID: {st.session_state.get('bot_runner_pid')}")
            else:
                st.info("Status: Not Running")

        st.markdown("### 🔄 Refresh Data (Selected Period)")
        st.caption(f"Selected period: {start_date} to {end_date}")
        refresh_col1, refresh_col2, refresh_col3 = st.columns([2, 1, 3])
        with refresh_col1:
            refresh_password_input = st.text_input(
                "RefreshData Password",
                type="password",
                key="bot_refresh_data_password_input",
                help="Required to run refresh for selected period."
            )
        with refresh_col2:
            st.markdown("<div style='height: 1.85rem;'></div>", unsafe_allow_html=True)
            if st.button("🔁 RefreshData", key="bot_refresh_data_btn"):
                if _is_pid_running(st.session_state.get("bot_runner_pid")):
                    st.warning("Bot is already running. Stop it first, then run RefreshData.")
                else:
                    expected_password = _get_refresh_data_password()
                    if not expected_password:
                        st.error("Refresh password is not configured. Set REFRESH_DATA_PASSWORD in secrets or environment.")
                    elif str(refresh_password_input or "") != expected_password:
                        st.error("Invalid refresh password.")
                    else:
                        try:
                            current_log_path = _get_primary_bot_log_path()
                            current_log_start = _safe_file_size(current_log_path)
                            refresh_env = _get_bot_runtime_env()
                            refresh_env.update(
                                {
                                    "FORCE_START_DATE": str(start_date),
                                    "FORCE_END_DATE": str(end_date),
                                }
                            )
                            new_pid = _start_bot_process(
                                extra_env=refresh_env
                            )
                            st.session_state["bot_runner_pid"] = int(new_pid)
                            st.session_state["bot_runner_log_path"] = current_log_path
                            st.session_state["bot_runner_log_start_pos"] = int(current_log_start)
                            st.session_state["bot_open_live_popup"] = True
                            st.success(f"RefreshData started for {start_date} to {end_date} (PID: {new_pid}).")
                        except Exception as refresh_exc:
                            st.error(f"Could not start RefreshData: {refresh_exc}")
        with refresh_col3:
            st.caption("Runs bot for only selected period and updates DB rows for that range.")

        lines_to_show = st.slider(
            "Log lines",
            min_value=50,
            max_value=1000,
            value=250,
            step=50,
            key="bot_runner_log_lines"
        )

        logs_text, log_path = _read_bot_logs_tail(
            max_lines=int(lines_to_show),
            start_pos=st.session_state.get("bot_runner_log_start_pos", 0),
            forced_log_path=st.session_state.get("bot_runner_log_path"),
        )
        if log_path:
            st.caption(f"Log file: {log_path}")
        else:
            st.caption("Log file not found yet. Start the bot to generate logs.")

        st.markdown(_render_colored_log_html(logs_text), unsafe_allow_html=True)

        if st.button("📺 Open Live Log Popup", key="bot_runner_open_popup_btn"):
            st.session_state["bot_open_live_popup"] = True

        if _is_pid_running(st.session_state.get("bot_runner_pid")) and st.session_state.get("bot_open_live_popup"):
            show_bot_logs_dialog(lines_to_show)

    st.markdown("---")
    st.markdown("© 2026 Bazaar Prime Analytics Dashboard | Powered by Streamlit")

if __name__ == "__main__":
    main()