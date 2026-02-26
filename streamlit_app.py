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
def fetch_Channel_dm_sunburst_data(start, end, town_code):
    
    """Fetch channel-DM sunburst data"""
    db_name = "db42280"
    
    query = f"""
    SELECT
    u.`Channel Type` as Channel,
		s.Brand,
		o.DM,
		count(DISTINCT o.`Store Code`) as StoreCount,
        town	
FROM
	(SELECT DISTINCT `Deliveryman Name` as DM ,`Store Code`,`SKU Code`,`Delivery Date`,Case when `Distributor Code`='D70002202' then 'Karachi'
				when `Distributor Code`='D70002246' then 'Lahore' else 'CBL' end Town from ordervsdelivered where `Distributor Code` = '{town_code}') o
	LEFT JOIN sku_master s ON s.Sku_Code = o.`SKU Code` 
	LEFT JOIN universe u on u.`Store Code`=o.`Store Code`
	where `Delivery Date` BETWEEN '{start}' AND '{end}'
	GROUP BY u.`Channel Type`,s.Brand,o.dm,o.town"""
    
    return read_sql_cached(query, db_name)

@st.cache_data(ttl=3600)
def Channelwise_performance_data(start, end, town_code):
    """Fetch channel-wise performance data for comparison charts"""
    db_name = "db42280"
    
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
    where o.`Distributor Code`= '{town_code}'
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
def Brand_wise_performance_growth_data(start, end, town_code):
    """Fetch brand-wise performance data for comparison charts"""
    db_name = "db42280"
    
    query = f"""
 WITH raw AS (
    SELECT
        s.brand AS Brand,
        SUM(o.`Delivered Amount` + o.`Total Discount`) AS Amount,
        o.`Delivery Date` AS D_Date,
				case when SUM(o.`Delivered (Litres)`)=0 then SUM(o.`Delivered (Kg)`) else SUM(o.`Delivered (Litres)`) END Ltr,
				Case when o.`Distributor Code`='D70002202' then 'Karachi' else 'Lahore' end Town
    FROM
        ordervsdelivered o
    LEFT JOIN sku_master s ON s.sku_code= o.`sku code`
    where o.`Distributor Code`= '{town_code}'
    GROUP BY
        s.brand,
        o.`Delivery Date`,o.`Distributor Code`
)

, selected_period AS (
    SELECT
		Town,
        raw.brand,
        SUM(raw.Amount) AS NMV,
				SUM(ltr) as Ltr
    FROM
        raw
    WHERE
        raw.D_Date BETWEEN '{start}' AND '{end}'
    GROUP BY
        raw.brand,
				Town
)
, last_year_period AS (
    SELECT
		town,
        raw.brand,
        SUM(raw.Amount) AS NMV,
				sum(ltr) as Ltr
    FROM
        raw
    WHERE
        raw.D_Date BETWEEN DATE_SUB('{start}', INTERVAL 1 YEAR) AND DATE_SUB('{end}', INTERVAL 1 YEAR)
    GROUP BY
        raw.brand,town
),
last_month AS (
    SELECT
		town,
        raw.brand,
        SUM(raw.Amount) AS NMV,
				sum(ltr) as Ltr
    FROM
        raw
    WHERE
        raw.D_Date BETWEEN DATE_SUB('{start}', INTERVAL 1 month) AND DATE_SUB('{end}', INTERVAL 1 month)
    GROUP BY
        raw.brand,town
)

SELECT
		sp.town,
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
    last_year_period ly ON sp.brand = ly.brand
LEFT JOIN
    last_month lm ON sp.brand = lm.brand

		
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
        ordervsdelivered o
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



 from ordervsdelivered o
 LEFT JOIN (SELECT month,year,sum(Target_In_Value) as Target_In_Value,sum(Target_In_Volume) as Target_In_Volume,Distributor_Code  from targets group by year,month,Distributor_Code) t on t.month= month(o.`Delivery Date`) and t.year=YEAR(o.`Delivery Date`) and t.Distributor_Code = o.`Distributor Code`
where o.`Distributor Code` = '{town_code}'
 GROUP BY MONTH(`Delivery Date`),YEAR(`Delivery Date`),Town
order by YEAR(`Delivery Date`) desc,MONTH(`Delivery Date`) desc
 limit 8


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



 from ordervsdelivered o
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
    u.`Channel Type` AS Channel,
    ROUND(SUM(`Delivered Amount` + `Total Discount`)) AS NMV,
    ROUND(SUM(`Delivered (Litres)` + `Delivered (KG)`)) AS Ltr
FROM 
    ordervsdelivered o
INNER JOIN 
    universe u ON u.`Store Code` = o.`Store Code`
WHERE 
    `Delivery Date` >= DATE_SUB(CURDATE(), INTERVAL 7 MONTH)  -- Get data from 7 months ago
    AND (YEAR(`Delivery Date`) < YEAR(CURDATE()) OR (YEAR(`Delivery Date`) = YEAR(CURDATE()) AND MONTH(`Delivery Date`) < MONTH(CURDATE())))  -- Exclude current month
    AND o.`Distributor Code` = '{town_code}'
    GROUP BY 
    period, u.`Channel Type`
ORDER BY 
    period, u.`Channel Type`
    """
    return read_sql_cached(query, db_name)

@st.cache_data(ttl=3600)
def tgtvsach_brand_level(town_code, selected_period):

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
        FROM ordervsdelivered o
        LEFT JOIN sku_master s ON s.Sku_Code = o.`SKU Code`
        WHERE o.`Distributor Code` = '{town_code}'
                    AND o.`Delivery Date` BETWEEN STR_TO_DATE(CONCAT('{selected_period}', '-01'), '%%Y-%%m-%%d')
                                                                    AND LAST_DAY(STR_TO_DATE(CONCAT('{selected_period}', '-01'), '%%Y-%%m-%%d'))
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
def fetch_treemap_period_options(town_code):
        """Fetch available monthly periods for Booker treemap."""
        query = f"""
        select distinct concat(`Year`,"-",`Month`) as period from targets_new
WHERE Distributor_Code ='{town_code}'
        ORDER BY period DESC
        """
        return read_sql_cached(query, "db42280")
    

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
    
    fig = px.sunburst(
        df_plot,
        path=['Channel', 'Brand', 'DM'],
        values='StoreCount',
        title="☀️ Channel, Brand & Deliveryman Hierarchy"
    )
    
    fig.update_traces(
        textinfo="label+percent entry",
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{label}</b><br>Stores: %{value:.0f}<extra></extra>'
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
    
    # Format values for display
    if metric_type == 'Value':
        current_vals = df_processed[current_col].round(2)
        last_year_vals = df_processed[last_year_col].round(2)
        last_month_vals = df_processed[last_month_col].round(2)
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
            marker=dict(color='#1f77b4')
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
            marker=dict(color='#ff7f0e')
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
            marker=dict(color='#2ca02c')
        )
    ])
    
    y_axis_title = 'Sales (in Millions)' if metric_type == 'Value' else 'Litres'
    
    fig.update_layout(
        # title=f"📊 Channel Performance Comparison - {metric_type}",
        yaxis=dict(title=y_axis_title),
        xaxis=dict(title='Channel'),
        height=600,
        barmode='group',
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, weight='bold'),
        legend=dict(title='Period', orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
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

    ly_colors = ['#39A039' if value >= 0 else '#FF6B6B' for value in df_processed[growth_ly_col]]
    lm_colors = ['#2E8B57' if value >= 0 else '#E74C3C' for value in df_processed[growth_lm_col]]

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
        height=500,
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, weight='bold'),
        legend=dict(title='Growth Comparison', orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )

    return apply_theme_aware_bar_labels(fig)

def create_brand_wise_growth_chart(df, metric_type='Value'):
    """Create brand-wise growth percentage chart with value/ltr hover details."""
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
        'brand', current_col, last_year_col, last_month_col,
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

    ly_colors = ['#39A039' if value >= 0 else '#FF6B6B' for value in df_processed[growth_ly_col]]
    lm_colors = ['#2E8B57' if value >= 0 else '#E74C3C' for value in df_processed[growth_lm_col]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_processed['brand'],
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
            x=df_processed['brand'],
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
        title=f"📈 Brand-wise Growth Percentage - {metric_type}",
        yaxis=dict(title='Growth %'),
        xaxis=dict(title='Brand'),
        height=600,
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, weight='bold'),
        legend=dict(title='Growth Comparison', orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    return apply_theme_aware_bar_labels(fig)

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

    ly_colors = ['#39A039' if value >= 0 else '#FF6B6B' for value in df_processed[growth_ly_col]]
    lm_colors = ['#2E8B57' if value >= 0 else '#E74C3C' for value in df_processed[growth_lm_col]]
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
        height=600,
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, weight='bold'),
        legend=dict(title='Growth Comparison', orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
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
            marker=dict(color='#1f77b4'),
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
            marker=dict(color='#ff7f0e'),
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
        height=600,
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
        color_discrete_sequence=['#1f77b4'] * len(brand_prod),
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
        xaxis_title="Period",
        yaxis_title="Booker",
        height=600,
        annotations=annotations,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
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
        xaxis_title="Period",
        yaxis_title="Channel",
        # height=700,
        annotations=annotations,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig


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
    base_theme = st.get_option("theme.base")
    return "#111827" if base_theme == "light" else "#F8FAFC"

def apply_theme_aware_bar_labels(fig):
    """Apply theme-aware text color on all bar-trace data labels."""
    label_color = get_theme_text_color()
    fig.update_traces(selector=dict(type="bar"), textfont=dict(color=label_color))
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
    




# ======================
# 🎯 MAIN APP
# ======================

def main():
    # Check authentication
    check_authentication()

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # Display username if available
    if st.session_state.get("username"):
        st.sidebar.markdown(f"**User:** {st.session_state.username}")

    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

    
    st.sidebar.markdown("---")

    # Period selector
    st.sidebar.subheader("📅 Period")
    period_option = st.sidebar.selectbox(
        "Select Period",
        options=["Last 7 Days", "Last 30 Days", "This Month", "Last Month", "Last 3 Months", "YTD", "Custom"],
        index=1,
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


    st.balloons()
    # Main content
    st.title(f"📊 Bazaar Prime Analytics Dashboard - {town}")
    tab1,tab2=st.tabs(["📈 Sales Growth Analysis","🎯 Booker Performance"])
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
                                <div style='font-size:13px; font-weight:600; margin:0;color:#333; font-weight:600;' >
                                🛍️ {str(r['Channel'])}
                                </div>
                                    <div style='font-size:28px; font-weight:700; margin:2px 0; line-height:1;color:#333;font-weight:700;'>
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
        # if "selected_dms" not in st.session_state:
        #     st.session_state.selected_dms = []
        # if "selected_channels" not in st.session_state:
        #     st.session_state.selected_channels = []

        # try:
        #     (
        #         _kpis,
        #         fig_daily,
        #         fig_type,
        #         fig_sales_growth,
        #         fig_brand_comparison,
        #         fig_brand_growth_bar,
        #         brand_datatable,
        #         fig_dm_comparison,
        #         fig_tgt_vs_ach_ytd,
        #         fig_sku_heatmap,
        #         fig_channel_sunburst,
        #         dm_options,
        #         fig_brand_productivity,
        #         fig_tgt_vs_ach_booker,
        #         fig_channel_nmv_ytd,
        #         channel_options,
        #         fig_channel_treemap,
        #     ) = fetch_bazaarprime_dashboard(
        #         start_date,
        #         end_date,
        #         town_code,
        #         st.session_state.selected_dms,
        #         st.session_state.selected_channels,
        #     )
        # except Exception as exc:
        #     st.error(f"Failed to load BazaarPrime charts: {exc}")
        #     st.stop()

        # dm_values = [item["value"] for item in (dm_options or [])]
        # channel_values = [item["value"] for item in (channel_options or [])]

        # st.subheader("📊 Sales Overview")
        # c1, c2 = st.columns([2, 1])
        # with c1:
        #     st.plotly_chart(fig_daily, use_container_width=True)
        # with c2:
        #     st.plotly_chart(fig_type, use_container_width=True)
        # st.plotly_chart(fig_sales_growth, use_container_width=True)

        # st.markdown("---")
        # st.subheader("🏢 Brand Performance")
        # st.plotly_chart(fig_brand_comparison, use_container_width=True)
        # st.plotly_chart(fig_brand_growth_bar, use_container_width=True)
        # st.dataframe(pd.DataFrame(brand_datatable), use_container_width=True, height=420)

        # st.markdown("---")
        # st.subheader("🚚 Delivery & Operations Analytics")
        # st.plotly_chart(fig_dm_comparison, use_container_width=True)
        # st.plotly_chart(fig_tgt_vs_ach_ytd, use_container_width=True)
        # st.plotly_chart(fig_sku_heatmap, use_container_width=True)

        # st.markdown("---")
        # st.subheader("📊 Channel & Productivity Analysis")
        # col1, col2, col3 = st.columns([1,2,1])
        # with col2:
        #     selected_dms = st.multiselect(
        #     "Filter Deliverymen",
        #     options=dm_values,
        #     default=[dm for dm in st.session_state.selected_dms if dm in dm_values],
        #     )

        # if selected_dms != st.session_state.selected_dms :
        #     st.session_state.selected_dms = selected_dms
        #     st.rerun()
        
        # c3, c4 = st.columns([1.5, 1])
        # with c3:
        #     st.plotly_chart(fig_channel_sunburst, use_container_width=True)
        # with c4:
        #     st.plotly_chart(fig_brand_productivity, use_container_width=True)

        # st.markdown("---")
        # st.subheader("🎯 Target vs Achievement")
        # c5, c6 = st.columns(2)
        # with c5:
        #     st.plotly_chart(fig_tgt_vs_ach_booker, use_container_width=True)
        # with c6:
        #     st.plotly_chart(fig_channel_nmv_ytd, use_container_width=True)

        # st.markdown("---")
        # # st.subheader("🌳 Channel Performance - 6 Month Breakdown")
        # col1, col2, col3 = st.columns([1,2,1])
        # with col2:
        #     selected_channels = st.multiselect(
        #     "Filter Channels",
        #     options=channel_values,
        #     default=[ch for ch in st.session_state.selected_channels if ch in channel_values],
        # )
        # if selected_channels != st.session_state.selected_channels:
        #     st.session_state.selected_channels = selected_channels
        #     st.rerun()
        # st.plotly_chart(fig_channel_treemap, use_container_width=True)

        # st.markdown("---")

        # Booker Analysis Section
        
        #channel wise sales performance chart
        st.markdown("---")
        st.subheader(f"📊 Channel-wise Performance Comparison")
        
        # Filter for metric type
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

        #channel wise growth percentage chart
        st.markdown("---")
        st.subheader(f"📈 Channel-wise Growth Percentage")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col3:
            metric_filter = st.radio(
                "Select Metric for Growth",
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
        with col3:
            metric_filter = st.radio(
                "Select Metric for Brand Growth",
                options=['Value', 'Ltr'],
                horizontal=True,
                help="Toggle between Sales Value and Litres growth comparison"
            )

        brand_growth_df = Brand_wise_performance_growth_data(start_date, end_date, town_code)
        st.plotly_chart(create_brand_wise_growth_chart(brand_growth_df, metric_type=metric_filter), use_container_width=True, key="brand_growth_chart")
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

    with tab2:
        st.subheader("🎯 Booker Performance Analysis")
        period_options_df = fetch_treemap_period_options(town_code)

        if period_options_df is None or period_options_df.empty:
            st.warning("No periods available for treemap data.")
            st.plotly_chart(create_tgtach_brand_maptree(pd.DataFrame()), use_container_width=True, key="booker_treemap_chart")
        else:
            treemap_period_options = period_options_df['period'].dropna().astype(str).tolist()
            col_period, col_ach = st.columns([2, 1])
            with col_period:
                selected_treemap_period = st.selectbox(
                    "Select Period for Booker Treemap",
                    options=treemap_period_options,
                    index=0,
                    key="booker_treemap_period",
                    help="Upload Brand Wise Target Data to get the available periods in this dropdown"
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

            treemap_df = tgtvsach_brand_level(town_code, selected_treemap_period)
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
                    st.caption(f"Rows: {len(treemap_df)} | NMV: {debug_nmv:,.0f} | Target: {debug_target:,.0f}")
                else:
                    st.caption("Rows: 0 for selected period")
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




    st.markdown("---")
    st.markdown("© 2026 Bazaar Prime Analytics Dashboard | Powered by Streamlit")

if __name__ == "__main__":
    main()
