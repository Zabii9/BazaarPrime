import dash
from dash import dcc, html, Input, Output, State
import dash_auth
import pandas as pd
from sqlalchemy import create_engine
from io import StringIO
from functools import lru_cache
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dash.exceptions import PreventUpdate
from dash.dash_table import DataTable
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ======================
# 🔐 LOGIN CONFIG
# ======================
VALID_USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "viewer": {"password": "viewer123", "role": "viewer"},
}
auth_dict = {u: v["password"] for u, v in VALID_USERS.items()}

# ======================
# 🛢 DATABASE (Cloud Configuration)
# ======================
DB_CONFIG = {
    "username": os.getenv("DB_USER", "db42280"),
    "password": os.getenv("DB_PASSWORD", "admin2233"),
    "host": os.getenv("DB_HOST", "db42280.public.databaseasp.net"),
    "port": int(os.getenv("DB_PORT", "3306")),
}

# Map town values to database names
TOWN_TO_DB = {
    "all": "db42280",  # Default to main cloud database for 'All Locations'
    "prime": "db42280",  # Karachi
    "primelhr": "db42280",  # Lahore (update if separate DB exists)
}

def get_engine(town="prime"):
    """Get SQLAlchemy engine for the specified town/database."""
    db_name = TOWN_TO_DB.get(town, "db42280")
    connection_string = f"mysql+pymysql://{DB_CONFIG['username']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{db_name}"
    return create_engine(
        connection_string,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=10,
        max_overflow=20,
    )

# Default engine (cloud database)
engine = get_engine("db42280")


@lru_cache(maxsize=128)
def _read_sql_cached_json(query, db_name="db42280"):
    eng = get_engine(db_name)
    return pd.read_sql(query, eng).to_json(orient='split', date_format='iso')


def read_sql_cached(query, use_cache=True, db_name="db42280"):
    if use_cache:
        return pd.read_json(StringIO(_read_sql_cached_json(query, db_name)), orient='split')
    eng = get_engine(db_name)
    return pd.read_sql(query, eng)

def fetch_booker_less_ctn_data(months_back=3, town="prime"):
    try:
        months_back = int(months_back)
    except (TypeError, ValueError):
        months_back = 3

    if months_back not in [1, 2, 3, 4]:
        months_back = 3

    db_name = TOWN_TO_DB.get(town, "db42280")
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
)
"""

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
    booker_less_ctn_df = read_sql_cached(booker_less_ctn_query, db_name=db_name)

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
    booker_less_ctn_detail_df = read_sql_cached(booker_less_ctn_detail_query, db_name=db_name)

    if booker_less_ctn_df.empty:
        return [{'name': 'Booker Name', 'id': 'Booker_Name'}], [], booker_less_ctn_detail_df.to_dict('records')

    pivoted_df = pd.DataFrame(booker_less_ctn_df).pivot_table(
        index='Booker_Name',
        columns='brand',
        values='age',
        aggfunc='sum',
        fill_value=0
    )
    pivoted_df.reset_index(inplace=True)

    columns = [{'name': 'Booker Name', 'id': 'Booker_Name'}]
    for column in pivoted_df.columns[1:]:
        columns.append({'name': column, 'id': column})

    for col in pivoted_df.columns:
        if col != 'Booker_Name':
            pivoted_df[col] = (pivoted_df[col] * 100).round(2).astype(str) + '%'

    return columns, pivoted_df.to_dict('records'), booker_less_ctn_detail_df.to_dict('records')


booker_columns, booker_lessthanctn_datatable, booker_lessthanctn_detail_records = fetch_booker_less_ctn_data(3)




# ======================
# 🚀 APP INIT
# ======================
app = dash.Dash(__name__)
auth = dash_auth.BasicAuth(app, auth_dict)
server = app.server

# ======================
# 🎨 PROFESSIONAL THEME
# ======================
COLORS = {
    "bg": "#f8fafc",
    "card": "#ffffff",
    "primary": "#0f172a",
    "secondary": "#6366f1",
    "accent": "#ec4899",
    "success": "#10b981",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "text": "#1e293b",
    "text-light": "#64748b",
    "border": "#e2e8f0",
    "theme": px.colors.qualitative.Set3,  # Using Plotly's built-in Set3 color scale
}

FONTSIZE = {
    "title": 28,
    "subtitle": 18,
    "card_title": 15,
    "label": 12,
    "fontname": "'Segoe UI', 'Helvetica Neue', sans-serif"
}

CARD = {
    "background": COLORS["card"],
    "borderRadius": "12px",
    "padding": "24px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.06)",
    "border": f"1px solid {COLORS['border']}",
}

KPI_CARD = {
    "borderRadius": "12px",
    "padding": "20px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.06)",
    "border": f"1px solid {COLORS['border']}",
}


# ======================
# 📐 LAYOUT
# ======================
app.layout = html.Div(style={"background": COLORS["bg"], "padding": "28px", "minHeight": "100vh", "fontFamily": FONTSIZE["fontname"]}, children=[

    dcc.Store(id='booker-lessthanctn-detail-store', data=booker_lessthanctn_detail_records),
    dcc.Store(id='selected-town-store', data='prime'),

    # Header Section
    html.Div(style={
        "marginBottom": "32px",
        "paddingBottom": "24px",
        "borderBottom": f"2px solid {COLORS['border']}",
    }, children=[
        html.H1("📊 Sales Dashboard", style={
            "color": COLORS["text"],
            "fontSize": FONTSIZE["title"],
            "fontWeight": "700",
            "margin": "0 0 8px 0",
            "letterSpacing": "-0.5px",
        }),
        html.P("Real-time sales, inventory & performance analytics", style={
            "color": COLORS["text-light"],
            "fontSize": "14px",
            "margin": "0",
            "fontWeight": "400",
        }),
    ]),

    # Filters Section
    html.Div(style={
        "background": COLORS["card"],
        "borderRadius": "12px",
        "padding": "20px",
        "marginBottom": "24px",
        "border": f"1px solid {COLORS['border']}",
        "display": "flex",
        "gap": "16px",
        "alignItems": "center",
        "flexWrap": "wrap",
    }, children=[
        html.Label("📅 Date Range:", style={"fontWeight": "600", "color": COLORS["text"], "fontSize": "14px"}),
        dcc.DatePickerRange(
            id="date-range",
            min_date_allowed=datetime(2026, 1, 1),
            max_date_allowed=datetime.today(),
            start_date=(datetime.today() - pd.Timedelta(days=30)),
            end_date=datetime.today(),
            start_date_placeholder_text="Start Date",
            end_date_placeholder_text="End Date",
            display_format="DD MMM YYYY",
            style={
                "background": "white",
                "borderRadius": "8px",
                "border": f"1px solid {COLORS['border']}",
                "padding": "8px",
                "fontFamily": FONTSIZE["fontname"],
            }
        ),

        html.Label("📍 Location:", style={"fontWeight": "600", "color": COLORS["text"], "fontSize": "14px", "marginLeft": "12px"}),
        dcc.Dropdown(
            id="town",
            options=[
                {"label": "All Locations", "value": "all"},
                {"label": "🏙️ Karachi", "value": "prime"},
                {"label": "🏢 Lahore", "value": "primelhr"},
            ],
            value="prime",
            clearable=False,
            style={
                "width": "200px",
                "borderRadius": "8px",
                "border": f"1px solid {COLORS['border']}",
                "fontFamily": FONTSIZE["fontname"],
            }
        )
    ]),

    # KPI Row with Loading
    html.Div(style={"marginBottom": "24px"}, children=[
        html.H3("📈 Key Performance Indicators", style={
            "fontSize": FONTSIZE["subtitle"],
            "fontWeight": "700",
            "color": COLORS["text"],
            "margin": "0 0 16px 0",
        }),
        dcc.Loading(
            type="dot",
            children=html.Div(id="kpi-row", style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
                "gap": "16px",
            })
        ),
    ]),

    # Sales Overview Section
    html.H3("📊 Sales Overview", style={"marginTop": "24px", "marginBottom": "16px", "color": COLORS["primary"], "fontSize": FONTSIZE["card_title"]}),
    dcc.Loading(
        type="dot",
        children=html.Div(style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "16px"}, children=[
            html.Div(style=CARD, children=[dcc.Graph(id="daily-sales")]),
            html.Div(style=CARD, children=[dcc.Graph(id="order-type-chart")]),
            html.Div(style=CARD, children=[dcc.Graph(id="sales-growth")])
        ])
    ),

    # Brand Performance Section
    html.H3("🏢 Brand Performance", style={"marginTop": "24px", "marginBottom": "16px", "color": COLORS["primary"], "fontSize": FONTSIZE["card_title"]}),
    dcc.Loading(
        type="dot",
        children=html.Div(style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "16px"}, children=[
            html.Div(style=CARD, children=[dcc.Graph(id="brand-compare")]),
            html.Div(style=CARD, children=[dcc.Graph(id="brand-growth-bar")]),
            html.Div(style=CARD,# Table: Top 10 SKUs by Sales
                children=[ html.H2("🏷️ Brands Wise NMV"),
                        DataTable(
                            id='sku-sales-table',
                            columns=[
                                {'name': 'Brand', 'id': 'brand'},
                                {'name': 'Current NMV', 'id': 'Current_Period_Sales'},
                                {'name': 'LastYear NMV', 'id': 'Last_Year_Sales'},
                                {'name': 'Current Ltr', 'id': 'Current_Period_Ltr'},
                                {'name': 'LastYear Ltr', 'id': 'Last_Year_Ltr'},
                                {'name': 'Ltr Comparsion LY', 'id': 'Ltr_Growth_LY'},
                                # {'name': 'Ltr Comparsion', 'id': 'sparkline'},
                                

                                
                            ],
                            style_table={'overflowX': 'auto', 'borderCollapse': 'collapse'},
                            style_cell={'textAlign': 'center', 'padding': '12px', 'border': f'1px solid {COLORS["border"]}'},
                            style_header={
                                'backgroundColor': COLORS["secondary"],
                                'color': 'white',
                                'fontWeight': 'bold',
                                'textAlign': 'center',
                                'fontSize': '13px',
                                'border': f'1px solid {COLORS["border"]}'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#f8fafc'
                                },
                                {
                                    'if': {
                                        'filter_query': '{Ltr_Growth_LY} contains "+"',
                                        'column_id': 'Ltr_Growth_LY'
                                    },
                                    'color': COLORS["success"],
                                    'fontWeight': 'bold'
                                },
                                {
                                    'if': {
                                        'filter_query': '{Ltr_Growth_LY} contains "-"',
                                        'column_id': 'Ltr_Growth_LY'
                                    },
                                    'color': COLORS["danger"],
                                    'fontWeight': 'bold'
                                },
                                 {
                                    'if': {
                                        'filter_query': '{brand} contains "Grand Total"',
                                        'column_id': ['brand','Last_Year_Sales','Current_Period_Sales','Ltr_Growth_LY','Current_Period_Ltr','Last_Year_Ltr']
                                    },
                                    'fontWeight': 'bold',
                                    'backgroundColor': '#f0f4ff'
                                }

                            ],

                            page_size=12,
                        )
                        ]),
        ])
    ),

    # Delivery & Operations Section
    html.H3("🚚 Delivery & Operations Analytics", style={"marginTop": "24px", "marginBottom": "16px", "color": COLORS["primary"], "fontSize": FONTSIZE["card_title"]}),
    dcc.Loading(
        type="dot",
        children=html.Div(style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "16px"}, children=[
            html.Div(style=CARD, children=[dcc.Graph(id="dm-compare")]),
            html.Div(style=CARD, children=[dcc.Graph(id="TgtvsAch-YTD")]),
            html.Div(style=CARD, children=[dcc.Graph(id="SKU-HeatMap")])
        ])
    ),

    # Channel & Productivity Section
    html.H3("📊 Channel & Productivity Analysis", style={"marginTop": "24px", "marginBottom": "16px", "color": COLORS["primary"], "fontSize": FONTSIZE["card_title"]}),
    dcc.Loading(
        type="dot",
        children=html.Div(style={"display": "grid", "gridTemplateColumns": "1.5fr 1fr", "gap": "16px"}, children=[
            html.Div(style=CARD, children=[
                html.Div(style={"display": "flex", "justifyContent": "flex-end", "marginBottom": "12px"}, children=[
                    dcc.Dropdown(
                        id="dm-sunburst-filter",
                        options=[],
                        multi=True,
                        placeholder="Filter deliveryman",
                        style={"width": "320px", "fontSize": FONTSIZE["label"]}
                    )
                ]),
                dcc.Graph(id="Channel-SunBrust")
            ]),
            html.Div(style=CARD, children=[dcc.Graph(id="Brand-Productivity-Chart")])
        ])
    ),

    # Target Achievement Section
    html.H3("🎯 Target vs Achievement", style={"marginTop": "24px", "marginBottom": "16px", "color": COLORS["primary"], "fontSize": FONTSIZE["card_title"]}),
    dcc.Loading(
        type="dot",
        children=html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
            html.Div(style=CARD, children=[dcc.Graph(id="TgtVsAch-Booker")]),
            html.Div(style=CARD, children=[dcc.Graph(id="Channel_NMV_YTD")])
        ])
        ),

    # Channel Treemap Section (6-Month Period-wise Breakdown)
    html.H3("🌳 Channel Performance - 6 Month Breakdown", style={"marginTop": "24px", "marginBottom": "16px", "color": COLORS["primary"], "fontSize": FONTSIZE["card_title"]}),
    dcc.Loading(
        type="dot",
        children=html.Div(style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "16px"}, children=[
            html.Div(style=CARD, children=[
                html.Div(style={"display": "flex", "justifyContent": "flex-end", "marginBottom": "12px"}, children=[
                    dcc.Dropdown(
                        id="channel-treemap-filter",
                        options=[],
                        multi=True,
                        placeholder="Filter channels",
                        style={"width": "320px", "fontSize": FONTSIZE["label"]}
                    )
                ]),
                dcc.Graph(id="channel-treemap")
            ])
        ])
    ),

    # Booker Less-Than-Half-Carton Analysis
    html.H3("🏷️ Booker Wise Less Than Half Carton Deliveries", style={"marginTop": "24px", "marginBottom": "16px", "color": COLORS["primary"], "fontSize": FONTSIZE["card_title"]}),
    html.Div(style=CARD, children=[
        html.Div(style={"display": "flex", "justifyContent": "flex-end", "alignItems": "center", "marginBottom": "16px"}, children=[
            dcc.Dropdown(
                id="booker-month-filter",
                options=[
                    {"label": "Last 1 Month", "value": 1},
                    {"label": "Last 2 Months", "value": 2},
                    {"label": "Last 3 Months", "value": 3},
                    {"label": "Last 4 Months", "value": 4},
                ],
                value=3,
                clearable=False,
                style={"width": "220px", "fontSize": FONTSIZE["label"]}
            )
        ]),
        DataTable(
            id='booker-lessthanctn-table',
            columns=booker_columns,  # Dynamically created columns
            data=booker_lessthanctn_datatable,  # Data passed dynamically
            style_table={'overflowX': 'auto', 'borderCollapse': 'collapse'},
            style_cell={'textAlign': 'center', 'padding': '12px', 'border': f'1px solid {COLORS["border"]}'},
            style_header={
                'backgroundColor': COLORS["secondary"],
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'fontSize': '13px',
                'border': f'1px solid {COLORS["border"]}'
            },
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8fafc'},
                {
                    'if': {
                        'filter_query': '{age} >= 0.5',
                        'column_id': 'age'
                    },
                    'color': COLORS["success"],
                    'fontWeight': 'bold'
                },
            {
                'if': {
                    'filter_query': '{age} < 0.5',
                    'column_id': 'age'
                },
                'color': COLORS["danger"],
                'fontWeight': 'bold'
            }
            ],
            page_size=5,
        )
    ]),

    html.Div(style=CARD, children=[
        html.H3("📋 Booker Less-Than-Half-Carton Subreport", id="booker-subreport-title", style={"marginBottom": "16px"}),
        DataTable(
            id='booker-lessthanctn-subreport-table',
            columns=[
                {'name': 'Brand', 'id': 'brand'},
                {'name': 'Store Code', 'id': 'StoreCode'},
                {'name': 'Store Name', 'id': 'StoreName'},
                {'name': 'Total Deliveries', 'id': 'Total_Deliveries'},
                {'name': 'Half-Carton Deliveries', 'id': 'HalfCtnDel'},
                {'name': 'Age', 'id': 'age'}
            ],
            data=[],
            style_table={'overflowX': 'auto', 'borderCollapse': 'collapse'},
            style_cell={'textAlign': 'center', 'padding': '12px', 'border': f'1px solid {COLORS["border"]}'},
            style_header={
                'backgroundColor': COLORS["secondary"],
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'fontSize': '13px',
                'border': f'1px solid {COLORS["border"]}'
            },
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8fafc'}
            ],
            page_size=10,
        )
    ]),

    # Footer
    html.Div(style={
        "marginTop": "40px",
        "paddingTop": "20px",
        "borderTop": f"1px solid {COLORS['border']}",
        "textAlign": "center",
        "color": COLORS["text-light"],
        "fontSize": "13px",
        "fontFamily": FONTSIZE.get("fontname", "Segoe UI")
    }, children=[
        html.P("© 2026 Bazaar Prime Analytics Dashboard | Last Updated: Real-time", style={"margin": 0})
    ])

])
        



# ======================
# 🔁 CALLBACK
# ======================
@app.callback(
    Output("kpi-row", "children"),
    Output("daily-sales", "figure"),
    Output("order-type-chart", "figure"),
    Output("sales-growth", "figure"),
    Output("brand-compare", "figure"),
    Output("brand-growth-bar", "figure"),
    Output("sku-sales-table", "data"),
    Output("dm-compare","figure"),
    Output("TgtvsAch-YTD","figure"),
    Output("SKU-HeatMap","figure"),
    Output("Channel-SunBrust","figure"),
    Output("dm-sunburst-filter", "options"),
    Output("Brand-Productivity-Chart", "figure"),
    Output("TgtVsAch-Booker","figure"),
    Output("Channel_NMV_YTD","figure"),
    Output("channel-treemap-filter", "options"),
    Output("channel-treemap", "figure"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("town", "value"),
    Input("dm-sunburst-filter", "value"),
    Input("channel-treemap-filter", "value"),
    # Input("growth-dropdown", "value")
    
)




def update_dashboard(start, end, town, selected_dms, selected_channels):

    if not start or not end:
        raise PreventUpdate
    
    # Map town value to database name
    db_name = TOWN_TO_DB.get(town, "db42280")
    
    # Handle 'All' value by adjusting query condition
    if town == 'all':
        town_condition = "IN ('Karachi', 'Lahore')"  # Check for both Karachi and Lahore
    else:
        town_condition = f"= '{town}'"  # Filter for specific town

    # ---------- FETCH DATA ----------
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
# WHERE sp.town {town_condition}

		
ORDER BY
    sp.Channel;
    """
    sales_df = read_sql_cached(query, db_name=db_name)
    brand_sale_query = f"""
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
# WHERE sp.town {town_condition}

		
ORDER BY
    sp.nmv DESC;

    """  
    brand_df = read_sql_cached(brand_sale_query, db_name=db_name)
    DeliveryMan_Sales_query=f"""
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
# where sp.town {town_condition}
		
ORDER BY
    sp.DeliveryMan;
    """
    DeliveryMan_Sales_df = read_sql_cached(DeliveryMan_Sales_query, db_name=db_name)
    TargetVsAch_Query="""
    with a as (SELECT 
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
 LEFT JOIN (SELECT month,year,sum(Target_In_Value) as Target_In_Value,sum(Target_In_Volume) as Target_In_Volume  from targets group by year,month) t on t.month= month(o.`Delivery Date`) and t.year=YEAR(o.`Delivery Date`)
 
 GROUP BY MONTH(`Delivery Date`),YEAR(`Delivery Date`),Town
 order by YEAR(`Delivery Date`),MONTH(`Delivery Date`))
 select * from a
#  where a.Town {town_condition}

"""
    TargetVsAch_df = read_sql_cached(TargetVsAch_Query, db_name=db_name)
    SkuWIseSales_query=f"""
    with a as (SELECT
    SUBSTRING_INDEX(s.Master_Sku, ' [', 1) AS SKU,  -- Get everything before the first '['
	Round(sum( `Delivered Amount` ),0) AS NMV ,
    Case when o.`Distributor Code`='D70002202' then 'Karachi'
				when o.`Distributor Code`='D70002246' then 'Lahore' else 'CBL' end Town
FROM
	ordervsdelivered o
	LEFT JOIN sku_master s ON s.Sku_Code = o.`SKU Code` 
	where `Delivery Date` BETWEEN '{start}' AND '{end}'
GROUP BY
	s.Master_Sku, Town
HAVING sum( `Delivered Amount` ) >0
order by sum(`Delivered Amount`) desc)
select * from a 
# where town {town_condition}

    """
    SKUWise_Sale_df = read_sql_cached(SkuWIseSales_query, db_name=db_name)
    Channel_DM_Sunbrust_Query=f"""
SELECT
    u.`Channel Type` as Channel,
		s.Brand,
		o.DM,
		count(DISTINCT o.`Store Code`) as StoreCount,
        town	
FROM
	(SELECT DISTINCT `Deliveryman Name` as DM ,`Store Code`,`SKU Code`,`Delivery Date`,Case when `Distributor Code`='D70002202' then 'Karachi'
				when `Distributor Code`='D70002246' then 'Lahore' else 'CBL' end Town from ordervsdelivered) o
	LEFT JOIN sku_master s ON s.Sku_Code = o.`SKU Code` 
	LEFT JOIN universe u on u.`Store Code`=o.`Store Code`
	where `Delivery Date` BETWEEN '{start}' AND '{end}'
    # and town {town_condition}
	GROUP BY u.`Channel Type`,s.Brand,o.dm,o.town
    """
    Channel_DM_Sunbrust_df = read_sql_cached(Channel_DM_Sunbrust_Query, db_name=db_name)
    dm_options = [{"label": d, "value": d} for d in sorted(Channel_DM_Sunbrust_df["DM"].dropna().unique())]
    if selected_dms:
        if isinstance(selected_dms, str):
            selected_dms = [selected_dms]
        Channel_DM_Sunbrust_df = Channel_DM_Sunbrust_df[Channel_DM_Sunbrust_df["DM"].isin(selected_dms)]
    TGTVsAch_Booker_query=f"""
     
with a as (SELECT concat(MONTH(`Delivery Date`),"-",YEAR(`Delivery Date`)) as period,
o.`Order Booker Name` as Booker,
round(t.Target_In_Value) as Target_Value,
round(sum(`Delivered Amount`+`Total Discount`)) as NMV,
Round((sum(`Delivered Amount`+`Total Discount`)/t.Target_In_Value)*100) as Value_Ach,
round(t.Target_In_Volume) as Target_Ltr,
round(sum(`Delivered (Litres)`+`Delivered (KG)`)) as Ltr,
Round((sum(`Delivered (Litres)`+`Delivered (KG)`)/t.Target_In_Volume)*100) as Ltr_Ach,
Case when o.`Distributor Code`='D70002202' then 'Karachi'
				when o.`Distributor Code`='D70002246' then 'Lahore' else 'CBL' end Town



 from ordervsdelivered o
 LEFT JOIN (SELECT month,year,sum(Target_In_Value) as Target_In_Value,sum(Target_In_Volume) as Target_In_Volume,order_booker_code  from targets group by year,month,order_booker_code) t on t.month= month(o.`Delivery Date`) and t.year=YEAR(o.`Delivery Date`) and t.order_booker_code=o.`Order Booker Code`
 
 
 
 GROUP BY MONTH(`Delivery Date`),YEAR(`Delivery Date`),o.`Order Booker Name`,town 
 order by YEAR(`Delivery Date`),MONTH(`Delivery Date`),o.`Order Booker Name`)
 select * from a 
# where town {town_condition}

    """
    TGTVsAch_Booker_df = read_sql_cached(TGTVsAch_Booker_query, db_name=db_name)

    Channel_NMV_YTD_query=f"""
with a as (SELECT
    CONCAT(YEAR(`Delivery Date`), '-', LPAD(MONTH(`Delivery Date`), 2, '0')) AS period,
    u.`Channel Type` AS Channel,
    ROUND(SUM(`Delivered Amount` + `Total Discount`)) AS NMV,
    ROUND(SUM(`Delivered (Litres)` + `Delivered (KG)`)) AS Ltr,
    Case when o.`Distributor Code`='D70002202' then 'Karachi'
				when o.`Distributor Code`='D70002246' then 'Lahore' else 'CBL' end Town
FROM 
    ordervsdelivered o
INNER JOIN 
    universe u ON u.`Store Code` = o.`Store Code`  -- Use INNER JOIN if no missing records

GROUP BY 
    period, u.`Channel Type`,town  -- Group by calculated period
ORDER BY 
    period, u.`Channel Type`)  -- Sorting by period and Channel Type
    select * from a 
    # where town {town_condition}
    
"""
    Channel_NMV_YTD_df = read_sql_cached(Channel_NMV_YTD_query, db_name=db_name)

    Channel_Treemap_query=f"""
SELECT
    CONCAT(YEAR(`Delivery Date`), '-', LPAD(MONTH(`Delivery Date`), 2, '0')) AS period,
    u.`Channel Type` AS Channel,
    ROUND(SUM(`Delivered Amount` + `Total Discount`)) AS NMV
FROM 
    ordervsdelivered o
INNER JOIN 
    universe u ON u.`Store Code` = o.`Store Code`
WHERE 
    `Delivery Date` >= DATE_SUB(CURDATE(), INTERVAL 7 MONTH)  -- Get data from 7 months ago
    AND (YEAR(`Delivery Date`) < YEAR(CURDATE()) OR (YEAR(`Delivery Date`) = YEAR(CURDATE()) AND MONTH(`Delivery Date`) < MONTH(CURDATE())))  -- Exclude current month
GROUP BY 
    period, u.`Channel Type`
ORDER BY 
    period, u.`Channel Type`
"""
    Channel_Treemap_df = read_sql_cached(Channel_Treemap_query, db_name=db_name)
    
    if brand_df.empty  or sales_df.empty or Channel_NMV_YTD_df.empty or TGTVsAch_Booker_df.empty or DeliveryMan_Sales_df.empty or TargetVsAch_df.empty or SKUWise_Sale_df.empty or Channel_DM_Sunbrust_df.empty or Channel_Treemap_df.empty:
        raise PreventUpdate

    # ---------- KPIs ----------
    current_sales = sales_df["Current_Period_Sales"].sum()
    total_orders = sales_df.shape[0]  # Number of channels
    aov = current_sales / total_orders if total_orders else 0

    # Last same period comparison
    last_sales = sales_df["Last_Year_Sales"].sum()
    last_month_sales = sales_df["Last_Month_Sales"].sum()

    growth_pct = ((current_sales - last_sales) / last_sales) * 100 if last_sales > 0 else 0
    growth_pct1 = ((current_sales - last_month_sales) / last_month_sales) * 100 if last_month_sales > 0 else 0

    arrow = "▲" if growth_pct >= 0 else "▼"
    arrow_color = COLORS["success"] if growth_pct >= 0 else COLORS["danger"]
    arrow1 = "▲" if growth_pct1 >= 0 else "▼"
    arrow_color1 = COLORS["success"] if growth_pct1 >= 0 else COLORS["danger"]

    # ------------ltr calculation----
    current_ltr = sales_df["Current_Period_Ltr"].sum()
    last_ltr = sales_df["Last_Year_Ltr"].sum()
    last_month_ltr = sales_df["Last_Month_Ltr"].sum()
    # --------------------------------

    growth_pct_ltr = ((current_ltr - last_ltr) / last_ltr) * 100 if last_ltr > 0 else 0
    growth_pct1_ltr = ((current_ltr - last_month_ltr) / last_month_ltr) * 100 if last_month_ltr > 0 else 0
    # --------------------------------
    arrow_ltr = "▲" if growth_pct_ltr >= 0 else "▼"
    arrow_color_ltr = COLORS["success"] if growth_pct_ltr >= 0 else COLORS["danger"]
    arrow1_ltr = "▲" if growth_pct1_ltr >= 0 else "▼"
    arrow_color1_ltr = COLORS["success"] if growth_pct1_ltr >= 0 else COLORS["danger"]

    # start_date = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%f").date()  # Adjust format to handle milliseconds
    # end_date = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S.%f").date()    # Adjust format to handle milliseconds

    # order calculation for KPI
    current_orders = sales_df["Current_Orders"].sum()
    last_year_orders = sales_df["Last_Year_Orders"].sum()
    last_month_orders = sales_df["Last_Month_Orders"].sum()
    growth_pct_orders_ly = ((current_orders - last_year_orders) / last_year_orders) * 100 if last_year_orders > 0 else 0
    growth_pct_orders_lm = ((current_orders - last_month_orders) / last_month_orders) * 100 if last_month_orders > 0 else 0
    arrow_orders_ly = "▲" if growth_pct_orders_ly >= 0 else "▼"
    arrow_color_orders_ly = COLORS["success"] if growth_pct_orders_ly >= 0 else COLORS["danger"]
    arrow_orders_lm = "▲" if growth_pct_orders_lm >= 0 else "▼"
    arrow_color_orders_lm = COLORS["success"] if growth_pct_orders_lm >= 0 else COLORS["danger"]

    #calculate averge order value for current period, last year and last month
    aov_current = current_sales / current_orders if current_orders else 0
    aov_last_year = last_sales / last_year_orders if last_year_orders else 0
    aov_last_month = last_month_sales / last_month_orders if last_month_orders else 0
    growth_pct_aov_ly = ((aov_current - aov_last_year) / aov_last_year) * 100 if aov_last_year > 0 else 0
    growth_pct_aov_lm = ((aov_current - aov_last_month) / aov_last_month) * 100 if aov_last_month > 0 else 0
    arrow_aov_ly = "▲" if growth_pct_aov_ly >= 0 else "▼"
    arrow_color_aov_ly = COLORS["success"] if growth_pct_aov_ly >= 0 else COLORS["danger"]
    arrow_aov_lm = "▲" if growth_pct_aov_lm >= 0 else "▼"
    arrow_color_aov_lm = COLORS["success"] if growth_pct_aov_lm >= 0 else COLORS["danger"]

    
    # ------------------- Fetch Brand Wise Sales Data----------------
    df_brand_sales = brand_df[['brand', 'Current_Period_Sales', 'Last_Year_Sales','Current_Period_Ltr','Last_Year_Ltr','Ltr_Growth_LY']].sort_values(by='Current_Period_Sales', ascending=False).head(10)

    # Ensure the sales columns are numeric (if they are not already)
    df_brand_sales['Current_Period_Sales'] = pd.to_numeric(df_brand_sales['Current_Period_Sales'], errors='coerce')
    df_brand_sales['Last_Year_Sales'] = pd.to_numeric(df_brand_sales['Last_Year_Sales'], errors='coerce')
    df_brand_sales['Current_Period_Ltr'] = pd.to_numeric(df_brand_sales['Current_Period_Ltr'], errors='coerce')
    df_brand_sales['Last_Year_Ltr'] = pd.to_numeric(df_brand_sales['Last_Year_Ltr'], errors='coerce')
    df_brand_sales['Ltr_Growth_LY'] = pd.to_numeric(df_brand_sales['Ltr_Growth_LY'], errors='coerce')


    # Calculate Percentage Change between Current Period Sales and Last Year Sales
    df_brand_sales['Percentage_Change'] = ((df_brand_sales['Current_Period_Sales'] / df_brand_sales['Last_Year_Sales']) - 1) 

    # Calculate Grand Totals
    grand_total_sales = {
        'Current_Period_Sales': df_brand_sales['Current_Period_Sales'].sum(),
        'Last_Year_Sales': df_brand_sales['Last_Year_Sales'].sum(),
        'Current_Period_Ltr': df_brand_sales['Current_Period_Ltr'].sum(),
        'Last_Year_Ltr': df_brand_sales['Last_Year_Ltr'].sum(),
        'Ltr_Growth_LY': df_brand_sales['Ltr_Growth_LY'].mean()  # Average growth for the grand total row
    }


    # Add Grand Total Row
    grand_total_row = pd.DataFrame({
        'brand': ['Grand Total'],
        'Current_Period_Sales': [f" {grand_total_sales['Current_Period_Sales'] / 1_000_000:,.2f} M" if grand_total_sales['Current_Period_Sales'] >= 1_000_000 else f" {grand_total_sales['Current_Period_Sales'] / 1_000:,.2f} K"],
        'Last_Year_Sales': [f" {grand_total_sales['Last_Year_Sales'] / 1_000_000:,.2f} M" if grand_total_sales['Last_Year_Sales'] >= 1_000_000 else f" {grand_total_sales['Last_Year_Sales'] / 1_000:,.2f} K"],
        'Current_Period_Ltr': [f" {grand_total_sales['Current_Period_Ltr'] / 1_000_000:,.2f} M" if grand_total_sales['Current_Period_Ltr'] >= 1_000_000 else f" {grand_total_sales['Current_Period_Ltr'] / 1_000:,.2f} K"],
        'Last_Year_Ltr': [f" {grand_total_sales['Last_Year_Ltr'] / 1_000_000:,.2f} M" if grand_total_sales['Last_Year_Ltr'] >= 1_000_000 else f" {grand_total_sales['Last_Year_Ltr'] / 1_000:,.2f} K"],
        'Ltr_Growth_LY': [f"{grand_total_sales['Ltr_Growth_LY']:.2f}%"],
    })

     # Format Current and Last Y    ear Sales in Millions (M) or Thousands (K)
    df_brand_sales['Current_Period_Sales'] = df_brand_sales['Current_Period_Sales'].apply(lambda x: f" {x / 1_000_000:,.2f} M" if x >= 1_000_000 else f" {x / 1_000:,.2f} K")
    df_brand_sales['Last_Year_Sales'] = df_brand_sales['Last_Year_Sales'].apply(lambda x: f" {x / 1_000_000:,.2f} M" if x >= 1_000_000 else f" {x / 1_000:,.2f} K")
    df_brand_sales['Current_Period_Ltr'] = df_brand_sales['Current_Period_Ltr'].apply(lambda x: f" {x / 1_000_000:,.2f} M" if x >= 1_000_000 else f" {x / 1_000:,.2f} K")
    df_brand_sales['Last_Year_Ltr'] = df_brand_sales['Last_Year_Ltr'].apply(lambda x: f" {x / 1_000_000:,.2f} M" if x >= 1_000_000 else f" {x / 1_000:,.2f} K")
    
    
     # Add '+' sign for positive growth and format the Percentage_Change column to show 2 decimals
    df_brand_sales['Ltr_Growth_LY'] = df_brand_sales['Ltr_Growth_LY'].apply(lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")
     # Sort the DataFrame first based on 'Ltr_Growth_LY' column
    df_sorted = df_brand_sales.sort_values(by='Ltr_Growth_LY', ascending=False).head(10)

    # Append the Grand Total Row to the DataFrame
    df_brand_sales = pd.concat([df_sorted, grand_total_row], ignore_index=True)
    
    # Format the SKU data to be passed to the DataTable
    brand_datatable = df_brand_sales.to_dict('records')
    # -----------------------------------------------
    # Monthly Sales Comparison--Grwoth/DeGrowth-----
    
    df_Growth_Degrowth=sales_df[['Channel', 'Current_Period_Sales', 'Last_Year_Sales','Current_Period_Ltr','Last_Year_Ltr','Last_Month_Sales','Last_Month_Ltr']]
    df_Growth_Degrowth['Sales_Growth_%'] = ((df_Growth_Degrowth['Current_Period_Sales']/df_Growth_Degrowth['Last_Year_Sales']) - 1) * 100
    df_Growth_Degrowth['Ltr_Growth_%'] = ((df_Growth_Degrowth['Current_Period_Ltr'] / df_Growth_Degrowth['Last_Year_Ltr']) - 1) * 100 
    df_Growth_Degrowth['LM_Ltr_Growth_%'] = ((df_Growth_Degrowth['Current_Period_Ltr'] / df_Growth_Degrowth['Last_Month_Ltr']) - 1) * 100 
    df_Growth_Degrowth['LM_Sales_Growth_%'] = ((df_Growth_Degrowth['Current_Period_Sales'] / df_Growth_Degrowth['Last_Month_Sales']) - 1) * 100 
    
    # Format the SKU data to be passed to the DataTable
    growth_datatable = df_Growth_Degrowth.to_dict('records')



    kpis = [
        html.Div(style={**CARD, 'background': COLORS["theme"][0]}, children=[
            html.H3("💰 Current Period Sales"),
            html.H2(f"Rs {current_sales / 1_000_000:,.2f} M "),
            html.P([f"vs Last Year {last_sales/ 1_000_000:,.2f} M",html.Span(f"{arrow} {growth_pct:.2f}%", style={"color": arrow_color, "fontSize": "16px"})]),
            html.P([f"vs Last Month {last_month_sales/ 1_000_000:,.2f} M", html.Span(f"{arrow1} {growth_pct1:.2f}%", style={"color": arrow_color1, "fontSize": "13px"})])

        ]),
        html.Div(style={**CARD, 'background': COLORS["theme"][1]}, children=[
            html.H3("🛢 Current Period Litres"),
            html.H2(f"{current_ltr:,.0f} Ltr "),
            html.P([f"vs Last Year {last_ltr:,.0f} Ltr",html.Span(f"{arrow_ltr} {growth_pct_ltr:.2f}%", style={"color": arrow_color_ltr, "fontSize": "16px"})]),
            html.P([f"vs Last Month {last_month_ltr:,.0f} Ltr", html.Span(f"{arrow1_ltr} {growth_pct1_ltr:.2f}%", style={"color": arrow_color1_ltr, "fontSize": "13px"})])

        ]),

        html.Div(style={**CARD, 'background': COLORS["theme"][2]},
                 children=[html.H3("🧾 Orders"), html.H2(f"{current_orders:,.0f}"),
                           html.P([f"vs Last Year {last_year_orders:,.0f}",
                                   html.Span(f"{arrow_orders_ly} {growth_pct_orders_ly:.2f}%",
                                             style={"color": arrow_color_orders_ly, "fontSize": "16px"})]),
                           html.P([f"vs Last Month {last_month_orders:,.0f}",
                                   html.Span(f"{arrow_orders_lm} {growth_pct_orders_lm:.2f}%",
                                             style={"color": arrow_color_orders_lm, "fontSize": "13px"})])]),
        html.Div(style={**CARD, 'background': COLORS["theme"][3]}, children=[html.H3("📦 Avg Order Value"), html.H2(f"Rs {aov_current:,.0f}"), html.P([f"vs Last Year Rs {aov_last_year:,.0f}", html.Span(f"{arrow_aov_ly} {growth_pct_aov_ly:.2f}%", style={"color": arrow_color_aov_ly, "fontSize": "16px"})]), html.P([f"vs Last Month Rs {aov_last_month:,.0f}", html.Span(f"{arrow_aov_lm} {growth_pct_aov_lm:.2f}%", style={"color": arrow_color_aov_lm, "fontSize": "13px"})])])
    ]
 
    # ---------- Channel Wise Sale Type ----------
    fig_daily=go.Figure()
    fig_daily.add_trace(go.Scatter(x=sales_df["Channel"], y=sales_df["Current_Period_Sales"]/ 1_000_000, mode="lines+markers+text"
                                   ,name="Current Period Sales",fill='tozeroy',line=dict(color=COLORS["theme"][0], width=3)
                                   ,marker=dict(size=7), hovertemplate=': Rs %{y:,.2f} M'))
    fig_daily.add_trace(go.Scatter  (x=sales_df["Channel"], y=sales_df["Last_Month_Sales"]/ 1_000_000, mode="lines+markers"
                                    ,name="Last Month Sales", hovertemplate=': Rs %{y:,.2f} M',line=dict(color=COLORS["theme"][6], width=3)))
    fig_daily.add_trace(go.Scatter(x=sales_df["Channel"], y=sales_df["Last_Year_Sales"]/ 1_000_000, mode="lines+markers"
                                   ,name="Last Year Sales", hovertemplate=': Rs %{y:,.2f} M',line=dict(color=COLORS["theme"][2], width=3)))
    
    fig_daily.update_layout(template="plotly_white", yaxis_title="Sales (in Million Rs)", hovermode="x unified"
                            ,legend=dict(
        orientation="h",  # Horizontal layout
        yanchor="bottom",  # Align the legend to the bottom
        y=-0.2,  # Adjust this to control vertical spacing
        xanchor="center",  # Center the legend
        x=0.5  # Position at the center horizontally
        ),
        title={
        'font': {'size': FONTSIZE["title"]},
        'text': 'Channel Wise Sales Comparison (Last Month/Last Year) vs Current Period',  # Main title
        'x': 0.5,  # Center the title horizontally
        'xanchor': 'center',  # Align title at the center
        'yanchor': 'top',  # Position title at the top
    }
    )  

    # ---------- sale Type Chart (Pie Chart) ----------
    fig_type = px.pie( sales_df, names="Channel", values=sales_df["Current_Period_Sales"]/ 1_000_000, hole=0.6, template="plotly_white"
                      ,color_discrete_sequence=COLORS["theme"])
    fig_type.update_traces(textposition='outside', textinfo='percent+label', hovertemplate='%{label}: Rs %{value:,.0f} M <extra></extra>')
    fig_type.update_layout(showlegend=False, legend=dict(
        orientation="h",  # Horizontal layout
        yanchor="bottom",  # Align the legend to the bottom
        y=-0.2,  # Adjust this to control vertical spacing
        xanchor="center",  # Center the legend
        x=0.5  # Position at the center horizontally
        ),
        title={
        'font': {'size': FONTSIZE["title"]},
        'text': 'Channel Wise Sales',  # Main title
        'x': 0.5,  # Center the title horizontally
        'xanchor': 'center',  # Align title at the center
        'yanchor': 'top',  # Position title at the top
    }

    )
    
     # -----------Channel Wise Growth/DeGrowth Sale---------
    
    # Bar chart for Sales Growth and Ltr Growth
    growth_datatable_df = pd.DataFrame(growth_datatable)
    growth_datatable_df["Sales_Growth_%"] = growth_datatable_df["Sales_Growth_%"].round(0)
    growth_datatable_df["Ltr_Growth_%"] = growth_datatable_df["Ltr_Growth_%"].round(0)
    growth_datatable_df["LM_Ltr_Growth_%"] = growth_datatable_df["LM_Ltr_Growth_%"].round(0)
    growth_datatable_df["LM_Sales_Growth_%"] = growth_datatable_df["LM_Sales_Growth_%"].round(0)

    growth_datatable = growth_datatable_df.to_dict('records')
    
    growth_datatable_df["Sales_Growth_%_color"] = growth_datatable_df["Sales_Growth_%"].apply(lambda x: "pink" if x < 0 else COLORS["theme"][3])
    growth_datatable_df["Ltr_Growth_%_color"] = growth_datatable_df["Ltr_Growth_%"].apply(lambda x: "pink" if x < 0 else COLORS["theme"][3])
    growth_datatable_df["LM_Sales_Growth_%_color"] = growth_datatable_df["LM_Sales_Growth_%"].apply(lambda x: "pink" if x < 0 else COLORS["theme"][6])
    growth_datatable_df["LM_Ltr_Growth_%_color"] = growth_datatable_df["LM_Ltr_Growth_%"].apply(lambda x: "pink" if x < 0 else COLORS["theme"][6])

    fig_sales_growth = go.Figure()
    # Adding Sales Growth % trace with dynamic color based on value
    fig_sales_growth.add_trace(go.Bar(
            x=growth_datatable_df["Channel"], 
            y=growth_datatable_df["Sales_Growth_%"], 
            name="LY Sales Growth %",
            marker_color=growth_datatable_df["Sales_Growth_%_color"],
            text=growth_datatable_df["Sales_Growth_%"].astype(str) + "%",  # Add text for data labels
            textposition="inside",  # Position text outside the bars
            hovertemplate=' %{y:.2f}%'


        ))

        # Adding Ltr Growth % trace with dynamic color based on value
    fig_sales_growth.add_trace(go.Bar(
            x=growth_datatable_df["Channel"], 
            y=growth_datatable_df["LM_Sales_Growth_%"], 
            name="LM Sales Growth %",
            marker_color=growth_datatable_df["LM_Sales_Growth_%_color"],
            text=growth_datatable_df["LM_Sales_Growth_%"].astype(str) + "%",  # Add text for data labels
            textposition="inside",  # Position text outside the bars
            hovertemplate='%{y:.2f}%'
        ))
    fig_sales_growth.update_layout(
        title={
            'font': {'size': FONTSIZE["title"]},
            'text': 'Channel Wise Sales Growth/De-Growth Comparison',  # Main title
            'x': 0.5,  # Center the title horizontally
            'xanchor': 'center',  # Align title at the center
            'yanchor': 'top',  # Position title at the top
        },
    barmode='group', 
    template="plotly_white",
    hovermode="x unified",
    legend=dict(
        orientation="h",  # Horizontal layout
        yanchor="bottom",  # Align the legend to the bottom
        y=-0.2,  # Adjust this to control vertical spacing
        xanchor="center",  # Center the legend
        x=0.5  # Position at the center horizontally
    )
    )
    
    # ---------- Brand-wise Liters Comparison Chart ----------
    fig_Brand_Comparsion = go.Figure()
    fig_Brand_Comparsion.add_trace(go.Scatter(x=brand_df["brand"],
                                               y=brand_df["Current_Period_Ltr"].apply(lambda x: x/1000), mode="lines+markers+text"
                                               ,text=brand_df["Current_Period_Ltr"].apply(lambda x: f"{x/1000:,.0f} T"), textposition="top center",name="Current Ltr",fill='tozeroy',line=dict(color=COLORS["theme"][0], width=3),
                                                        marker=dict(size=7), hovertemplate=' : %{y:.2f}T'))
    fig_Brand_Comparsion.add_trace(go.Scatter(x=brand_df["brand"], y=brand_df["Last_Month_Ltr"].apply(lambda x: x/1000), textposition="top center",
                                              text=brand_df["Last_Month_Ltr"].apply(lambda x: f"{x/1000:,.0f}T"), 
                                              mode="lines+markers", name="Last Month Ltr",line=dict(color=COLORS["theme"][6], width=3),
                                              hovertemplate=' : %{y:.2f}T'))
    fig_Brand_Comparsion.add_trace(go.Scatter(x=brand_df["brand"], y=brand_df["Last_Year_Ltr"].apply(lambda x: x/1000),text=brand_df["Last_Year_Ltr"]
                                              .apply(lambda x: f"{x/1000:,.0f} T"), mode="lines+markers", textposition="top center"
                                              , name="Last Year Ltr", hovertemplate=' : %{y:.2f}T',line=dict(color=COLORS["theme"][3], width=3)))
    
    fig_Brand_Comparsion.update_layout(
    title={
        'font': {'size': FONTSIZE["title"]},
        'text': 'Brand-wise Liters Comparison',  # Main title
        'x': 0.5,  # Center the title horizontally
        'xanchor': 'center',  # Align title at the center
        'yanchor': 'top',  # Position title at the top
    },
    title_font=dict(size=24),  # Title font size
    hovermode="x unified",
    legend=dict(
        orientation="h",  # Horizontal layout
        yanchor="bottom",  # Align the legend to the bottom
        y=-0.2,  # Adjust this to control vertical spacing
        xanchor="center",  # Center the legend
        x=0.5  # Position at the center horizontally
    ),
    annotations=[  # Add description below the main title
        {
            'font': {'size': FONTSIZE["subtitle"]},
            'text': 'Current, Last Year, and Last Month liters Comparison across brands',  # Description text
            'x': 0.5,  # Center the description horizontally
            'xref': 'paper',  # Align description with paper coordinates
            'y': 1.05,  # Position description slightly above the title
            'yref': 'paper',
            'showarrow': False,  # No arrow needed
            'font': {
                'size': 14,  # Description font size
                'color': 'gray'  # Description color
            }
        }
    ],template="plotly_white")


# ------------------------brand wise comparsion chart--------
    fig_Brand_Comparison_Bar = go.Figure()
    brand_growth_datatable_df = pd.DataFrame(brand_df)
    brand_growth_datatable_df["Ltr_Growth_LY"] = brand_growth_datatable_df["Ltr_Growth_LY"].round(0)
    brand_growth_datatable_df["Ltr_Growth_LM"] = brand_growth_datatable_df["Ltr_Growth_LM"].round(0)
    brand_growth_datatable_df["Ltr_Growth_LY_color"] = brand_growth_datatable_df["Ltr_Growth_LY"].apply(lambda x: "pink" if x < 0 else COLORS["theme"][3] )
    brand_growth_datatable_df["Ltr_Growth_LM_color"] = brand_growth_datatable_df["Ltr_Growth_LM"].apply(lambda x: "pink" if x < 0 else COLORS["theme"][6])
    
    fig_Brand_Comparison_Bar.add_trace(go.Bar(
            x=brand_growth_datatable_df["brand"], 
            y=brand_growth_datatable_df["Ltr_Growth_LY"], 
            name="LY Ltr Growth %",
            marker_color=brand_growth_datatable_df["Ltr_Growth_LY_color"],
            # marker=dict(color=COLORS["theme"][3]),
            text=brand_growth_datatable_df["Ltr_Growth_LY"].astype(str) + "%",  # Add text for data labels
            textposition="inside",  # Position text outside the bars
            hovertemplate='%{y:.0f}%',
            # textfont=dict(color=brand_growth_datatable_df["Ltr_Growth_LY_color"])  #Change font color based on growth or de-growth
        ))

        # Adding Ltr Growth % trace with dynamic color based on value
    fig_Brand_Comparison_Bar.add_trace(go.Bar(
            x=brand_growth_datatable_df["brand"], 
            y=brand_growth_datatable_df["Ltr_Growth_LM"], 
            name="LM Ltr Growth %",
            marker_color=brand_growth_datatable_df["Ltr_Growth_LM_color"],
            # marker=dict(color=COLORS["theme"][6]),
            text=brand_growth_datatable_df["Ltr_Growth_LM"].astype(str) + "%",  # Add text for data labels
            textposition="inside",  # Position text outside the bars
            hovertemplate='%{y:.0f}%',
            # textfont=dict(color=brand_growth_datatable_df["Ltr_Growth_LM_color"])
        ))
    fig_Brand_Comparison_Bar.update_layout(
    barmode='group', 
    template="plotly_white",
    hovermode="x unified",
    legend=dict(
        orientation="h",  # Horizontal layout
        yanchor="bottom",  # Align the legend to the bottom
        y=-0.2,  # Adjust this to control vertical spacing
        xanchor="center",  # Center the legend
        x=0.5  # Position at the center horizontally
    ),
    title={
        'font': {'size': FONTSIZE["title"]},
        'text': '🏷️ Brand Wise Sales Growth/De-Growth (Ltr)',  # Main title
        'x': 0.5,  # Center the title horizontally
        'xanchor': 'center',  # Align title at the center
        'yanchor': 'top',  # Position title at the top
    },
    # annotations=[  # Add description below the main title
    #     {
    #         'font': {'size': FONTSIZE["subtitle"], 'family': FONTSIZE["Fontname"]},
    #         'text': 'Last Year, and Last Month liters Comparison across Brands',  # Description text
    #         'x': 0.5,  # Center the description horizontally
    #         'xref': 'paper',  # Align description with paper coordinates
    #         'y': 1.05,  # Position description slightly above the title
    #         'yref': 'paper',
    #         'showarrow': False,  # No arrow needed
    #     }
    # ]   
)
    
    # ------------------------deliveryman wise comparsion chart--------
    fig_DM_Comparison_Bar = go.Figure()
    dm_growth_datatable_df = pd.DataFrame(DeliveryMan_Sales_df)
    dm_growth_datatable_df["Ltr_Growth_LY"] = dm_growth_datatable_df["Ltr_Growth_LY"].round(0)
    dm_growth_datatable_df["Ltr_Growth_LM"] = dm_growth_datatable_df["Ltr_Growth_LM"].round(0)
    dm_growth_datatable_df["Ltr_Growth_LY_color"] = dm_growth_datatable_df["Ltr_Growth_LY"].apply(lambda x: "pink" if x < 0 else COLORS["theme"][3])
    dm_growth_datatable_df["Ltr_Growth_LM_color"] = dm_growth_datatable_df["Ltr_Growth_LM"].apply(lambda x: "pink" if x < 0 else COLORS["theme"][6])

    fig_DM_Comparison_Bar.add_trace(go.Bar(
        x=dm_growth_datatable_df["DeliveryMan"], 
        y=dm_growth_datatable_df["Ltr_Growth_LY"], 
        name="LY Ltr Growth %",
        marker_color=dm_growth_datatable_df["Ltr_Growth_LY_color"],  # Correct the reference here
        text=dm_growth_datatable_df["Ltr_Growth_LY"].astype(str) + "%",
        #  "<br>" +  # This creates a line break between data labels
        #  (dm_growth_datatable_df["Last_Year_Ltr"]/1000).astype(str) + "T", # Add text for data labels
        textposition="inside",  # Position text inside the bars
        hovertemplate='%{y:.0f}%',  # Format the hovertext
    ))
        # Adding Ltr Growth % trace with dynamic color based on value
    fig_DM_Comparison_Bar.add_trace(go.Bar(
            x=dm_growth_datatable_df["DeliveryMan"], 
            y=dm_growth_datatable_df["Ltr_Growth_LM"], 
            name="LM Ltr Growth %",
            marker_color=dm_growth_datatable_df["Ltr_Growth_LM_color"],
            # marker=dict(color=COLORS["theme"][6]),
            text=dm_growth_datatable_df["Ltr_Growth_LM"].astype(str) + "%",
            # + "<br>" + (dm_growth_datatable_df["Last_Month_Ltr"]/1000).round(0).astype(str) + "T",  # Add text for data labels
            textposition="inside",  # Position text outside the bars
            hovertemplate='%{y:.0f}%',
            # textfont=dict(color=brand_growth_datatable_df["Ltr_Growth_LM_color"])
        ))
    fig_DM_Comparison_Bar.update_layout(
    barmode='group', 
    template="plotly_white",
    hovermode="x unified",
    legend=dict(
        orientation="h",  # Horizontal layout
        yanchor="top",  # Align the legend to the bottom
        y=-0.2,  # Adjust this to control vertical spacing
        xanchor="center",  # Center the legend
        x=0.5  # Position at the center horizontally
    ),
    title={
        'font': {'size': FONTSIZE["title"]},
        'text': 'Deliveryman-wise Liters Growth Comparison',  # Main title
        'x': 0.5,  # Center the title horizontally
        'xanchor': 'center',  # Align title at the center
        'yanchor': 'top',  # Position title at the top
    },
    annotations=[  # Add description below the main title
        {
            'font': {'size': FONTSIZE["subtitle"]},
            'text': 'Last Year, and Last Month liters Comparison across DM',  # Description text
            'x': 0.5,  # Center the description horizontally
            'xref': 'paper',  # Align description with paper coordinates
            'y': 1.05,  # Position description slightly above the title
            'yref': 'paper',
            'showarrow': False,  # No arrow needed
            'font': {
                'size': 14,  # Description font size
                'color': 'gray'  # Description color
            }
        }
    ]
)

    # -------------------TargetVsAch Bar Chart
    fig_TgtvsAch = go.Figure()
    TgtVsAch_dt_df = pd.DataFrame(TargetVsAch_df)
    fig_TgtvsAch.add_trace(go.Bar(
        x=TgtVsAch_dt_df["period"], 
        y=TgtVsAch_dt_df["Target_Value"].apply(lambda x: x/1000000), 
        name="Target In Value",
        marker=dict(color=COLORS["theme"][2],cornerradius=30),
        text=TgtVsAch_dt_df["Target_Value"].apply(lambda x: x/1000000).round(0).astype(str) + "M",
        textposition="inside",  # Position text inside the bars
        # hovertemplate='%{y:.0f}M',  # Format the hovertext
    ))
        # Adding Ltr Growth % trace with dynamic color based on value
    fig_TgtvsAch.add_trace(go.Bar(
            x=TgtVsAch_dt_df["period"], 
            y=TgtVsAch_dt_df["NMV"].apply(lambda x: x/1000000), 
            name="Achivement",
            # marker_color=TgtVsAch_dt_df["Ltr_Growth_LM_color"],
            marker=dict(color=COLORS["theme"][6],cornerradius=30),
            text=TgtVsAch_dt_df["NMV"].apply(lambda x: x/1000000).round(0).astype(str) + "M | "  + TgtVsAch_dt_df["Value_Ach"].astype(str) +"%",
            textposition="inside",  # Position text outside the bars
            # hovertemplate='%{y:.0f}M',
            # textfont=dict(color=brand_growth_datatable_df["Ltr_Growth_LM_color"])
        ))
    fig_TgtvsAch.update_layout(
    barmode='group', 
    template="plotly_white",
    legend=dict(
        orientation="h",  # Horizontal layout
        yanchor="top",  # Align the legend to the bottom
        y=-0.2,  # Adjust this to control vertical spacing
        xanchor="center",  # Center the legend
        x=0.5  # Position at the center horizontally
    ),
    title={
        'font': {'size': FONTSIZE["title"]},
        'text': 'Target Vs Achivement YTD',  # Main title
        'x': 0.5,  # Center the title horizontally
        'xanchor': 'center',  # Align title at the center
        'yanchor': 'top',  # Position title at the top
    },
    
)

    #---------------------SKU HeatMap
    # Convert NMV to millions
    SKUWise_Sale_df["NMV_in_millions"] = SKUWise_Sale_df["NMV"] / 1000000

    # Reshape the data into a grid (for demonstration, it will be a single row)
    matrix_data = np.array(SKUWise_Sale_df["NMV_in_millions"]).reshape(1, -1)

    # Create a heatmap using Plotly
    fig_SKU_HeatMap = px.imshow(matrix_data, 
                labels=dict(x="SKU", y="NMV", color="Amount (in Millions)"),
                x=SKUWise_Sale_df["SKU"],  # Use SKU as the x-axis labels
                color_continuous_scale=COLORS["theme"],  # Choose a color scale
                title="Heatmap of NMV")
    fig_SKU_HeatMap.update_layout(
        title={
            'font': {'size': FONTSIZE["title"]},
            'text': 'SKU Wise NMV Heatmap',  # Main title
            'x': 0.5,  # Center the title horizontally
            'xanchor': 'center',  # Align title at the center
            'yanchor': 'top',  # Position title at the top
        },
        xaxis_title="SKU",
        yaxis_title="",
        yaxis=dict(showticklabels=False),  # Hide y-axis tick labels since it's just a single row
    )
    fig_SKU_HeatMap.update_traces(hovertemplate='SKU: %{x}<br>NMV: %{z:.2f}M')  # Customize hover text
    
    #---------------------Channel, Brand, DM Sunburst Chart
    fig_Channel_SunBrust = px.sunburst(Channel_DM_Sunbrust_df, path=['Brand', 'DM'], values='StoreCount'
                                       , title="Brand, and DM Sunburst (Unique Productivity)", template="plotly_white",color='StoreCount'
                                       ,color_continuous_scale=COLORS["theme"])
    fig_Channel_SunBrust.update_layout(
        title={
            'font': {'size': FONTSIZE["title"]},
            'text': 'Brand, and Deliveryman Hierarchical (Unique Productivity)',  # Main title
            'x': 0.5,  # Center the title horizontally
            'xanchor': 'center',  # Align title at the center
            'yanchor': 'top',  # Position title at the top
        },width=700
        ,height=600
    )

    #summarize the total store count for each brand
    brand_productivity = Channel_DM_Sunbrust_df.groupby("Brand")["StoreCount"].sum().reset_index()
    #sort the brands based on store count in ascending order
    brand_productivity = brand_productivity.sort_values(by="StoreCount", ascending=True) 

    # Create Bar Chart for Brand & Their Productivity
    fig_Brand_Productivity = go.Figure()
    fig_Brand_Productivity.add_trace(go.Bar(
        x=brand_productivity["StoreCount"],
        y=brand_productivity["Brand"],
        name="Unique Productivity",
        marker=dict(color=COLORS["theme"][3],cornerradius=30),
        text=brand_productivity["StoreCount"].astype(str),  # Add text for data labels
        textposition="inside",  # Position text inside the bars
        hovertemplate='Brand: %{y}<br>Unique Productivity: %{x}<extra></extra>',  # Customize hover text
        orientation='h',
    ))
    fig_Brand_Productivity.update_layout(
        title={
            'font': {'size': FONTSIZE["title"]},
            'text': 'Brand-wise Unique Productivity',  # Main title
            'x': 0.5,  # Center the title horizontally
            'xanchor': 'center',  # Align title at the center
            'yanchor': 'top',  # Position title at the top
        },
        template="plotly_white"
    )

    #Create heatmap with Booker on the y-axis and periods on the x-axis
    # Convert 'period' to datetime using inferencing by pandas
    TGTVsAch_Booker_df['period'] = pd.to_datetime(TGTVsAch_Booker_df['period'], errors='coerce')

    # Extract year and month for sorting
    TGTVsAch_Booker_df['Year'] = TGTVsAch_Booker_df['period'].dt.year
    TGTVsAch_Booker_df['Month'] = TGTVsAch_Booker_df['period'].dt.month

    # Sort the data by Year, then by Month, then by Booker
    TGTVsAch_Booker_df.sort_values(by=['Year', 'Month', 'Booker'], inplace=True)

    # Pivot the data for the heatmap
    pivot_df = TGTVsAch_Booker_df.pivot(index='Booker', columns='period', values='NMV')
    pivot_df_rounded = (pivot_df / 1_000_000).round(1)
    # Fill NaN values with 0 (or another value if needed)
    # pivot_df = pivot_df.fillna(0)

    fig_TgtvsAch_Booker = px.imshow(pivot_df_rounded,
                labels=dict(x="Period", y="Booker", color="NMV"),
                x=pivot_df_rounded.columns,  # Use period as the x-axis labels
                y=pivot_df_rounded.index,  # Use Booker as the y-axis labels
                color_continuous_scale=COLORS["theme"],  # Choose a color scale
                title="Heatmap of NMV by Booker (YTD) Millions",
                text_auto=".1f",  # Show values on the heatmap
                )
                    
    fig_TgtvsAch_Booker.update_layout(
        title={
            'font': {'size': FONTSIZE["title"]},
            'text': 'Heatmap of NMV by Booker (YTD) Millions',  # Main title
            'x': 0.5,  # Center the title horizontally
            'xanchor': 'center',  # Align title at the center
            'yanchor': 'top',  # Position title at the top
        },
        width=800,
        height=600,
    )
    fig_TgtvsAch_Booker.update_traces(hovertemplate='Booker: %{y}<br>Period: %{x|%Y-%m}<br>NMV: %{z:.1f}M')  # Customize hover text
    
    pivot_df1 = Channel_NMV_YTD_df.pivot(index='Channel', columns='period', values='Ltr')
    pivot_df1_rounded = (pivot_df1 / 1000).round(1)
    #channel wise NMV Heatmap YTD
    fig_Channel_NMV_YTD = px.imshow(pivot_df1_rounded,
                labels=dict(x="Period", y="Channel", color="Ltr"),
                x=pivot_df1_rounded.columns,  # Use period as the x-axis labels
                y=pivot_df1_rounded.index,  # Use Channel as the y-axis labels
                color_continuous_scale=COLORS["theme"],  # Choose a color scale
                title="Heatmap of NMV by Channel (YTD) Ton",
                text_auto=".1f",  # Show values on the heatmap
                )
    fig_Channel_NMV_YTD.update_layout(
        title={
            'font': {'size': FONTSIZE["title"]},
            'text': 'Heatmap of NMV by Channel (YTD) Ton',  # Main title
            'x': 0.5,  # Center the title horizontally
            'xanchor': 'center',  # Align title at the center
            'yanchor': 'top',  # Position title at the top
        },
       
    )
    fig_Channel_NMV_YTD.update_traces(hovertemplate='Channel: %{y}<br>Period: %{x|%Y-%m}<br>Ltr: %{z:.1f}T')  # Customize hover text    

    # ---------- Channel Treemap (6-Month Period-wise Breakdown) ----------
    treemap_df = Channel_Treemap_df.copy()
    treemap_df['period'] = pd.to_datetime(treemap_df['period'], errors='coerce')
    treemap_df = treemap_df.dropna(subset=['period', 'Channel', 'NMV'])
    treemap_df['period_label'] = treemap_df['period'].dt.strftime('%b-%y')
    treemap_df['NMV_M'] = treemap_df['NMV'] / 1_000_000
    channel_options = [{"label": c, "value": c} for c in sorted(treemap_df['Channel'].dropna().unique())]
    if selected_channels:
        treemap_df = treemap_df[treemap_df['Channel'].isin(selected_channels)]
    treemap_df.sort_values(by=['period', 'Channel'], inplace=True)

    if treemap_df.empty:
        fig_channel_treemap = go.Figure().add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color=COLORS["text-light"])
        )
        return kpis, fig_daily, fig_type,fig_sales_growth,fig_Brand_Comparsion, fig_Brand_Comparison_Bar,brand_datatable,fig_DM_Comparison_Bar,fig_TgtvsAch,fig_SKU_HeatMap,fig_Channel_SunBrust, dm_options, fig_Brand_Productivity, fig_TgtvsAch_Booker, fig_Channel_NMV_YTD, channel_options, fig_channel_treemap
    else:
        period_totals = treemap_df.groupby('period_label')['NMV_M'].sum()
        period_order = treemap_df.sort_values('period')['period_label'].drop_duplicates().tolist()
        treemap_df.sort_values(by=['period', 'Channel'], inplace=True)

        labels = ['All Periods']
        parents = ['']
        values = [period_totals.sum()]
        ids = ['root']

        for period_label in period_order:
            period_total = period_totals[period_label]
            period_label_total = f"{period_label} | {period_total:.2f}M"
            period_id = f"period:{period_label}"
            labels.append(period_label_total)
            parents.append('root')
            values.append(period_total)
            ids.append(period_id)

            period_rows = treemap_df[treemap_df['period_label'] == period_label].sort_values('Channel')
            for _, row in period_rows.iterrows():
                channel_label = f"{row['Channel']} - {row['NMV_M']:.2f}M"
                labels.append(channel_label)
                parents.append(period_id)
                values.append(row['NMV_M'])
                ids.append(f"{period_id}|channel:{row['Channel']}")

        fig_channel_treemap = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            ids=ids,
            branchvalues='total',
            sort=False,
            marker=dict(
                colors=values,
                colorscale=COLORS["theme"],
                line=dict(width=2, color='white')
            ),
            textinfo='label',
            textfont_size=12,
            hovertemplate='<b>%{label}</b><br>NMV: %{value:.2f}M<extra></extra>'
        ))

        fig_channel_treemap.update_layout(
            title={
                'font': {'size': FONTSIZE["title"]},
                'text': 'Channel-wise NMV Treemap (6-Month Breakdown)',
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
            }
        )





    return kpis, fig_daily, fig_type,fig_sales_growth,fig_Brand_Comparsion, fig_Brand_Comparison_Bar,brand_datatable,fig_DM_Comparison_Bar,fig_TgtvsAch,fig_SKU_HeatMap,fig_Channel_SunBrust, dm_options, fig_Brand_Productivity, fig_TgtvsAch_Booker, fig_Channel_NMV_YTD, channel_options, fig_channel_treemap


@app.callback(
    Output("selected-town-store", "data"),
    Input("town", "value"),
)
def store_selected_town(town):
    """Store the selected town/database in dcc.Store for use in other callbacks."""
    return TOWN_TO_DB.get(town, "db42280")


@app.callback(
    Output("booker-lessthanctn-table", "columns"),
    Output("booker-lessthanctn-table", "data"),
    Output("booker-lessthanctn-detail-store", "data"),
    Input("booker-month-filter", "value"),
    State("selected-town-store", "data"),
)
def update_booker_monthly_report(booker_month_filter, selected_town):
    selected_town = selected_town or "prime"
    return fetch_booker_less_ctn_data(booker_month_filter, selected_town)


@app.callback(
    Output("booker-subreport-title", "children"),
    Output("booker-lessthanctn-subreport-table", "data"),
    Input("booker-lessthanctn-table", "active_cell"),
    State("booker-lessthanctn-table", "data"),
    State("booker-lessthanctn-detail-store", "data"),
)
def update_booker_subreport(active_cell, table_data, detail_data):
    if not active_cell or not table_data or not detail_data:
        return "📋 Booker Less-Than-Half-Carton Subreport (select any row/cell above)", []

    row_index = active_cell.get("row")
    column_id = active_cell.get("column_id")

    if row_index is None or row_index >= len(table_data):
        return "📋 Booker Less-Than-Half-Carton Subreport (select any row/cell above)", []

    selected_row = table_data[row_index]
    selected_booker = selected_row.get("Booker_Name")
    if not selected_booker:
        return "📋 Booker Less-Than-Half-Carton Subreport (select any row/cell above)", []

    drill_source_df = pd.DataFrame(detail_data)
    drill_df = drill_source_df[drill_source_df["Booker_Name"] == selected_booker].copy()
    title_suffix = f"Booker: {selected_booker}"

    if column_id and column_id != "Booker_Name":
        selected_brand = column_id
        drill_df = drill_df[drill_df["brand"] == selected_brand].copy()
        title_suffix = f"Booker: {selected_booker} | Brand: {selected_brand}"

    if drill_df.empty:
        return f"📋 Booker Less-Than-Half-Carton Subreport ({title_suffix}) - No records", []

    drill_df = drill_df.sort_values(["age", "HalfCtnDel"], ascending=[False, False])
    drill_df["age"] = (drill_df["age"] * 100).round(2).astype(str) + "%"

    subreport_data = drill_df[["brand", "StoreCode", "StoreName", "Total_Deliveries", "HalfCtnDel", "age"]].to_dict("records")
    return f"📋 Booker Less-Than-Half-Carton Subreport ({title_suffix})", subreport_data



# ======================
# ▶ RUN
# ======================
if __name__ == "__main__":
    # For local testing use debug=True
    # For cloud deployment, debug should be False
    app.run(debug=False)
	# app.run_server(host='0.0.0.0', port=80, debug=False)
