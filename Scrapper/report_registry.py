"""
report_registry.py
==================
Dynamic Report Registry for Salesflo Scraper.

HOW TO ADD A NEW REPORT
------------------------
1. Add an entry to REPORT_REGISTRY dict below.
2. Define `nav_steps` (menu path after "Reports").
3. Define `filters` with control type + selector/value for each filter you want set.
4. Define `columns` list so the parser knows what to extract.
5. Define `db_table` for where rows go.
6. Done — no other code changes needed.

FILTER TYPES SUPPORTED
-----------------------
- radio       : input[type="radio"]  — set by name+value
- select      : <select>             — set by select option value/label
- date        : date picker input    — filled with formatted date string
- checkbox    : <input type="checkbox"> — check or uncheck
- text        : plain text input     — filled with value

COLUMN DEFINITION
-----------------
Each column dict:
  {
    "name":     "Column Header Label in report",   # canonical name for DB
    "aliases":  ["alt label", ...],                # alternate header spellings
    "db_col":   "db_column_name",                  # snake_case DB column name
    "db_type":  "DATE|DECIMAL|INT|VARCHAR(n)",     # MySQL type
    "is_date":  True/False,                        # parse as date?
    "is_float": True/False,                        # parse as float?
    "is_int":   True/False,                        # parse as int?
  }
"""

from typing import Any

# ── Nav ───────────────────────────────────────────────────────────────────────
# Base login URL — shared across all reports
LOGIN_URL  = "https://engrofoods.salesflo.com/OB/login/"
BASE_URL   = "https://engrofoods.salesflo.com/OB/reports/"

# ── Filter helpers (reusable filter blocks) ───────────────────────────────────

DATE_FILTER_START = {
    "type": "date",
    "role": "start_date",
    "selectors": [
        'input[name="std2"]', 'input#std2',
        'input[name="dt1"]',  'input#dt1',
        'input[name*="start" i]', 'input[id*="start" i]',
    ],
    "label": "Start Date",
}

DATE_FILTER_END = {
    "type": "date",
    "role": "end_date",
    "selectors": [
        'input[name="end2"]', 'input#end2',
        'input[name="dt2"]',  'input#dt2',
        'input[name*="end" i]', 'input[id*="end" i]',
    ],
    "label": "End Date",
}

RADIO_DAILY = {
    "type": "radio",
    "name": "TP",
    "value": "2",
    "selectors": ['input#Daily', 'input[name="TP"][value="2"]'],
    "label": "Daily",
}

RADIO_SUMMARY = {
    "type": "radio",
    "name": "TOR",
    "value": "1",
    "selectors": ['input#Summary', 'input[name="TOR"][value="1"]'],
    "label": "Summary",
}

# ══════════════════════════════════════════════════════════════════════════════
# REPORT REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

REPORT_REGISTRY: dict[str, dict[str, Any]] = {

    # ── 1. End Stock Trend ────────────────────────────────────────────────────
    "end_stock_trend": {
        "title":      "End Stock Trend",
        "db_table":   "end_stock_summary",
        "parse_mode": "end_stock",          # special unpivot logic
        "save_mode":  "end_stock",

        # Menu path AFTER clicking "Reports"
        # Each step is a CSS/text selector clicked in order
        "nav_steps": [
            "text=Inventory Reports",
            "text=End Stock Trend",
        ],
        # Selectors that confirm the report form is loaded
        "ready_selectors": [
            'input#dt1', 'input[name="dt1"]',
            'select#su', 'select[name="su"]',
        ],

        "filters": [
            RADIO_DAILY,
            RADIO_SUMMARY,
            DATE_FILTER_START,
            DATE_FILTER_END,
            {
                "type": "select",
                "selectors": ['select#su', 'select[name="su"]'],
                "value": "5",          # 5 = Value
                "label": "Stock Unit",
            },
        ],

        # Fixed columns (before date pivot columns)
        "columns": [
            {"name": "S#",                 "aliases": ["s#","s.no","sno"],                         "db_col": "sno",              "db_type": "VARCHAR(50)"},
            {"name": "Distributor Code",   "aliases": ["distributor code"],                         "db_col": "distributor_code", "db_type": "VARCHAR(100)"},
            {"name": "Distributor Name",   "aliases": ["distributor name","distributor"],           "db_col": "distributor_name", "db_type": "VARCHAR(255)"},
            {"name": "SKU Description",    "aliases": ["sku description","sku desc","description"], "db_col": "sku_description",  "db_type": "VARCHAR(500)"},
            {"name": "SKU Code",           "aliases": ["sku code","sku"],                           "db_col": "sku_code",         "db_type": "VARCHAR(100)"},
            {"name": "Brand Name",         "aliases": ["brand name","brand"],                       "db_col": "brand_name",       "db_type": "VARCHAR(255)"},
            {"name": "Brand Code",         "aliases": ["brand code"],                              "db_col": "brand_code",       "db_type": "VARCHAR(100)"},
            # date columns are auto-detected and unpivoted — no need to list them
        ],
    },

    # ── 2. Visits Summary ─────────────────────────────────────────────────────
    "visits_summary": {
        "title":      "Visits Summary",
        "db_table":   "visits_summary_rows",
        "parse_mode": "generic",
        "save_mode":  "generic",

        "nav_steps": [
            "text=Visits Reports",
            "text=Visits Summary",
        ],
        "ready_selectors": [
            'input#std2', 'input[name="std2"]',
            'input#end2', 'input[name="end2"]',
        ],

        "filters": [
            DATE_FILTER_START,
            DATE_FILTER_END,
            # Show both Visit Complete = Yes and No
            {"type": "checkbox", "selectors": ['input#ShowVisitComplete_1', 'input[name="ShowVisitComplete"][value="1"]'], "checked": True,  "label": "Visit Complete Yes"},
            {"type": "checkbox", "selectors": ['input#ShowVisitComplete_0', 'input[name="ShowVisitComplete"][value="0"]'], "checked": True,  "label": "Visit Complete No"},
            # Show Invoice Number column
            {"type": "checkbox", "selectors": ['input#ShowInvoiceNumber_1', 'input[name="ShowInvoiceNumber"][value="1"]'], "checked": True,  "label": "Show Invoice Number"},
        ],

        "columns": [
            {"name": "Distributor",                        "aliases": ["distributor","distributor name"],             "db_col": "Distributor",                        "db_type": "VARCHAR(255)"},
            {"name": "Visit Date",                         "aliases": ["visit date"],                                 "db_col": "Visit Date",                         "db_type": "DATE",    "is_date": True},
            {"name": "Delivery Date",                      "aliases": ["delivery date"],                              "db_col": "Delivery Date",                      "db_type": "DATE",    "is_date": True},
            {"name": "PJP Name",                           "aliases": ["pjp name"],                                   "db_col": "PJP Name",                           "db_type": "VARCHAR(255)"},
            {"name": "App User",                           "aliases": ["app user"],                                   "db_col": "App User",                           "db_type": "VARCHAR(255)"},
            {"name": "Store Name",                         "aliases": ["store name"],                                 "db_col": "Store Name",                         "db_type": "VARCHAR(255)"},
            {"name": "Store Code",                         "aliases": ["store code"],                                 "db_col": "Store Code",                         "db_type": "VARCHAR(100)"},
            {"name": "Store Company Code",                 "aliases": ["store company code"],                         "db_col": "Store Company Code",                 "db_type": "VARCHAR(100)"},
            {"name": "Sync Down",                          "aliases": ["sync down"],                                  "db_col": "Sync Down",                          "db_type": "VARCHAR(50)"},
            {"name": "Sync Down Date",                     "aliases": ["sync down date"],                             "db_col": "Sync Down Date",                     "db_type": "DATE",    "is_date": True},
            {"name": "Sync Down Time",                     "aliases": ["sync down time"],                             "db_col": "Sync Down Time",                     "db_type": "VARCHAR(50)"},
            {"name": "Sync Up Date",                       "aliases": ["sync up date"],                               "db_col": "Sync Up Date",                       "db_type": "DATE",    "is_date": True},
            {"name": "Sync Up Time",                       "aliases": ["sync up time"],                               "db_col": "Sync Up Time",                       "db_type": "VARCHAR(50)"},
            {"name": "Visit Complete",                     "aliases": ["visit complete"],                             "db_col": "Visit Complete",                     "db_type": "VARCHAR(100)"},
            {"name": "Order Number",                       "aliases": ["order number"],                               "db_col": "Order Number",                       "db_type": "VARCHAR(255)"},
            {"name": "Invoice Number",                     "aliases": ["invoice number"],                             "db_col": "Invoice Number",                     "db_type": "VARCHAR(255)"},
            {"name": "Total Units",                        "aliases": ["total units"],                                "db_col": "Total Units",                        "db_type": "DECIMAL(18,4)", "is_float": True},
            {"name": "Total Value",                        "aliases": ["total value"],                                "db_col": "Total Value",                        "db_type": "DECIMAL(18,4)", "is_float": True},
            {"name": "Total SKU Sold",                     "aliases": ["total sku sold"],                             "db_col": "Total SKU Sold",                     "db_type": "INT",     "is_int": True},
            {"name": "Non Productive w.r.t Order",         "aliases": ["non productive w.r.t order","non productive"],"db_col": "Non Productive w.r.t Order",         "db_type": "VARCHAR(255)"},
            {"name": "Close Reason",                       "aliases": ["close reason"],                               "db_col": "Close Reason",                       "db_type": "VARCHAR(500)"},
            {"name": "Total Visits",                       "aliases": ["total visits"],                               "db_col": "Total Visits",                       "db_type": "INT",     "is_int": True},
            {"name": "First Check In Date",                "aliases": ["first check in date"],                        "db_col": "First Check In Date",                "db_type": "DATE",    "is_date": True},
            {"name": "First Check In Time",                "aliases": ["first check in time"],                        "db_col": "First Check In Time",                "db_type": "VARCHAR(50)"},
            {"name": "First Check Out Date",               "aliases": ["first check out date"],                       "db_col": "First Check Out Date",               "db_type": "DATE",    "is_date": True},
            {"name": "First Check Out Time",               "aliases": ["first check out time"],                       "db_col": "First Check Out Time",               "db_type": "VARCHAR(50)"},
            {"name": "First Spent Time",                   "aliases": ["first spent time"],                           "db_col": "First Spent Time",                   "db_type": "VARCHAR(50)"},
            {"name": "Total Spent Time",                   "aliases": ["total spent time"],                           "db_col": "Total Spent Time",                   "db_type": "VARCHAR(50)"},
            {"name": "Store Latitude",                     "aliases": ["store latitude"],                             "db_col": "Store Latitude",                     "db_type": "DECIMAL(12,8)", "is_float": True},
            {"name": "Store Longitude",                    "aliases": ["store longitude"],                            "db_col": "Store Longitude",                    "db_type": "DECIMAL(12,8)", "is_float": True},
            {"name": "Visit Latitude",                     "aliases": ["visit latitude"],                             "db_col": "Visit Latitude",                     "db_type": "DECIMAL(12,8)", "is_float": True},
            {"name": "Visit Longitude",                    "aliases": ["visit longitude"],                            "db_col": "Visit Longitude",                    "db_type": "DECIMAL(12,8)", "is_float": True},
            {"name": "Distance From Original Location (m)","aliases": ["distance from original location (m)"],        "db_col": "Distance From Original Location (m)","db_type": "DECIMAL(18,2)", "is_float": True},
            {"name": "Order Added From",                   "aliases": ["order added from"],                           "db_col": "Order Added From",                   "db_type": "VARCHAR(255)"},
        ],
    },

    # ── 3. Ordered Vs Delivered ───────────────────────────────────────────────
    "ordered_vs_delivered": {
        "title":      "Ordered Vs Delivered Report",
        "db_table":   "ordered_vs_delivered_rows",
        "parse_mode": "generic",
        "save_mode":  "generic",

        "nav_steps": [
            "text=Ordering Invoicing Reports",
            "text=Ordered Vs Delivered Report",
        ],
        "ready_selectors": [
            'input#dt1', 'input[name="dt1"]',
            'input#dt2', 'input[name="dt2"]',
        ],

        "filters": [
            DATE_FILTER_START,
            DATE_FILTER_END,
            # QTY Type = Units (value=3)
            {"type": "radio",    "name": "QTY",  "value": "3", "selectors": ['input[name="QTY"][value="3"]'],                             "label": "QTY Units"},
            # Uncheck order statuses 1 and 2
            {"type": "checkbox", "selectors": ['input.ord_chk_boxes[value="1"]', 'input[name*="ord"][value="1"]'],                         "checked": False, "label": "Order Status 1"},
            {"type": "checkbox", "selectors": ['input.ord_chk_boxes[value="2"]', 'input[name*="ord"][value="2"]'],                         "checked": False, "label": "Order Status 2"},
            # Show Delivery Man column
            {"type": "checkbox", "selectors": ['input#ShowDeliveryman_1', 'input[name="ShowDeliveryMan"][value="1"]'],                     "checked": True,  "label": "Show Deliveryman"},
            # Show SKU Weight Type columns
            {"type": "checkbox", "selectors": ['input#show_sku_weight', 'input[name="show_sku_weight"]', 'tr#ShowWeightTypes input[type="checkbox"]'], "checked": True, "label": "Show SKU Weight"},
        ],

        "columns": [
            {"name": "S.No#",                "aliases": ["s.no#","s no#","s no","serial no"],       "db_col": "S.No#",                "db_type": "VARCHAR(50)"},
            {"name": "Distributor Name",     "aliases": ["distributor name"],                        "db_col": "Distributor Name",     "db_type": "VARCHAR(255)"},
            {"name": "Distributor Code",     "aliases": ["distributor code"],                        "db_col": "Distributor Code",     "db_type": "VARCHAR(100)"},
            {"name": "Store Name",           "aliases": ["store name"],                              "db_col": "Store Name",           "db_type": "VARCHAR(255)"},
            {"name": "Store Code",           "aliases": ["store code"],                              "db_col": "Store Code",           "db_type": "VARCHAR(100)"},
            {"name": "SKU Code",             "aliases": ["sku code"],                               "db_col": "SKU Code",             "db_type": "VARCHAR(100)"},
            {"name": "SKU Name",             "aliases": ["sku name"],                               "db_col": "SKU Name",             "db_type": "VARCHAR(255)"},
            {"name": "SKU Manufacturer Code","aliases": ["sku manufacturer code"],                  "db_col": "SKU Manufacturer Code","db_type": "VARCHAR(100)"},
            {"name": "Category Code",        "aliases": ["category code"],                           "db_col": "Category Code",        "db_type": "VARCHAR(100)"},
            {"name": "Category Name",        "aliases": ["category name"],                           "db_col": "Category Name",        "db_type": "VARCHAR(255)"},
            {"name": "Order Booker Code",    "aliases": ["order booker code"],                       "db_col": "Order Booker Code",    "db_type": "VARCHAR(100)"},
            {"name": "Order Booker Name",    "aliases": ["order booker name"],                       "db_col": "Order Booker Name",    "db_type": "VARCHAR(255)"},
            {"name": "Deliveryman Code",     "aliases": ["deliveryman code"],                        "db_col": "Deliveryman Code",     "db_type": "VARCHAR(100)"},
            {"name": "Deliveryman Name",     "aliases": ["deliveryman name"],                        "db_col": "Deliveryman Name",     "db_type": "VARCHAR(255)"},
            {"name": "Order Number",         "aliases": ["order number"],                            "db_col": "Order Number",         "db_type": "VARCHAR(255)"},
            {"name": "Invoice Number",       "aliases": ["invoice number"],                          "db_col": "Invoice Number",       "db_type": "VARCHAR(255)"},
            {"name": "Status",               "aliases": ["status"],                                  "db_col": "Status",               "db_type": "VARCHAR(100)"},
            {"name": "Order Date",           "aliases": ["order date"],                              "db_col": "Order Date",           "db_type": "DATE",          "is_date": True},
            {"name": "Delivery Date",        "aliases": ["delivery date"],                           "db_col": "Delivery Date",        "db_type": "DATE",          "is_date": True},
            {"name": "Order Units",          "aliases": ["order units"],                             "db_col": "Order Units",          "db_type": "DECIMAL(18,4)", "is_float": True},
            {"name": "Order (Grams)",        "aliases": ["order (grams)","order grams"],             "db_col": "Order (Grams)",        "db_type": "DECIMAL(18,4)", "is_float": True},
            {"name": "Order (ML)",           "aliases": ["order (ml)","order ml"],                  "db_col": "Order (ML)",           "db_type": "DECIMAL(18,4)", "is_float": True},
            {"name": "Order (KG)",           "aliases": ["order (kg)","order kg"],                  "db_col": "Order (KG)",           "db_type": "DECIMAL(18,4)", "is_float": True},
            {"name": "Order (Litres)",       "aliases": ["order (litres)","order litres"],           "db_col": "Order (Litres)",       "db_type": "DECIMAL(18,4)", "is_float": True},
            {"name": "Order Amount",         "aliases": ["order amount"],                            "db_col": "Order Amount",         "db_type": "DECIMAL(18,4)", "is_float": True},
            {"name": "Delivered Units",      "aliases": ["delivered units"],                         "db_col": "Delivered Units",      "db_type": "DECIMAL(18,4)", "is_float": True},
            {"name": "Delivered (Grams)",    "aliases": ["delivered (grams)","delivered grams"],     "db_col": "Delivered (Grams)",    "db_type": "DECIMAL(18,4)", "is_float": True},
            {"name": "Delivered (ML)",       "aliases": ["delivered (ml)","delivered ml"],           "db_col": "Delivered (ML)",       "db_type": "DECIMAL(18,4)", "is_float": True},
            {"name": "Delivered (KG)",       "aliases": ["delivered (kg)","delivered kg"],           "db_col": "Delivered (KG)",       "db_type": "DECIMAL(18,4)", "is_float": True},
            {"name": "Delivered (Litres)",   "aliases": ["delivered (litres)","delivered litres"],   "db_col": "Delivered (Litres)",   "db_type": "DECIMAL(18,4)", "is_float": True},
            {"name": "Delivered Amount",     "aliases": ["delivered amount"],                        "db_col": "Delivered Amount",     "db_type": "DECIMAL(18,4)", "is_float": True},
            {"name": "Returned Units",       "aliases": ["returned units"],                          "db_col": "Returned Units",       "db_type": "DECIMAL(18,4)", "is_float": True},
            {"name": "Returned Amount",      "aliases": ["returned amount"],                         "db_col": "Returned Amount",      "db_type": "DECIMAL(18,4)", "is_float": True},
            {"name": "Total Discount",       "aliases": ["total discount"],                          "db_col": "Total Discount",       "db_type": "DECIMAL(18,4)", "is_float": True},
        ],
    },

    # ── 4. Current End Stock ──────────────────────────────────────────────────
    # Example: adding a completely new report takes only this block
    "current_end_stock": {
        "title":      "Current End Stock",
        "db_table":   "end_stock_summary",
        "parse_mode": "generic",
        "save_mode":  "generic",

        "nav_steps": [
            "text=Inventory Reports",
            "text=Current End Stock",
        ],
        "ready_selectors": [
            'select[name="national"]', 'select#national',
            'button:has-text("View Report")', 'input[value="View Report"]',
        ],

        "filters": [
            # No date filters — this is a snapshot report
            # Only show active SKUs (checkbox with tr id="DynamicCheckBoxTr_ShowBoxes")
            {
                "type": "checkbox",
                "selectors": [
                    'tr#DynamicCheckBoxTr_ShowBoxes input[type="checkbox"]',
                    'input[name="ShowBoxes"]',
                ],
                "checked": True,
                "label": "Show Boxes",
            },
        ],

        "columns": [
            {"name": "Distributor Code",  "aliases": ["distributor code"],          "db_col": "distributor_code",  "db_type": "VARCHAR(100)"},
            {"name": "Distributor Name",  "aliases": ["distributor name"],           "db_col": "distributor_name",  "db_type": "VARCHAR(255)"},
            {"name": "SKU Code",          "aliases": ["sku code","sku"],             "db_col": "sku_code",          "db_type": "VARCHAR(100)"},
            {"name": "SKU Description",   "aliases": ["sku description","sku name"], "db_col": "sku_description",   "db_type": "VARCHAR(500)"},
            {"name": "Brand Code",        "aliases": ["brand code"],                 "db_col": "brand_code",        "db_type": "VARCHAR(100)"},
            {"name": "Brand Name",        "aliases": ["brand name"],                 "db_col": "brand_name",        "db_type": "VARCHAR(255)"},
            {"name": "Cases",             "aliases": ["cases","qty (cases)"],        "db_col": "cases",             "db_type": "DECIMAL(18,4)", "is_float": True},
            {"name": "Units",             "aliases": ["units","qty (units)"],        "db_col": "units",             "db_type": "DECIMAL(18,4)", "is_float": True},
            {"name": "Value",             "aliases": ["value","amount"],             "db_col": "value",             "db_type": "DECIMAL(18,4)", "is_float": True},
        ],
    },

    # ── HOW TO ADD YOUR OWN REPORT ────────────────────────────────────────────
    # Copy the block above, change:
    #   - key name (e.g. "daily_end_stock")
    #   - title, db_table, parse_mode, save_mode
    #   - nav_steps: menu text sequence after "Reports"
    #   - ready_selectors: any selector that proves the form loaded
    #   - filters: list of filter actions
    #   - columns: list of column defs (name + aliases for flexible header matching)
    # ─────────────────────────────────────────────────────────────────────────
}