"""
Salesflo End Stock Summary Bot
- Runs daily at 10:00 PM
- Fetches End Stock Summary (Daily | Summary | Yesterday's date | Value)
- Downloads Excel, unpivots date columns (col H onward), excludes total row
- Saves to remote MySQL database
- Smart date logic: checks last saved date in DB, fetches all missing days up to N-1
"""

import asyncio
import html
import json
import logging
import os
import re
import hashlib
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta, date as date_type
from typing import Optional
from urllib.parse import urljoin, urlparse, unquote

import aiomysql
from dotenv import load_dotenv
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("salesflo_bot.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SITE_URL = "https://engrofoods.salesflo.com/OB/reports/?page=InventoryReports#noanchor"


def _load_local_secrets_toml() -> dict:
    """Load Streamlit-style secrets from user home for non-Streamlit scripts."""
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
                    return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            log.warning("Unable to read secrets file %s: %s", candidate, exc)
            continue
    return {}


_SECRETS = _load_local_secrets_toml()
_DB_SECRETS = _SECRETS.get("database", {}) if isinstance(_SECRETS, dict) else {}
_SALESFLO_SECRETS = _SECRETS.get("salesflo", {}) if isinstance(_SECRETS, dict) else {}
_SCHEDULER_SECRETS = _SECRETS.get("scheduler", {}) if isinstance(_SECRETS, dict) else {}


def _resolve_secret(section_dict: dict, key_options: list[str], env_options: list[str], default: str = "") -> str:
    if isinstance(section_dict, dict):
        for key in key_options:
            value = section_dict.get(key)
            if value not in (None, ""):
                return str(value).strip()

    if isinstance(_SECRETS, dict):
        for key in key_options:
            value = _SECRETS.get(key)
            if value not in (None, ""):
                return str(value).strip()

    for env_key in env_options:
        value = os.getenv(env_key)
        if value not in (None, ""):
            return str(value).strip()

    return str(default).strip()

USERNAME = _resolve_secret(
    _SALESFLO_SECRETS,
    ["SALESFLO_USERNAME", "username", "user"],
    ["SALESFLO_USERNAME"],
    "",
)
PASSWORD = _resolve_secret(
    _SALESFLO_SECRETS,
    ["SALESFLO_PASSWORD", "password", "pass"],
    ["SALESFLO_PASSWORD"],
    "",
)
USERNAME_2 = _resolve_secret(
    _SALESFLO_SECRETS,
    ["SALESFLO_USERNAME2", "SALESFLO_USERNAME_2", "username2", "user2"],
    ["SALESFLO_USERNAME2", "SALESFLO_USERNAME_2"],
    "",
)
PASSWORD_2 = _resolve_secret(
    _SALESFLO_SECRETS,
    ["SALESFLO_PASSWORD2", "SALESFLO_PASSWORD_2", "password2", "pass2"],
    ["SALESFLO_PASSWORD2", "SALESFLO_PASSWORD_2"],
    "",
)

DB_HOST = _resolve_secret(
    _DB_SECRETS,
    ["host", "DB_HOST"],
    ["DB_HOST"],
    "",
)
DB_PORT = int(_resolve_secret(
    _DB_SECRETS,
    ["port", "DB_PORT"],
    ["DB_PORT"],
    "3306",
))
DB_USER = _resolve_secret(
    _DB_SECRETS,
    ["username", "user", "DB_USER", "DB_USERNAME"],
    ["DB_USER", "DB_USERNAME"],
    "",
)
DB_PASS = _resolve_secret(
    _DB_SECRETS,
    ["password", "pass", "DB_PASSWORD", "DB_PASS"],
    ["DB_PASSWORD", "DB_PASS"],
    "",
)
DB_NAME = _resolve_secret(
    _DB_SECRETS,
    ["database", "db", "DB_NAME"],
    ["DB_NAME"],
    "salesflo_data",
)

ENABLED_REPORTS = os.getenv("ENABLED_REPORTS", "visits_summary,end_stock_trend,ordered_vs_delivered")

REPORT_CONFIGS = {
    "visits_summary": {
        "title": "Visits Summary",
        "table": "visits_summary_rows",
        "nav_page": "https://engrofoods.salesflo.com/OB/reports/?page=VisitReports#noanchor",
        "nav_steps": ["text=Reports", "text=Visits Reports", "text=Visits Summary"],
        "ready_selectors": ['input#std2', 'input[name="std2"]', 'input#end2', 'input[name="end2"]'],
        "require_daily": False,
        "require_summary": False,
        "visit_complete_both": True,
        "show_invoice": True,
        "stock_unit_value": False,
        "parse_mode": "visits",
        "save_mode": "visits",
    },
    "end_stock_trend": {
        "title": "End Stock Trend",
        "table": "end_stock_summary",
        "nav_page": "https://engrofoods.salesflo.com/OB/reports/?page=InventoryReports#noanchor",
        "nav_steps": ["text=Reports", "text=Inventory Reports", "text=End Stock Trend"],
        "ready_selectors": ['input#dt1', 'input[name="dt1"]', 'select#su', 'select[name="su"]'],
        "require_daily": True,
        "require_summary": True,
        "visit_complete_both": False,
        "show_invoice": False,
        "stock_unit_value": "5",
        "parse_mode": "end_stock",
        "save_mode": "end_stock",
    },
    "ordered_vs_delivered": {
        "title": "Ordered Vs Delivered Report",
        "table": "ordered_vs_delivered_rows",
        "nav_page": SITE_URL,
        "nav_steps": ["text=Reports", "text=Ordering Invoicing Reports", "text=Ordered Vs Delivered Report"],
        "ready_selectors": ['input#dt1', 'input[name="dt1"]', 'input#dt2', 'input[name="dt2"]'],
        "require_daily": False,
        "require_summary": False,
        "visit_complete_both": False,
        "show_invoice": False,
        "stock_unit_value": False,
        "qty_type_value": "3",
        "order_status_unchecked": ["1", "2"],
        "show_delivery_man": True,
        "show_sku_weight": True,
        "parse_mode": "ordered_vs_delivered",
        "save_mode": "ordered_vs_delivered",
    },
}

VISITS_SUMMARY_COLUMNS = [
    "Distributor",
    "Visit Date",
    "Delivery Date",
    "PJP Name",
    "App User",
    "Store Name",
    "Store Code",
    "Store Company Code",
    "Sync Down",
    "Sync Down Date",
    "Sync Down Time",
    "Sync Up Date",
    "Sync Up Time",
    "Visit Complete",
    "Order Number",
    "Invoice Number",
    "Total Units",
    "Total Value",
    "Total SKU Sold",
    "Non Productive w.r.t Order",
    "Close Reason",
    "Total Visits",
    "First Check In Date",
    "First Check In Time",
    "First Check Out Date",
    "First Check Out Time",
    "First Spent Time",
    "Total Spent Time",
    "Store Latitude",
    "Store Longitude",
    "Visit Latitude",
    "Visit Longitude",
    "Distance From Original Location (m)",
    "Order Added From",
]

VISITS_COLUMN_ALIASES = {
    "distributor": "Distributor",
    "distributor name": "Distributor",
    "visit date": "Visit Date",
    "delivery date": "Delivery Date",
    "pjp name": "PJP Name",
    "app user": "App User",
    "store name": "Store Name",
    "store code": "Store Code",
    "store company code": "Store Company Code",
    "sync down": "Sync Down",
    "sync down date": "Sync Down Date",
    "sync down time": "Sync Down Time",
    "sync up date": "Sync Up Date",
    "sync up time": "Sync Up Time",
    "visit complete": "Visit Complete",
    "order number": "Order Number",
    "invoice number": "Invoice Number",
    "total units": "Total Units",
    "total value": "Total Value",
    "total sku sold": "Total SKU Sold",
    "non productive w.r.t order": "Non Productive w.r.t Order",
    "close reason": "Close Reason",
    "total visits": "Total Visits",
    "first check in date": "First Check In Date",
    "first check in time": "First Check In Time",
    "first check out date": "First Check Out Date",
    "first check out time": "First Check Out Time",
    "first spent time": "First Spent Time",
    "total spent time": "Total Spent Time",
    "store latitude": "Store Latitude",
    "store longitude": "Store Longitude",
    "visit latitude": "Visit Latitude",
    "visit longitude": "Visit Longitude",
    "distance from original location (m)": "Distance From Original Location (m)",
    "order added from": "Order Added From",
}

ORDERED_VS_DELIVERED_COLUMNS = [
    "S.No#",
    "Distributor Name",
    "Distributor Code",
    "Store Name",
    "Store Code",
    "SKU Code",
    "SKU Name",
    "SKU Manufacturer Code",
    "Category Code",
    "Category Name",
    "Order Booker Code",
    "Order Booker Name",
    "Deliveryman Code",
    "Deliveryman Name",
    "Order Number",
    "Invoice Number",
    "Status",
    "Order Date",
    "Delivery Date",
    "Order Units",
    "Order (Grams)",
    "Order (ML)",
    "Order (KG)",
    "Order (Litres)",
    "Order Amount",
    "Delivered Units",
    "Delivered (Grams)",
    "Delivered (ML)",
    "Delivered (KG)",
    "Delivered (Litres)",
    "Delivered Amount",
    "Returned Units",
    "Returned Amount",
    "Total Discount",
]

ORDERED_VS_DELIVERED_ALIASES = {
    "s.no#": "S.No#",
    "s no#": "S.No#",
    "s no": "S.No#",
    "s.no": "S.No#",
    "serial no": "S.No#",
    "distributor name": "Distributor Name",
    "distributor code": "Distributor Code",
    "store name": "Store Name",
    "store code": "Store Code",
    "sku code": "SKU Code",
    "sku name": "SKU Name",
    "sku manufacturer code": "SKU Manufacturer Code",
    "category code": "Category Code",
    "category name": "Category Name",
    "order booker code": "Order Booker Code",
    "order booker name": "Order Booker Name",
    "deliveryman code": "Deliveryman Code",
    "deliveryman name": "Deliveryman Name",
    "order number": "Order Number",
    "invoice number": "Invoice Number",
    "status": "Status",
    "order date": "Order Date",
    "delivery date": "Delivery Date",
    "order units": "Order Units",
    "order (grams)": "Order (Grams)",
    "order grams": "Order (Grams)",
    "order (ml)": "Order (ML)",
    "order ml": "Order (ML)",
    "order (kg)": "Order (KG)",
    "order kg": "Order (KG)",
    "order (litres)": "Order (Litres)",
    "order litres": "Order (Litres)",
    "order amount": "Order Amount",
    "delivered units": "Delivered Units",
    "delivered (grams)": "Delivered (Grams)",
    "delivered grams": "Delivered (Grams)",
    "delivered (ml)": "Delivered (ML)",
    "delivered ml": "Delivered (ML)",
    "delivered (kg)": "Delivered (KG)",
    "delivered kg": "Delivered (KG)",
    "delivered (litres)": "Delivered (Litres)",
    "delivered litres": "Delivered (Litres)",
    "delivered amount": "Delivered Amount",
    "returned units": "Returned Units",
    "returned amount": "Returned Amount",
    "total discount": "Total Discount",
}


def _normalize_header_label(value: str) -> str:
    return " ".join(str(value or "").replace("\n", " ").replace("\r", " ").split()).strip().lower()


def _normalize_date_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.strip("'\" ")
    text = re.sub(r"\s+", " ", text)
    # Normalize variants like "February, 28 2026" -> "February 28, 2026"
    text = re.sub(r"^([A-Za-z]+)\s*,\s*(\d{1,2})\s+(\d{4})$", r"\1 \2, \3", text)
    return text


_DATE_PARSE_WARNED_LABELS: set[str] = set()


def get_salesflo_accounts() -> list[tuple[str, str, str]]:
    accounts: list[tuple[str, str, str]] = []

    primary_user = str(USERNAME or "").strip()
    primary_pass = str(PASSWORD or "").strip()
    if primary_user and primary_pass:
        accounts.append(("account_1", primary_user, primary_pass))

    second_user = str(USERNAME_2 or "").strip()
    second_pass = str(PASSWORD_2 or "").strip()
    if second_user and second_pass:
        accounts.append(("account_2", second_user, second_pass))

    # Remove duplicate credentials if both env entries point to same account.
    deduped: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for account_label, username, password in accounts:
        key = (username, password)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((account_label, username, password))

    return deduped


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

async def get_db_connection():
    return await aiomysql.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASS,
        db=DB_NAME, charset="utf8mb4",
        autocommit=True,
    )


async def ensure_tables(conn):
    """Create tables if they don't exist."""
    async with conn.cursor() as cur:
        # Unpivoted schema: one row per (date, distributor, SKU)
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS end_stock_summary (
                id               INT AUTO_INCREMENT PRIMARY KEY,
                report_date      DATE NOT NULL,
                account_label    VARCHAR(50),
                distributor_code VARCHAR(100),
                distributor_name VARCHAR(255),
                sku_code         VARCHAR(100),
                sku_description  VARCHAR(500),
                brand_code       VARCHAR(100),
                brand_name       VARCHAR(255),
                value            DECIMAL(18,4),
                unit             VARCHAR(50) DEFAULT 'Value',
                fetched_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uq_date_acc_dist_sku (report_date, account_label, distributor_code, sku_code)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS bot_run_log (
                id         INT AUTO_INCREMENT PRIMARY KEY,
                run_date   DATE NOT NULL,
                status     ENUM('success','failed','no_data') NOT NULL,
                rows_saved INT DEFAULT 0,
                message    TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS visits_summary_rows (
                id               INT AUTO_INCREMENT PRIMARY KEY,
                `Report Date`    DATE NULL,
                `Account Label`  VARCHAR(50),
                `Visit Date`     DATE NULL,
                `Delivery Date`  DATE NULL,
                `Distributor`    VARCHAR(255),
                `PJP Name`       VARCHAR(255),
                `App User`       VARCHAR(255),
                `Store Name`     VARCHAR(255),
                `Store Code`     VARCHAR(100),
                `Store Company Code` VARCHAR(100),
                `Sync Down`      VARCHAR(50),
                `Sync Down Date` DATE NULL,
                `Sync Down Time` VARCHAR(50),
                `Sync Up Date`   DATE NULL,
                `Sync Up Time`   VARCHAR(50),
                `Invoice Number` VARCHAR(255),
                `Visit Complete` VARCHAR(100),
                `Order Number`   VARCHAR(255),
                `Total Units`    DECIMAL(18,4),
                `Total Value`    DECIMAL(18,4),
                `Total SKU Sold` INT,
                `Non Productive w.r.t Order` VARCHAR(255),
                `Close Reason`   VARCHAR(500),
                `Total Visits`   INT,
                `First Check In Date`  DATE NULL,
                `First Check In Time`  VARCHAR(50),
                `First Check Out Date` DATE NULL,
                `First Check Out Time` VARCHAR(50),
                `First Spent Time`     VARCHAR(50),
                `Total Spent Time`     VARCHAR(50),
                `Store Latitude`       DECIMAL(12,8),
                `Store Longitude`      DECIMAL(12,8),
                `Visit Latitude`       DECIMAL(12,8),
                `Visit Longitude`      DECIMAL(12,8),
                `Distance From Original Location (m)` DECIMAL(18,4),
                `Order Added From`     VARCHAR(255),
                row_hash         CHAR(64) NOT NULL,
                row_json         LONGTEXT,
                fetched_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uq_row_hash (row_hash)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS ordered_vs_delivered_rows (
                id               INT AUTO_INCREMENT PRIMARY KEY,
                `Report Date`    DATE NULL,
                `Account Label`  VARCHAR(50),
                `S.No#`          VARCHAR(50),
                `Distributor Name` VARCHAR(255),
                `Distributor Code` VARCHAR(100),
                `Store Name`     VARCHAR(255),
                `Store Code`     VARCHAR(100),
                `SKU Code`       VARCHAR(100),
                `SKU Name`       VARCHAR(255),
                `SKU Manufacturer Code` VARCHAR(100),
                `Category Code`  VARCHAR(100),
                `Category Name`  VARCHAR(255),
                `Order Booker Code` VARCHAR(100),
                `Order Booker Name` VARCHAR(255),
                `Deliveryman Code` VARCHAR(100),
                `Deliveryman Name` VARCHAR(255),
                `Order Number`   VARCHAR(255),
                `Invoice Number` VARCHAR(255),
                `Status`         VARCHAR(100),
                `Order Date`     DATE NULL,
                `Delivery Date`  DATE NULL,
                `Order Units`    DECIMAL(18,4),
                `Order (Grams)`  DECIMAL(18,4),
                `Order (ML)`     DECIMAL(18,4),
                `Order (KG)`     DECIMAL(18,4),
                `Order (Litres)` DECIMAL(18,4),
                `Order Amount`   DECIMAL(18,4),
                `Delivered Units` DECIMAL(18,4),
                `Delivered (Grams)` DECIMAL(18,4),
                `Delivered (ML)` DECIMAL(18,4),
                `Delivered (KG)` DECIMAL(18,4),
                `Delivered (Litres)` DECIMAL(18,4),
                `Delivered Amount` DECIMAL(18,4),
                `Returned Units` DECIMAL(18,4),
                `Returned Amount` DECIMAL(18,4),
                `Total Discount` DECIMAL(18,4),
                row_hash         CHAR(64) NOT NULL,
                row_json         LONGTEXT,
                fetched_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uq_row_hash (row_hash)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        # Backward-compatible migrations for existing visits_summary_rows table.
        await cur.execute(
            """
            SELECT COLUMN_NAME
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA=%s AND TABLE_NAME='end_stock_summary'
            """,
            (DB_NAME,),
        )
        end_stock_cols = {row[0].lower() for row in await cur.fetchall()}
        if "account_label" not in end_stock_cols:
            await cur.execute("ALTER TABLE end_stock_summary ADD COLUMN account_label VARCHAR(50)")

        await cur.execute(
            """
            SELECT INDEX_NAME
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA=%s AND TABLE_NAME='end_stock_summary'
            """,
            (DB_NAME,),
        )
        end_stock_indexes = {row[0] for row in await cur.fetchall()}
        if "uq_date_dist_sku" in end_stock_indexes and "uq_date_acc_dist_sku" not in end_stock_indexes:
            await cur.execute("ALTER TABLE end_stock_summary DROP INDEX uq_date_dist_sku")
            end_stock_indexes.discard("uq_date_dist_sku")
        if "uq_date_acc_dist_sku" not in end_stock_indexes:
            await cur.execute(
                "ALTER TABLE end_stock_summary ADD UNIQUE KEY uq_date_acc_dist_sku (report_date, account_label, distributor_code, sku_code)"
            )

        rename_columns = {
            "report_date": ("Report Date", "DATE NULL"),
            "account_label": ("Account Label", "VARCHAR(50)"),
            "visit_date": ("Visit Date", "DATE NULL"),
            "delivery_date": ("Delivery Date", "DATE NULL"),
            "distributor": ("Distributor", "VARCHAR(255)"),
            "pjp_name": ("PJP Name", "VARCHAR(255)"),
            "app_user": ("App User", "VARCHAR(255)"),
            "store_name": ("Store Name", "VARCHAR(255)"),
            "store_code": ("Store Code", "VARCHAR(100)"),
            "store_company_code": ("Store Company Code", "VARCHAR(100)"),
            "sync_down": ("Sync Down", "VARCHAR(50)"),
            "sync_down_date": ("Sync Down Date", "DATE NULL"),
            "sync_down_time": ("Sync Down Time", "VARCHAR(50)"),
            "sync_up_date": ("Sync Up Date", "DATE NULL"),
            "sync_up_time": ("Sync Up Time", "VARCHAR(50)"),
            "invoice_number": ("Invoice Number", "VARCHAR(255)"),
            "visit_complete": ("Visit Complete", "VARCHAR(100)"),
            "order_number": ("Order Number", "VARCHAR(255)"),
            "total_units": ("Total Units", "DECIMAL(18,4)"),
            "total_value": ("Total Value", "DECIMAL(18,4)"),
            "total_sku_sold": ("Total SKU Sold", "INT"),
            "non_productive_wrt_order": ("Non Productive w.r.t Order", "VARCHAR(255)"),
            "close_reason": ("Close Reason", "VARCHAR(500)"),
            "total_visits": ("Total Visits", "INT"),
            "first_check_in_date": ("First Check In Date", "DATE NULL"),
            "first_check_in_time": ("First Check In Time", "VARCHAR(50)"),
            "first_check_out_date": ("First Check Out Date", "DATE NULL"),
            "first_check_out_time": ("First Check Out Time", "VARCHAR(50)"),
            "first_spent_time": ("First Spent Time", "VARCHAR(50)"),
            "total_spent_time": ("Total Spent Time", "VARCHAR(50)"),
            "store_latitude": ("Store Latitude", "DECIMAL(12,8)"),
            "store_longitude": ("Store Longitude", "DECIMAL(12,8)"),
            "visit_latitude": ("Visit Latitude", "DECIMAL(12,8)"),
            "visit_longitude": ("Visit Longitude", "DECIMAL(12,8)"),
            "distance_from_original_location_m": ("Distance From Original Location (m)", "DECIMAL(18,4)"),
            "order_added_from": ("Order Added From", "VARCHAR(255)"),
        }

        columns_to_add = {
            "Report Date": "DATE NULL",
            "Account Label": "VARCHAR(50)",
            "Visit Date": "DATE NULL",
            "Delivery Date": "DATE NULL",
            "Distributor": "VARCHAR(255)",
            "PJP Name": "VARCHAR(255)",
            "App User": "VARCHAR(255)",
            "Store Name": "VARCHAR(255)",
            "Store Code": "VARCHAR(100)",
            "Store Company Code": "VARCHAR(100)",
            "Sync Down": "VARCHAR(50)",
            "Sync Down Date": "DATE NULL",
            "Sync Down Time": "VARCHAR(50)",
            "Sync Up Date": "DATE NULL",
            "Sync Up Time": "VARCHAR(50)",
            "Order Number": "VARCHAR(255)",
            "Invoice Number": "VARCHAR(255)",
            "Visit Complete": "VARCHAR(100)",
            "Total Units": "DECIMAL(18,4)",
            "Total Value": "DECIMAL(18,4)",
            "Total SKU Sold": "INT",
            "Non Productive w.r.t Order": "VARCHAR(255)",
            "Close Reason": "VARCHAR(500)",
            "Total Visits": "INT",
            "First Check In Date": "DATE NULL",
            "First Check In Time": "VARCHAR(50)",
            "First Check Out Date": "DATE NULL",
            "First Check Out Time": "VARCHAR(50)",
            "First Spent Time": "VARCHAR(50)",
            "Total Spent Time": "VARCHAR(50)",
            "Store Latitude": "DECIMAL(12,8)",
            "Store Longitude": "DECIMAL(12,8)",
            "Visit Latitude": "DECIMAL(12,8)",
            "Visit Longitude": "DECIMAL(12,8)",
            "Distance From Original Location (m)": "DECIMAL(18,4)",
            "Order Added From": "VARCHAR(255)",
        }

        await cur.execute(
            """
            SELECT COLUMN_NAME
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA=%s AND TABLE_NAME='visits_summary_rows'
            """,
            (DB_NAME,),
        )
        existing_cols = {row[0].lower() for row in await cur.fetchall()}

        for old_name, (new_name, col_def) in rename_columns.items():
            if old_name.lower() in existing_cols and new_name.lower() not in existing_cols:
                await cur.execute(
                    f"ALTER TABLE visits_summary_rows CHANGE COLUMN `{old_name}` `{new_name}` {col_def}"
                )
                existing_cols.discard(old_name.lower())
                existing_cols.add(new_name.lower())

        for col_name, col_def in columns_to_add.items():
            if col_name.lower() in existing_cols:
                continue
            await cur.execute(f"ALTER TABLE visits_summary_rows ADD COLUMN `{col_name}` {col_def}")

        await cur.execute(
            """
            SELECT COLUMN_NAME
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA=%s AND TABLE_NAME='ordered_vs_delivered_rows'
            """,
            (DB_NAME,),
        )
        ordered_existing_cols = {row[0].lower() for row in await cur.fetchall()}
        ordered_rename_columns = {
            "report_date": ("Report Date", "DATE NULL"),
            "account_label": ("Account Label", "VARCHAR(50)"),
            "order_from_date": ("Order From Date", "DATE NULL"),
            "order_to_date": ("Order To Date", "DATE NULL"),
            "s_no": ("S.No#", "VARCHAR(50)"),
            "distributor_name": ("Distributor Name", "VARCHAR(255)"),
            "distributor_code": ("Distributor Code", "VARCHAR(100)"),
            "store_name": ("Store Name", "VARCHAR(255)"),
            "store_code": ("Store Code", "VARCHAR(100)"),
            "sku_code": ("SKU Code", "VARCHAR(100)"),
            "sku_name": ("SKU Name", "VARCHAR(255)"),
            "sku_manufacturer_code": ("SKU Manufacturer Code", "VARCHAR(100)"),
            "category_code": ("Category Code", "VARCHAR(100)"),
            "category_name": ("Category Name", "VARCHAR(255)"),
            "order_booker_code": ("Order Booker Code", "VARCHAR(100)"),
            "order_booker_name": ("Order Booker Name", "VARCHAR(255)"),
            "deliveryman_code": ("Deliveryman Code", "VARCHAR(100)"),
            "deliveryman_name": ("Deliveryman Name", "VARCHAR(255)"),
            "order_number": ("Order Number", "VARCHAR(255)"),
            "invoice_number": ("Invoice Number", "VARCHAR(255)"),
            "status": ("Status", "VARCHAR(100)"),
            "order_date": ("Order Date", "DATE NULL"),
            "delivery_date": ("Delivery Date", "DATE NULL"),
            "order_units": ("Order Units", "DECIMAL(18,4)"),
            "order_grams": ("Order (Grams)", "DECIMAL(18,4)"),
            "order_ml": ("Order (ML)", "DECIMAL(18,4)"),
            "order_kg": ("Order (KG)", "DECIMAL(18,4)"),
            "order_litres": ("Order (Litres)", "DECIMAL(18,4)"),
            "order_amount": ("Order Amount", "DECIMAL(18,4)"),
            "delivered_units": ("Delivered Units", "DECIMAL(18,4)"),
            "delivered_grams": ("Delivered (Grams)", "DECIMAL(18,4)"),
            "delivered_ml": ("Delivered (ML)", "DECIMAL(18,4)"),
            "delivered_kg": ("Delivered (KG)", "DECIMAL(18,4)"),
            "delivered_litres": ("Delivered (Litres)", "DECIMAL(18,4)"),
            "delivered_amount": ("Delivered Amount", "DECIMAL(18,4)"),
            "returned_units": ("Returned Units", "DECIMAL(18,4)"),
            "returned_amount": ("Returned Amount", "DECIMAL(18,4)"),
            "total_discount": ("Total Discount", "DECIMAL(18,4)"),
        }

        for old_name, (new_name, col_def) in ordered_rename_columns.items():
            if old_name.lower() in ordered_existing_cols and new_name.lower() not in ordered_existing_cols:
                await cur.execute(
                    f"ALTER TABLE ordered_vs_delivered_rows CHANGE COLUMN `{old_name}` `{new_name}` {col_def}"
                )
                ordered_existing_cols.discard(old_name.lower())
                ordered_existing_cols.add(new_name.lower())

        ordered_drop_columns = [
            "order_from_date",
            "order_to_date",
            "Order From Date",
            "Order To Date",
        ]
        for col_name in ordered_drop_columns:
            if col_name.lower() in ordered_existing_cols:
                await cur.execute(f"ALTER TABLE ordered_vs_delivered_rows DROP COLUMN `{col_name}`")
                ordered_existing_cols.discard(col_name.lower())

        ordered_columns_to_add = {
            "Report Date": "DATE NULL",
            "Account Label": "VARCHAR(50)",
            "S.No#": "VARCHAR(50)",
            "Distributor Name": "VARCHAR(255)",
            "Distributor Code": "VARCHAR(100)",
            "Store Name": "VARCHAR(255)",
            "Store Code": "VARCHAR(100)",
            "SKU Code": "VARCHAR(100)",
            "SKU Name": "VARCHAR(255)",
            "SKU Manufacturer Code": "VARCHAR(100)",
            "Category Code": "VARCHAR(100)",
            "Category Name": "VARCHAR(255)",
            "Order Booker Code": "VARCHAR(100)",
            "Order Booker Name": "VARCHAR(255)",
            "Deliveryman Code": "VARCHAR(100)",
            "Deliveryman Name": "VARCHAR(255)",
            "Order Number": "VARCHAR(255)",
            "Invoice Number": "VARCHAR(255)",
            "Status": "VARCHAR(100)",
            "Order Date": "DATE NULL",
            "Delivery Date": "DATE NULL",
            "Order Units": "DECIMAL(18,4)",
            "Order (Grams)": "DECIMAL(18,4)",
            "Order (ML)": "DECIMAL(18,4)",
            "Order (KG)": "DECIMAL(18,4)",
            "Order (Litres)": "DECIMAL(18,4)",
            "Order Amount": "DECIMAL(18,4)",
            "Delivered Units": "DECIMAL(18,4)",
            "Delivered (Grams)": "DECIMAL(18,4)",
            "Delivered (ML)": "DECIMAL(18,4)",
            "Delivered (KG)": "DECIMAL(18,4)",
            "Delivered (Litres)": "DECIMAL(18,4)",
            "Delivered Amount": "DECIMAL(18,4)",
            "Returned Units": "DECIMAL(18,4)",
            "Returned Amount": "DECIMAL(18,4)",
            "Total Discount": "DECIMAL(18,4)",
            "row_hash": "CHAR(64) NOT NULL",
            "row_json": "LONGTEXT",
            "fetched_at": "DATETIME DEFAULT CURRENT_TIMESTAMP",
        }
        for col_name, col_def in ordered_columns_to_add.items():
            if col_name.lower() in ordered_existing_cols:
                continue
            await cur.execute(f"ALTER TABLE ordered_vs_delivered_rows ADD COLUMN `{col_name}` {col_def}")
    log.info("Database tables verified/created.")


async def get_last_saved_date(conn, table_name: str, account_label: str = "") -> Optional[date_type]:
    """Return the most recent report_date for a report table."""
    async with conn.cursor() as cur:
        if table_name == "visits_summary_rows":
            if account_label:
                await cur.execute(
                    "SELECT MAX(`Visit Date`) FROM visits_summary_rows WHERE `Account Label`=%s",
                    (account_label,),
                )
            else:
                await cur.execute("SELECT MAX(`Visit Date`) FROM visits_summary_rows")
        elif table_name == "ordered_vs_delivered_rows":
            if account_label:
                await cur.execute(
                    "SELECT MAX(`Order Date`) FROM ordered_vs_delivered_rows WHERE `Account Label`=%s",
                    (account_label,),
                )
            else:
                await cur.execute("SELECT MAX(`Order Date`) FROM ordered_vs_delivered_rows")
        else:
            if account_label:
                await cur.execute(
                    f"SELECT MAX(report_date) FROM {table_name} WHERE account_label=%s",
                    (account_label,),
                )
            else:
                await cur.execute(f"SELECT MAX(report_date) FROM {table_name}")
        row = await cur.fetchone()
        return row[0] if row and row[0] else None


async def save_rows(conn, rows: list[dict]) -> int:
    """Upsert unpivoted rows into end_stock_summary."""
    if not rows:
        return 0

    rows = rows[:-1]
    if not rows:
        return 0

    deduped: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row.get("report_date"),
            str(row.get("account_label", "") or "").strip(),
            str(row.get("distributor_code", "") or "").strip(),
            str(row.get("sku_code", "") or "").strip(),
        )
        deduped[key] = row

    unique_rows = list(deduped.values())
    if len(unique_rows) != len(rows):
        log.info(f"Deduplicated parsed rows before save: {len(rows)} -> {len(unique_rows)}")

    async with conn.cursor() as cur:
        sql = """
            INSERT INTO end_stock_summary
                (report_date, account_label, distributor_code, distributor_name,
                 sku_code, sku_description, brand_code, brand_name, value, unit)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                account_label    = VALUES(account_label),
                distributor_name = VALUES(distributor_name),
                sku_description  = VALUES(sku_description),
                brand_name       = VALUES(brand_name),
                value            = VALUES(value),
                fetched_at       = CURRENT_TIMESTAMP
        """
        data = [
            (
                r["report_date"],
                r.get("account_label", ""),
                r.get("distributor_code", ""),
                r.get("distributor_name", ""),
                r.get("sku_code", ""),
                r.get("sku_description", ""),
                r.get("brand_code", ""),
                r.get("brand_name", ""),
                r.get("value"),
                r.get("unit", "Value"),
            )
            for r in unique_rows
        ]
        await cur.executemany(sql, data)
    return len(unique_rows)


async def save_visit_rows(conn, rows: list[dict]) -> int:
    """Save Visits Summary rows as normalized JSON with dedup by row_hash."""
    if not rows:
        return 0

    def _pick(row_map: dict, key: str) -> str:
        return str(row_map.get(key, "") or "").strip()

    def _to_int(value) -> Optional[int]:
        number = _to_float(value)
        return int(number) if number is not None else None

    def _to_date(value: str) -> Optional[date_type]:
        if not value:
            return None
        return _parse_date_label(value)

    deduped: dict[str, dict] = {}
    for row in rows:
        row_map = row.get("row", {}) or {}
        hash_payload = {
            "row": row_map,
            "raw_row": row.get("raw_row", []) or [],
            "report_date": str(row.get("report_date") or ""),
            "invoice_number": str(row.get("invoice_number") or ""),
            "visit_complete": str(row.get("visit_complete") or ""),
            "account_label": str(row.get("account_label") or ""),
        }
        row_json = json.dumps(hash_payload, ensure_ascii=False, sort_keys=True)
        row_hash = hashlib.sha256(row_json.encode("utf-8")).hexdigest()
        deduped[row_hash] = {
            "report_date": row.get("report_date"),
            "account_label": str(row.get("account_label") or ""),
            "visit_date": _to_date(_pick(row_map, "Visit Date")),
            "delivery_date": _to_date(_pick(row_map, "Delivery Date")),
            "distributor": _pick(row_map, "Distributor"),
            "pjp_name": _pick(row_map, "PJP Name"),
            "app_user": _pick(row_map, "App User"),
            "store_name": _pick(row_map, "Store Name"),
            "store_code": _pick(row_map, "Store Code"),
            "store_company_code": _pick(row_map, "Store Company Code"),
            "sync_down": _pick(row_map, "Sync Down"),
            "sync_down_date": _to_date(_pick(row_map, "Sync Down Date")),
            "sync_down_time": _pick(row_map, "Sync Down Time"),
            "sync_up_date": _to_date(_pick(row_map, "Sync Up Date")),
            "sync_up_time": _pick(row_map, "Sync Up Time"),
            "invoice_number": row.get("invoice_number", ""),
            "visit_complete": row.get("visit_complete", ""),
            "order_number": _pick(row_map, "Order Number"),
            "total_units": _to_float(_pick(row_map, "Total Units")),
            "total_value": _to_float(_pick(row_map, "Total Value")),
            "total_sku_sold": _to_int(_pick(row_map, "Total SKU Sold")),
            "non_productive_wrt_order": _pick(row_map, "Non Productive w.r.t Order"),
            "close_reason": _pick(row_map, "Close Reason"),
            "total_visits": _to_int(_pick(row_map, "Total Visits")),
            "first_check_in_date": _to_date(_pick(row_map, "First Check In Date")),
            "first_check_in_time": _pick(row_map, "First Check In Time"),
            "first_check_out_date": _to_date(_pick(row_map, "First Check Out Date")),
            "first_check_out_time": _pick(row_map, "First Check Out Time"),
            "first_spent_time": _pick(row_map, "First Spent Time"),
            "total_spent_time": _pick(row_map, "Total Spent Time"),
            "store_latitude": _to_float(_pick(row_map, "Store Latitude")),
            "store_longitude": _to_float(_pick(row_map, "Store Longitude")),
            "visit_latitude": _to_float(_pick(row_map, "Visit Latitude")),
            "visit_longitude": _to_float(_pick(row_map, "Visit Longitude")),
            "distance_from_original_location_m": _to_float(_pick(row_map, "Distance From Original Location (m)")),
            "order_added_from": _pick(row_map, "Order Added From"),
            "row_hash": row_hash,
            "row_json": row_json,
        }

    payload = list(deduped.values())
    async with conn.cursor() as cur:
        sql = """
            INSERT INTO visits_summary_rows
                (
                    `Report Date`, `Account Label`, `Visit Date`, `Delivery Date`, `Distributor`, `PJP Name`, `App User`, `Store Name`, `Store Code`,
                    `Store Company Code`, `Sync Down`, `Sync Down Date`, `Sync Down Time`,
                    `Sync Up Date`, `Sync Up Time`, `Visit Complete`, `Order Number`, `Invoice Number`,
                    `Total Units`, `Total Value`, `Total SKU Sold`, `Non Productive w.r.t Order`,
                    `Close Reason`, `Total Visits`, `First Check In Date`, `First Check In Time`,
                    `First Check Out Date`, `First Check Out Time`, `First Spent Time`,
                    `Total Spent Time`, `Store Latitude`, `Store Longitude`, `Visit Latitude`,
                    `Visit Longitude`, `Distance From Original Location (m)`, `Order Added From`,
                    row_hash, row_json
                )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                `Report Date` = VALUES(`Report Date`),
                `Account Label` = VALUES(`Account Label`),
                `Visit Date` = VALUES(`Visit Date`),
                `Delivery Date` = VALUES(`Delivery Date`),
                `Distributor` = VALUES(`Distributor`),
                `PJP Name` = VALUES(`PJP Name`),
                `App User` = VALUES(`App User`),
                `Store Name` = VALUES(`Store Name`),
                `Store Code` = VALUES(`Store Code`),
                `Store Company Code` = VALUES(`Store Company Code`),
                `Sync Down` = VALUES(`Sync Down`),
                `Sync Down Date` = VALUES(`Sync Down Date`),
                `Sync Down Time` = VALUES(`Sync Down Time`),
                `Sync Up Date` = VALUES(`Sync Up Date`),
                `Sync Up Time` = VALUES(`Sync Up Time`),
                `Invoice Number` = VALUES(`Invoice Number`),
                `Visit Complete` = VALUES(`Visit Complete`),
                `Order Number` = VALUES(`Order Number`),
                `Total Units` = VALUES(`Total Units`),
                `Total Value` = VALUES(`Total Value`),
                `Total SKU Sold` = VALUES(`Total SKU Sold`),
                `Non Productive w.r.t Order` = VALUES(`Non Productive w.r.t Order`),
                `Close Reason` = VALUES(`Close Reason`),
                `Total Visits` = VALUES(`Total Visits`),
                `First Check In Date` = VALUES(`First Check In Date`),
                `First Check In Time` = VALUES(`First Check In Time`),
                `First Check Out Date` = VALUES(`First Check Out Date`),
                `First Check Out Time` = VALUES(`First Check Out Time`),
                `First Spent Time` = VALUES(`First Spent Time`),
                `Total Spent Time` = VALUES(`Total Spent Time`),
                `Store Latitude` = VALUES(`Store Latitude`),
                `Store Longitude` = VALUES(`Store Longitude`),
                `Visit Latitude` = VALUES(`Visit Latitude`),
                `Visit Longitude` = VALUES(`Visit Longitude`),
                `Distance From Original Location (m)` = VALUES(`Distance From Original Location (m)`),
                `Order Added From` = VALUES(`Order Added From`),
                row_json = VALUES(row_json),
                fetched_at = CURRENT_TIMESTAMP
        """
        await cur.executemany(
            sql,
            [
                (
                    item.get("report_date"),
                    item.get("account_label", ""),
                    item.get("visit_date"),
                    item.get("delivery_date"),
                    item.get("distributor", ""),
                    item.get("pjp_name", ""),
                    item.get("app_user", ""),
                    item.get("store_name", ""),
                    item.get("store_code", ""),
                    item.get("store_company_code", ""),
                    item.get("sync_down", ""),
                    item.get("sync_down_date"),
                    item.get("sync_down_time", ""),
                    item.get("sync_up_date"),
                    item.get("sync_up_time", ""),
                    item.get("visit_complete", ""),
                    item.get("order_number", ""),
                    item.get("invoice_number", ""),
                    item.get("total_units"),
                    item.get("total_value"),
                    item.get("total_sku_sold"),
                    item.get("non_productive_wrt_order", ""),
                    item.get("close_reason", ""),
                    item.get("total_visits"),
                    item.get("first_check_in_date"),
                    item.get("first_check_in_time", ""),
                    item.get("first_check_out_date"),
                    item.get("first_check_out_time", ""),
                    item.get("first_spent_time", ""),
                    item.get("total_spent_time", ""),
                    item.get("store_latitude"),
                    item.get("store_longitude"),
                    item.get("visit_latitude"),
                    item.get("visit_longitude"),
                    item.get("distance_from_original_location_m"),
                    item.get("order_added_from", ""),
                    item.get("row_hash"),
                    item.get("row_json", "{}"),
                )
                for item in payload
            ],
        )

    return len(payload)


async def save_ordered_vs_delivered_rows(conn, rows: list[dict]) -> int:
    """Save Ordered Vs Delivered rows as normalized JSON with dedup by row_hash."""
    if not rows:
        return 0

    def _pick(row_map: dict, key: str) -> str:
        return str(row_map.get(key, "") or "").strip()

    def _to_date(value: str) -> Optional[date_type]:
        if not value:
            return None
        normalized = _normalize_date_text(value)
        if normalized.lower() in {"(null)", "null", "", "-", "--", "n/a", "na"}:
            return None

        parsed = _parse_date_label(normalized, warn=False)
        if parsed is not None:
            return parsed

        candidate = normalized.replace("T", " ").replace("Z", "")
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%d-%m-%Y %H:%M:%S",
            "%d-%m-%Y %H:%M",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%d-%b-%Y",
            "%d-%B-%Y",
            "%d %b %Y",
            "%d %B %Y",
            "%b %d, %Y",
            "%B %d, %Y",
        ):
            try:
                return datetime.strptime(candidate, fmt).date()
            except ValueError:
                continue

        return None

    deduped: dict[str, dict] = {}
    for row in rows:
        row_map = row.get("row", {}) or {}
        hash_payload = {
            "row": row_map,
            "raw_row": row.get("raw_row", []) or [],
            "report_date": str(row.get("report_date") or ""),
            "account_label": str(row.get("account_label") or ""),
        }
        row_json = json.dumps(hash_payload, ensure_ascii=False, sort_keys=True)
        row_hash = hashlib.sha256(row_json.encode("utf-8")).hexdigest()
        deduped[row_hash] = {
            "report_date": row.get("report_date"),
            "account_label": str(row.get("account_label") or ""),
            "s_no": _pick(row_map, "S.No#"),
            "distributor_name": _pick(row_map, "Distributor Name"),
            "distributor_code": _pick(row_map, "Distributor Code"),
            "store_name": _pick(row_map, "Store Name"),
            "store_code": _pick(row_map, "Store Code"),
            "sku_code": _pick(row_map, "SKU Code"),
            "sku_name": _pick(row_map, "SKU Name"),
            "sku_manufacturer_code": _pick(row_map, "SKU Manufacturer Code"),
            "category_code": _pick(row_map, "Category Code"),
            "category_name": _pick(row_map, "Category Name"),
            "order_booker_code": _pick(row_map, "Order Booker Code"),
            "order_booker_name": _pick(row_map, "Order Booker Name"),
            "deliveryman_code": _pick(row_map, "Deliveryman Code"),
            "deliveryman_name": _pick(row_map, "Deliveryman Name"),
            "order_number": _pick(row_map, "Order Number"),
            "invoice_number": _pick(row_map, "Invoice Number"),
            "status": _pick(row_map, "Status"),
            "order_date": _to_date(_pick(row_map, "Order Date")),
            "delivery_date": _to_date(_pick(row_map, "Delivery Date")),
            "order_units": _to_float(_pick(row_map, "Order Units")),
            "order_grams": _to_float(_pick(row_map, "Order (Grams)")),
            "order_ml": _to_float(_pick(row_map, "Order (ML)")),
            "order_kg": _to_float(_pick(row_map, "Order (KG)")),
            "order_litres": _to_float(_pick(row_map, "Order (Litres)")),
            "order_amount": _to_float(_pick(row_map, "Order Amount")),
            "delivered_units": _to_float(_pick(row_map, "Delivered Units")),
            "delivered_grams": _to_float(_pick(row_map, "Delivered (Grams)")),
            "delivered_ml": _to_float(_pick(row_map, "Delivered (ML)")),
            "delivered_kg": _to_float(_pick(row_map, "Delivered (KG)")),
            "delivered_litres": _to_float(_pick(row_map, "Delivered (Litres)")),
            "delivered_amount": _to_float(_pick(row_map, "Delivered Amount")),
            "returned_units": _to_float(_pick(row_map, "Returned Units")),
            "returned_amount": _to_float(_pick(row_map, "Returned Amount")),
            "total_discount": _to_float(_pick(row_map, "Total Discount")),
            "row_hash": row_hash,
            "row_json": row_json,
        }

    payload = list(deduped.values())
    async with conn.cursor() as cur:
        sql = """
            INSERT INTO ordered_vs_delivered_rows
                (
                    `Report Date`, `Account Label`,
                    `S.No#`, `Distributor Name`, `Distributor Code`, `Store Name`, `Store Code`,
                    `SKU Code`, `SKU Name`, `SKU Manufacturer Code`, `Category Code`, `Category Name`,
                    `Order Booker Code`, `Order Booker Name`, `Deliveryman Code`, `Deliveryman Name`,
                    `Order Number`, `Invoice Number`, `Status`, `Order Date`, `Delivery Date`,
                    `Order Units`, `Order (Grams)`, `Order (ML)`, `Order (KG)`, `Order (Litres)`, `Order Amount`,
                    `Delivered Units`, `Delivered (Grams)`, `Delivered (ML)`, `Delivered (KG)`, `Delivered (Litres)`,
                    `Delivered Amount`, `Returned Units`, `Returned Amount`, `Total Discount`,
                    row_hash, row_json
                )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                `Report Date` = VALUES(`Report Date`),
                `Account Label` = VALUES(`Account Label`),
                `S.No#` = VALUES(`S.No#`),
                `Distributor Name` = VALUES(`Distributor Name`),
                `Distributor Code` = VALUES(`Distributor Code`),
                `Store Name` = VALUES(`Store Name`),
                `Store Code` = VALUES(`Store Code`),
                `SKU Code` = VALUES(`SKU Code`),
                `SKU Name` = VALUES(`SKU Name`),
                `SKU Manufacturer Code` = VALUES(`SKU Manufacturer Code`),
                `Category Code` = VALUES(`Category Code`),
                `Category Name` = VALUES(`Category Name`),
                `Order Booker Code` = VALUES(`Order Booker Code`),
                `Order Booker Name` = VALUES(`Order Booker Name`),
                `Deliveryman Code` = VALUES(`Deliveryman Code`),
                `Deliveryman Name` = VALUES(`Deliveryman Name`),
                `Order Number` = VALUES(`Order Number`),
                `Invoice Number` = VALUES(`Invoice Number`),
                `Status` = VALUES(`Status`),
                `Order Date` = VALUES(`Order Date`),
                `Delivery Date` = VALUES(`Delivery Date`),
                `Order Units` = VALUES(`Order Units`),
                `Order (Grams)` = VALUES(`Order (Grams)`),
                `Order (ML)` = VALUES(`Order (ML)`),
                `Order (KG)` = VALUES(`Order (KG)`),
                `Order (Litres)` = VALUES(`Order (Litres)`),
                `Order Amount` = VALUES(`Order Amount`),
                `Delivered Units` = VALUES(`Delivered Units`),
                `Delivered (Grams)` = VALUES(`Delivered (Grams)`),
                `Delivered (ML)` = VALUES(`Delivered (ML)`),
                `Delivered (KG)` = VALUES(`Delivered (KG)`),
                `Delivered (Litres)` = VALUES(`Delivered (Litres)`),
                `Delivered Amount` = VALUES(`Delivered Amount`),
                `Returned Units` = VALUES(`Returned Units`),
                `Returned Amount` = VALUES(`Returned Amount`),
                `Total Discount` = VALUES(`Total Discount`),
                row_json = VALUES(row_json),
                fetched_at = CURRENT_TIMESTAMP
        """
        await cur.executemany(
            sql,
            [
                (
                    item.get("report_date"),
                    item.get("account_label", ""),
                    item.get("s_no", ""),
                    item.get("distributor_name", ""),
                    item.get("distributor_code", ""),
                    item.get("store_name", ""),
                    item.get("store_code", ""),
                    item.get("sku_code", ""),
                    item.get("sku_name", ""),
                    item.get("sku_manufacturer_code", ""),
                    item.get("category_code", ""),
                    item.get("category_name", ""),
                    item.get("order_booker_code", ""),
                    item.get("order_booker_name", ""),
                    item.get("deliveryman_code", ""),
                    item.get("deliveryman_name", ""),
                    item.get("order_number", ""),
                    item.get("invoice_number", ""),
                    item.get("status", ""),
                    item.get("order_date"),
                    item.get("delivery_date"),
                    item.get("order_units"),
                    item.get("order_grams"),
                    item.get("order_ml"),
                    item.get("order_kg"),
                    item.get("order_litres"),
                    item.get("order_amount"),
                    item.get("delivered_units"),
                    item.get("delivered_grams"),
                    item.get("delivered_ml"),
                    item.get("delivered_kg"),
                    item.get("delivered_litres"),
                    item.get("delivered_amount"),
                    item.get("returned_units"),
                    item.get("returned_amount"),
                    item.get("total_discount"),
                    item.get("row_hash"),
                    item.get("row_json", "{}"),
                )
                for item in payload
            ],
        )

    return len(payload)


async def log_run(conn, run_date, status, rows_saved=0, message=""):
    async with conn.cursor() as cur:
        await cur.execute(
            "INSERT INTO bot_run_log (run_date, status, rows_saved, message) VALUES (%s,%s,%s,%s)",
            (run_date, status, rows_saved, message),
        )


# ══════════════════════════════════════════════════════════════════════════════
# BROWSER / SCRAPING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

async def login(page, username: str, password: str, account_label: str = "account"):
    log.info(f"[{account_label}] Navigating to login page...")
    login_url = "https://engrofoods.salesflo.com/OB/login/"
    nav_attempts = 3
    nav_timeout_ms = int(os.getenv("LOGIN_NAV_TIMEOUT_MS", "45000"))
    last_nav_error: Optional[Exception] = None
    for attempt in range(1, nav_attempts + 1):
        try:
            await page.goto(login_url, wait_until="domcontentloaded", timeout=nav_timeout_ms)
            last_nav_error = None
            break
        except PlaywrightTimeout as exc:
            last_nav_error = exc
            log.warning(
                f"[{account_label}] Login navigation timed out (attempt {attempt}/{nav_attempts}). Retrying..."
            )
            if attempt == nav_attempts:
                break
            await asyncio.sleep(3)

    if last_nav_error is not None:
        # Final fallback: accept partial load if the server is slow to finish domcontentloaded.
        try:
            await page.goto(login_url, wait_until="commit", timeout=nav_timeout_ms)
        except Exception:
            raise last_nav_error

    async def _iter_roots():
        return [page] + list(page.frames)

    async def _find_first(selectors: list[str], timeout_ms: int = 25000):
        end_time = time.time() + (timeout_ms / 1000)
        while time.time() < end_time:
            roots = await _iter_roots()
            for root in roots:
                for selector in selectors:
                    try:
                        loc = root.locator(selector).first
                        if await loc.count() > 0:
                            return loc
                    except Exception:
                        continue
            await asyncio.sleep(0.4)
        return None

    async def _find_login_inputs():
        username_selectors = [
            'input[name="username"]',
            'input#username',
            'input[name="email"]',
            'input#email',
            'input[name="mobile"]',
            'input[name*="user" i]',
            'input[id*="user" i]',
            'input[type="text"]',
        ]
        password_selectors = [
            'input[name="password"]',
            'input#password',
            'input[name*="pass" i]',
            'input[id*="pass" i]',
            'input[type="password"]',
        ]

        username_loc = await _find_first(username_selectors, timeout_ms=25000)
        password_loc = await _find_first(password_selectors, timeout_ms=25000)
        return username_loc, password_loc

    username_input, password_input = await _find_login_inputs()

    if username_input is None or password_input is None:
        # One hard refresh retry for transient/slow login shells in cloud.
        log.warning(f"[{account_label}] Login inputs not found on first pass. Retrying login page load...")
        await page.goto(login_url, wait_until="commit", timeout=nav_timeout_ms)
        username_input, password_input = await _find_login_inputs()

    if username_input is None or password_input is None:
        page_url = page.url
        page_title = ""
        try:
            page_title = await page.title()
        except Exception:
            page_title = ""

        # If already authenticated, continue without explicit login form.
        try:
            if await page.locator("text=Reports").count() > 0:
                log.info(f"[{account_label}] Login form not shown, but Reports is visible. Assuming active session.")
                return
        except Exception:
            pass

        raise RuntimeError(
            f"Login form not found. URL={page_url} Title={page_title}. "
            "Selectors for username/password did not appear."
        )

    async def _fast_fill(target, value: str):
        try:
            await target.fill(value, timeout=1200)
            return
        except Exception:
            pass

        await target.evaluate(
            """
            (el, val) => {
                el.removeAttribute('readonly');
                el.readOnly = false;
                el.disabled = false;
                el.focus();
                el.value = val;
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }
            """,
            value,
        )

    await _fast_fill(username_input, username)
    await _fast_fill(password_input, password)
    clicked = False
    submit_selectors = [
        'input[type="image"][src*="btnLogin.png"]',
        'input[type="image"][src*="btnLogin"]',
        'button[type="submit"]',
        'input[type="submit"]',
        'button:has-text("Login")',
        'button:has-text("Sign In")',
        'input[value*="Login" i]',
        'input[value*="Sign" i]',
        '.login-btn, .btn-login, [id*="login" i], [name*="login" i]',
    ]

    roots = await _iter_roots()
    for selector in submit_selectors:
        for root in roots:
            button = root.locator(selector).first
            if await button.count() == 0:
                continue
            try:
                await button.click(timeout=2000)
                clicked = True
                break
            except PlaywrightTimeout:
                continue
            except Exception:
                continue
        if clicked:
            break

    if not clicked:
        try:
            await password_input.press("Enter")
            clicked = True
        except Exception:
            pass

    if not clicked:
        submitted = await page.evaluate(
            """
            () => {
                const form = document.querySelector('form');
                if (!form) return false;
                if (typeof form.requestSubmit === 'function') {
                    form.requestSubmit();
                } else {
                    form.submit();
                }
                return true;
            }
            """
        )
        if not submitted:
            raise RuntimeError("Unable to submit login form: no clickable submit control found.")

    try:
        await page.wait_for_selector("text=Reports", timeout=30000)
    except PlaywrightTimeout:
        await page.wait_for_load_state("networkidle")
    log.info(f"[{account_label}] Logged in successfully.")


async def navigate_to_report(page, report_cfg: dict):
    log.info(f"Navigating to report: {report_cfg['title']}...")
    await page.goto("https://engrofoods.salesflo.com/OB/login", wait_until="networkidle")

    async def _is_report_ready() -> bool:
        selectors = report_cfg.get("ready_selectors", [])
        if not selectors:
            return False
        roots = [page] + list(page.frames)
        for root in roots:
            for selector in selectors:
                loc = root.locator(selector).first
                if await loc.count() > 0:
                    return True
        return False

    async def _click_first(selectors: list[str], pause: float = 0.6) -> bool:
        for selector in selectors:
            try:
                loc = page.locator(selector).first
                if await loc.count() > 0:
                    await loc.click(timeout=8000)
                    await asyncio.sleep(pause)
                    return True
            except Exception:
                continue
        return False

    # Open Reports shell first.
    await _click_first(["text=Reports", "a:has-text('Reports')"], pause=2.0)

    # Report-specific navigation through side menu + tile/button.
    if report_cfg.get("title") == "Visits Summary":
        await _click_first([
            "a:has-text('Visits Reports')",
            "li:has-text('Visits Reports') a",
            "text=Visits Reports",
        ], pause=1.0)
        await _click_first([
            "a:has-text('Visits Summary')",
            "text=Visits Summary",
            "* :has-text('Visits Summary')",
        ], pause=1.0)
    else:
        for step in report_cfg.get("nav_steps", []):
            if step.strip().lower() == "text=reports":
                continue
            try:
                await page.locator(step).first.click(timeout=8000)
                await asyncio.sleep(0.6)
            except Exception:
                pass

    if not await _is_report_ready():
        try:
            await page.goto(report_cfg.get("nav_page", SITE_URL), wait_until="networkidle")

            if report_cfg.get("title") == "Visits Summary":
                await _click_first([
                    "a:has-text('Visits Reports')",
                    "li:has-text('Visits Reports') a",
                    "text=Visits Reports",
                ], pause=1.0)
                await _click_first([
                    "a:has-text('Visits Summary')",
                    "text=Visits Summary",
                ], pause=1.0)
            else:
                report_name = report_cfg.get("title", "")
                if report_name:
                    await _click_first([f"text={report_name}", f"a:has-text('{report_name}')"], pause=0.8)
        except Exception:
            pass

    # Small settle time for dynamic panel rendering.
    for _ in range(6):
        if await _is_report_ready():
            break
        await asyncio.sleep(0.5)

    await page.wait_for_load_state("networkidle")
    log.info(f"On report page: {report_cfg['title']}")


async def set_filters(page, start_date: date_type, end_date: date_type, report_cfg: dict):
    """Set common report filters with report-specific toggles from config."""
    start_str = start_date.strftime("%m/%d/%Y")
    end_str   = end_date.strftime("%m/%d/%Y")
    start_str_long = start_date.strftime("%B %d, %Y")
    end_str_long   = end_date.strftime("%B %d, %Y")

    log.info(f"Setting filters: {start_str} -> {end_str}")

    def _roots():
        return [page] + list(page.frames)

    async def _find_target(selectors: list[str]):
        for root in _roots():
            for selector in selectors:
                candidate = root.locator(selector).first
                if await candidate.count() > 0:
                    return candidate
        return None

    async def _select_radio(selectors: list[str], label: str, group_name: str, expected_value: str) -> bool:
        target = None

        # Wait briefly for controls to render (Salesflo can load params asynchronously).
        for _ in range(12):
            target = await _find_target(selectors)
            if target is not None:
                break
            await asyncio.sleep(0.5)

        if target is None:
            # Fallback: select by radio group name + value.
            for root in _roots():
                by_value = root.locator(f'input[type="radio"][name="{group_name}"][value="{expected_value}"]').first
                if await by_value.count() > 0:
                    target = by_value
                    break

                radios = root.locator(f'input[type="radio"][name="{group_name}"]')
                radio_count = await radios.count()
                if radio_count > 0:
                    # For TP, Daily is usually second option if value mapping is missing.
                    target = radios.nth(1) if group_name == "TP" and radio_count > 1 else radios.first
                    break

        if target is None:
            log.warning(f"{label} option not found.")
            return False

        try:
            await target.check(timeout=2000)
        except Exception:
            try:
                await target.click(timeout=2000)
            except Exception:
                await target.evaluate(
                    """
                    (el) => {
                        el.removeAttribute('disabled');
                        el.disabled = false;
                        el.checked = true;
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                    """
                )

        checked = await target.is_checked()
        if not checked:
            log.warning(f"Could not confirm {label} option selected.")
            return False

        log.info(f"Selected {label} option.")
        return True

    async def _set_checkbox(selectors: list[str], label: str, checked: bool = True) -> bool:
        target = None
        for _ in range(12):
            target = await _find_target(selectors)
            if target is not None:
                break
            await asyncio.sleep(0.5)

        if target is None:
            log.warning(f"{label} checkbox not found.")
            return False

        try:
            if checked:
                await target.check(timeout=3000)
            else:
                await target.uncheck(timeout=3000)
        except Exception:
            try:
                await target.click(timeout=2000)
            except Exception:
                await target.evaluate(
                    """
                    (el, desired) => {
                        el.removeAttribute('disabled');
                        el.disabled = false;
                        el.checked = !!desired;
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                    """,
                    checked,
                )

        try:
            current = await target.is_checked()
            if current != checked:
                log.warning(f"Could not set {label} checkbox to {checked}.")
                return False
        except Exception:
            pass

        log.info(f"Set {label} checkbox: {checked}.")
        return True

    if report_cfg.get("require_daily", False):
        await _select_radio(
            [
                'input#Daily',
                'input[name="TP"][value="2"]',
                'input[name="TP"][id*="Daily" i]',
                'label:has-text("Daily") input[type="radio"]',
                'input[type="radio"][value="Daily" i]',
            ],
            "Daily",
            "TP",
            "2",
        )

    if report_cfg.get("require_summary", False):
        await _select_radio(
            [
                'input#Summary',
                'input[name="TOR"][value="1"]',
                'input[name="TOR"][id*="Summary" i]',
                'label:has-text("Summary") input[type="radio"]',
                'input[type="radio"][value="Summary" i]',
            ],
            "Summary",
            "TOR",
            "1",
        )

    qty_type_value = report_cfg.get("qty_type_value")
    if qty_type_value:
        qty_value = str(qty_type_value)
        unit_label = "Unit" if qty_value == "3" else f"Value {qty_value}"
        await _select_radio(
            [
                f'input[name="QTY"][value="{qty_value}"]',
                f'input#Unit' if qty_value == "3" else f'input[id*="{qty_value}"]',
                f'label:has-text("{unit_label}") input[type="radio"]',
            ],
            f"QTY Type {unit_label}",
            "QTY",
            qty_value,
        )

    async def _fill_date(selectors: list[str], value_primary: str, value_alt: str, label: str):
        target = None
        for _ in range(30):
            target = await _find_target(selectors)
            if target is not None:
                break
            await asyncio.sleep(0.5)

        if target is None:
            # Fallback for Visits Summary layout: date row contains two text inputs.
            for root in _roots():
                try:
                    date_row_inputs = root.locator(
                        '#dateTr input[type="text"], tr#dateTr input, input.dateFrom, input.dateTo, input.hasDatepicker'
                    )
                    count = await date_row_inputs.count()
                    if count > 0:
                        if "start" in label.lower():
                            target = date_row_inputs.nth(0)
                        else:
                            target = date_row_inputs.nth(1) if count > 1 else date_row_inputs.nth(0)
                        break
                except Exception:
                    continue

        if target is None:
            raise RuntimeError(f"{label} field not found.")

        try:
            await target.click(timeout=5000)
        except Exception:
            pass

        try:
            await target.fill("")
            await target.fill(value_primary)
            await target.press("Enter")
            return
        except Exception:
            pass

        try:
            await target.fill("")
            await target.fill(value_alt)
            await target.press("Enter")
            return
        except Exception:
            pass

        await target.evaluate(
            """
            (el, val) => {
                el.removeAttribute('readonly');
                el.readOnly = false;
                el.disabled = false;
                el.focus();
                el.value = val;
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
                el.dispatchEvent(new KeyboardEvent('keyup', { key: 'Enter', bubbles: true }));
            }
            """,
            value_primary,
        )

    # Start Date (Salesflo expects format like "February 27, 2026")
    await _fill_date(
        [
            'input[name="std2"]',
            'input#std2',
            'input[name="dt1"]',
            'input#dt1',
            'input[name*="start" i]',
            'input[id*="start" i]',
            'input[placeholder*="Start" i]',
            'input#startDate',
        ],
        start_str_long,
        start_str,
        "Start date",
    )

    # End Date (Salesflo expects format like "February 27, 2026")
    await _fill_date(
        [
            'input[name="end2"]',
            'input#end2',
            'input[name="dt2"]',
            'input#dt2',
            'input[name*="end" i]',
            'input[id*="end" i]',
            'input[placeholder*="End" i]',
            'input#endDate',
        ],
        end_str_long,
        end_str,
        "End date",
    )

    if report_cfg.get("visit_complete_both", False):
        try:
            yes_visit_complete = await _find_target(['input#ShowVisitComplete_1', 'input[name="ShowVisitComplete"][value="1"]'])
            no_visit_complete = await _find_target(['input#ShowVisitComplete_0', 'input[name="ShowVisitComplete"][value="0"]'])
            if await yes_visit_complete.count() > 0:
                await yes_visit_complete.check(timeout=3000)
            if await no_visit_complete.count() > 0:
                await no_visit_complete.check(timeout=3000)
        except Exception:
            log.warning("Visit Complete checkboxes not found or not clickable.")

    if report_cfg.get("show_invoice", False):
        try:
            show_invoice = await _find_target(['input#ShowInvoiceNumber_1', 'input[name="ShowInvoiceNumber"][value="1"]'])
            if show_invoice is not None and await show_invoice.count() > 0:
                await show_invoice.check(timeout=3000)
            else:
                log.warning("Show Invoice Number checkbox not found.")
        except Exception:
            log.warning("Could not set Show Invoice Number checkbox.")

    stock_unit_value = report_cfg.get("stock_unit_value")
    if stock_unit_value:
        try:
            stock_unit = await _find_target(['select#su', 'select[name="su"]', 'select[name*="unit"]', 'select[id*="unit"]'])
            await stock_unit.select_option(value=str(stock_unit_value))
        except Exception:
            try:
                stock_unit = await _find_target(['select#su', 'select[name="su"]', 'select[name*="unit"]', 'select[id*="unit"]'])
                await stock_unit.select_option(label="Value")
            except Exception:
                log.warning("Stock unit dropdown not found or could not be set.")

    for status_value in report_cfg.get("order_status_unchecked", []):
        value = str(status_value)
        await _set_checkbox(
            [
                f'input.ord_chk_boxes[value="{value}"]',
                f'input[name*="ord"][value="{value}"]',
            ],
            f"Order status {value}",
            checked=False,
        )

    if report_cfg.get("show_delivery_man", False):
        await _set_checkbox(
            [
                'input#ShowDeliveryman_1',
                'input[name="ShowDeliveryMan"][value="1"]',
                'input.ShowDeliveryMan',
            ],
            "Show Delivery man",
            checked=True,
        )

    if report_cfg.get("show_sku_weight", False):
        await _set_checkbox(
            [
                'input#show_sku_weight',
                'input[name="show_sku_weight"]',
                'tr#ShowWeightTypes input[type="checkbox"]',
            ],
            "Show SKU Weight Type",
            checked=True,
        )

    log.info("Filters set.")


async def click_view_report_and_wait(page) -> bool:
    """Click View Report, poll until Completed (max 5 min)."""
    log.info("Clicking 'View Report'...")
    await page.click("button:has-text('View Report'), input[value='View Report']")

    async def _handle_report_method_popup():
        popup_title = page.locator("text=Please Select Report Method")
        if await popup_title.count() == 0:
            return

        download_btn = page.locator(
            "button:has-text('Download'), a:has-text('Download'), input[value='Download']"
        ).first

        try:
            await download_btn.click(timeout=3000)
            log.info("Selected report method: Download")
        except Exception:
            pass

    await _handle_report_method_popup()

    timeout, poll_interval, elapsed = 300, 10, 0
    while elapsed < timeout:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
        await _handle_report_method_popup()
        if await page.locator("text=Completed").count() > 0:
            log.info(f"Report completed after ~{elapsed}s.")
            return True
        log.info(f"Still pending... ({elapsed}s)")

    log.error("Report did not complete within timeout.")
    return False


async def generate_and_parse(page, start_date: date_type, end_date: date_type, parse_mode: str) -> list[dict]:
    """Click View Report -> Generate and parse report table from generated window."""
    page_load_timeout_ms = int(os.getenv("REPORT_PAGE_LOAD_TIMEOUT_MS", "120000"))
    end_stock_wait_seconds = int(os.getenv("END_STOCK_GENERATE_WAIT_SECONDS", "240"))
    visits_wait_seconds = int(os.getenv("VISITS_GENERATE_WAIT_SECONDS", "420"))
    ordered_wait_seconds = int(os.getenv("ORDERED_GENERATE_WAIT_SECONDS", "420"))

    log.info("Clicking 'View Report' and selecting Generate...")
    existing_pages = list(page.context.pages)
    await page.click("button:has-text('View Report'), input[value='View Report']")

    report_page = page
    modal_visible = False
    try:
        await page.wait_for_selector("text=Please Select Report Method", timeout=6000)
        modal_visible = True
    except PlaywrightTimeout:
        modal_visible = False

    if modal_visible:
        generate_btn = page.locator(
            "button:has-text('Generate'), a:has-text('Generate'), input[value='Generate']"
        ).first
        try:
            async with page.context.expect_page(timeout=15000) as page_info:
                await generate_btn.click(timeout=5000)
            report_page = await page_info.value
        except PlaywrightTimeout:
            await generate_btn.click(timeout=5000)
    else:
        # Some runs skip the method modal and open report window directly.
        for _ in range(30):
            new_pages = [p for p in page.context.pages if p not in existing_pages]
            if new_pages:
                report_page = new_pages[-1]
                break
            await asyncio.sleep(0.5)

    try:
        try:
            await report_page.wait_for_load_state("domcontentloaded", timeout=page_load_timeout_ms)
        except PlaywrightTimeout:
            log.warning("Generated report page did not reach domcontentloaded in time; continuing with available DOM.")
        async def _read_table_payload(root) -> dict:
            try:
                return await root.evaluate(
                    r"""
                    () => {
                        const clean = (value) => (value || '').replace(/\s+/g, ' ').trim();
                        const pageText = clean(document.body?.innerText || '').toLowerCase();
                        const noRecordsPatterns = [
                            'sorry! no record found',
                            'sorry! no records found',
                            'sorry no record found',
                            'sorry no records found',
                            'no record found',
                            'no records found',
                        ];
                        const noRecords = noRecordsPatterns.some((p) => pageText.includes(p));

                        const tables = Array.from(document.querySelectorAll('#StickyTable, .ui-jqgrid-btable, table'));
                        if (!tables.length) return { rows: [], externalHeader: [], noRecords };

                        const tablePayloads = tables.map((table) => {
                            const rows = Array.from(table.querySelectorAll('tr')).map((tr) =>
                                Array.from(tr.querySelectorAll('th,td')).map((cell) => clean(cell.innerText || cell.textContent))
                            ).filter((r) => r.length > 0);

                            const meaningfulRows = rows.filter((r) => {
                                const joined = clean(r.join(' ')).toLowerCase();
                                if (!joined) return false;
                                if (joined === 'print' || joined.startsWith('print ')) return false;
                                if (joined.includes('please select report method')) return false;
                                return true;
                            });

                            const rowCount = meaningfulRows.length;
                            const maxCols = meaningfulRows.reduce((m, r) => Math.max(m, r.length), 0);

                            let externalHeader = [];
                            const gridRoot = table.closest('.ui-jqgrid-view') || table.closest('.ui-jqgrid');
                            if (gridRoot) {
                                externalHeader = Array.from(gridRoot.querySelectorAll('.ui-jqgrid-htable th'))
                                    .map((cell) => clean(cell.innerText || cell.textContent))
                                    .filter(Boolean);
                            }
                            if (!externalHeader.length) {
                                externalHeader = Array.from(table.querySelectorAll('thead th'))
                                    .map((cell) => clean(cell.innerText || cell.textContent))
                                    .filter(Boolean);
                            }

                            const score = (rowCount * 100) + maxCols + (externalHeader.length * 5);
                            return { rows: meaningfulRows, externalHeader, score, noRecords };
                        });

                        tablePayloads.sort((a, b) => b.score - a.score);
                        return tablePayloads[0] || { rows: [], externalHeader: [], noRecords };
                    }
                    """
                )
            except Exception:
                return {"rows": [], "externalHeader": [], "noRecords": False}

        async def _is_table_loading(root) -> bool:
            try:
                return await root.evaluate(
                    """
                    () => {
                        const loadingSelectors = [
                            '.ui-jqgrid-loading',
                            '.loading',
                            '.loader',
                            '.spinner',
                            '[aria-busy="true"]'
                        ];

                        for (const selector of loadingSelectors) {
                            const nodes = Array.from(document.querySelectorAll(selector));
                            if (nodes.some((el) => {
                                const style = window.getComputedStyle(el);
                                return style && style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
                            })) {
                                return true;
                            }
                        }

                        const pageText = (document.body?.innerText || '').toLowerCase();
                        return pageText.includes('loading') || pageText.includes('please wait') || pageText.includes('processing');
                    }
                    """
                )
            except Exception:
                return False

        def _payload_signature(payload: dict) -> tuple:
            rows = payload.get("rows", []) or []
            header = payload.get("externalHeader", []) or []
            head = tuple(tuple(str(c) for c in r[:4]) for r in rows[:2])
            tail = tuple(tuple(str(c) for c in r[:4]) for r in rows[-2:])
            return (len(rows), len(header), head, tail)

        table_payload = {"rows": [], "externalHeader": [], "noRecords": False}
        last_signature = None
        stable_polls = 0
        if parse_mode == "visits":
            max_wait_seconds = visits_wait_seconds
        elif parse_mode == "ordered_vs_delivered":
            max_wait_seconds = ordered_wait_seconds
        else:
            max_wait_seconds = end_stock_wait_seconds
        no_records_detected = False

        for _ in range(max_wait_seconds):
            roots = [report_page] + list(report_page.frames)
            loading_detected = False
            best_payload = table_payload

            for root in roots:
                payload = await _read_table_payload(root)
                if await _is_table_loading(root):
                    loading_detected = True
                if payload.get("noRecords"):
                    no_records_detected = True

                if payload.get("rows") or payload.get("externalHeader"):
                    if (
                        not best_payload.get("rows")
                        and not best_payload.get("externalHeader")
                    ):
                        best_payload = payload
                    else:
                        current_size = len(best_payload.get("rows", [])) + len(best_payload.get("externalHeader", []))
                        candidate_size = len(payload.get("rows", [])) + len(payload.get("externalHeader", []))
                        if candidate_size >= current_size:
                            best_payload = payload

            if best_payload.get("rows") or best_payload.get("externalHeader"):
                table_payload = best_payload
                current_signature = _payload_signature(table_payload)
                if not loading_detected and current_signature == last_signature:
                    stable_polls += 1
                else:
                    stable_polls = 0
                    last_signature = current_signature

                if stable_polls >= 2:
                    break

            await asyncio.sleep(1)

        table_rows = table_payload.get("rows", [])
        external_header = table_payload.get("externalHeader", [])

        if no_records_detected and not table_rows and not external_header:
            log.info("Report returned no records for selected filters/date range.")
            return []

        if not table_rows and not external_header:
            raise RuntimeError("Generated report table not found after waiting in page/frames.")

        header_idx = None
        if parse_mode == "end_stock":
            required_header_terms = ["distributor code", "sku code"]
        elif parse_mode == "ordered_vs_delivered":
            required_header_terms = ["order", "deliver"]
        else:
            required_header_terms = ["visit", "invoice"]

        def _end_stock_header_score(cells: list[str]) -> int:
            norm_cells = [_normalize_header_label(c) for c in cells]
            score = 0
            score += 2 if any("sku code" in c for c in norm_cells) else 0
            score += 2 if any("sku description" in c for c in norm_cells) else 0
            score += 1 if any("distributor" in c for c in norm_cells) else 0
            score += 1 if any("brand" in c for c in norm_cells) else 0
            score += 2 * sum(1 for c in norm_cells if _parse_date_label(c, warn=False) is not None)
            return score

        def _visits_header_score(cells: list[str]) -> int:
            score = 0
            for col in cells:
                normalized = _normalize_header_label(col)
                if normalized in VISITS_COLUMN_ALIASES:
                    score += 1
            return score

        def _is_visits_header_candidate(cells: list[str]) -> bool:
            score = _visits_header_score(cells)
            norm_cells = [_normalize_header_label(c) for c in cells]
            has_date_term = any("date" in c for c in norm_cells)
            has_visit_or_invoice = any(("visit" in c) or ("invoice" in c) for c in norm_cells)
            return score >= 4 or (score >= 2 and has_date_term and has_visit_or_invoice)

        def _ordered_header_score(cells: list[str]) -> int:
            norm_cells = [_normalize_header_label(c) for c in cells]
            score = 0
            score += 2 if any("order" in c for c in norm_cells) else 0
            score += 2 if any("deliver" in c for c in norm_cells) else 0
            score += 1 if any("sku" in c for c in norm_cells) else 0
            score += 1 if any("store" in c for c in norm_cells) else 0
            score += 1 if any("date" in c for c in norm_cells) else 0
            score += sum(1 for c in norm_cells if c in ORDERED_VS_DELIVERED_ALIASES)
            return score

        for idx, row in enumerate(table_rows):
            norm = [str(col).strip().lower() for col in row]
            if parse_mode == "visits":
                if _is_visits_header_candidate([str(col) for col in row]):
                    header_idx = idx
                    break
            elif parse_mode == "ordered_vs_delivered":
                if _ordered_header_score([str(col) for col in row]) >= 3:
                    header_idx = idx
                    break
            else:
                score = _end_stock_header_score([str(col) for col in row])
                if all(any(term in col for col in norm) for term in required_header_terms) or score >= 4:
                    header_idx = idx
                    break

        synthetic_header_used = False

        if header_idx is not None:
            header = [str(col).strip() for col in table_rows[header_idx]]
            data_rows = table_rows[header_idx + 1 :]
        else:
            norm_external = [str(col).strip().lower() for col in external_header]
            if parse_mode == "visits":
                if external_header and _is_visits_header_candidate([str(col) for col in external_header]):
                    header = [str(col).strip() for col in external_header]
                    data_rows = table_rows
                elif table_rows:
                    widest_row_len = max(len(r) for r in table_rows)
                    if widest_row_len >= len(VISITS_SUMMARY_COLUMNS):
                        header = ["S#"] + VISITS_SUMMARY_COLUMNS
                    else:
                        header = VISITS_SUMMARY_COLUMNS[:widest_row_len]
                    data_rows = table_rows
                    synthetic_header_used = True
                    log.warning("Could not locate explicit Visits header row; using canonical Visits column mapping.")
                else:
                    raise RuntimeError("Could not locate header row in generated report table.")
            elif parse_mode == "ordered_vs_delivered":
                if external_header and _ordered_header_score([str(col) for col in external_header]) >= 3:
                    header = [str(col).strip() for col in external_header]
                    data_rows = table_rows
                elif table_rows:
                    widest_row_len = max(len(r) for r in table_rows)
                    header = ORDERED_VS_DELIVERED_COLUMNS[:widest_row_len]
                    data_rows = table_rows
                    synthetic_header_used = True
                    log.warning("Could not locate explicit Ordered Vs Delivered header row; using report column mapping.")
                else:
                    raise RuntimeError("Could not locate header row in generated report table.")
            elif external_header and _end_stock_header_score([str(col) for col in external_header]) >= 4:
                header = [str(col).strip() for col in external_header]
                data_rows = table_rows
            elif external_header and all(any(term in col for col in norm_external) for term in required_header_terms):
                header = [str(col).strip() for col in external_header]
                data_rows = table_rows
            elif table_rows:
                widest_row_len = max(len(r) for r in table_rows)
                if parse_mode == "end_stock":
                    fixed = [
                        "S#",
                        "Distributor Code",
                        "Distributor Name",
                        "SKU Description",
                        "SKU Code",
                        "Brand Name",
                        "Brand Code",
                    ]
                    if widest_row_len > len(fixed):
                        extra_cols = widest_row_len - len(fixed)
                        range_days = (end_date - start_date).days + 1
                        if range_days >= extra_cols:
                            inferred_start = start_date
                        else:
                            inferred_start = end_date - timedelta(days=extra_cols - 1)
                        inferred_dates = [inferred_start + timedelta(days=i) for i in range(extra_cols)]
                        synthetic_dates = [d.isoformat() for d in inferred_dates]
                        header = fixed + synthetic_dates
                    else:
                        header = fixed[:widest_row_len]
                else:
                    header = [f"col_{i}" for i in range(widest_row_len)]
                data_rows = table_rows
                synthetic_header_used = True
                log.warning("Could not locate explicit header row; using synthetic header mapping.")
            else:
                raise RuntimeError("Could not locate header row in generated report table.")

        rows: list[dict] = []
        normalized_header = [str(h).strip() for h in header]

        def _lookup_value(row_map: dict, keys: list[str]) -> str:
            for key, value in row_map.items():
                lk = key.lower()
                if any(k in lk for k in keys):
                    return str(value or "").strip()
            return ""

        if parse_mode == "end_stock":
            norm_header = [h.lower() for h in normalized_header]

            def _find_col(*keys: str) -> Optional[int]:
                for i, col in enumerate(norm_header):
                    if any(k in col for k in keys):
                        return i
                return None

            idx_dist_code = _find_col("distributor code")
            idx_dist_name = _find_col("distributor name")
            idx_sku_desc = _find_col("sku description")
            idx_sku_code = _find_col("sku code")
            idx_brand_name = _find_col("brand name")
            idx_brand_code = _find_col("brand code")

            date_indices = [i for i, label in enumerate(normalized_header) if _parse_date_label(label, warn=False) is not None]
            if not date_indices:
                start_from = (idx_brand_code + 1) if idx_brand_code is not None else 7
                date_indices = list(range(start_from, len(normalized_header)))

            for row in data_rows:
                if not row or all((str(v).strip() == "") for v in row):
                    continue
                first_cell = str(row[0]).strip().lower() if len(row) > 0 else ""
                if "total" in first_cell:
                    continue

                def _cell(i: Optional[int]) -> str:
                    if i is None or i >= len(row):
                        return ""
                    return str(row[i] or "").strip()

                for col_idx in date_indices:
                    if col_idx >= len(row):
                        continue
                    date_label = normalized_header[col_idx] if col_idx < len(normalized_header) else ""
                    parsed_dt = _parse_date_label(date_label) or end_date
                    rows.append(
                        {
                            "report_date": parsed_dt,
                            "distributor_code": _cell(idx_dist_code),
                            "distributor_name": _cell(idx_dist_name),
                            "sku_code": _cell(idx_sku_code),
                            "sku_description": _cell(idx_sku_desc),
                            "brand_code": _cell(idx_brand_code),
                            "brand_name": _cell(idx_brand_name),
                            "value": _to_float(row[col_idx]),
                            "unit": "Value",
                        }
                    )
        elif parse_mode == "visits":
            for row in data_rows:
                if not row or all((str(v).strip() == "") for v in row):
                    continue
                first_cell = str(row[0]).strip().lower() if len(row) > 0 else ""
                if "total" in first_cell:
                    continue

                full_row_text = " ".join(str(v or "").strip().lower() for v in row)
                if full_row_text in {"print", ""} or full_row_text.startswith("print "):
                    continue

                row_map_raw: dict[str, str] = {}
                for idx, col_name in enumerate(normalized_header):
                    if idx < len(row):
                        row_map_raw[col_name or f"col_{idx}"] = str(row[idx] or "").strip()

                row_map: dict[str, str] = {col: "" for col in VISITS_SUMMARY_COLUMNS}
                for source_key, source_val in row_map_raw.items():
                    normalized_key = _normalize_header_label(source_key)
                    canonical_key = VISITS_COLUMN_ALIASES.get(normalized_key)
                    if canonical_key:
                        row_map[canonical_key] = source_val

                date_text = (
                    row_map.get("Visit Date", "")
                    or row_map.get("Delivery Date", "")
                    or _lookup_value(row_map_raw, ["visit date", "delivery date", "date"])
                )
                parsed_date = _parse_date_label(date_text, warn=False) if date_text else None

                row_is_meaningful = any(
                    row_map.get(field, "").strip()
                    for field in ("Distributor", "PJP Name", "Store Name", "Store Code", "Order Number", "Invoice Number")
                )
                if not row_is_meaningful:
                    continue

                if parsed_date is None:
                    if date_text and _normalize_date_text(date_text).lower() not in {"print", ""}:
                        _parse_date_label(date_text, warn=True)
                    parsed_date = end_date

                rows.append(
                    {
                        "report_date": parsed_date,
                        "invoice_number": row_map.get("Invoice Number", ""),
                        "visit_complete": row_map.get("Visit Complete", ""),
                        "raw_row": [str(v or "").strip() for v in row],
                        "row": row_map,
                    }
                )
        else:
            for row in data_rows:
                if not row or all((str(v).strip() == "") for v in row):
                    continue
                first_cell = str(row[0]).strip().lower() if len(row) > 0 else ""
                if "total" in first_cell:
                    continue

                full_row_text = " ".join(str(v or "").strip().lower() for v in row)
                if full_row_text in {"print", ""} or full_row_text.startswith("print "):
                    continue

                row_map_raw: dict[str, str] = {}
                for idx, col_name in enumerate(normalized_header):
                    if idx < len(row):
                        key = str(col_name or f"Column {idx + 1}").strip() or f"Column {idx + 1}"
                        row_map_raw[key] = str(row[idx] or "").strip()

                row_map: dict[str, str] = {col: "" for col in ORDERED_VS_DELIVERED_COLUMNS}
                for source_key, source_val in row_map_raw.items():
                    normalized_source = _normalize_header_label(source_key)
                    canonical_key = ORDERED_VS_DELIVERED_ALIASES.get(normalized_source)
                    if canonical_key is None:
                        cleaned_source = re.sub(r"[^a-z0-9]+", " ", normalized_source).strip()
                        for alias_key, alias_target in ORDERED_VS_DELIVERED_ALIASES.items():
                            cleaned_alias = re.sub(r"[^a-z0-9]+", " ", alias_key).strip()
                            if cleaned_source == cleaned_alias or cleaned_source in cleaned_alias or cleaned_alias in cleaned_source:
                                canonical_key = alias_target
                                break
                    if canonical_key:
                        row_map[canonical_key] = source_val

                row_is_meaningful = any(
                    row_map.get(field, "").strip()
                    for field in ("Distributor Name", "Store Name", "SKU Code", "Order Number", "Invoice Number")
                )
                if not row_is_meaningful:
                    continue

                date_text = row_map.get("Order Date", "") or row_map.get("Delivery Date", "")
                parsed_date = _parse_date_label(date_text, warn=False) if date_text else None
                if parsed_date is None:
                    parsed_date = end_date

                rows.append(
                    {
                        "report_date": parsed_date,
                        "raw_row": [str(v or "").strip() for v in row],
                        "row": row_map,
                    }
                )

        log.info(f"Generated report parsed -> {len(rows)} rows.")
        return rows
    finally:
        if report_page is not page:
            try:
                await report_page.close()
            except Exception:
                pass


async def download_and_parse(page, fallback_date: date_type) -> list[dict]:
    """
    Download the completed Excel report and unpivot it.

    Excel layout (from screenshot):
      Rows 1-25  : metadata / filter summary  (skipped)
      Row 26     : headers → S# | Distributor Code | Distributor Name |
                              SKU Description | SKU Code | Brand Name |
                              Brand Code | [date cols DD-MM-YY …]
      Row 27+    : data rows
      Last row   : TOTAL row                  (excluded)
      Cols H+    : one column per date        (unpivoted)
    """
    import openpyxl

    download_dir = os.path.abspath("downloads")
    os.makedirs(download_dir, exist_ok=True)

    # In some report views, a modal asks for report method after View Report.
    popup_title = page.locator("text=Please Select Report Method")
    if await popup_title.count() > 0:
        try:
            await page.locator(
                "button:has-text('Download'), a:has-text('Download'), input[value='Download']"
            ).first.click(timeout=3000)
            log.info("Report method popup handled with Download option.")
        except Exception:
            pass

    default_filename = f"end_stock_{fallback_date.isoformat()}.xlsx"
    filename = default_filename
    save_path = os.path.join(download_dir, filename)

    log.info("Downloading report...")
    download_link = page.locator(
        "a[title*='Download Report File'], a[title*='Download'], td a[href*='download']"
    ).first
    await download_link.wait_for(state="visible", timeout=15000)
    await download_link.scroll_into_view_if_needed()

    downloaded = False

    # Prefer direct S3 pre-signed URL from report panel when available.
    direct_s3_url = None
    candidate_links = page.locator(
        "a[href*='salesflo.s3.amazonaws.com'], a[href*='GenerateReportExcel'], a[title*='Download Report File']"
    )
    candidate_count = await candidate_links.count()
    for index in range(candidate_count):
        href_candidate = await candidate_links.nth(index).get_attribute("href")
        if href_candidate and "salesflo.s3.amazonaws.com" in href_candidate:
            direct_s3_url = html.unescape(href_candidate)
            break

    if direct_s3_url:
        try:
            response = await page.context.request.get(direct_s3_url, timeout=30000)
            if response.ok:
                parsed = urlparse(direct_s3_url)
                base_name = unquote(os.path.basename(parsed.path))
                if base_name:
                    filename = base_name
                    save_path = os.path.join(download_dir, filename)
                body = await response.body()
                if body[:200].lstrip().lower().startswith(b"<html"):
                    log.warning("Direct S3 URL returned HTML; falling back to click-based download.")
                else:
                    with open(save_path, "wb") as report_file:
                        report_file.write(body)
                    downloaded = True
            else:
                log.warning(f"Direct S3 URL returned HTTP {response.status}; falling back to click-based download.")
        except Exception:
            log.warning("Direct S3 URL fetch failed; falling back to click-based download.")

    try:
        if not downloaded:
            async with page.expect_download(timeout=10000) as dl_info:
                try:
                    await download_link.click(timeout=5000)
                except Exception:
                    await download_link.click(timeout=5000, force=True)
            download = await dl_info.value
            suggested = (download.suggested_filename or "").strip()
            if suggested:
                filename = suggested
                save_path = os.path.join(download_dir, filename)
            await download.save_as(save_path)
            downloaded = True
    except PlaywrightTimeout:
        log.warning("UI click did not emit a download event. Trying direct file URL download.")

    if not downloaded:
        href = await download_link.get_attribute("href")
        if not href:
            raise RuntimeError("Download link has no href; cannot fetch report file.")
        href = html.unescape(href)

        def _resolve_name(resp_url: str, content_disposition: str) -> tuple[str, str]:
            lower_url = (resp_url or "").lower()
            resolved_name = filename
            resolved_path = save_path
            if ".xls" in lower_url and ".xlsx" not in lower_url:
                resolved_name = f"end_stock_{fallback_date.isoformat()}.xls"
                resolved_path = os.path.join(download_dir, resolved_name)
            elif ".xlsx" in lower_url:
                resolved_name = f"end_stock_{fallback_date.isoformat()}.xlsx"
                resolved_path = os.path.join(download_dir, resolved_name)
            elif "filename=" in content_disposition.lower():
                raw_name = content_disposition.split("filename=")[-1].strip().strip('"')
                if raw_name:
                    resolved_name = raw_name
                    resolved_path = os.path.join(download_dir, resolved_name)
            return resolved_name, resolved_path

        file_url = urljoin(page.url, href)
        response = await page.context.request.get(file_url, timeout=30000)
        if not response.ok:
            raise RuntimeError(f"Direct file download failed: HTTP {response.status} ({file_url})")

        content_disposition = response.headers.get("content-disposition", "")
        content_type = (response.headers.get("content-type") or "").lower()
        body = await response.body()

        is_excel_binary = body.startswith(b"PK") or body.startswith(b"\xD0\xCF\x11\xE0")
        is_html_response = ("text/html" in content_type) or body[:200].lstrip().lower().startswith(b"<html")

        if not is_excel_binary and is_html_response:
            log.warning("Direct URL returned HTML. Capturing file response from UI download click.")
            try:
                async with page.expect_response(
                    lambda r: r.status == 200 and (
                        ".xls" in r.url.lower() or ".xlsx" in r.url.lower() or "download" in r.url.lower()
                    ),
                    timeout=15000,
                ) as resp_info:
                    await download_link.click(timeout=5000, force=True)

                response = await resp_info.value
                content_disposition = response.headers.get("content-disposition", "")
                body = await response.body()
                filename, save_path = _resolve_name(response.url, content_disposition)
            except PlaywrightTimeout:
                sample = body[:200].decode("utf-8", errors="ignore")
                raise RuntimeError(
                    "Download returned HTML instead of Excel file. "
                    f"Sample response: {sample[:120]}"
                )
        else:
            filename, save_path = _resolve_name(file_url, content_disposition)

        with open(save_path, "wb") as report_file:
            report_file.write(body)

    log.info(f"Saved download -> {save_path}")

    with open(save_path, "rb") as file_check:
        magic = file_check.read(8)

    if magic.startswith(b"PK") or save_path.lower().endswith(".xlsx"):
        wb  = openpyxl.load_workbook(save_path, data_only=True)
        ws  = wb.active
        all_rows = list(ws.iter_rows(values_only=True))
    elif magic.startswith(b"\xD0\xCF\x11\xE0") or save_path.lower().endswith(".xls"):
        try:
            import xlrd
        except ImportError as exc:
            raise RuntimeError("Downloaded file is .xls. Install xlrd==2.0.1 to parse it.") from exc

        workbook = xlrd.open_workbook(save_path)
        sheet = workbook.sheet_by_index(0)
        all_rows = [tuple(sheet.row_values(i)) for i in range(sheet.nrows)]
    else:
        with open(save_path, "rb") as bad_file:
            sample = bad_file.read(200).decode("utf-8", errors="ignore")
        raise RuntimeError(
            "Downloaded file is not a valid Excel file. "
            f"First bytes/text sample: {sample[:120]}"
        )

    # Row 26 (0-based index 25) is the header
    HEADER_IDX  = 25
    FIXED_COLS  = 7   # A–G are fixed identifier columns
    header      = [str(c).strip() if c else "" for c in all_rows[HEADER_IDX]]
    date_labels = header[FIXED_COLS:]   # H onwards

    # Data: rows after header, exclude the last (totals) row
    data_rows = all_rows[HEADER_IDX + 1 : -1]

    rows = []
    for row in data_rows:
        if all(v is None for v in row):
            continue  # skip blank rows

        dist_code  = str(row[1] or "").strip()
        dist_name  = str(row[2] or "").strip()
        sku_desc   = str(row[3] or "").strip()
        sku_code   = str(row[4] or "").strip()
        brand_name = str(row[5] or "").strip()
        brand_code = str(row[6] or "").strip()

        # Unpivot date columns
        for i, date_label in enumerate(date_labels):
            if not date_label:
                continue
            raw_val    = row[FIXED_COLS + i] if (FIXED_COLS + i) < len(row) else None
            value      = _to_float(raw_val)
            parsed_dt  = _parse_date_label(date_label) or fallback_date

            rows.append({
                "report_date":      parsed_dt,
                "distributor_code": dist_code,
                "distributor_name": dist_name,
                "sku_code":         sku_code,
                "sku_description":  sku_desc,
                "brand_code":       brand_code,
                "brand_name":       brand_name,
                "value":            value,
                "unit":             "Value",
            })

    log.info(f"Unpivoted -> {len(rows)} rows from {filename}.")
    return rows


def _parse_date_label(label: str, warn: bool = True) -> Optional[date_type]:
    """Parse date labels with numeric and month-name variants."""
    label = _normalize_date_text(label)
    if not label:
        return None

    null_like_values = {
        "0000-00-00",
        "00/00/0000",
        "-",
        "--",
        "n/a",
        "na",
        "none",
        "null",
        "november 30, -0001",
    }
    if label.lower() in null_like_values:
        return None

    if re.search(r"-\d{4}$", label):
        return None

    # Trim common trailing time portions from date-time strings.
    label_date_part = re.sub(r"\s+\d{1,2}:\d{2}(:\d{2})?\s*([AaPp][Mm])?$", "", label).strip()
    if label_date_part and label_date_part != label:
        label = label_date_part

    for fmt in (
        "%d-%m-%y", "%d-%m-%Y",
        "%d/%m/%Y", "%d/%m/%y",
        "%m/%d/%Y", "%m/%d/%y",
        "%Y-%m-%d", "%Y/%m/%d",
        "%B %d, %Y", "%b %d, %Y",
        "%B %d %Y", "%b %d %Y",
    ):
        try:
            return datetime.strptime(label, fmt).date()
        except ValueError:
            continue

    if warn and label not in _DATE_PARSE_WARNED_LABELS:
        _DATE_PARSE_WARNED_LABELS.add(label)
        log.warning(f"Could not parse date label: '{label}'")
    return None


def _to_float(val) -> Optional[float]:
    try:
        return float(str(val).replace(",", "").strip()) if val is not None else None
    except (ValueError, TypeError):
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CORE FETCH LOGIC
# ══════════════════════════════════════════════════════════════════════════════

async def fetch_date_range(page, conn, report_key: str, report_cfg: dict, start_date: date_type, end_date: date_type, account_label: str = ""):
    """
    Fetch one report covering start_date → end_date (the site supports ranges).
    End Stock saves unpivoted rows (one per date×distributor×SKU);
    Visits Summary saves one row per visit record.
    """
    try:
        await navigate_to_report(page, report_cfg)
        await set_filters(page, start_date, end_date, report_cfg)

        rows = await generate_and_parse(page, start_date, end_date, report_cfg.get("parse_mode", "visits"))
        if not rows:
            await log_run(conn, end_date, "no_data", 0, f"{report_key}: Generated report returned 0 rows.")
            return "no_data", 0

        if account_label:
            for row in rows:
                row["account_label"] = account_label

        if report_cfg.get("save_mode") == "end_stock":
            saved = await save_rows(conn, rows)
        elif report_cfg.get("save_mode") == "ordered_vs_delivered":
            saved = await save_ordered_vs_delivered_rows(conn, rows)
        else:
            saved = await save_visit_rows(conn, rows)

        await log_run(conn, end_date, "success", saved, f"{report_key}")
        log.info(f"SUCCESS [{report_key}] {start_date} -> {end_date} | {saved} rows saved.")
        return "success", saved

    except Exception as e:
        msg = str(e)
        log.error(f"ERROR [{report_key}] ({start_date}->{end_date}): {msg}")
        await log_run(conn, end_date, "failed", 0, f"{report_key}: {msg}")
        return "failed", 0


def _parse_forced_period() -> tuple[Optional[date_type], Optional[date_type]]:
    forced_start_raw = str(os.getenv("FORCE_START_DATE", "") or "").strip()
    forced_end_raw = str(os.getenv("FORCE_END_DATE", "") or "").strip()

    if not forced_start_raw and not forced_end_raw:
        return None, None

    if not forced_start_raw or not forced_end_raw:
        raise RuntimeError("Both FORCE_START_DATE and FORCE_END_DATE are required together (YYYY-MM-DD).")

    try:
        forced_start = datetime.strptime(forced_start_raw, "%Y-%m-%d").date()
        forced_end = datetime.strptime(forced_end_raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise RuntimeError("Invalid forced period format. Use YYYY-MM-DD for FORCE_START_DATE/FORCE_END_DATE.") from exc

    if forced_start > forced_end:
        raise RuntimeError("FORCE_START_DATE cannot be after FORCE_END_DATE.")

    return forced_start, forced_end


def _ensure_playwright_browser_available() -> None:
    browser_cache_dir = str(Path.home() / ".cache" / "ms-playwright")
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", browser_cache_dir)

    install_commands = [
        [sys.executable, "-m", "playwright", "install", "chromium"],
        [sys.executable, "-m", "playwright", "install"],
    ]

    for install_cmd in install_commands:
        try:
            proc = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                log.info("Playwright browser install check passed using: %s", " ".join(install_cmd))
                return

            log.warning(
                "Playwright install attempt failed (exit=%s): %s",
                proc.returncode,
                " ".join(install_cmd),
            )
            if proc.stderr:
                log.warning("Playwright install stderr: %s", proc.stderr[-2000:])
        except Exception as exc:
            log.warning("Playwright install command error (%s): %s", " ".join(install_cmd), exc)


async def _launch_browser_with_bootstrap(pw):
    _ensure_playwright_browser_available()

    try:
        return await pw.chromium.launch(headless=True)
    except Exception as exc:
        err_text = str(exc)
        if "error while loading shared libraries" in err_text or "libglib-2.0.so.0" in err_text:
            log.error(
                "Chromium launch failed due to missing Linux shared libraries. "
                "On Streamlit Cloud, add required apt packages in packages.txt and redeploy."
            )
        missing_browser = (
            "Executable doesn't exist" in err_text
            or "Please run the following command to download new browsers" in err_text
        )
        if not missing_browser:
            raise

        log.warning("Playwright Chromium not found. Attempting one-time install for this environment...")
        install_cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
        install_proc = subprocess.run(
            install_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if install_proc.returncode != 0:
            log.error("Playwright install failed (exit=%s).", install_proc.returncode)
            if install_proc.stdout:
                log.error("Playwright install stdout: %s", install_proc.stdout[-2000:])
            if install_proc.stderr:
                log.error("Playwright install stderr: %s", install_proc.stderr[-2000:])
            raise RuntimeError("Playwright browser install failed on cloud runtime.") from exc

        log.info("Playwright Chromium installed successfully. Retrying browser launch...")
        return await pw.chromium.launch(headless=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BOT RUN
# ══════════════════════════════════════════════════════════════════════════════

async def run_bot():
    log.info("BOT RUN STARTED")
    log.info(
        "Config check | DB_HOST=%s | DB_PORT=%s | DB_NAME=%s | DB_USER_SET=%s | DB_PASS_SET=%s | SALESFLO_USER_SET=%s",
        "set" if str(DB_HOST or "").strip() else "missing",
        str(DB_PORT or "").strip() or "missing",
        str(DB_NAME or "").strip() or "missing",
        "yes" if str(DB_USER or "").strip() else "no",
        "yes" if str(DB_PASS or "").strip() else "no",
        "yes" if str(USERNAME or "").strip() else "no",
    )

    status = "failed"
    saved = 0

    conn = await get_db_connection()

    yesterday  = datetime.now().date() - timedelta(days=1)
    forced_start_date, forced_end_date = _parse_forced_period()
    if forced_start_date and forced_end_date:
        log.info(f"Forced refresh period active: {forced_start_date} -> {forced_end_date}")

    enabled_report_keys = [r.strip() for r in ENABLED_REPORTS.split(",") if r.strip()]
    selected_reports = [(key, REPORT_CONFIGS[key]) for key in enabled_report_keys if key in REPORT_CONFIGS]
    if not selected_reports:
        raise RuntimeError("No valid reports configured in ENABLED_REPORTS.")

    accounts = get_salesflo_accounts()
    if not accounts:
        raise RuntimeError("No valid Salesflo credentials found. Set SALESFLO_USERNAME/SALESFLO_PASSWORD and optional SALESFLO_USERNAME2/SALESFLO_PASSWORD2.")

    try:
        await ensure_tables(conn)

        async with async_playwright() as pw:
            browser = await _launch_browser_with_bootstrap(pw)

            try:
                aggregate_status = "no_data"
                aggregate_saved = 0

                for account_label, account_username, account_password in accounts:
                    context = await browser.new_context(accept_downloads=True)
                    page = await context.new_page()
                    try:
                        await login(page, account_username, account_password, account_label=account_label)

                        for report_key, report_cfg in selected_reports:
                            if forced_start_date and forced_end_date:
                                start_date = forced_start_date
                                report_end_date = forced_end_date
                                log.info(
                                    f"[{account_label}][{report_key}] Forced refresh range {start_date} -> {report_end_date}."
                                )
                            else:
                                report_end_date = yesterday
                                last_saved = await get_last_saved_date(conn, report_cfg["table"], account_label=account_label)
                                if last_saved is None:
                                    start_date = yesterday - timedelta(days=7)
                                    log.info(f"[{account_label}][{report_key}] No existing data. Backfilling from {start_date} to {report_end_date}.")
                                else:
                                    start_date = last_saved + timedelta(days=1)
                                    log.info(f"[{account_label}][{report_key}] Last saved: {last_saved}. Fetching {start_date} -> {report_end_date}.")

                            if start_date > report_end_date:
                                log.info(f"[{account_label}][{report_key}] Already up-to-date. Skipping.")
                                continue

                            rep_status, rep_saved = await fetch_date_range(
                                page,
                                conn,
                                report_key,
                                report_cfg,
                                start_date,
                                report_end_date,
                                account_label=account_label,
                            )
                            aggregate_saved += rep_saved
                            if rep_status == "failed":
                                aggregate_status = "failed"
                            elif rep_status == "success" and aggregate_status != "failed":
                                aggregate_status = "success"
                    finally:
                        try:
                            await context.close()
                        except Exception:
                            pass

                status = aggregate_status
                saved = aggregate_saved
            finally:
                try:
                    await browser.close()
                except Exception as close_err:
                    log.warning(f"Browser close warning: {close_err}")
    finally:
        try:
            await conn.ensure_closed()
        except Exception:
            pass
    if status == "success":
        log.info(f"Bot run complete. Rows saved: {saved}")
    elif status == "no_data":
        log.warning("Bot run finished with no data.")
    else:
        log.error("Bot run finished with failures. Check logs above for details.")


def main():
    log.info("Starting one-time bot run...")
    asyncio.run(run_bot())
    log.info("Bot process exiting after run completion.")


if __name__ == "__main__":
    main()