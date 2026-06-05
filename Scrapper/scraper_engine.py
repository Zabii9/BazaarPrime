"""
scraper_engine.py
=================
Dynamic Salesflo Scraper Engine.

Reads report definitions from report_registry.py and handles:
  - Login (shared per account)
  - Navigation to any report
  - Filter application (radio, select, checkbox, date, text)
  - Report generation (click View Report → Generate → wait for table)
  - Table parsing (generic + end_stock unpivot)
  - DB save (auto-creates/migrates table from column definitions)

No report-specific code is needed outside report_registry.py.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, date as date_type
from typing import Optional, Any

import aiomysql
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

from report_registry import REPORT_REGISTRY, LOGIN_URL

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# DATE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

_DATE_WARN_SEEN: set[str] = set()


def _normalize_date_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("\u2019", "'").replace("\u2018", "'").strip("'\" ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^([A-Za-z]+)\s*,\s*(\d{1,2})\s+(\d{4})$", r"\1 \2, \3", text)
    return text


def parse_date(label: str, warn: bool = True) -> Optional[date_type]:
    label = _normalize_date_text(label)
    if not label:
        return None
    null_like = {"0000-00-00", "00/00/0000", "-", "--", "n/a", "na", "none", "null"}
    if label.lower() in null_like:
        return None
    if re.search(r"-\d{4}$", label):
        return None
    # Strip trailing time
    label = re.sub(r"\s+\d{1,2}:\d{2}(:\d{2})?\s*([AaPp][Mm])?$", "", label).strip()
    for fmt in (
        "%d-%m-%y", "%d-%m-%Y", "%d/%m/%Y", "%d/%m/%y",
        "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%Y/%m/%d",
        "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
    ):
        try:
            return datetime.strptime(label, fmt).date()
        except ValueError:
            continue
    if warn and label not in _DATE_WARN_SEEN:
        _DATE_WARN_SEEN.add(label)
        log.warning("Could not parse date label: '%s'", label)
    return None


def to_float(val) -> Optional[float]:
    try:
        return float(str(val).replace(",", "").strip()) if val is not None else None
    except (ValueError, TypeError):
        return None


def to_int(val) -> Optional[int]:
    f = to_float(val)
    return int(f) if f is not None else None


# ══════════════════════════════════════════════════════════════════════════════
# DB HELPERS
# ══════════════════════════════════════════════════════════════════════════════

async def ensure_report_table(conn, report_cfg: dict) -> None:
    """
    Auto-create or migrate the DB table based on column definitions.
    Always adds: id, report_date, account_label, row_hash, row_json, fetched_at.
    """
    table = report_cfg["db_table"]
    columns: list[dict] = report_cfg.get("columns", [])
    parse_mode = report_cfg.get("parse_mode", "generic")

    async with conn.cursor() as cur:
        # ── Build CREATE TABLE ────────────────────────────────────────────────
        col_defs = [
            "id INT AUTO_INCREMENT PRIMARY KEY",
            "report_date DATE NULL",
            "account_label VARCHAR(50)",
        ]

        if parse_mode == "end_stock":
            # End stock has fixed cols + pivot col
            for col in columns:
                name = col["db_col"]
                dtype = col["db_type"]
                col_defs.append(f"`{name}` {dtype}")
            col_defs += [
                "value DECIMAL(18,4)",
                "unit VARCHAR(50) DEFAULT 'Value'",
            ]
        else:
            for col in columns:
                name = col["db_col"]
                dtype = col["db_type"]
                col_defs.append(f"`{name}` {dtype}")

        col_defs += [
            "row_hash CHAR(64) NOT NULL",
            "row_json MEDIUMTEXT",
            "fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP",
        ]

        if parse_mode == "end_stock":
            unique_key = "UNIQUE KEY uq_date_acc_dist_sku (report_date, account_label, distributor_code, sku_code)"
        else:
            unique_key = "UNIQUE KEY uq_row_hash (row_hash)"

        col_defs.append(unique_key)
        ddl = f"CREATE TABLE IF NOT EXISTS `{table}` (\n    " + ",\n    ".join(col_defs) + "\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
        await cur.execute(ddl)

        # ── Migrate missing columns ───────────────────────────────────────────
        await cur.execute(
            "SELECT COLUMN_NAME FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s",
            (conn.db, table),
        )
        existing = {row[0].lower() for row in await cur.fetchall()}

        for col in columns:
            if col["db_col"].lower() not in existing:
                await cur.execute(f"ALTER TABLE `{table}` ADD COLUMN `{col['db_col']}` {col['db_type']}")
                log.info("Migrated: added column %s.%s", table, col["db_col"])

        # ── Fix known bad column types from older schema ──────────────────────
        # 1. Distance From Original Location (m) was DECIMAL(18,4) — too few integer
        #    digits for large GPS distances (e.g. 23456.7890 overflows). Widen to (18,2).
        # 2. row_json was LONGTEXT on some installs — ensure at least MEDIUMTEXT.
        _type_fixes = {
            "distance from original location (m)": ("DECIMAL(18,2)", ["decimal(18,4)"]),
            "row_json": ("MEDIUMTEXT", ["longtext"]),
        }
        await cur.execute(
            "SELECT COLUMN_NAME, COLUMN_TYPE FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s",
            (conn.db, table),
        )
        live_types = {row[0].lower(): row[1].lower() for row in await cur.fetchall()}
        for col_name_lower, (new_type, bad_types) in _type_fixes.items():
            current_type = live_types.get(col_name_lower, "")
            if any(bt in current_type for bt in bad_types):
                # Find exact case column name from live schema
                exact_name = next(
                    (r for r in live_types if r == col_name_lower), col_name_lower
                )
                await cur.execute(
                    f"ALTER TABLE `{table}` MODIFY COLUMN `{exact_name}` {new_type}"
                )
                log.info("Migrated: fixed column type %s.%s -> %s", table, exact_name, new_type)

    log.info("Table ready: %s", table)


async def save_generic_rows(conn, table: str, rows: list[dict]) -> int:
    """Upsert generic rows by row_hash."""
    if not rows:
        return 0

    # Collect all keys across rows
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    all_keys.discard("row_hash")
    all_keys.discard("row_json")
    sorted_keys = sorted(all_keys)

    insert_cols = sorted_keys + ["row_hash", "row_json"]
    placeholders = ", ".join(["%s"] * len(insert_cols))
    update_clause = ", ".join(
        [f"`{k}` = VALUES(`{k}`)" for k in sorted_keys]
        + ["row_json = VALUES(row_json)", "fetched_at = CURRENT_TIMESTAMP"]
    )
    col_clause = ", ".join(f"`{k}`" for k in insert_cols)

    sql = f"""
        INSERT INTO `{table}` ({col_clause})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {update_clause}
    """

    data = []
    for row in rows:
        data.append(tuple(row.get(k) for k in sorted_keys) + (row["row_hash"], row["row_json"]))

    async with conn.cursor() as cur:
        await cur.executemany(sql, data)
    return len(rows)


async def save_end_stock_rows(conn, table: str, rows: list[dict]) -> int:
    """Upsert end-stock unpivoted rows."""
    if not rows:
        return 0

    # Remove last row (totals)
    rows = rows[:-1] if rows else rows

    # Dedup by (date, account, dist_code, sku_code)
    deduped: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row.get("report_date"),
            str(row.get("account_label", "") or ""),
            str(row.get("distributor_code", "") or ""),
            str(row.get("sku_code", "") or ""),
        )
        deduped[key] = row

    rows = list(deduped.values())

    sql = f"""
        INSERT INTO `{table}`
            (report_date, account_label, distributor_code, distributor_name,
             sku_code, sku_description, brand_code, brand_name, value, unit)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            distributor_name = VALUES(distributor_name),
            sku_description  = VALUES(sku_description),
            brand_name       = VALUES(brand_name),
            value            = VALUES(value),
            fetched_at       = CURRENT_TIMESTAMP
    """
    data = [
        (
            r.get("report_date"), r.get("account_label", ""),
            r.get("distributor_code", ""), r.get("distributor_name", ""),
            r.get("sku_code", ""), r.get("sku_description", ""),
            r.get("brand_code", ""), r.get("brand_name", ""),
            r.get("value"), r.get("unit", "Value"),
        )
        for r in rows
    ]
    async with conn.cursor() as cur:
        await cur.executemany(sql, data)
    return len(rows)


async def get_last_saved_date(conn, table: str, account_label: str = "") -> Optional[date_type]:
    async with conn.cursor() as cur:
        if account_label:
            await cur.execute(f"SELECT MAX(report_date) FROM `{table}` WHERE account_label=%s", (account_label,))
        else:
            await cur.execute(f"SELECT MAX(report_date) FROM `{table}`")
        row = await cur.fetchone()
        return row[0] if row and row[0] else None


async def log_run(conn, **kwargs):
    async with conn.cursor() as cur:
        await cur.execute(
            """INSERT INTO bot_run_log
               (run_date,status,rows_saved,rows_deleted,account_label,report_key,table_name,period_start,period_end,action_type,message)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                kwargs.get("run_date"), kwargs.get("status"), kwargs.get("rows_saved", 0),
                kwargs.get("rows_deleted", 0), kwargs.get("account_label", ""),
                kwargs.get("report_key", ""), kwargs.get("table_name", ""),
                kwargs.get("period_start"), kwargs.get("period_end"),
                kwargs.get("action_type", "run"), kwargs.get("message", ""),
            ),
        )


# ══════════════════════════════════════════════════════════════════════════════
# BROWSER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

async def login(page, username: str, password: str, account_label: str = "account"):
    log.info("[%s] Logging in...", account_label)
    nav_timeout = int(os.getenv("LOGIN_NAV_TIMEOUT_MS", "45000"))

    for attempt in range(1, 4):
        try:
            await page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=nav_timeout)
            break
        except PlaywrightTimeout:
            if attempt == 3:
                await page.goto(LOGIN_URL, wait_until="commit", timeout=nav_timeout)

    async def _find(selectors: list[str], timeout_ms: int = 25000):
        deadline = time.time() + timeout_ms / 1000
        while time.time() < deadline:
            for root in [page] + list(page.frames):
                for sel in selectors:
                    try:
                        loc = root.locator(sel).first
                        if await loc.count() > 0:
                            return loc
                    except Exception:
                        pass
            await asyncio.sleep(0.4)
        return None

    user_input = await _find(['input[name="username"]', 'input#username', 'input[name="email"]', 'input[type="text"]'])
    pass_input = await _find(['input[name="password"]', 'input#password', 'input[type="password"]'])

    if not user_input or not pass_input:
        raise RuntimeError(f"[{account_label}] Login form not found at {page.url}")

    async def _fill(target, value: str):
        try:
            await target.fill(value, timeout=1500)
            return
        except Exception:
            pass
        await target.evaluate(
            "(el, v) => { el.readOnly=false; el.disabled=false; el.focus(); el.value=v; "
            "el.dispatchEvent(new Event('input',{bubbles:true})); el.dispatchEvent(new Event('change',{bubbles:true})); }",
            value,
        )

    await _fill(user_input, username)
    await _fill(pass_input, password)

    # Try click submit button
    clicked = False
    for sel in [
        'input[type="image"][src*="btnLogin"]',
        'button[type="submit"]', 'input[type="submit"]',
        'button:has-text("Login")', 'button:has-text("Sign In")',
    ]:
        try:
            loc = page.locator(sel).first
            if await loc.count() > 0:
                await loc.click(timeout=3000)
                clicked = True
                break
        except Exception:
            pass

    if not clicked:
        try:
            await pass_input.press("Enter")
        except Exception:
            pass

    try:
        await page.wait_for_selector("text=Reports", timeout=30000)
    except PlaywrightTimeout:
        await page.wait_for_load_state("networkidle")
    log.info("[%s] Login successful.", account_label)


async def navigate_to_report(page, report_cfg: dict):
    """Navigate through the menu to reach the report form."""
    title = report_cfg["title"]
    log.info("Navigating to: %s", title)

    ready_sels = report_cfg.get("ready_selectors", [])

    async def _is_ready() -> bool:
        for root in [page] + list(page.frames):
            for sel in ready_sels:
                try:
                    if await root.locator(sel).first.count() > 0:
                        return True
                except Exception:
                    pass
        return False

    async def _click(sel: str, pause: float = 0.7) -> bool:
        for root in [page] + list(page.frames):
            try:
                loc = root.locator(sel).first
                if await loc.count() > 0:
                    await loc.click(timeout=8000)
                    await asyncio.sleep(pause)
                    return True
            except Exception:
                pass
        return False

    # Step 1: Open Reports top menu
    await _click("text=Reports", pause=2.0)

    # Step 2: Walk nav_steps from config
    for step in report_cfg.get("nav_steps", []):
        await _click(step, pause=1.0)

    # Check readiness; fallback to direct URL if needed
    for _ in range(8):
        if await _is_ready():
            break
        await asyncio.sleep(0.5)

    if not await _is_ready():
        await page.goto(f"https://engrofoods.salesflo.com/OB/reports/", wait_until="networkidle")
        await _click("text=Reports", pause=2.0)
        for step in report_cfg.get("nav_steps", []):
            await _click(step, pause=1.0)
        for _ in range(8):
            if await _is_ready():
                break
            await asyncio.sleep(0.5)

    await page.wait_for_load_state("networkidle")
    log.info("Report form loaded: %s", title)


async def apply_filters(page, start_date: date_type, end_date: date_type, report_cfg: dict):
    """
    Apply all filters defined in report_cfg['filters'].
    Each filter specifies: type, selectors, value/checked, label.
    """
    start_long = f"{start_date.strftime('%B')} {start_date.day}, {start_date.year}"
    end_long   = f"{end_date.strftime('%B')} {end_date.day}, {end_date.year}"
    start_short = start_date.strftime("%m/%d/%Y")
    end_short   = end_date.strftime("%m/%d/%Y")

    filters: list[dict] = report_cfg.get("filters", [])

    async def _find(selectors: list[str], timeout_s: float = 6.0) -> Optional[Any]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            for root in [page] + list(page.frames):
                for sel in selectors:
                    try:
                        loc = root.locator(sel).first
                        if await loc.count() > 0:
                            return loc
                    except Exception:
                        pass
            await asyncio.sleep(0.4)
        return None

    async def _apply_radio(f: dict):
        sels = f.get("selectors", [])
        name = f.get("name", "")
        value = f.get("value", "")
        label = f.get("label", "radio")
        target = await _find(sels, 6.0)
        if target is None and name and value:
            target = await _find([f'input[type="radio"][name="{name}"][value="{value}"]'])
        if target is None:
            log.warning("Radio not found: %s", label)
            return
        try:
            await target.check(timeout=2000)
        except Exception:
            try:
                await target.click(timeout=2000)
            except Exception:
                await target.evaluate(
                    "(el) => { el.checked=true; el.dispatchEvent(new Event('change',{bubbles:true})); }"
                )
        log.info("Radio set: %s", label)

    async def _apply_select(f: dict):
        sels = f.get("selectors", [])
        value = str(f.get("value", ""))
        label = f.get("label", "select")
        target = await _find(sels, 6.0)
        if target is None:
            log.warning("Select not found: %s", label)
            return
        try:
            await target.select_option(value=value)
        except Exception:
            try:
                await target.select_option(label=label)
            except Exception:
                log.warning("Could not set select: %s", label)
        log.info("Select set: %s = %s", label, value)

    async def _apply_checkbox(f: dict):
        sels = f.get("selectors", [])
        checked = f.get("checked", True)
        label = f.get("label", "checkbox")
        target = await _find(sels, 6.0)
        if target is None:
            log.warning("Checkbox not found: %s", label)
            return
        try:
            if checked:
                await target.check(timeout=2000)
            else:
                await target.uncheck(timeout=2000)
        except Exception:
            await target.evaluate(
                "(el, desired) => { el.checked=!!desired; el.dispatchEvent(new Event('change',{bubbles:true})); }",
                checked,
            )
        log.info("Checkbox set: %s = %s", label, checked)

    async def _apply_date(f: dict, primary: str, alt: str):
        sels = f.get("selectors", [])
        label = f.get("label", "date")
        target = await _find(sels, 15.0)
        if target is None:
            log.warning("Date input not found: %s", label)
            return

        for value in (primary, alt):
            try:
                await target.click(timeout=3000)
                await target.fill("")
                await target.fill(value)
                await target.press("Enter")
                current = await target.input_value()
                if primary in current or alt in current:
                    await target.evaluate("(el) => { el.blur(); el.dispatchEvent(new Event('blur',{bubbles:true})); }")
                    log.info("Date set: %s = %s", label, value)
                    return
            except Exception:
                pass

        # Force via JS
        await target.evaluate(
            "(el, v) => { el.readOnly=false; el.value=v; "
            "el.dispatchEvent(new Event('input',{bubbles:true})); "
            "el.dispatchEvent(new Event('change',{bubbles:true})); "
            "el.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true})); "
            "el.blur(); }",
            primary,
        )
        log.info("Date force-set: %s = %s", label, primary)

    for f in filters:
        ftype = f.get("type")
        role  = f.get("role", "")

        if ftype == "radio":
            await _apply_radio(f)
        elif ftype == "select":
            await _apply_select(f)
        elif ftype == "checkbox":
            await _apply_checkbox(f)
        elif ftype == "date":
            if role == "start_date":
                await _apply_date(f, start_long, start_short)
            elif role == "end_date":
                await _apply_date(f, end_long, end_short)
            else:
                # If no role, use start date by convention
                await _apply_date(f, start_long, start_short)
        elif ftype == "text":
            target = await _find(f.get("selectors", []))
            if target:
                await target.fill(str(f.get("value", "")))

    log.info("All filters applied.")


# ══════════════════════════════════════════════════════════════════════════════
# REPORT TABLE PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(s: str) -> str:
    return " ".join(str(s or "").replace("\n", " ").replace("\r", " ").split()).strip().lower()


def _build_alias_map(columns: list[dict]) -> dict[str, str]:
    """Maps normalized alias/name -> canonical column name."""
    m: dict[str, str] = {}
    for col in columns:
        m[_normalize(col["name"])] = col["name"]
        for alias in col.get("aliases", []):
            m[_normalize(alias)] = col["name"]
    return m


def _is_header_candidate(row: list[str], alias_map: dict[str, str], min_hits: int = 2) -> bool:
    hits = sum(1 for cell in row if _normalize(cell) in alias_map)
    return hits >= min_hits


def parse_table(
    table_rows: list[list[str]],
    external_header: list[str],
    report_cfg: dict,
    fallback_date: date_type,
    start_date: date_type,
    end_date: date_type,
) -> list[dict]:
    columns: list[dict] = report_cfg.get("columns", [])
    parse_mode: str = report_cfg.get("parse_mode", "generic")
    alias_map = _build_alias_map(columns)

    # ── Find header row ───────────────────────────────────────────────────────
    header_idx = None
    for i, row in enumerate(table_rows):
        if _is_header_candidate([str(c) for c in row], alias_map):
            header_idx = i
            break

    if header_idx is not None:
        header = [str(c).strip() for c in table_rows[header_idx]]
        data_rows = table_rows[header_idx + 1:]
    elif external_header and _is_header_candidate([str(c) for c in external_header], alias_map):
        header = [str(c).strip() for c in external_header]
        data_rows = table_rows
    else:
        # Synthetic: use canonical column order
        widest = max((len(r) for r in table_rows), default=0)
        header = [col["name"] for col in columns[:widest]]
        data_rows = table_rows
        log.warning("Using synthetic header for %s", report_cfg["title"])

    normalized_header = [str(h).strip() for h in header]

    # ── End stock: unpivot date columns ───────────────────────────────────────
    if parse_mode == "end_stock":
        return _parse_end_stock(normalized_header, data_rows, columns, fallback_date, start_date, end_date)

    # ── Generic: map columns, build row dicts ─────────────────────────────────
    return _parse_generic(normalized_header, data_rows, columns, alias_map, fallback_date)


def _parse_end_stock(
    header: list[str],
    data_rows: list[list],
    columns: list[dict],
    fallback_date: date_type,
    start_date: date_type,
    end_date: date_type,
) -> list[dict]:
    """Unpivot: one output row per (fixed_cols × date_col)."""
    norm_header = [h.lower() for h in header]

    def _find_col(*keys: str) -> Optional[int]:
        for i, col in enumerate(norm_header):
            if any(k in col for k in keys):
                return i
        return None

    idx_dist_code  = _find_col("distributor code")
    idx_dist_name  = _find_col("distributor name")
    idx_sku_desc   = _find_col("sku description", "sku desc")
    idx_sku_code   = _find_col("sku code")
    idx_brand_name = _find_col("brand name")
    idx_brand_code = _find_col("brand code")

    # Date columns start after all fixed columns
    fixed_boundary = max(
        [i for i in [idx_dist_code, idx_dist_name, idx_sku_desc, idx_sku_code, idx_brand_name, idx_brand_code]
         if i is not None],
        default=6,
    ) + 1

    date_indices = [
        i for i, label in enumerate(header)
        if i >= fixed_boundary and parse_date(label, warn=False) is not None
    ]
    if not date_indices:
        date_indices = list(range(fixed_boundary, len(header)))

    rows = []
    for row in data_rows:
        if not row or all(str(v).strip() == "" for v in row):
            continue
        if "total" in str(row[0]).strip().lower():
            continue

        def _cell(i: Optional[int]) -> str:
            return str(row[i] or "").strip() if i is not None and i < len(row) else ""

        for col_idx in date_indices:
            if col_idx >= len(row):
                continue
            label = header[col_idx] if col_idx < len(header) else ""
            parsed_dt = parse_date(label, warn=False) or fallback_date
            rows.append({
                "report_date":      parsed_dt,
                "distributor_code": _cell(idx_dist_code),
                "distributor_name": _cell(idx_dist_name),
                "sku_code":         _cell(idx_sku_code),
                "sku_description":  _cell(idx_sku_desc),
                "brand_code":       _cell(idx_brand_code),
                "brand_name":       _cell(idx_brand_name),
                "value":            to_float(row[col_idx]),
                "unit":             "Value",
            })
    return rows


def _parse_generic(
    header: list[str],
    data_rows: list[list],
    columns: list[dict],
    alias_map: dict[str, str],
    fallback_date: date_type,
) -> list[dict]:
    """Map each data row to canonical column names, type-cast, hash."""
    col_meta = {col["name"]: col for col in columns}
    rows = []

    for row in data_rows:
        if not row or all(str(v).strip() == "" for v in row):
            continue
        if "total" in str(row[0]).strip().lower():
            continue
        full_text = " ".join(str(v).strip().lower() for v in row)
        if full_text in {"print", ""} or full_text.startswith("print "):
            continue

        # Map raw cells to canonical names
        raw_map: dict[str, str] = {}
        for i, col_name in enumerate(header):
            if i < len(row):
                raw_map[col_name or f"col_{i}"] = str(row[i] or "").strip()

        canonical: dict[str, Any] = {}
        for raw_key, raw_val in raw_map.items():
            mapped = alias_map.get(_normalize(raw_key))
            if mapped:
                meta = col_meta.get(mapped, {})
                if meta.get("is_date"):
                    canonical[mapped] = parse_date(raw_val, warn=False)
                elif meta.get("is_float"):
                    canonical[mapped] = to_float(raw_val)
                elif meta.get("is_int"):
                    canonical[mapped] = to_int(raw_val)
                else:
                    canonical[mapped] = raw_val

        # Ensure all columns present
        for col in columns:
            if col["name"] not in canonical:
                canonical[col["name"]] = None

        # Derive report_date from first date column found
        report_date = fallback_date
        for col in columns:
            if col.get("is_date") and canonical.get(col["name"]):
                report_date = canonical[col["name"]]
                break

        # Skip entirely empty rows
        has_data = any(
            v not in (None, "", 0)
            for k, v in canonical.items()
            if not col_meta.get(k, {}).get("is_date")
        )
        if not has_data:
            continue

        row_json = json.dumps({"canonical": {k: str(v) for k, v in canonical.items()}, "raw": list(row)}, ensure_ascii=False, sort_keys=True)
        row_hash = hashlib.sha256(row_json.encode()).hexdigest()

        out = {
            "report_date":   report_date,
            "row_hash":      row_hash,
            "row_json":      row_json,
        }
        # Use db_col names as keys for DB insert
        for col in columns:
            out[col["db_col"]] = canonical.get(col["name"])

        rows.append(out)

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION (click View Report → Generate → wait for table)
# ══════════════════════════════════════════════════════════════════════════════

async def generate_and_parse(
    page,
    start_date: date_type,
    end_date: date_type,
    report_cfg: dict,
) -> list[dict]:
    """Click View Report, choose Generate, wait for table, parse and return rows."""
    page_load_ms = int(os.getenv("REPORT_PAGE_LOAD_TIMEOUT_MS", "120000"))
    max_wait_s   = int(os.getenv("REPORT_GENERATE_WAIT_SECONDS", "300"))

    log.info("Generating report: %s", report_cfg["title"])
    existing_pages = list(page.context.pages)
    await page.click("button:has-text('View Report'), input[value='View Report']")

    # Modal: choose Generate vs Download
    report_page = page
    modal_visible = False
    try:
        await page.wait_for_selector("text=Please Select Report Method", timeout=6000)
        modal_visible = True
    except PlaywrightTimeout:
        pass

    if modal_visible:
        gen_btn = page.locator("button:has-text('Generate'), a:has-text('Generate'), input[value='Generate']").first
        try:
            async with page.context.expect_page(timeout=15000) as page_info:
                await gen_btn.click(timeout=5000)
            report_page = await page_info.value
        except PlaywrightTimeout:
            await gen_btn.click(timeout=5000)
    else:
        for _ in range(30):
            new_pages = [p for p in page.context.pages if p not in existing_pages]
            if new_pages:
                report_page = new_pages[-1]
                break
            await asyncio.sleep(0.5)

    try:
        try:
            await report_page.wait_for_load_state("domcontentloaded", timeout=page_load_ms)
        except PlaywrightTimeout:
            log.warning("Report page slow to load; continuing with available DOM.")

        # ── Poll for stable table ─────────────────────────────────────────────
        async def _read_payload(root) -> dict:
            try:
                return await root.evaluate(r"""
                () => {
                    const clean = s => (s || '').replace(/\s+/g,' ').trim();
                    const noRec = ['no record found','no records found','sorry! no record','sorry no record'].some(p => clean(document.body?.innerText||'').toLowerCase().includes(p));
                    const tables = Array.from(document.querySelectorAll('#StickyTable,.ui-jqgrid-btable,table'));
                    if (!tables.length) return {rows:[],extHeader:[],noRecords:noRec};
                    const scored = tables.map(tbl => {
                        const rows = Array.from(tbl.querySelectorAll('tr'))
                            .map(tr => Array.from(tr.querySelectorAll('th,td')).map(td => clean(td.innerText||td.textContent)))
                            .filter(r => r.length && r.join('').trim());
                        const gridRoot = tbl.closest('.ui-jqgrid-view')||tbl.closest('.ui-jqgrid');
                        const extHeader = gridRoot
                            ? Array.from(gridRoot.querySelectorAll('.ui-jqgrid-htable th')).map(th => clean(th.innerText||th.textContent)).filter(Boolean)
                            : Array.from(tbl.querySelectorAll('thead th')).map(th => clean(th.innerText||th.textContent)).filter(Boolean);
                        return {rows, extHeader, score: rows.length*100 + Math.max(...rows.map(r=>r.length),0) + extHeader.length*5, noRecords:noRec};
                    });
                    scored.sort((a,b)=>b.score-a.score);
                    return scored[0]||{rows:[],extHeader:[],noRecords:noRec};
                }
                """)
            except Exception:
                return {"rows": [], "extHeader": [], "noRecords": False}

        best = {"rows": [], "extHeader": [], "noRecords": False}
        last_sig = None
        stable = 0
        no_records = False

        for _ in range(max_wait_s):
            for root in [report_page] + list(report_page.frames):
                payload = await _read_payload(root)
                if payload.get("noRecords"):
                    no_records = True
                size = len(payload.get("rows", [])) + len(payload.get("extHeader", []))
                best_size = len(best.get("rows", [])) + len(best.get("extHeader", []))
                if size >= best_size:
                    best = payload

            rows = best.get("rows", [])
            ext = best.get("extHeader", [])
            sig = (len(rows), len(ext), tuple(tuple(c[:3] for c in r[:4]) for r in rows[:2]))
            if sig == last_sig:
                stable += 1
            else:
                stable = 0
                last_sig = sig
            if rows and stable >= 2:
                break
            await asyncio.sleep(1)

        if no_records and not best.get("rows"):
            log.info("Report returned no records.")
            return []

        if not best.get("rows") and not best.get("extHeader"):
            raise RuntimeError("Report table not found after waiting.")

        # Parse
        parsed = parse_table(
            table_rows=best["rows"],
            external_header=best.get("extHeader", []),
            report_cfg=report_cfg,
            fallback_date=end_date,
            start_date=start_date,
            end_date=end_date,
        )
        log.info("Parsed %d rows from generated report.", len(parsed))
        return parsed

    finally:
        if report_page is not page:
            try:
                await report_page.close()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# FETCH ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

async def fetch_report(
    page,
    conn,
    report_key: str,
    report_cfg: dict,
    start_date: date_type,
    end_date: date_type,
    account_label: str = "",
) -> tuple[str, int]:
    """Navigate → filter → generate → parse → save one date range."""
    try:
        await navigate_to_report(page, report_cfg)
        await apply_filters(page, start_date, end_date, report_cfg)

        rows = await generate_and_parse(page, start_date, end_date, report_cfg)
        if not rows:
            await log_run(conn, run_date=end_date, status="no_data", rows_saved=0,
                          account_label=account_label, report_key=report_key,
                          table_name=report_cfg["db_table"], period_start=start_date,
                          period_end=end_date, action_type="run",
                          message=f"{report_key}: 0 rows returned")
            return "no_data", 0

        for row in rows:
            row["account_label"] = account_label

        parse_mode = report_cfg.get("parse_mode", "generic")
        if parse_mode == "end_stock":
            saved = await save_end_stock_rows(conn, report_cfg["db_table"], rows)
        else:
            saved = await save_generic_rows(conn, report_cfg["db_table"], rows)

        await log_run(conn, run_date=end_date, status="success", rows_saved=saved,
                      account_label=account_label, report_key=report_key,
                      table_name=report_cfg["db_table"], period_start=start_date,
                      period_end=end_date, action_type="run",
                      message=f"{report_key}: {saved} rows saved")
        log.info(
            "[OK] [%s][%s] %s -> %s | %d rows saved",
            account_label, report_key, start_date, end_date, saved,
        )
        return "success", saved

    except Exception as exc:
        msg = str(exc)
        log.error("[%s][%s] ERROR: %s", account_label, report_key, msg)
        await log_run(conn, run_date=end_date, status="failed", rows_saved=0,
                      account_label=account_label, report_key=report_key,
                      table_name=report_cfg.get("db_table", ""), period_start=start_date,
                      period_end=end_date, action_type="run", message=msg)
        return "failed", 0