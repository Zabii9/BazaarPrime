"""
main.py
=======
Salesflo Dynamic Scraper — Entry Point.

NORMAL RUN
  python main.py
  python main.py --reports end_stock_trend
  python main.py --reports visits_summary,ordered_vs_delivered
  python main.py --start 2025-01-01 --end 2025-01-31

FORCED REFRESH  (delete existing rows then re-scrape)
  # All accounts, all reports for a date range
  python main.py --force-refresh --start 2025-03-01 --end 2025-03-31

  # Only one report
  python main.py --force-refresh --reports end_stock_trend --start 2025-03-01 --end 2025-03-31

  # Only one account
  python main.py --force-refresh --accounts account_2 --start 2025-03-01 --end 2025-03-31

  # Specific report + specific account
  python main.py --force-refresh --reports visits_summary --accounts account_1 --start 2025-03-01 --end 2025-03-31

  # Multiple accounts comma-separated
  python main.py --force-refresh --accounts account_1,account_3 --reports end_stock_trend --start 2025-03-01 --end 2025-03-31

DELETE ONLY  (wipe rows without re-scraping — useful before a manual CSV import)
  python main.py --delete-only --reports end_stock_trend --accounts account_2 --start 2025-03-01 --end 2025-03-31

UTILITIES
  python main.py --list-reports              # show all registered reports
  python main.py --list-accounts             # show all configured account labels
  python main.py --preview --reports end_stock_trend --accounts account_1 --start 2025-03-01 --end 2025-03-31
                                             # dry-run: show what WOULD be deleted/fetched, no DB changes
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, date as date_type
from pathlib import Path
from typing import Optional

import aiomysql
from dotenv import load_dotenv
from playwright.async_api import async_playwright

from report_registry import REPORT_REGISTRY
from scraper_engine import (
    ensure_report_table,
    fetch_report,
    get_last_saved_date,
    log_run,
    login,
)

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
# Windows cmd/PowerShell uses CP1252 which can't encode unicode symbols like ✓ ✗ ⚠ 🗑
# Fix: use a SafeStreamHandler that replaces unencodable chars instead of crashing.

class _SafeStreamHandler(logging.StreamHandler):
    """Stream handler that never crashes on Windows CP1252 / non-UTF8 consoles."""
    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            stream = self.stream
            # Replace unencodable chars with closest ASCII equivalent
            enc = getattr(stream, "encoding", "utf-8") or "utf-8"
            safe_msg = msg.encode(enc, errors="replace").decode(enc)
            stream.write(safe_msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

_LOG_FMT = "%(asctime)s [%(levelname)-8s] %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_file_handler   = logging.FileHandler("salesflo_scraper.log", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(_LOG_FMT, _DATE_FMT))

_console_handler = _SafeStreamHandler(sys.stdout)
_console_handler.setFormatter(logging.Formatter(_LOG_FMT, _DATE_FMT))

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _console_handler])
# Suppress noisy aiomysql "Table already exists" warnings from console (still in log file)
logging.getLogger("aiomysql").setLevel(logging.ERROR)

log = logging.getLogger(__name__)

# ── Status symbols (ASCII-safe for console, full unicode in log file) ─────────
OK   = "[OK]"
WARN = "[WARN]"
ERR  = "[ERR]"
DEL  = "[DEL]"
SKIP = "[SKIP]"
DRY  = "[DRY]"


# ══════════════════════════════════════════════════════════════════════════════
# SECRETS / CONFIG LOADER
# ══════════════════════════════════════════════════════════════════════════════

def _load_toml_secrets() -> dict:
    try:
        import tomllib
    except ImportError:
        return {}
    for path in [
        os.getenv("STREAMLIT_SECRETS_FILE", ""),
        str(Path.home() / ".streamlit" / "secrets.toml"),
    ]:
        if path and Path(path).exists():
            try:
                with open(path, "rb") as f:
                    return tomllib.load(f)
            except Exception:
                pass
    return {}


_SECRETS = _load_toml_secrets()


def _secret(section: str, keys: list[str], env_keys: list[str], default: str = "") -> str:
    sec = _SECRETS.get(section, {}) if isinstance(_SECRETS, dict) else {}
    for k in keys:
        v = sec.get(k) or _SECRETS.get(k)
        if v not in (None, ""):
            return str(v).strip()
    for k in env_keys:
        v = os.getenv(k)
        if v not in (None, ""):
            return str(v).strip()
    return default


# ── DB config ─────────────────────────────────────────────────────────────────
DB_HOST = _secret("database", ["host"],               ["DB_HOST"])
DB_PORT = int(_secret("database", ["port"],           ["DB_PORT"], "3306"))
DB_USER = _secret("database", ["username", "user"],   ["DB_USER", "DB_USERNAME"])
DB_PASS = _secret("database", ["password", "pass"],   ["DB_PASSWORD", "DB_PASS"])
DB_NAME = _secret("database", ["database", "db"],     ["DB_NAME"], "salesflo_data")


# ══════════════════════════════════════════════════════════════════════════════
# ACCOUNT LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_accounts() -> list[tuple[str, str, str]]:
    """
    Returns list of (label, username, password).

    secrets.toml:
      [accounts]
      account_1 = { username = "u1@x.com", password = "p1" }
      account_2 = { username = "u2@x.com", password = "p2" }

    .env:
      ACCOUNT_1_USERNAME=u1@x.com
      ACCOUNT_1_PASSWORD=p1
      ACCOUNT_2_USERNAME=u2@x.com
      ACCOUNT_2_PASSWORD=p2
    """
    accounts: list[tuple[str, str, str]] = []

    toml_accounts = _SECRETS.get("accounts", {}) if isinstance(_SECRETS, dict) else {}
    if isinstance(toml_accounts, dict):
        for label, creds in sorted(toml_accounts.items()):
            if isinstance(creds, dict):
                u = str(creds.get("username", "") or "").strip()
                p = str(creds.get("password", "") or "").strip()
                if u and p:
                    accounts.append((label, u, p))

    if not accounts:
        i = 1
        while True:
            u = os.getenv(f"ACCOUNT_{i}_USERNAME", "").strip()
            p = os.getenv(f"ACCOUNT_{i}_PASSWORD", "").strip()
            if not u or not p:
                break
            accounts.append((f"account_{i}", u, p))
            i += 1

    # Legacy fallback
    if not accounts:
        salesflo_sec = _SECRETS.get("salesflo", {}) if isinstance(_SECRETS, dict) else {}
        for label, u_key, p_key in [
            ("account_1", "SALESFLO_USERNAME",  "SALESFLO_PASSWORD"),
            ("account_2", "SALESFLO_USERNAME2", "SALESFLO_PASSWORD2"),
        ]:
            u = str(salesflo_sec.get(u_key.lower(), "") or os.getenv(u_key, "") or "").strip()
            p = str(salesflo_sec.get(p_key.lower(), "") or os.getenv(p_key, "") or "").strip()
            if u and p:
                accounts.append((label, u, p))

    if not accounts:
        raise RuntimeError(
            "No Salesflo accounts found.\n"
            "Add to .env:  ACCOUNT_1_USERNAME / ACCOUNT_1_PASSWORD\n"
            "or to ~/.streamlit/secrets.toml under [accounts]."
        )

    return accounts


def filter_accounts(
    all_accounts: list[tuple[str, str, str]],
    account_filter: Optional[list[str]],
) -> list[tuple[str, str, str]]:
    """If --accounts given, keep only those labels; otherwise return all."""
    if not account_filter:
        return all_accounts
    allowed = {a.strip().lower() for a in account_filter}
    matched = [a for a in all_accounts if a[0].lower() in allowed]
    if not matched:
        available = [a[0] for a in all_accounts]
        raise ValueError(
            f"No accounts matched filter {account_filter}.\n"
            f"Available account labels: {available}"
        )
    return matched


# ══════════════════════════════════════════════════════════════════════════════
# DB HELPERS
# ══════════════════════════════════════════════════════════════════════════════

async def get_db() -> aiomysql.Connection:
    return await aiomysql.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASS,
        db=DB_NAME, charset="utf8mb4",
        autocommit=True,
    )


async def ensure_bot_log_table(conn):
    async with conn.cursor() as cur:
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS bot_run_log (
                id            INT AUTO_INCREMENT PRIMARY KEY,
                run_date      DATE NOT NULL,
                status        ENUM('success','failed','no_data') NOT NULL,
                rows_saved    INT DEFAULT 0,
                rows_deleted  INT DEFAULT 0,
                account_label VARCHAR(50),
                report_key    VARCHAR(100),
                table_name    VARCHAR(100),
                period_start  DATE NULL,
                period_end    DATE NULL,
                action_type   VARCHAR(50) DEFAULT 'run',
                message       TEXT,
                created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)


async def delete_period(
    conn,
    table: str,
    start_date: date_type,
    end_date: date_type,
    account_label: str = "",
    dry_run: bool = False,
) -> int:
    """
    Delete rows in table where report_date BETWEEN start and end.
    If account_label given, also filters by that column.
    Returns number of rows that would be / were deleted.
    """
    date_col = "report_date"
    acc_col  = "account_label"

    # Build WHERE
    sql_count = f"SELECT COUNT(*) FROM `{table}` WHERE `{date_col}` BETWEEN %s AND %s"
    sql_delete = f"DELETE FROM `{table}` WHERE `{date_col}` BETWEEN %s AND %s"
    params: list = [start_date, end_date]

    if account_label:
        sql_count  += f" AND `{acc_col}` = %s"
        sql_delete += f" AND `{acc_col}` = %s"
        params.append(account_label)

    async with conn.cursor() as cur:
        await cur.execute(sql_count, tuple(params))
        row = await cur.fetchone()
        count = int(row[0]) if row else 0

        if dry_run:
            return count

        await cur.execute(sql_delete, tuple(params))
        return int(cur.rowcount or 0)


# ══════════════════════════════════════════════════════════════════════════════
# FORCED REFRESH
# ══════════════════════════════════════════════════════════════════════════════

async def forced_refresh(
    page,
    conn,
    report_key: str,
    report_cfg: dict,
    start_date: date_type,
    end_date: date_type,
    account_label: str = "",
    dry_run: bool = False,
) -> tuple[str, int]:
    """
    1. Fetch fresh data from Salesflo.
    2. Atomically delete the period from DB.
    3. Insert fresh rows.

    dry_run=True: only shows what would happen, no DB writes.
    """
    from scraper_engine import (
        navigate_to_report, apply_filters, generate_and_parse,
        save_end_stock_rows, save_generic_rows,
    )

    table = report_cfg["db_table"]

    if dry_run:
        would_delete = await delete_period(conn, table, start_date, end_date, account_label, dry_run=True)
        log.info(
            "[DRY-RUN][forced_refresh][%s][%s] Would delete %d rows for %s -> %s",
            account_label, report_key, would_delete, start_date, end_date,
        )
        return "dry_run", 0

    try:
        await navigate_to_report(page, report_cfg)
        await apply_filters(page, start_date, end_date, report_cfg)
        rows = await generate_and_parse(page, start_date, end_date, report_cfg)

        if not rows:
            await log_run(
                conn, run_date=end_date, status="no_data",
                account_label=account_label, report_key=report_key,
                table_name=table, period_start=start_date, period_end=end_date,
                action_type="forced_refresh", message="0 rows returned from report",
            )
            return "no_data", 0

        for row in rows:
            row["account_label"] = account_label

        parse_mode = report_cfg.get("parse_mode", "generic")

        await conn.begin()
        try:
            deleted = await delete_period(conn, table, start_date, end_date, account_label)

            if parse_mode == "end_stock":
                saved = await save_end_stock_rows(conn, table, rows)
            else:
                saved = await save_generic_rows(conn, table, rows)

            await conn.commit()
        except Exception:
            await conn.rollback()
            raise

        await log_run(
            conn, run_date=end_date, status="success",
            rows_saved=saved, rows_deleted=deleted,
            account_label=account_label, report_key=report_key,
            table_name=table, period_start=start_date, period_end=end_date,
            action_type="forced_refresh",
            message=f"deleted={deleted} saved={saved}",
        )
        log.info(
            "%s [forced_refresh][%s][%s] %s -> %s | deleted=%d saved=%d", OK,
            account_label, report_key, start_date, end_date, deleted, saved,
        )
        return "success", saved

    except Exception as exc:
        log.error("[forced_refresh][%s][%s] ERROR: %s", account_label, report_key, exc)
        await log_run(
            conn, run_date=end_date, status="failed",
            account_label=account_label, report_key=report_key,
            table_name=table, period_start=start_date, period_end=end_date,
            action_type="forced_refresh", message=str(exc),
        )
        return "failed", 0


# ══════════════════════════════════════════════════════════════════════════════
# DELETE-ONLY  (no re-scrape)
# ══════════════════════════════════════════════════════════════════════════════

async def delete_only(
    conn,
    report_key: str,
    report_cfg: dict,
    start_date: date_type,
    end_date: date_type,
    account_label: str = "",
    dry_run: bool = False,
) -> int:
    """Delete rows for the period without re-fetching anything."""
    table = report_cfg["db_table"]
    deleted = await delete_period(conn, table, start_date, end_date, account_label, dry_run=dry_run)

    if dry_run:
        log.info(
            "[DRY-RUN][delete_only][%s][%s] Would delete %d rows for %s -> %s",
            account_label, report_key, deleted, start_date, end_date,
        )
        return deleted

    log.info(
        "%s [delete_only][%s][%s] Deleted %d rows for %s -> %s", DEL,
        account_label, report_key, deleted, start_date, end_date,
    )
    await log_run(
        conn, run_date=end_date, status="success" if deleted >= 0 else "failed",
        rows_saved=0, rows_deleted=deleted,
        account_label=account_label, report_key=report_key,
        table_name=table, period_start=start_date, period_end=end_date,
        action_type="delete_only",
        message=f"delete_only: {deleted} rows deleted for {start_date} -> {end_date}",
    )
    return deleted


# ══════════════════════════════════════════════════════════════════════════════
# BROWSER LAUNCH
# ══════════════════════════════════════════════════════════════════════════════

async def launch_browser(pw):
    browser_path = str(Path.home() / ".cache" / "ms-playwright")
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", browser_path)
    try:
        return await pw.chromium.launch(headless=True)
    except Exception as exc:
        if "Executable doesn't exist" in str(exc) or "Please run" in str(exc):
            import subprocess
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
            return await pw.chromium.launch(headless=True)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

async def run(
    report_keys: list[str],
    start_override: Optional[date_type] = None,
    end_override: Optional[date_type]   = None,
    force_refresh: bool                 = False,
    do_delete_only: bool                = False,
    account_filter: Optional[list[str]] = None,
    dry_run: bool                       = False,
):
    # ── Header ────────────────────────────────────────────────────────────────
    mode = "DELETE-ONLY" if do_delete_only else ("FORCED-REFRESH" if force_refresh else "NORMAL")
    log.info("=" * 65)
    log.info("SALESFLO SCRAPER  |  mode=%-15s  dry_run=%s", mode, dry_run)
    log.info("  reports  : %s", report_keys)
    log.info("  accounts : %s", account_filter or "ALL")
    log.info("  dates    : %s -> %s", start_override or "auto", end_override or "auto")
    log.info("=" * 65)

    # ── Validate report keys ──────────────────────────────────────────────────
    invalid = [k for k in report_keys if k not in REPORT_REGISTRY]
    if invalid:
        raise ValueError(f"Unknown report key(s): {invalid}. Available: {list(REPORT_REGISTRY.keys())}")

    yesterday  = datetime.now().date() - timedelta(days=1)
    all_accts  = load_accounts()
    accounts   = filter_accounts(all_accts, account_filter)
    log.info("Running for %d account(s): %s", len(accounts), [a[0] for a in accounts])

    conn = await get_db()
    try:
        await ensure_bot_log_table(conn)
        if not do_delete_only:
            for key in report_keys:
                await ensure_report_table(conn, REPORT_REGISTRY[key])

        # ── DELETE-ONLY mode — no browser needed ──────────────────────────────
        if do_delete_only:
            if not (start_override and end_override):
                raise ValueError("--delete-only requires --start and --end.")

            total_deleted = 0
            for account_label, _, _ in accounts:
                for report_key in report_keys:
                    report_cfg = REPORT_REGISTRY[report_key]
                    deleted = await delete_only(
                        conn, report_key, report_cfg,
                        start_override, end_override,
                        account_label=account_label, dry_run=dry_run,
                    )
                    total_deleted += deleted

            if dry_run:
                log.info("[DRY-RUN] Would delete %d rows total.", total_deleted)
            else:
                log.info("%s Delete-only complete. Total rows deleted: %d", DEL, total_deleted)
            return

        # ── Normal / Forced-refresh mode — browser needed ─────────────────────
        results: list[dict] = []  # track per-report outcome for summary

        async with async_playwright() as pw:
            browser = await launch_browser(pw)
            try:
                for account_label, username, password in accounts:
                    context = await browser.new_context(accept_downloads=True)
                    page    = await context.new_page()
                    try:
                        await login(page, username, password, account_label=account_label)

                        for report_key in report_keys:
                            report_cfg = REPORT_REGISTRY[report_key]

                            # ── Resolve date range ────────────────────────────
                            if start_override and end_override:
                                start_date = start_override
                                end_date   = end_override
                            else:
                                end_date   = yesterday
                                last_saved = await get_last_saved_date(
                                    conn, report_cfg["db_table"], account_label
                                )
                                if last_saved is None:
                                    start_date = yesterday - timedelta(days=7)
                                    log.info(
                                        "  [%s][%s] No prior data -- backfilling from %s",
                                        account_label, report_key, start_date,
                                    )
                                else:
                                    start_date = last_saved + timedelta(days=1)
                                    log.info(
                                        "  [%s][%s] Last saved: %s -- fetching from %s",
                                        account_label, report_key, last_saved, start_date,
                                    )

                            if start_date > end_date:
                                log.info(
                                    "  %s [%s][%s] Already up-to-date.",
                                    SKIP, account_label, report_key,
                                )
                                results.append({
                                    "account": account_label, "report": report_key,
                                    "status": "skipped",
                                    "start": start_date, "end": end_date,
                                    "saved": 0, "deleted": 0,
                                })
                                continue

                            # ── Execute ───────────────────────────────────────
                            _deleted = 0
                            if force_refresh:
                                status, saved = await forced_refresh(
                                    page, conn, report_key, report_cfg,
                                    start_date, end_date, account_label,
                                    dry_run=dry_run,
                                )
                                try:
                                    async with conn.cursor() as _c:
                                        await _c.execute(
                                            "SELECT rows_deleted FROM bot_run_log "
                                            "WHERE account_label=%s AND report_key=%s "
                                            "AND action_type='forced_refresh' "
                                            "ORDER BY id DESC LIMIT 1",
                                            (account_label, report_key),
                                        )
                                        _r = await _c.fetchone()
                                        _deleted = int(_r[0]) if _r else 0
                                except Exception:
                                    pass
                            else:
                                if dry_run:
                                    log.info(
                                        "  %s [%s][%s] Would fetch %s -> %s",
                                        DRY, account_label, report_key, start_date, end_date,
                                    )
                                    status, saved = "dry_run", 0
                                else:
                                    status, saved = await fetch_report(
                                        page, conn, report_key, report_cfg,
                                        start_date, end_date, account_label,
                                    )

                            results.append({
                                "account": account_label, "report": report_key,
                                "status": status,
                                "start": start_date, "end": end_date,
                                "saved": saved, "deleted": _deleted,
                            })

                    finally:
                        try:
                            await context.close()
                        except Exception:
                            pass
            finally:
                try:
                    await browser.close()
                except Exception:
                    pass
    finally:
        try:
            await conn.ensure_closed()
        except Exception:
            pass

    _print_summary(results, dry_run)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(results: list[dict], dry_run: bool = False) -> None:
    """Print a clean, readable summary table after every run."""
    if not results:
        log.info("No reports were processed.")
        return

    # ── Status icon map (ASCII-safe) ──────────────────────────────────────────
    _icon = {
        "success":  "[OK]    ",
        "failed":   "[FAIL]  ",
        "no_data":  "[EMPTY] ",
        "skipped":  "[SKIP]  ",
        "dry_run":  "[DRY]   ",
    }

    # ── Column widths ─────────────────────────────────────────────────────────
    w_acc  = max(len(r["account"]) for r in results) + 2
    w_rep  = max(len(r["report"])  for r in results) + 2
    w_acc  = max(w_acc, 12)
    w_rep  = max(w_rep, 24)

    divider = "-" * (w_acc + w_rep + 52)

    lines = [
        "",
        "=" * (w_acc + w_rep + 52),
        f"  SCRAPER RUN SUMMARY{'  [DRY-RUN -- no DB changes]' if dry_run else ''}",
        "=" * (w_acc + w_rep + 52),
        f"  {'Account':<{w_acc}} {'Report':<{w_rep}} {'Status':<9} {'Date Range':<24} {'Saved':>7} {'Deleted':>8}",
        divider,
    ]

    total_saved   = 0
    total_deleted = 0
    has_fail      = False
    has_data      = False

    for r in results:
        icon    = _icon.get(r["status"], "[?]     ")
        date_range = f"{r['start']} -> {r['end']}"
        saved   = r["saved"]
        deleted = r["deleted"]
        total_saved   += saved
        total_deleted += deleted
        if r["status"] == "failed":
            has_fail = True
        if r["status"] == "success":
            has_data = True

        lines.append(
            f"  {icon}{r['account']:<{w_acc}} {r['report']:<{w_rep}} "
            f"{date_range:<24} {saved:>7,} {deleted:>8,}"
        )

    lines += [
        divider,
        f"  {'TOTAL':<{w_acc + w_rep + 10}} {total_saved:>7,} {total_deleted:>8,}",
        "=" * (w_acc + w_rep + 52),
        "",
    ]

    summary_text = "\n".join(lines)

    # Log to file (always) and print to console
    logging.getLogger(__name__).info(summary_text)

    # Final one-liner status
    if dry_run:
        log.info("%s Dry-run complete. No rows were written.", DRY)
    elif has_fail:
        log.error("%s Run finished WITH ERRORS. Check log above.", ERR)
    elif has_data:
        log.info("%s Run finished successfully. %d rows saved.", OK, total_saved)
    else:
        log.warning("%s Run finished -- all reports returned no data or were skipped.", WARN)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Salesflo Dynamic Report Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
--------
# Normal daily run (all accounts, all reports, auto date)
  python main.py

# Only specific reports
  python main.py --reports end_stock_trend
  python main.py --reports visits_summary,ordered_vs_delivered

# Custom date range
  python main.py --start 2025-01-01 --end 2025-01-31

# Forced refresh — delete + re-scrape
  python main.py --force-refresh --start 2025-03-01 --end 2025-03-31
  python main.py --force-refresh --reports end_stock_trend --start 2025-03-01 --end 2025-03-31
  python main.py --force-refresh --accounts account_2 --start 2025-03-01 --end 2025-03-31
  python main.py --force-refresh --reports visits_summary --accounts account_1 --start 2025-03-01 --end 2025-03-31
  python main.py --force-refresh --accounts account_1,account_3 --reports end_stock_trend --start 2025-03-01 --end 2025-03-31

# Delete only (wipe rows, no re-scrape)
  python main.py --delete-only --reports end_stock_trend --accounts account_2 --start 2025-03-01 --end 2025-03-31

# Dry-run preview (no DB changes — just shows what WOULD happen)
  python main.py --force-refresh --preview --accounts account_1 --reports end_stock_trend --start 2025-03-01 --end 2025-03-31
  python main.py --delete-only --preview --accounts account_2 --start 2025-03-01 --end 2025-03-31

# Utility
  python main.py --list-reports
  python main.py --list-accounts
        """,
    )

    # ── Core options ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--reports", "-r",
        type=str,
        default=os.getenv("ENABLED_REPORTS", ",".join(REPORT_REGISTRY.keys())),
        help="Comma-separated report keys. Default: all registered reports.",
    )
    parser.add_argument(
        "--accounts", "-a",
        type=str,
        default="",
        help=(
            "Comma-separated account labels to include (e.g. account_1,account_3). "
            "Default: ALL configured accounts. Use --list-accounts to see labels."
        ),
    )
    parser.add_argument("--start", "-s", type=str, default="", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   "-e", type=str, default="", help="End date YYYY-MM-DD")

    # ── Mode flags ────────────────────────────────────────────────────────────
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--force-refresh", "-f",
        action="store_true",
        help="Delete existing rows for the period then re-scrape. Requires --start/--end.",
    )
    mode_group.add_argument(
        "--delete-only", "-d",
        action="store_true",
        help="Delete rows for the period WITHOUT re-scraping. Requires --start/--end.",
    )

    # ── Safety flags ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Dry-run: print what would be deleted/fetched without making any DB changes.",
    )

    # ── Info flags ────────────────────────────────────────────────────────────
    parser.add_argument("--list-reports",  "-l", action="store_true", help="List all registered reports and exit.")
    parser.add_argument("--list-accounts", "-L", action="store_true", help="List all configured account labels and exit.")

    args = parser.parse_args()

    # ── --list-reports ────────────────────────────────────────────────────────
    if args.list_reports:
        print("\nRegistered Reports")
        print("=" * 60)
        for key, cfg in REPORT_REGISTRY.items():
            print(f"\n  key       : {key}")
            print(f"  title     : {cfg['title']}")
            print(f"  db_table  : {cfg['db_table']}")
            print(f"  parse_mode: {cfg.get('parse_mode','generic')}")
            print(f"  nav_steps : {cfg.get('nav_steps', [])}")
            print(f"  filters   : {len(cfg.get('filters', []))} defined")
            print(f"  columns   : {len(cfg.get('columns', []))} defined")
        print()
        sys.exit(0)

    # ── --list-accounts ───────────────────────────────────────────────────────
    if args.list_accounts:
        try:
            accts = load_accounts()
        except RuntimeError as exc:
            print(f"Error loading accounts: {exc}")
            sys.exit(1)
        print("\nConfigured Accounts")
        print("=" * 40)
        for label, username, _ in accts:
            print(f"  {label:<20}  {username}")
        print()
        sys.exit(0)

    # ── Parse report keys ─────────────────────────────────────────────────────
    report_keys = [k.strip() for k in args.reports.split(",") if k.strip()]

    # ── Parse account filter ──────────────────────────────────────────────────
    account_filter: Optional[list[str]] = None
    if args.accounts.strip():
        account_filter = [a.strip() for a in args.accounts.split(",") if a.strip()]

    # ── Parse dates ───────────────────────────────────────────────────────────
    start_date: Optional[date_type] = None
    end_date:   Optional[date_type] = None

    if args.start:
        try:
            start_date = datetime.strptime(args.start.strip(), "%Y-%m-%d").date()
        except ValueError:
            parser.error("--start must be YYYY-MM-DD.")
    if args.end:
        try:
            end_date = datetime.strptime(args.end.strip(), "%Y-%m-%d").date()
        except ValueError:
            parser.error("--end must be YYYY-MM-DD.")

    if bool(start_date) != bool(end_date):
        parser.error("--start and --end must both be provided together.")
    if start_date and end_date and start_date > end_date:
        parser.error("--start cannot be after --end.")

    # ── Validate mode constraints ─────────────────────────────────────────────
    if args.force_refresh and not (start_date and end_date):
        parser.error("--force-refresh requires --start and --end.")
    if args.delete_only and not (start_date and end_date):
        parser.error("--delete-only requires --start and --end.")

    # ── Run ───────────────────────────────────────────────────────────────────
    asyncio.run(run(
        report_keys    = report_keys,
        start_override = start_date,
        end_override   = end_date,
        force_refresh  = args.force_refresh,
        do_delete_only = args.delete_only,
        account_filter = account_filter,
        dry_run        = args.preview,
    ))


if __name__ == "__main__":
    main()