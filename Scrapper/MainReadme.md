# Salesflo Dynamic Report Scraper

A **configuration-driven** scraper for Salesflo.  
Adding a new report requires **zero new Python code** — only a config block in `report_registry.py`.

---

## Architecture

```
salesflo_scraper/
├── report_registry.py   ← ONLY file you edit to add/change reports
├── scraper_engine.py    ← Generic engine: login, nav, filters, parse, save
├── main.py              ← CLI entry point + account loader + orchestration
├── .env.example         ← Copy to .env and fill credentials
└── secrets.toml.example ← Alternative: Streamlit Cloud / TOML credentials
```

### How It Works

```
main.py
  └─ load_accounts()          ← reads all distributor credentials
  └─ for each account:
       login(page)
       for each report:
         navigate_to_report() ← uses nav_steps from registry
         apply_filters()      ← uses filters list from registry
         generate_and_parse() ← generic table reader + parser
         save_*_rows()        ← auto-creates/migrates DB table
```

The key insight: **everything report-specific lives in `report_registry.py`**.  
`scraper_engine.py` is a generic driver that never needs to change.

---

## Installation

```bash
pip install playwright aiomysql python-dotenv openpyxl
playwright install chromium
```

---

## Credentials

### Option A — `.env` file
```bash
cp .env.example .env
# Edit .env with your DB and account credentials
```

### Option B — Streamlit `secrets.toml`
```bash
cp secrets.toml.example ~/.streamlit/secrets.toml
# Edit with your credentials
```

Multiple accounts — just add more numbered entries:
```env
ACCOUNT_1_USERNAME=dist1@example.com
ACCOUNT_1_PASSWORD=pass1
ACCOUNT_2_USERNAME=dist2@example.com
ACCOUNT_2_PASSWORD=pass2
# ...unlimited accounts
```

---

## Usage

```bash
# Run all reports (auto date range)
python main.py

# Run specific reports
python main.py --reports end_stock_trend
python main.py --reports visits_summary,ordered_vs_delivered

# Custom date range
python main.py --start 2025-01-01 --end 2025-01-31

# Force refresh (delete + reload) a period
python main.py --reports end_stock_trend --force-refresh --start 2025-03-01 --end 2025-03-31

# List all configured reports
python main.py --list-reports
```

---

## Adding a New Report

Open `report_registry.py` and add a new entry to `REPORT_REGISTRY`.  
That's it. No other file needs to change.

```python
"my_new_report": {
    "title":      "My New Report",           # Display name (used for nav click)
    "db_table":   "my_new_report_rows",      # MySQL table (auto-created)
    "parse_mode": "generic",                 # "generic" or "end_stock" (pivot)
    "save_mode":  "generic",

    # Menu path: what to click after "Reports"
    "nav_steps": [
        "text=Sale Reports",                 # click the section heading
        "text=My New Report",                # click the report tile/link
    ],

    # Any selector that proves the report form is loaded
    "ready_selectors": [
        'input[name="dt1"]',
        'button:has-text("View Report")',
    ],

    # Filters to apply — each dict is one UI interaction
    "filters": [
        # Date range (role tells engine which date to fill)
        {"type": "date", "role": "start_date", "selectors": ['input[name="dt1"]'], "label": "Start Date"},
        {"type": "date", "role": "end_date",   "selectors": ['input[name="dt2"]'], "label": "End Date"},

        # Radio button (inspect element → find name + value attributes)
        {"type": "radio", "name": "TP", "value": "2", "selectors": ['input[name="TP"][value="2"]'], "label": "Daily"},

        # Select dropdown (inspect element → find <select> id + option value)
        {"type": "select", "selectors": ['select#su'], "value": "5", "label": "Value"},

        # Checkbox — use id from DevTools (like tr#DynamicCheckBoxTr_ShowBoxes)
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

    # Column definitions — controls DB schema AND header matching
    "columns": [
        # name     : exact column header text in the report
        # aliases  : alternate spellings (lowercase) — for flexible matching
        # db_col   : column name in MySQL
        # db_type  : MySQL data type
        # is_date / is_float / is_int : type casting flags
        {"name": "Distributor Code", "aliases": ["distributor code"],  "db_col": "distributor_code", "db_type": "VARCHAR(100)"},
        {"name": "SKU Code",         "aliases": ["sku code", "sku"],   "db_col": "sku_code",         "db_type": "VARCHAR(100)"},
        {"name": "Order Date",       "aliases": ["order date"],        "db_col": "order_date",       "db_type": "DATE",          "is_date": True},
        {"name": "Total Value",      "aliases": ["total value","value"],"db_col":"total_value",       "db_type": "DECIMAL(18,4)", "is_float": True},
        {"name": "Total Units",      "aliases": ["total units","units"],"db_col":"total_units",       "db_type": "DECIMAL(18,4)", "is_float": True},
    ],
},
```

### Finding Selector IDs (DevTools)

As shown in the screenshots:
1. Open the report page in browser.
2. Right-click the filter control → **Inspect**.
3. Note the `id`, `name`, or parent `tr id` attribute.
4. Use it as a selector in your filter's `selectors` list.

Examples from the screenshots:
- `tr#DynamicCheckBoxTr_ShowBoxes input[type="checkbox"]` → Show Boxes checkbox
- `tr#DynamicCheckBoxTr_ShowTonnage input[type="checkbox"]` → Show Tonnage
- `tr#DynamicCheckBoxTr_ShowHSCode input[type="checkbox"]` → Show HS Code
- `tr#ShowWeightTypes input[type="checkbox"]` → Show SKU Weight Type

---

## Filter Types Reference

| type       | Required fields                           | Description                    |
|------------|-------------------------------------------|--------------------------------|
| `date`     | `role` (start_date/end_date), `selectors` | Fills a date picker input      |
| `radio`    | `name`, `value`, `selectors`              | Selects a radio button         |
| `select`   | `value`, `selectors`                      | Sets a `<select>` dropdown     |
| `checkbox` | `checked` (bool), `selectors`             | Checks or unchecks a checkbox  |
| `text`     | `value`, `selectors`                      | Types text into an input       |

---

## DB Table Auto-Creation

When you run a new report for the first time, the engine:
1. Reads the `columns` list from the registry.
2. `CREATE TABLE IF NOT EXISTS` with those columns + standard fields (`id`, `report_date`, `account_label`, `row_hash`, `row_json`, `fetched_at`).
3. On subsequent runs it `ALTER TABLE ADD COLUMN` for any new columns you add — zero downtime migration.

---

## Scheduling (cron)

```cron
# Every day at 10 PM
0 22 * * * cd /path/to/salesflo_scraper && python main.py >> /var/log/salesflo.log 2>&1
```