# Bazaar Prime Analytics — End User Guide

This guide explains all dashboard features for business users.

## 1) Access & Login

- Open the app URL shared by your admin.
- Enter username/password on the login screen.
- After login, your username appears in the sidebar.
- Use **Logout** from sidebar any time.

## 2) Global Controls (Apply across dashboard)

### Sidebar: Period
Choose one of:
- Last 7 Days
- Last 30 Days
- This Month
- Last Month
- Last 3 Months
- YTD
- Custom (manual start/end dates)

### Sidebar: Location
Select city/distributor:
- Karachi (`D70002202`)
- Lahore (`D70002246`)

> Tip: Change period/location first, then analyze any tab.

## 3) Tab Overview

The app has 4 tabs:
1. **📈 Sales Growth Analysis**
2. **🎯 Booker Performance**
3. **🧭 Booker & Field Force Deep Analysis**
4. **🧪 Custom Query**

---

## 4) Tab 1 — Sales Growth Analysis

### A) KPI Cards
Top KPI cards show:
- Total Revenue
- Total Litres
- Total Orders
- Avg Order Value (AOV)

Each KPI includes comparison vs:
- Last Year
- Last Month

### B) Channel-wise AOV Cards
- Channel-level AOV cards
- Growth vs last year/month

### C) Channel Performance Charts
- Channel-wise performance comparison (Value/Ltr toggle)
- Channel-wise growth chart (Value/Ltr toggle)

### D) Brand & Deliveryman Growth
- Brand-wise growth percentage (Value/Ltr toggle)
- Deliveryman-wise growth percentage (Value/Ltr toggle)

### E) Targets & Achievement
- Target vs Achievement comparison (Value/Ltr)
- Booker/period target achievement heatmap
- Channel heatmap panel

### F) Sunburst + Productivity
- Channel/Brand/DM sunburst chart
- Brand productivity companion chart
- DM/channel filters

### G) Booker Less-Than-Half-Carton Analysis
- Period selector (last 1–4 months)
- Summary table by booker/brand
- Optional drill-down detail table

### H) MOPU & Drop Size Analysis
- Combined chart for:
  - Total Orders
  - Drop Size
  - SKU per Bill
  - MOPU

---

## 5) Tab 2 — Booker Performance

### A) Booker Treemap Controls
- Period selector (based on available data)
- Channel filter
- Achievement filter (All / Below 50/60/70)
- Brand multi-filter

### B) Treemaps
- Booker-level treemap
- Brand→Booker treemap

### C) OB/Brand Sankey
- Flow direction toggle:
  - OB → Brand
  - Brand → OB
- Top N / Bottom N controls
- Split layout / force-left layout toggles

### D) Booker GMV Calendar Heatmap
- Month window (3/6/12 months)
- Booker filter
- Daily GMV/order intensity view

---

## 6) Tab 3 — Booker & Field Force Deep Analysis

### A) Deep Filters
- Booker multiselect
- Channel multiselect

### B) Performance Charts
- Route-wise achieved vs target
- Daily sales & orders trend
- Daily calls trend (with fallback to latest visit data if needed)

### C) Customer Activity Insights
- Weekly cohort retention heatmap
- Segmentation donut:
  - Power Users
  - Regular
  - Occasional
  - Dormant
- Booker-wise segmentation stacked chart
- Right-side segmentation table with table-only filters

### D) Brand Focus / Scoring
- **Booker Brand-Level Scoring**
- Top/Bottom brand by booker table
- Selected booker brand score detail chart
- Low-focus threshold slider
- Booker-wise low-focus brand summary with color chips

### E) KPI Blocks (Unified card style)
**Calls KPIs**:
- Avg Strike Rate
- Avg Calls/Day
- Productive Calls %
- SKU/Bill
- Total Visits

**Summary KPIs**:
- Active Bookers
- Total NMV
- Total Orders
- Avg AOV

### F) Leaderboard
- Top 5 / Bottom 5 performers
- Weighted performance score shown in tooltip logic

---

## 7) Tab 4 — Custom Query (Read-only SQL)

This tab is for business users who need ad-hoc analysis.

### A) Saved Sample Queries
- Prebuilt examples (Booker-wise NMV, Brand sales, Daily trend, Top outlets)
- Use **Load Sample** to fill SQL editor

### B) Table & Column Structure Browser
- Expand **Table & Column Structure**
- Search table/column by keyword
- Select a table to see all columns + data types
- Interactive column selection helps build SQL quickly

### C) Fast Query Builder Helpers
- **Insert Table Name** button
- Multi-select columns in table grid auto-builds query:
  - `SELECT col1, col2 ... FROM table LIMIT 100`
- Search-match grid can also inject selected columns

### D) Query Validation & Security Rules
Only read-only SQL is allowed.

Allowed:
- Single `SELECT` statement

Blocked:
- `UPDATE`, `DELETE`, `INSERT`, `DROP`, `ALTER`, `TRUNCATE`, `CREATE`, etc.
- Multiple statements in one run

### E) Output & Download
- Result preview table
- Row count success message
- CSV download for preview rows

---

## 8) How to Use (Recommended Workflow)

1. Select **Period** and **Location** from sidebar.
2. Review high-level KPIs in Tab 1.
3. Deep-dive booker behavior in Tab 2 and Tab 3.
4. Use Tab 4 for custom SQL questions.
5. Export preview data as CSV when needed.

---

## 9) Common Tips

- If a chart is empty, first check filters (date/booker/channel).
- For table-only filters (in some sections), those filters affect only that table.
- Hover tooltips on charts/chips for exact values and definitions.
- In Custom Query, start from a sample query and then edit.

---

## 10) Common Errors & Fixes

### "No data available"
- Relax filters (wider date range, remove strict channel/booker filters).

### "Only SELECT queries are allowed"
- Rewrite query as a single `SELECT` statement.

### Query failed
- Verify table/column names from the structure browser.
- Remove unsupported SQL syntax for current database.

---

## 11) Data Notes

- Metrics are computed from backend transactional/target tables configured by admin.
- Some comparisons use prior period windows (last month/year or previous date window).
- KPI/leaderboard calculations are automated in app logic.

---

## 12) Support

For access issues, data mismatches, or new feature requests, contact your dashboard admin/analytics team and share:
- selected period
- selected location
- tab name
- screenshot/query used
