# Bazaar Prime Analytics Dashboard

Streamlit-based analytics dashboard for sales growth, booker performance, field-force deep analysis, and read-only custom SQL exploration.

## Documentation

- End-user complete guide: [END_USER_GUIDE.md](END_USER_GUIDE.md)

## Features (High Level)

- Multi-tab analytics UI (Sales, Booker Performance, Deep Analysis, Custom Query)
- KPI cards with period-over-period deltas
- Target vs achievement views, heatmaps, treemaps, sankey, cohort, segmentation
- Booker-level scoring, leaderboard, and brand low-focus summaries
- Custom SQL tab with table/column browser and strict SELECT-only safety checks

## Run Locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start app:

   ```bash
   streamlit run streamlit_app.py
   ```

## Configuration

- Database credentials are read from Streamlit secrets and/or environment variables.
- Make sure DB connectivity is configured before running.
