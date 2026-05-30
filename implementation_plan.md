# Customer Churn Intelligence — Split into 2-3 Parts

The existing Churn Intelligence section (lines 7384–7468 in `streamlit_app.py`) is a flat, single block inside the Summary tab. The goal is to expand it into a **3-part tabbed layout** using `st.tabs()` so each component has dedicated visual space with richer detail.

## Proposed Layout

The current single block will be replaced with **3 sub-tabs** organized logically:

### Part 1: `📊 Overview & Trends`
| Component | What's included |
|---|---|
| **KPI Metrics Row** | Active Customers, Churned Stores, New Customers, At Risk Stores (existing cards) |
| **Monthly Trends** | Dual-axis chart: active customers line + revenue bars (existing chart enhanced with churn-rate overlay line) |
| **Churn Rate Gauge** | New gauge showing current month churn rate vs previous month |

### Part 2: `🔬 Deep Analysis`
| Component | What's included |
|---|---|
| **Cohort Retention Heatmap** | Existing cohort heatmap (promoted to half-width) |
| **Retention Buckets** | Existing horizontal bar chart (promoted to half-width) |
| **Churn Risk Table** | Top churn-risk stores table with risk score, revenue, lifecycle segment, days-since-last-order |

### Part 3: `🧠 ML Scoring & Lifecycle`
| Component | What's included |
|---|---|
| **ML Churn Scoring** | New detailed scatter plot: risk_score vs revenue, colored by lifecycle segment, with hover details |
| **Revenue-Weighted Risk** | New horizontal waterfall/bar breakdown per lifecycle segment showing revenue at risk |
| **Lifecycle Engine** | Existing donut chart (promoted to larger size) + distribution summary table |

---

## Proposed Changes

### [MODIFY] [streamlit_app.py](file:///d:/Dash/BazaarPrime/streamlit_app.py)

#### 1. New chart functions (insert after line ~7123, after `create_retention_bucket_chart`)

- **`create_churn_rate_gauge(churn_rate, prev_churn_rate)`** — semicircle gauge for churn rate
- **`create_ml_risk_scatter(df)`** — scatter: risk_score vs revenue, colored by lifecycle_segment  
- **`create_revenue_at_risk_chart(df)`** — horizontal bar showing revenue by lifecycle segment with risk weighting
- **`create_lifecycle_distribution_table(df)`** — HTML summary table of lifecycle segment counts + revenue share

#### 2. Replace existing churn block (lines 7384–7468)

**Current:** Flat layout with 2-column row, 3-column row, raw markdown text  
**New:** `st.tabs(["📊 Overview & Trends", "🔬 Deep Analysis", "🧠 ML Scoring & Lifecycle"])` with each tab containing the corresponding components arranged in a clean grid.

> [!IMPORTANT]
> No changes to the data-fetching layer (`fetch_churn_intelligence_data`, `_prepare_churn_scores`). All existing SQL queries and scoring logic remain untouched. Only the presentation layer changes.

## Verification Plan

### Automated Tests
- Run `python -c "import streamlit_app"` to verify no syntax/import errors
- Run `streamlit run streamlit_app.py` and verify the churn section renders with 3 tabs

### Manual Verification  
- Navigate to the Summary tab → Customer Churn Intelligence section
- Verify 3 sub-tabs appear and each contains the correct components
- Verify existing charts still render correctly
