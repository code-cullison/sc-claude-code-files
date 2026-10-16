# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements.txt
streamlit run dashboard.py              # dashboard
jupyter notebook EDA_Refactored.ipynb   # notebook analysis
```

There is no test suite, linter, or build step. All paths (e.g. `ecommerce_data/`) are relative, so run everything from this directory.

## Architecture

Flat, three-module Python project over the CSVs in `ecommerce_data/` (orders, order_items, products, customers, order_reviews, order_payments). The data flows in one direction:

1. **`data_loader.py` — `EcommerceDataLoader`**: `load_raw_data()` → `process_all_data()` (cleans orders, order_items, reviews into `processed_data`) → `create_sales_dataset(year_filter, month_filter, status_filter='delivered')`, which builds the single denormalized sales DataFrame. It starts from cleaned `order_items`, merges orders, then merges products, customers and reviews from `raw_data` (not `processed_data`), and derives `delivery_days`. `load_and_process_data()` is the convenience entry point returning `(loader, processed_data)`.
2. **`business_metrics.py`**: `BusinessMetricsCalculator(sales_data)` computes revenue, monthly trend, category, geographic, satisfaction and delivery metrics per year; `generate_comprehensive_report(current_year, previous_year)` bundles them. `MetricsVisualizer(report)` renders the report with matplotlib/plotly. It requires the columns `price`, `order_id`, `purchase_year` (checked in `_validate_data`).
3. **`dashboard.py`** (Streamlit): calls `load_and_process_data` through an `@st.cache_data` function, then calls `loader.create_sales_dataset` itself for the selected year/month and for the previous year. It builds its own plotly charts and KPI cards. It does **not** use `MetricsVisualizer`, and it has its own `format_currency` and delivery-categorization helpers that duplicate ones in `business_metrics.py` and `data_loader.py`. Change both places when changing shared behavior.

`EDA.ipynb` is the original unrefactored analysis; `EDA_Refactored.ipynb` is the version that uses the modules above and is configured via `ANALYSIS_YEAR`, `COMPARISON_YEAR`, `ANALYSIS_MONTH` and `DATA_PATH` in its first code cell.

## Gotchas

- Only `status_filter='delivered'` orders are counted by default, so revenue and order numbers exclude other statuses.
- Reviews are merged on `order_id`, so orders with several reviews or items can duplicate rows. Check this before summing or counting on `sales_data`.
- The dashboard defaults the year selector to 2023 when present in the data.
