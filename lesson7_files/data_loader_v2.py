"""Data loading, cleaning and preparation for the e-commerce analysis (v2).

This module only prepares data. All business metric calculations live in
``business_metrics_v2.py``.

Typical usage:

    tables = load_raw_tables('ecommerce_data/')
    sales = build_sales_dataset(tables, status='delivered')
    period = filter_by_year_month(sales, year=2023, month=None)
"""

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

# Logical table name -> CSV file name.
TABLE_FILES: Dict[str, str] = {
    'orders': 'orders_dataset.csv',
    'order_items': 'order_items_dataset.csv',
    'products': 'products_dataset.csv',
    'customers': 'customers_dataset.csv',
    'reviews': 'order_reviews_dataset.csv',
    'payments': 'order_payments_dataset.csv',
}

# Tables that build_sales_dataset cannot work without.
REQUIRED_TABLES = ('orders', 'order_items')


def load_raw_tables(data_path: Union[str, Path] = 'ecommerce_data/') -> Dict[str, pd.DataFrame]:
    """Read every known CSV file in ``data_path`` into a DataFrame.

    Args:
        data_path: Directory containing the CSV files listed in ``TABLE_FILES``.

    Returns:
        Dictionary mapping logical table name to its raw DataFrame. Files that
        are missing are skipped.

    Raises:
        FileNotFoundError: If a table in ``REQUIRED_TABLES`` is missing.
    """
    data_dir = Path(data_path)
    tables: Dict[str, pd.DataFrame] = {}
    for name, filename in TABLE_FILES.items():
        file_path = data_dir / filename
        if file_path.exists():
            tables[name] = pd.read_csv(file_path)

    missing = [name for name in REQUIRED_TABLES if name not in tables]
    if missing:
        raise FileNotFoundError(f"Required tables not found in '{data_dir}': {missing}")
    return tables


def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Parse order timestamps and derive purchase year, month and delivery time.

    Args:
        orders: Raw orders table.

    Returns:
        Copy of ``orders`` with datetime columns plus ``purchase_year``,
        ``purchase_month`` and ``delivery_days`` (whole days between purchase
        and delivery to the customer; missing for undelivered orders).
    """
    cleaned = orders.copy()
    timestamp_cols = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date',
    ]
    for col in timestamp_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_datetime(cleaned[col])

    purchase = cleaned['order_purchase_timestamp']
    cleaned['purchase_year'] = purchase.dt.year
    cleaned['purchase_month'] = purchase.dt.month
    cleaned['delivery_days'] = (cleaned['order_delivered_customer_date'] - purchase).dt.days
    return cleaned


def clean_order_items(order_items: pd.DataFrame) -> pd.DataFrame:
    """Parse the shipping limit date of the order items table.

    Args:
        order_items: Raw order items table.

    Returns:
        Copy of ``order_items`` with ``shipping_limit_date`` as datetime.
    """
    cleaned = order_items.copy()
    if 'shipping_limit_date' in cleaned.columns:
        cleaned['shipping_limit_date'] = pd.to_datetime(cleaned['shipping_limit_date'])
    return cleaned


def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Parse review dates and keep one review score per order.

    Orders can have more than one review row. Averaging the scores per order
    keeps the sales dataset at one row per order item, so revenue is not
    duplicated when reviews are joined.

    Args:
        reviews: Raw reviews table.

    Returns:
        DataFrame with one row per ``order_id`` and its mean ``review_score``.
    """
    cleaned = reviews.copy()
    for col in ('review_creation_date', 'review_answer_timestamp'):
        if col in cleaned.columns:
            cleaned[col] = pd.to_datetime(cleaned[col])
    return cleaned.groupby('order_id', as_index=False)['review_score'].mean()


def build_sales_dataset(
    tables: Dict[str, pd.DataFrame],
    status: Optional[str] = 'delivered',
) -> pd.DataFrame:
    """Join order items with orders, products, customers and reviews.

    The result has one row per order item and covers all periods. Use
    ``filter_by_year_month`` or ``filter_by_period`` afterwards to select a
    time window.

    Args:
        tables: Output of ``load_raw_tables``.
        status: Keep only orders with this ``order_status``. Pass ``None`` to
            keep every status.

    Returns:
        Sales DataFrame with order, product category, customer state and
        review score columns attached.
    """
    orders = clean_orders(tables['orders'])
    order_items = clean_order_items(tables['order_items'])

    order_cols = [
        'order_id', 'customer_id', 'order_status', 'order_purchase_timestamp',
        'order_delivered_customer_date', 'purchase_year', 'purchase_month', 'delivery_days',
    ]
    sales = order_items.merge(orders[order_cols], on='order_id', how='inner')

    if status is not None:
        sales = sales[sales['order_status'] == status]

    if 'products' in tables:
        sales = sales.merge(
            tables['products'][['product_id', 'product_category_name']],
            on='product_id', how='left',
        )
    if 'customers' in tables:
        sales = sales.merge(
            tables['customers'][['customer_id', 'customer_state']],
            on='customer_id', how='left',
        )
    if 'reviews' in tables:
        sales = sales.merge(clean_reviews(tables['reviews']), on='order_id', how='left')

    return sales.reset_index(drop=True)


def filter_by_year_month(
    data: pd.DataFrame,
    year: int,
    month: Optional[int] = None,
) -> pd.DataFrame:
    """Select rows purchased in a given year, optionally a single month.

    Args:
        data: DataFrame with ``purchase_year`` and ``purchase_month`` columns.
        year: Calendar year to keep.
        month: Calendar month (1-12) to keep, or ``None`` for the full year.

    Returns:
        Filtered copy of ``data``.
    """
    mask = data['purchase_year'] == year
    if month is not None:
        mask &= data['purchase_month'] == month
    return data[mask].copy()


def filter_by_period(
    data: pd.DataFrame,
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
) -> pd.DataFrame:
    """Select rows purchased between two dates, both inclusive.

    Args:
        data: DataFrame with an ``order_purchase_timestamp`` datetime column.
        start: First day of the period, e.g. ``'2023-04-01'``.
        end: Last day of the period, e.g. ``'2023-06-30'``.

    Returns:
        Filtered copy of ``data``.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    purchase = data['order_purchase_timestamp']
    return data[(purchase >= start_ts) & (purchase < end_ts)].copy()
