"""Business metric calculations for the e-commerce analysis (v2).

Every function takes an already-filtered sales DataFrame (see
``data_loader_v2.build_sales_dataset`` and the filter helpers) and returns a
scalar, Series or DataFrame. No plotting or printing happens here.

Required columns by function:
    price, order_id                          revenue and order metrics
    order_purchase_timestamp                 monthly metrics
    product_category_name                    revenue_by_category
    customer_state                           revenue_by_state
    review_score, delivery_days              customer experience metrics
    order_status                             order_status_mix (use unfiltered orders)
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def growth_rate(current: float, previous: float) -> float:
    """Return the relative change from ``previous`` to ``current``.

    Args:
        current: Value for the analysis period.
        previous: Value for the comparison period.

    Returns:
        ``(current - previous) / previous`` as a fraction, or ``nan`` if
        ``previous`` is zero or missing.
    """
    if previous is None or pd.isna(previous) or previous == 0:
        return float('nan')
    return (current - previous) / previous


def total_revenue(data: pd.DataFrame) -> float:
    """Sum of item ``price`` over all rows, in currency units."""
    return float(data['price'].sum())


def order_count(data: pd.DataFrame) -> int:
    """Number of distinct orders."""
    return int(data['order_id'].nunique())


def average_order_value(data: pd.DataFrame) -> float:
    """Mean revenue per order (item prices summed per order, then averaged).

    Returns ``nan`` when ``data`` is empty.
    """
    if data.empty:
        return float('nan')
    return float(data.groupby('order_id')['price'].sum().mean())


def monthly_revenue(data: pd.DataFrame) -> pd.Series:
    """Revenue per calendar month, indexed by year-month period.

    Grouping by year and month keeps months from different years separate, so
    the result is correct for periods that span more than one year. Only
    months present in ``data`` appear in the result.
    """
    months = data['order_purchase_timestamp'].dt.to_period('M')
    return data.groupby(months)['price'].sum().sort_index()


def mean_monthly_growth(data: pd.DataFrame) -> float:
    """Average month-over-month revenue growth as a fraction.

    Returns ``nan`` when ``data`` covers fewer than two months.
    """
    changes = monthly_revenue(data).pct_change().dropna()
    return float(changes.mean()) if not changes.empty else float('nan')


def revenue_by_category(data: pd.DataFrame, top_n: Optional[int] = None) -> pd.DataFrame:
    """Revenue and revenue share per product category, largest first.

    Args:
        data: Sales DataFrame with ``product_category_name``.
        top_n: Keep only the largest ``top_n`` categories; ``None`` keeps all.

    Returns:
        DataFrame with ``product_category_name``, ``revenue`` and
        ``revenue_share`` (fraction of the revenue in ``data``, so the share
        is relative to all categories even when ``top_n`` truncates).
    """
    revenue = (
        data.groupby('product_category_name')['price'].sum()
        .sort_values(ascending=False)
        .rename('revenue')
        .reset_index()
    )
    total = data['price'].sum()
    revenue['revenue_share'] = revenue['revenue'] / total if total else np.nan
    return revenue.head(top_n) if top_n else revenue


def revenue_by_state(data: pd.DataFrame) -> pd.DataFrame:
    """Revenue per customer state, largest first.

    Returns:
        DataFrame with ``customer_state`` and ``revenue``.
    """
    return (
        data.groupby('customer_state')['price'].sum()
        .sort_values(ascending=False)
        .rename('revenue')
        .reset_index()
    )


def _one_row_per_order(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Reduce item-level rows to one row per order for order-level metrics."""
    return data[['order_id', *columns]].drop_duplicates('order_id')


def average_review_score(data: pd.DataFrame) -> float:
    """Mean review score across orders that have a review (nan if none)."""
    scores = _one_row_per_order(data, ['review_score'])['review_score']
    return float(scores.mean()) if scores.notna().any() else float('nan')


def review_score_distribution(data: pd.DataFrame) -> pd.Series:
    """Share of reviewed orders per rounded review score, indexed by score.

    Scores are rounded to whole numbers before counting. Shares sum to 1.
    """
    scores = _one_row_per_order(data, ['review_score'])['review_score'].dropna().round()
    return scores.value_counts(normalize=True).sort_index().rename('share')


def average_delivery_days(data: pd.DataFrame) -> float:
    """Mean days from purchase to delivery across orders (nan if none)."""
    days = _one_row_per_order(data, ['delivery_days'])['delivery_days']
    return float(days.mean()) if days.notna().any() else float('nan')


def review_by_delivery_days(data: pd.DataFrame) -> pd.DataFrame:
    """Mean review score for each whole number of delivery days.

    Returns:
        DataFrame with ``delivery_days``, ``review_score`` (mean) and ``orders``
        (number of reviewed orders behind each mean).
    """
    orders = _one_row_per_order(data, ['delivery_days', 'review_score']).dropna()
    return (
        orders.groupby('delivery_days')['review_score']
        .agg(review_score='mean', orders='count')
        .reset_index()
    )


def delivery_bucket_edges_to_labels(edges: Sequence[int]) -> list:
    """Build labels such as ``['1-3 days', '4-7 days', '8+ days']`` from upper edges.

    Args:
        edges: Ascending inclusive upper bounds of every bucket except the
            last open-ended one, e.g. ``[3, 7]``.
    """
    labels, lower = [], 1
    for upper in edges:
        labels.append(f'{lower}-{upper} days')
        lower = upper + 1
    labels.append(f'{lower}+ days')
    return labels


def review_by_delivery_bucket(
    data: pd.DataFrame,
    edges: Sequence[int],
) -> pd.DataFrame:
    """Mean review score per delivery-time bucket.

    The bucket edges are an input, not a built-in business rule.

    Args:
        data: Sales DataFrame with ``delivery_days`` and ``review_score``.
        edges: Ascending inclusive upper bounds of each bucket except the last,
            e.g. ``[3, 7]`` gives 1-3, 4-7 and 8+ days.

    Returns:
        DataFrame with ``delivery_bucket``, ``review_score`` (mean) and
        ``orders``, in bucket order. Empty buckets are omitted.
    """
    orders = _one_row_per_order(data, ['delivery_days', 'review_score']).dropna()
    labels = delivery_bucket_edges_to_labels(edges)
    bins = [-np.inf, *edges, np.inf]
    orders['delivery_bucket'] = pd.cut(orders['delivery_days'], bins=bins, labels=labels)
    return (
        orders.groupby('delivery_bucket', observed=True)['review_score']
        .agg(review_score='mean', orders='count')
        .reset_index()
    )


def order_status_mix(orders: pd.DataFrame) -> pd.Series:
    """Share of orders per ``order_status``, largest first.

    Pass an orders table that was NOT filtered by status (for example
    ``data_loader_v2.clean_orders`` output restricted to the analysis period).
    """
    return orders['order_status'].value_counts(normalize=True).rename('share')
