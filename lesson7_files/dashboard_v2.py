"""E-commerce analytics dashboard (v2).

Run with:  streamlit run dashboard_v2.py

Data preparation comes from ``data_loader_v2`` and every metric from
``business_metrics_v2``; this file only handles filters, layout and charts.
"""

import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import business_metrics_v2 as bm
import data_loader_v2 as dl

DATA_DIR = Path(__file__).parent / 'ecommerce_data'
DEFAULT_RANGE = (date(2023, 1, 1), date(2023, 12, 31))
DELIVERY_BUCKET_EDGES = [3, 7]          # upper bounds in days: 1-3, 4-7, 8+
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

COLOR_CURRENT = '#1E3A8A'
COLOR_PREVIOUS = '#94A3B8'
BLUE_GRADIENT = ['#BFDBFE', '#60A5FA', '#2563EB', '#1E3A8A']   # light for low values (map only)
# One distinct colour per bar; colours follow bar order and carry no good/bad meaning.
BAR_PALETTE = ['#0F766E', '#B45309', '#4338CA', '#BE123C', '#4D7C0F',
               '#7E22CE', '#0E7490', '#A16207', '#9D174D', '#475569']
COLOR_POSITIVE = '#15803D'
COLOR_NEGATIVE = '#B91C1C'
COLOR_NEUTRAL = '#64748B'

st.set_page_config(page_title='E-commerce Analytics Dashboard', layout='wide')

st.markdown(f"""
<style>
    .block-container {{ padding-top: 2rem; max-width: 1400px; }}
    .card {{
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08); padding: 1rem 1.25rem;
        display: flex; flex-direction: column; justify-content: center; box-sizing: border-box;
    }}
    .kpi-card {{ height: 140px; }}
    .bottom-card {{ height: 170px; align-items: center; text-align: center; }}
    .card-label {{ font-size: 0.9rem; color: #475569; margin: 0 0 0.35rem 0; }}
    .card-value {{ font-size: 2.1rem; font-weight: 700; color: #0F172A; margin: 0; line-height: 1.15; }}
    .card-value.large {{ font-size: 3rem; }}
    .card-trend {{ font-size: 0.9rem; font-weight: 600; margin: 0.35rem 0 0 0; min-height: 1.2rem; }}
    .trend-positive {{ color: {COLOR_POSITIVE}; }}
    .trend-negative {{ color: {COLOR_NEGATIVE}; }}
    .trend-neutral {{ color: {COLOR_NEUTRAL}; }}
    .stars {{ color: #B45309; font-size: 1.5rem; letter-spacing: 0.15rem; margin: 0.25rem 0; }}
    .card-subtitle {{ font-size: 0.9rem; color: #475569; margin: 0; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data

@st.cache_data
def load_sales() -> pd.DataFrame:
    """Load all delivered sales (every period) once and cache them."""
    tables = dl.load_raw_tables(DATA_DIR)
    return dl.build_sales_dataset(tables, status='delivered')


def filter_period(sales: pd.DataFrame, start: date, end: date, months: list) -> pd.DataFrame:
    """Rows purchased between two dates (inclusive) in the selected calendar months."""
    period = dl.filter_by_period(sales, start, end)
    return period[period['purchase_month'].isin(months)]


def previous_window(start: date, end: date) -> tuple:
    """Equal-length window that ends the day before ``start``."""
    length = (end - start).days + 1
    prev_end = start - timedelta(days=1)
    return prev_end - timedelta(days=length - 1), prev_end


# ------------------------------------------------------------ formatting

def fmt_money(value: float) -> str:
    """Format currency compactly: $725, $300K, $3.4M."""
    if abs(value) >= 1e6:
        return f'${value / 1e6:.1f}M'
    if abs(value) >= 1e3:
        return f'${value / 1e3:.0f}K'
    return f'${value:,.0f}'


def tick_label(value: float) -> str:
    """Axis tick label such as $300K or $2M (no trailing zeros)."""
    if abs(value) >= 1e6:
        return f'${round(value / 1e6, 1):g}M'
    if abs(value) >= 1e3:
        return f'${round(value / 1e3, 1):g}K'
    return f'${value:g}'


def money_ticks(vmax: float, count: int = 5) -> tuple:
    """Return (tick values, tick labels) with round steps from 0 up to ``vmax``."""
    raw_step = vmax / (count - 1) if vmax > 0 else 1
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw_step)
    values = [step * i for i in range(int(vmax // step) + 2)]
    return values, [tick_label(v) for v in values]


def trend_html(current: float, previous, suffix: str = '') -> str:
    """Trend line with arrow, two-decimal percentage and colour; empty if not computable."""
    if previous is None:
        return ''
    change = bm.growth_rate(current, previous)
    if pd.isna(change):
        return ''
    pct = abs(change) * 100
    if round(pct, 2) == 0:
        return f'<span class="trend-neutral">0.00%{suffix}</span>'
    css, arrow = ('trend-positive', '&#9650;') if change > 0 else ('trend-negative', '&#9660;')
    return f'<span class="{css}">{arrow} {pct:.2f}%{suffix}</span>'


def card(label: str, value: str, trend: str = '', css_class: str = 'kpi-card') -> str:
    """HTML for a metric card."""
    return (f'<div class="card {css_class}"><p class="card-label">{label}</p>'
            f'<p class="card-value">{value}</p><p class="card-trend">{trend}</p></div>')


# ---------------------------------------------------------------- charts

LAYOUT = dict(template='plotly_white', font=dict(size=13), margin=dict(t=60, b=50, l=60, r=20), height=380)


def revenue_trend_chart(current, previous, start, prev_start):
    """Solid line for the current period, dashed for the previous one, aligned month by month."""
    base = pd.Period(start, 'M')
    prev_base = pd.Period(prev_start, 'M')

    def aligned(series, origin):
        offsets = [(p - origin).n for p in series.index]
        return [(base + n).strftime('%b %Y') for n in offsets]

    cur = bm.monthly_revenue(current)
    fig = go.Figure()
    if previous is not None:
        prev = bm.monthly_revenue(previous)
        fig.add_trace(go.Scatter(
            x=aligned(prev, prev_base), y=prev.values, mode='lines+markers', name='Previous period',
            line=dict(color=COLOR_PREVIOUS, width=3, dash='dash'),
            customdata=[p.strftime('%b %Y') for p in prev.index],
            hovertemplate='%{customdata}<br>Revenue: %{y:$,.0f}<extra>Previous period</extra>'))
    fig.add_trace(go.Scatter(
        x=aligned(cur, base), y=cur.values, mode='lines+markers', name='Current period',
        line=dict(color=COLOR_CURRENT, width=3),
        customdata=[p.strftime('%b %Y') for p in cur.index],
        hovertemplate='%{customdata}<br>Revenue: %{y:$,.0f}<extra>Current period</extra>'))

    ymax = max(s.max() for s in ([cur.values] + ([prev.values] if previous is not None else [])))
    tickvals, ticktext = money_ticks(ymax)
    fig.update_layout(title='Revenue Trend', xaxis_title='Month', yaxis_title='Revenue (USD)',
                      legend=dict(orientation='h', y=1.12, x=0), **LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor='#E5E7EB', type='category')
    fig.update_yaxes(showgrid=True, gridcolor='#E5E7EB', tickvals=tickvals, ticktext=ticktext, rangemode='tozero')
    return fig


def category_chart(current):
    """Top 10 categories by revenue, descending, one colour per bar."""
    top = bm.revenue_by_category(current, top_n=10)
    fig = go.Figure(go.Bar(
        x=top['product_category_name'], y=top['revenue'],
        marker_color=BAR_PALETTE[:len(top)],
        text=[fmt_money(v) for v in top['revenue']], textposition='outside', cliponaxis=False,
        hovertemplate='%{x}<br>Revenue: %{y:$,.0f}<extra></extra>'))
    tickvals, ticktext = money_ticks(top['revenue'].max() * 1.1)
    fig.update_layout(title='Top 10 Product Categories by Revenue', xaxis_title='Product category',
                      yaxis_title='Revenue (USD)', xaxis_tickangle=-35, **LAYOUT)
    fig.update_xaxes(categoryorder='array', categoryarray=list(top['product_category_name']))
    fig.update_yaxes(tickvals=tickvals, ticktext=ticktext)
    return fig


def state_map(current):
    """US choropleth of revenue by customer state."""
    states = bm.revenue_by_state(current)
    tickvals, ticktext = money_ticks(states['revenue'].max())
    fig = px.choropleth(states, locations='customer_state', color='revenue', locationmode='USA-states',
                        scope='usa', color_continuous_scale=BLUE_GRADIENT,
                        labels={'revenue': 'Revenue', 'customer_state': 'State'})
    fig.update_traces(hovertemplate='%{location}<br>Revenue: %{z:$,.0f}<extra></extra>')
    fig.update_layout(title='Revenue by State',
                      coloraxis_colorbar=dict(title='Revenue (USD)', tickvals=tickvals, ticktext=ticktext),
                      **{**LAYOUT, 'margin': dict(t=60, b=10, l=10, r=10)})
    return fig


def satisfaction_chart(current):
    """Mean review score per delivery-time bucket; None when there is nothing to show."""
    buckets = bm.review_by_delivery_bucket(current, DELIVERY_BUCKET_EDGES)
    if buckets.empty:
        return None
    fig = go.Figure(go.Bar(
        x=buckets['delivery_bucket'].astype(str), y=buckets['review_score'], marker_color=BAR_PALETTE[:len(buckets)],
        text=[f'{v:.2f}' for v in buckets['review_score']], textposition='outside', cliponaxis=False,
        customdata=buckets['orders'],
        hovertemplate='%{x}<br>Average review score: %{y:.2f}<br>Orders: %{customdata:,}<extra></extra>'))
    fig.update_layout(title='Review Score by Delivery Time', xaxis_title='Delivery time (days from purchase)',
                      yaxis_title='Average review score (1-5)', yaxis_range=[0, 5.5], **LAYOUT)
    return fig


def show(fig):
    """Render a Plotly figure at full column width."""
    st.plotly_chart(fig, width='stretch')


# ------------------------------------------------------------------ page

def main():
    sales = load_sales()
    bounds = (sales['order_purchase_timestamp'].min().date(), sales['order_purchase_timestamp'].max().date())
    default = (max(DEFAULT_RANGE[0], bounds[0]), min(DEFAULT_RANGE[1], bounds[1]))

    title_col, filter_col = st.columns([3, 2])
    with title_col:
        st.title('E-commerce Analytics Dashboard')
    with filter_col:
        range_col, month_col = st.columns([3, 2])
        picked = range_col.date_input('Date range', value=default, min_value=bounds[0], max_value=bounds[1])
        months = month_col.multiselect('Months', options=list(range(1, 13)), default=list(range(1, 13)),
                                       format_func=lambda m: MONTH_NAMES[m - 1])

    if not isinstance(picked, (tuple, list)) or len(picked) != 2:
        st.info('Select both a start date and an end date.')
        st.stop()
    start, end = picked

    current = filter_period(sales, start, end, months)
    if current.empty:
        st.warning('No delivered sales for the selected date range and months.')
        st.stop()

    prev_start, prev_end = previous_window(start, end)
    # Compare only when the previous window lies fully inside the period the data covers;
    # a partly covered window would give a misleading comparison.
    previous = filter_period(sales, prev_start, prev_end, months) if prev_start >= bounds[0] else pd.DataFrame()
    if previous.empty:
        previous = None

    month_text = '' if len(months) == 12 else ' (' + ', '.join(MONTH_NAMES[m - 1] for m in sorted(months)) + ' only)'
    st.caption(f'Showing {start:%b %d, %Y} to {end:%b %d, %Y}{month_text}'
               + (f'; compared with {prev_start:%b %d, %Y} to {prev_end:%b %d, %Y}.' if previous is not None
                  else '; the previous period is not fully covered by the data, so comparisons are hidden.'))

    revenue, aov, orders = bm.total_revenue(current), bm.average_order_value(current), bm.order_count(current)
    prev_revenue = bm.total_revenue(previous) if previous is not None else None
    prev_aov = bm.average_order_value(previous) if previous is not None else None
    prev_orders = bm.order_count(previous) if previous is not None else None
    monthly_growth = bm.mean_monthly_growth(current)

    # KPI row
    kpis = [card('Total Revenue', fmt_money(revenue), trend_html(revenue, prev_revenue))]
    if not pd.isna(monthly_growth):
        kpis.append(card('Monthly Growth', f'{monthly_growth * 100:.2f}%'))
    kpis.append(card('Average Order Value', fmt_money(aov), trend_html(aov, prev_aov)))
    kpis.append(card('Total Orders', f'{orders:,}', trend_html(orders, prev_orders)))
    for column, html in zip(st.columns(len(kpis)), kpis):
        column.markdown(html, unsafe_allow_html=True)

    st.write('')

    # 2x2 chart grid
    top_left, top_right = st.columns(2)
    bottom_left, bottom_right = st.columns(2)
    with top_left:
        show(revenue_trend_chart(current, previous, start, prev_start))
    with top_right:
        if current['product_category_name'].notna().any():
            show(category_chart(current))
    with bottom_left:
        if current['customer_state'].notna().any():
            show(state_map(current))
    with bottom_right:
        fig = satisfaction_chart(current)
        if fig is not None:
            show(fig)

    # Bottom row
    delivery = bm.average_delivery_days(current)
    review = bm.average_review_score(current)
    cards = []
    if not pd.isna(delivery):
        prev_delivery = bm.average_delivery_days(previous) if previous is not None else None
        if prev_delivery is not None and pd.isna(prev_delivery):
            prev_delivery = None
        cards.append(card('Average Delivery Time', f'{delivery:.2f} days',
                          trend_html(delivery, prev_delivery), 'bottom-card'))
    if not pd.isna(review):
        filled = int(round(review))
        stars = '&#9733;' * filled + '&#9734;' * (5 - filled)
        cards.append(f'<div class="card bottom-card"><p class="card-value large">{review:.2f}</p>'
                     f'<p class="stars">{stars}</p><p class="card-subtitle">Average Review Score</p></div>')
    if cards:
        for column, html in zip(st.columns(2), cards):
            column.markdown(html, unsafe_allow_html=True)


main()
