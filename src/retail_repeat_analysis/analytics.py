"""Shared computations for the interactive dashboard and its tests."""
import numpy as np
import pandas as pd


def filter_orders(orders, start, end, countries):
    start, stop = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
    if start >= stop:
        raise ValueError("Start date must not follow end date")
    return orders.loc[(orders.InvoiceDate >= start) & (orders.InvoiceDate < stop)
                      & orders.Country.isin(countries)].copy()


def overview(orders):
    return {
        "revenue": float(orders.OrderRevenue.sum()), "orders": int(orders.InvoiceNo.nunique()),
        "customers": int(orders.CustomerID.nunique()),
        "average_order": float(orders.OrderRevenue.mean()) if not orders.empty else 0.0,
    }


def rfm_as_of(orders, as_of):
    from .pipeline import compute_rfm_table
    history = orders.loc[orders.InvoiceDate < pd.Timestamp(as_of)].copy()
    return compute_rfm_table(history.rename(columns={"OrderRevenue": "Revenue"}), analysis_date=as_of)


def cohort_retention(frame, observation_end=None):
    """Observed zero retention is zero; future and incomplete subsequent months are NaN."""
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    data = frame[["CustomerID", "InvoiceDate"]].copy()
    observed_day = pd.Timestamp(observation_end if observation_end is not None else data.InvoiceDate.max()).normalize()
    data = data.loc[data.InvoiceDate < observed_day + pd.Timedelta(days=1)]
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    data["month"] = data.InvoiceDate.dt.to_period("M")
    data["cohort"] = data.groupby("CustomerID")["month"].transform("min")
    data["age"] = (data["month"] - data["cohort"]).apply(lambda value: value.n)
    last_month = observed_day.to_period("M")
    max_age = int((last_month - data.cohort.min()).n)
    counts = data.groupby(["cohort", "age"]).CustomerID.nunique().unstack(fill_value=0)
    counts = counts.reindex(columns=range(max_age + 1), fill_value=0).astype(float)
    for cohort in counts.index:
        for age in counts.columns:
            month = cohort + age
            if month > last_month or (age > 0 and month.end_time.normalize() > observed_day):
                counts.loc[cohort, age] = np.nan
    retention = counts.div(counts[0], axis=0)
    counts.index = counts.index.astype(str)
    retention.index = retention.index.astype(str)
    return counts, retention
