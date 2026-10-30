import numpy as np
import pandas as pd
import pytest

from retail_repeat_analysis.analytics import cohort_retention, filter_orders, overview, rfm_as_of


def orders():
    frame = pd.DataFrame([
        ("a", "1", "2011-01-01 10:00", 10., "UK"),
        ("b", "2", "2011-01-02 10:00", 20., "France"),
        ("a", "3", "2011-02-28 23:59", 30., "UK"),
        ("c", "4", "2011-03-01 00:00", 40., "UK"),
    ], columns=["CustomerID", "InvoiceNo", "InvoiceDate", "OrderRevenue", "Country"])
    frame["InvoiceDate"] = pd.to_datetime(frame.InvoiceDate)
    return frame


def test_period_is_inclusive_and_country_filter_applies():
    selected = filter_orders(orders(), "2011-02-01", "2011-02-28", ["UK"])
    assert selected.InvoiceNo.tolist() == ["3"]
    assert overview(selected) == {"revenue": 30., "orders": 1, "customers": 1, "average_order": 30.}
    assert filter_orders(orders(), "2011-01-01", "2011-02-28", []).empty


def test_empty_selection_and_reversed_dates():
    assert overview(orders().iloc[:0])["average_order"] == 0
    with pytest.raises(ValueError):
        filter_orders(orders(), "2011-03-01", "2011-02-01", ["UK"])


def test_observed_zero_is_not_unobserved_future():
    _, retention = cohort_retention(orders(), "2011-03-31")
    assert retention.loc["2011-01", 0] == 1
    assert retention.loc["2011-01", 1] == .5
    assert retention.loc["2011-01", 2] == 0
    assert np.isnan(retention.loc["2011-03", 1])


def test_incomplete_next_month_is_hidden():
    _, retention = cohort_retention(orders(), "2011-03-15")
    assert np.isnan(retention.loc["2011-01", 2])
    assert retention.loc["2011-01", 1] == .5


def test_small_rfm_population_and_explicit_cutoff():
    rfm, summary = rfm_as_of(orders().query("CustomerID == 'a'"), "2011-02-01")
    assert len(rfm) == 1
    assert rfm.iloc[0].Monetary == 10
    assert rfm.iloc[0].Recency == 30
    assert summary["Клиенты"].sum() == 1

