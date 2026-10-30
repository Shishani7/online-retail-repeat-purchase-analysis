from pathlib import Path

import pandas as pd
import pytest

from retail_repeat_analysis.modeling import (
    FEATURE_COLUMNS,
    build_customer_features,
    build_snapshot_dataset,
    chronological_split,
)
from retail_repeat_analysis.pipeline import build_order_table


def transactions(rows):
    df = pd.DataFrame(rows, columns=["CustomerID", "InvoiceNo", "InvoiceDate", "Revenue", "StockCode"])
    df["InvoiceDate"] = pd.to_datetime(df.InvoiceDate)
    df["Quantity"] = 1
    df["Country"] = "UK"
    return df


def test_future_transactions_cannot_change_historical_features():
    original = transactions([("a", "1", "2011-01-05", 10., "x"), ("a", "2", "2011-02-05", 20., "y")])
    future = transactions([("a", "99", "2011-10-05", 1_000_000., "future-only")])
    combined = pd.concat([original, future], ignore_index=True)
    before = build_customer_features(original, build_order_table(original), "2011-03-01")
    after = build_customer_features(combined, build_order_table(combined), "2011-03-01")
    pd.testing.assert_frame_equal(before, after)
    assert before.iloc[0].monetary_total == 30
    assert before.iloc[0].unique_products == 2


def test_label_window_includes_cutoff_and_excludes_end():
    tx = transactions([
        ("a", "1", "2011-01-01", 10., "x"), ("b", "2", "2011-01-02", 10., "x"),
        ("a", "3", "2011-03-01", 10., "x"), ("b", "4", "2011-05-30", 10., "x"),
        ("new", "5", "2011-03-01", 10., "x"), ("a", "6", "2011-12-09", 10., "x"),
    ])
    snapshots = build_snapshot_dataset(tx, build_order_table(tx), min_history_months=1)
    march = snapshots.loc[snapshots.snapshot_date == pd.Timestamp("2011-03-01")].set_index("CustomerID")
    assert march.loc["a", "target_repeat_purchase"] == 1
    assert march.loc["b", "target_repeat_purchase"] == 0
    assert "new" not in march.index
    assert (snapshots.label_window_end <= tx.InvoiceDate.max()).all()


def test_published_split_has_mature_labels_and_untouched_test():
    root = Path(__file__).resolve().parents[1]
    snapshots = pd.read_csv(root / "data/processed/ml_snapshot_dataset.csv", dtype={"CustomerID": str})
    selection, validation, final_train, test = chronological_split(snapshots)
    assert selection.label_window_end.max() <= validation.snapshot_date.min()
    assert validation.label_window_end.max() <= test.snapshot_date.min()
    assert final_train.label_window_end.max() <= test.snapshot_date.min()
    assert final_train.snapshot_date.max() < test.snapshot_date.min()
    assert test.snapshot_date.nunique() == 1
    assert not {"CustomerID", "target_repeat_purchase", "snapshot_date", "label_window_end"}.intersection(FEATURE_COLUMNS)


def test_refuses_insufficient_history_for_model_selection():
    data = pd.DataFrame({"snapshot_date": ["2011-08-01", "2011-09-01"], "target_repeat_purchase": [0, 1]})
    with pytest.raises(ValueError, match="validation"):
        chronological_split(data)

