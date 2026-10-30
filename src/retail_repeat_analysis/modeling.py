"""History-only features and chronological model selection with matured labels."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HORIZON_DAYS = 90
THRESHOLD = 0.5
NUMERIC_FEATURES = [
    "recency_days", "frequency_orders", "monetary_total", "avg_order_value", "max_order_value",
    "avg_items_per_order", "active_months", "unique_products", "total_units",
    "avg_days_between_orders", "customer_tenure_days",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + ["country"]


def build_customer_features(clean_df, order_df, cutoff):
    """Build an existing-customer profile using transactions strictly before cutoff."""
    cutoff = pd.Timestamp(cutoff)
    orders = order_df.loc[order_df.InvoiceDate < cutoff].copy()
    tx = clean_df.loc[clean_df.InvoiceDate < cutoff]
    if orders.empty:
        return pd.DataFrame(columns=["CustomerID"] + FEATURE_COLUMNS)
    orders = orders.sort_values(["CustomerID", "InvoiceDate", "InvoiceNo"])
    orders["gap_days"] = orders.groupby("CustomerID").InvoiceDate.diff().dt.total_seconds() / 86400
    profiles = orders.groupby("CustomerID", as_index=False).agg(
        last_purchase_date=("InvoiceDate", "max"), first_purchase_date=("InvoiceDate", "min"),
        frequency_orders=("InvoiceNo", "nunique"), monetary_total=("OrderRevenue", "sum"),
        avg_order_value=("OrderRevenue", "mean"), max_order_value=("OrderRevenue", "max"),
        avg_items_per_order=("Items", "mean"),
        active_months=("InvoiceDate", lambda s: s.dt.to_period("M").nunique()),
        country=("Country", lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"),
        avg_days_between_orders=("gap_days", "mean"),
    )
    tx_profiles = tx.groupby("CustomerID", as_index=False).agg(
        unique_products=("StockCode", "nunique"), total_units=("Quantity", "sum"),
    )
    profiles = profiles.merge(tx_profiles, on="CustomerID", how="left", validate="one_to_one")
    profiles["recency_days"] = (cutoff - profiles.last_purchase_date).dt.days
    profiles["customer_tenure_days"] = (cutoff - profiles.first_purchase_date).dt.days
    return profiles


def build_snapshot_dataset(clean_df, order_df, horizon_days=HORIZON_DAYS, min_history_months=3):
    start = clean_df.InvoiceDate.min().to_period("M").to_timestamp() + pd.DateOffset(months=min_history_months)
    observation_end = clean_df.InvoiceDate.max()
    last_cutoff = observation_end.normalize() - pd.Timedelta(days=horizon_days)
    frames = []
    for cutoff in pd.date_range(start, last_cutoff, freq="MS"):
        profile = build_customer_features(clean_df, order_df, cutoff)
        if profile.empty:
            continue
        label_end = cutoff + pd.Timedelta(days=horizon_days)
        future = order_df.loc[(order_df.InvoiceDate >= cutoff) & (order_df.InvoiceDate < label_end)]
        profile["snapshot_date"] = cutoff
        profile["label_window_end"] = label_end
        profile["target_repeat_purchase"] = profile.CustomerID.isin(future.CustomerID).astype(int)
        frames.append(profile)
    if not frames:
        raise ValueError("Not enough observed history for fully matured 90-day labels")
    return pd.concat(frames, ignore_index=True)


def chronological_split(snapshot_df, horizon_days=HORIZON_DAYS):
    """Validation labels and every final training label must mature before the test cutoff."""
    data = snapshot_df.copy()
    data["snapshot_date"] = pd.to_datetime(data.snapshot_date)
    if "label_window_end" not in data:
        data["label_window_end"] = data.snapshot_date + pd.Timedelta(days=horizon_days)
    else:
        data["label_window_end"] = pd.to_datetime(data.label_window_end)
    if not (data.label_window_end == data.snapshot_date + pd.Timedelta(days=horizon_days)).all():
        raise ValueError("Label windows do not match the prediction horizon")
    test_cutoff = data.snapshot_date.max()
    eligible_validation = data.loc[
        (data.label_window_end <= test_cutoff) & (data.snapshot_date < test_cutoff), "snapshot_date"
    ]
    if eligible_validation.empty:
        raise ValueError("No fully matured validation window before test")
    validation_cutoff = eligible_validation.max()
    selection_train = data.loc[
        (data.label_window_end <= validation_cutoff) & (data.snapshot_date < validation_cutoff)
    ].copy()
    validation = data.loc[data.snapshot_date == validation_cutoff].copy()
    final_train = data.loc[
        (data.label_window_end <= test_cutoff) & (data.snapshot_date < test_cutoff)
    ].copy()
    test = data.loc[data.snapshot_date == test_cutoff].copy()
    for name, split in [("selection train", selection_train), ("validation", validation),
                        ("final train", final_train), ("test", test)]:
        if split.empty or split.target_repeat_purchase.nunique() != 2:
            raise ValueError(f"{name} requires nonempty data containing both classes")
    return selection_train, validation, final_train, test


def make_pipeline(kind):
    preprocessing = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                           ("scaler", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("encoder", OneHotEncoder(handle_unknown="ignore"))]), ["country"]),
    ])
    if kind == "Логистическая регрессия":
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    elif kind == "Random Forest":
        estimator = RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=10,
                                           random_state=42, class_weight="balanced_subsample", n_jobs=-1)
    else:
        raise ValueError(f"Unknown model: {kind}")
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])


def score_metrics(y_true, scores, threshold=THRESHOLD):
    predicted = scores >= threshold
    return {
        "ROC-AUC": float(roc_auc_score(y_true, scores)),
        "F1": float(f1_score(y_true, predicted, zero_division=0)),
        "Precision": float(precision_score(y_true, predicted, zero_division=0)),
        "Recall": float(recall_score(y_true, predicted, zero_division=0)),
    }


def train_on_snapshots(snapshot_df):
    selection_train, validation, final_train, test = chronological_split(snapshot_df)
    names = ["Логистическая регрессия", "Random Forest"]
    validation_rows = []
    for name in names:
        candidate = make_pipeline(name)
        candidate.fit(selection_train[FEATURE_COLUMNS], selection_train.target_repeat_purchase)
        scores = candidate.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
        validation_rows.append({"Модель": name, **score_metrics(validation.target_repeat_purchase, scores)})
    validation_metrics = pd.DataFrame(validation_rows).sort_values("ROC-AUC", ascending=False)
    selected_name = validation_metrics.iloc[0]["Модель"]
    metrics_rows, curves, trained_models = [], {}, {}
    for name in names:
        model = make_pipeline(name)
        model.fit(final_train[FEATURE_COLUMNS], final_train.target_repeat_purchase)
        scores = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        measured = score_metrics(test.target_repeat_purchase, scores)
        metrics_rows.append({"Модель": name, **measured})
        fpr, tpr, _ = roc_curve(test.target_repeat_purchase, scores)
        curves[name] = (fpr, tpr, measured["ROC-AUC"])
        trained_models[name] = model
    selected = trained_models[selected_name]
    feature_names = selected.named_steps["preprocess"].get_feature_names_out()
    estimator = selected.named_steps["model"]
    importance = (estimator.feature_importances_ if hasattr(estimator, "feature_importances_")
                  else np.abs(estimator.coef_[0]))
    feature_importance = pd.DataFrame({"Признак": feature_names, "Важность": importance})
    feature_importance = feature_importance.sort_values("Важность", ascending=False).reset_index(drop=True)

    def describe(split):
        return {
            "rows": len(split), "customers": int(split.CustomerID.nunique()),
            "snapshot_dates": [d.strftime("%Y-%m-%d") for d in sorted(split.snapshot_date.unique())],
            "max_label_window_end": split.label_window_end.max().isoformat(),
            "positive_rate": float(split.target_repeat_purchase.mean()),
        }

    report = {
        "horizon_days": HORIZON_DAYS, "threshold": THRESHOLD,
        "selection_rule": "Highest validation ROC-AUC; test metrics never select the winner",
        "selection_train": describe(selection_train), "validation": describe(validation),
        "final_train": describe(final_train), "test": describe(test),
        "selected_model": selected_name,
        "validation_metrics": validation_metrics.to_dict(orient="records"),
        "test_metrics": metrics_rows,
        "limitations": [
            "One validation cutoff and one held-out test cutoff in a short historical dataset.",
            "The same known customers can occur in successive snapshots; rows are dependent.",
            "Scores are not calibrated causal treatment effects or proof of campaign uplift.",
        ],
    }
    return {
        "snapshot_df": snapshot_df, "train_df": final_train, "test_df": test,
        "metrics_df": pd.DataFrame(metrics_rows).round(4), "validation_metrics_df": validation_metrics.round(4),
        "roc_curves": curves, "feature_importance": feature_importance,
        "best_model_name": selected_name, "selected_pipeline": selected, "validation_report": report,
    }


def build_ml_artifacts(clean_df, order_df):
    return train_on_snapshots(build_snapshot_dataset(clean_df, order_df))


def persist_model(artifacts, directory, source_sha256=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "repeat_purchase.joblib"
    joblib.dump(artifacts["selected_pipeline"], model_path, compress=3)
    checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = {
        "model_name": artifacts["best_model_name"], "model_version": checksum[:12], "sha256": checksum,
        "horizon_days": HORIZON_DAYS, "threshold": THRESHOLD, "feature_columns": FEATURE_COLUMNS,
        "feature_contract": "history-only-v1", "source_csv_sha256": source_sha256,
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ["scikit-learn", "numpy", "pandas", "joblib"]},
        "evaluation": artifacts["validation_report"],
        "warning": "Load only trusted repository artifacts. Scores have not been probability-calibrated.",
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata
