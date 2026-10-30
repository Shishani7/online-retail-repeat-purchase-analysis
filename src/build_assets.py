import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

# добавляем src в путь, чтобы импорт работал при запуске из корня проекта
sys.path.insert(0, str(Path(__file__).resolve().parent))

from retail_repeat_analysis import (
    build_cohort_retention,
    build_ml_artifacts,
    build_order_table,
    calculate_overview_metrics,
    clean_transactions,
    compute_customer_concentration,
    compute_rfm_table,
    configure_plotting,
    load_transactions,
    plot_cohort_heatmap,
    plot_customer_concentration,
    plot_feature_importance,
    plot_model_roc_curves,
    plot_monthly_active_customers,
    plot_monthly_revenue,
    plot_order_value_distribution,
    plot_rfm_segments,
    plot_top_countries,
)
from retail_repeat_analysis.experiments import historical_baseline, sample_size, synthetic_demo
from retail_repeat_analysis.modeling import FEATURE_COLUMNS, persist_model

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "online_retail.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
IMAGES_DIR = ROOT_DIR / "images"


def main():
    configure_plotting()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("Загружаем данные...")
    raw_df = load_transactions(RAW_DATA_PATH)
    clean_df, cleaning_summary = clean_transactions(raw_df)
    order_df = build_order_table(clean_df)
    overview_metrics = calculate_overview_metrics(clean_df, order_df)
    concentration_df = compute_customer_concentration(clean_df)
    rfm_df, rfm_summary = compute_rfm_table(clean_df)
    cohort_counts, retention = build_cohort_retention(clean_df)

    print("Обучаем модели...")
    ml_artifacts = build_ml_artifacts(clean_df, order_df)

    # сохраняем таблицы
    clean_df.to_csv(PROCESSED_DIR / "clean_transactions.csv", index=False)
    order_df.to_csv(PROCESSED_DIR / "orders.csv", index=False)
    cleaning_summary.to_csv(PROCESSED_DIR / "cleaning_summary.csv", index=False)
    overview_metrics.to_csv(PROCESSED_DIR / "overview_metrics.csv", index=False)
    concentration_df.to_csv(PROCESSED_DIR / "customer_concentration.csv", index=False)
    rfm_df.to_csv(PROCESSED_DIR / "rfm_segments.csv", index=False)
    rfm_summary.to_csv(PROCESSED_DIR / "rfm_segment_summary.csv", index=False)
    cohort_counts.to_csv(PROCESSED_DIR / "cohort_counts.csv")
    retention.to_csv(PROCESSED_DIR / "cohort_retention.csv")
    ml_artifacts["snapshot_df"].to_csv(PROCESSED_DIR / "ml_snapshot_dataset.csv", index=False)
    ml_artifacts["metrics_df"].to_csv(PROCESSED_DIR / "model_metrics.csv", index=False)
    ml_artifacts["validation_metrics_df"].to_csv(PROCESSED_DIR / "validation_metrics.csv", index=False)
    (PROCESSED_DIR / "temporal_validation.json").write_text(
        json.dumps(ml_artifacts["validation_report"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = persist_model(ml_artifacts, ROOT_DIR / "models",
                             hashlib.sha256(RAW_DATA_PATH.read_bytes()).hexdigest())
    examples = ROOT_DIR / "examples"
    examples.mkdir(exist_ok=True)
    request = json.loads(ml_artifacts["test_df"][FEATURE_COLUMNS].head(1).to_json(orient="records"))[0]
    (examples / "predict_request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    score = float(ml_artifacts["selected_pipeline"].predict_proba(pd.DataFrame([request]))[0, 1])
    response = {"score": score, "prediction": int(score >= metadata["threshold"]),
                **{key: metadata[key] for key in ["threshold", "horizon_days", "model_name", "model_version"]}}
    (examples / "predict_response.json").write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    (PROCESSED_DIR / "ab_simulation.json").write_text(json.dumps(synthetic_demo(), indent=2), encoding="utf-8")
    baseline_cutoff = (order_df.InvoiceDate.max() - pd.Timedelta(days=30)).to_period("M").to_timestamp()
    baseline = historical_baseline(order_df, baseline_cutoff)
    planning = {"historical_baseline": baseline, "sample_size": sample_size(baseline=baseline["repeat_rate"])}
    (PROCESSED_DIR / "ab_planning.json").write_text(json.dumps(planning, ensure_ascii=False, indent=2), encoding="utf-8")
    ml_artifacts["feature_importance"].to_csv(PROCESSED_DIR / "feature_importance.csv", index=False)

    top_products = (
        clean_df.groupby("Description", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    top_products.to_csv(PROCESSED_DIR / "top_products.csv", index=False)

    top_countries = (
        clean_df.groupby("Country", as_index=False)
        .agg(Выручка=("Revenue", "sum"), Заказы=("InvoiceNo", "nunique"), Клиенты=("CustomerID", "nunique"))
        .sort_values("Выручка", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    top_countries.to_csv(PROCESSED_DIR / "top_countries.csv", index=False)

    # строим и сохраняем графики
    print("Строим графики...")
    plot_monthly_revenue(clean_df, IMAGES_DIR / "monthly_revenue.png")
    plot_monthly_active_customers(clean_df, IMAGES_DIR / "monthly_active_customers.png")
    plot_order_value_distribution(order_df, IMAGES_DIR / "order_value_distribution.png")
    plot_top_countries(clean_df, IMAGES_DIR / "top_countries_revenue.png")
    plot_customer_concentration(concentration_df, IMAGES_DIR / "customer_concentration.png")
    plot_rfm_segments(rfm_summary, IMAGES_DIR / "rfm_segments.png")
    plot_cohort_heatmap(retention, IMAGES_DIR / "cohort_retention.png")
    plot_model_roc_curves(ml_artifacts["roc_curves"], IMAGES_DIR / "model_roc_curves.png")
    plot_feature_importance(ml_artifacts["feature_importance"], IMAGES_DIR / "feature_importance.png")

    key_metrics = {
        "raw_rows": len(raw_df),
        "clean_rows": len(clean_df),
        "customers": int(clean_df["CustomerID"].nunique()),
        "orders": int(order_df["InvoiceNo"].nunique()),
        "revenue": round(float(clean_df["Revenue"].sum()), 2),
        "avg_order_value": round(float(order_df["OrderRevenue"].mean()), 2),
        "median_order_value": round(float(order_df["OrderRevenue"].median()), 2),
        "best_model": ml_artifacts["best_model_name"],
    }
    (PROCESSED_DIR / "key_metrics.json").write_text(
        json.dumps(key_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Готово.")
    print(f"  Данные: {PROCESSED_DIR}")
    print(f"  Графики: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
