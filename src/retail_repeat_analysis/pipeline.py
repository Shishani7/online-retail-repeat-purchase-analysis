import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.switch_backend("Agg")


def configure_plotting():
    sns.set_theme(style="whitegrid", palette="Set2")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["font.family"] = "DejaVu Sans"


def save_figure(fig, output_path=None):
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")


def load_transactions(data_path):
    df = pd.read_csv(data_path)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
    return df


def clean_transactions(raw_df):
    df = raw_df.copy()
    initial_rows = len(df)
    cleaning_records = []

    # удаляем строки без CustomerID — без них нельзя делать клиентскую аналитику
    before = len(df)
    df = df[df["CustomerID"].notna()].copy()
    cleaning_records.append({
        "Этап": "Удаление строк без CustomerID",
        "Строк до": before,
        "Строк после": len(df),
        "Удалено строк": before - len(df),
        "Доля от исходного датасета, %": round((before - len(df)) / initial_rows * 100, 2),
    })

    # InvoiceNo начинается с C — это отмененные заказы
    before = len(df)
    df = df[~df["InvoiceNo"].str.startswith("C", na=False)].copy()
    cleaning_records.append({
        "Этап": "Удаление отмененных заказов",
        "Строк до": before,
        "Строк после": len(df),
        "Удалено строк": before - len(df),
        "Доля от исходного датасета, %": round((before - len(df)) / initial_rows * 100, 2),
    })

    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    cleaning_records.append({
        "Этап": "Удаление строк с неположительными Quantity и UnitPrice",
        "Строк до": before,
        "Строк после": len(df),
        "Удалено строк": before - len(df),
        "Доля от исходного датасета, %": round((before - len(df)) / initial_rows * 100, 2),
    })

    before = len(df)
    df = df.drop_duplicates().copy()
    cleaning_records.append({
        "Этап": "Удаление полных дубликатов",
        "Строк до": before,
        "Строк после": len(df),
        "Удалено строк": before - len(df),
        "Доля от исходного датасета, %": round((before - len(df)) / initial_rows * 100, 2),
    })

    df["CustomerID"] = df["CustomerID"].astype("int64").astype(str)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["InvoiceDay"] = df["InvoiceDate"].dt.date.astype(str)

    cleaning_summary = pd.DataFrame(cleaning_records)
    return df, cleaning_summary


def build_order_table(clean_df):
    order_df = (
        clean_df.groupby("InvoiceNo", as_index=False)
        .agg(
            CustomerID=("CustomerID", "first"),
            InvoiceDate=("InvoiceDate", "min"),
            Country=("Country", "first"),
            OrderRevenue=("Revenue", "sum"),
            Items=("Quantity", "sum"),
            UniqueProducts=("StockCode", "nunique"),
        )
        .sort_values("InvoiceDate")
        .reset_index(drop=True)
    )
    order_df["InvoiceMonth"] = order_df["InvoiceDate"].dt.to_period("M").astype(str)
    return order_df


def calculate_overview_metrics(clean_df, order_df):
    orders_per_customer = order_df.groupby("CustomerID").size()
    revenue_by_customer = clean_df.groupby("CustomerID")["Revenue"].sum().sort_values(ascending=False)

    n_customers = len(revenue_by_customer)
    top10 = revenue_by_customer.head(max(1, int(n_customers * 0.10))).sum() / revenue_by_customer.sum()
    top20 = revenue_by_customer.head(max(1, int(n_customers * 0.20))).sum() / revenue_by_customer.sum()

    metrics = [
        {"Метрика": "Период данных", "Значение": f"{clean_df['InvoiceDate'].min():%d.%m.%Y} - {clean_df['InvoiceDate'].max():%d.%m.%Y}"},
        {"Метрика": "Строк после очистки", "Значение": len(clean_df)},
        {"Метрика": "Уникальные клиенты", "Значение": int(clean_df["CustomerID"].nunique())},
        {"Метрика": "Уникальные заказы", "Значение": int(order_df["InvoiceNo"].nunique())},
        {"Метрика": "Общая выручка", "Значение": round(float(clean_df["Revenue"].sum()), 2)},
        {"Метрика": "Средний чек", "Значение": round(float(order_df["OrderRevenue"].mean()), 2)},
        {"Метрика": "Медианный чек", "Значение": round(float(order_df["OrderRevenue"].median()), 2)},
        {"Метрика": "Среднее число заказов на клиента", "Значение": round(float(orders_per_customer.mean()), 2)},
        {"Метрика": "Медианное число заказов на клиента", "Значение": round(float(orders_per_customer.median()), 2)},
        {"Метрика": "Доля выручки от топ-10% клиентов", "Значение": round(float(top10), 4)},
        {"Метрика": "Доля выручки от топ-20% клиентов", "Значение": round(float(top20), 4)},
    ]
    return pd.DataFrame(metrics)


def plot_monthly_revenue(clean_df, output_path=None):
    monthly_revenue = clean_df.groupby(clean_df["InvoiceDate"].dt.to_period("M"))["Revenue"].sum().sort_index()
    monthly_revenue.index = monthly_revenue.index.to_timestamp()

    fig, ax = plt.subplots()
    sns.lineplot(x=monthly_revenue.index, y=monthly_revenue.values, marker="o", linewidth=2.5, color="#2f7ed8", ax=ax)
    ax.set_title("Динамика месячной выручки")
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Выручка")
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, output_path)
    return fig


def plot_monthly_active_customers(clean_df, output_path=None):
    monthly_active = clean_df.groupby(clean_df["InvoiceDate"].dt.to_period("M"))["CustomerID"].nunique().sort_index()
    monthly_active.index = monthly_active.index.to_timestamp()

    fig, ax = plt.subplots()
    sns.barplot(x=monthly_active.index.strftime("%Y-%m"), y=monthly_active.values, color="#5cb85c", ax=ax)
    ax.set_title("Активные клиенты по месяцам")
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Число клиентов")
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, output_path)
    return fig


def plot_order_value_distribution(order_df, output_path=None):
    # срезаем верхний хвост для лучшей читаемости графика
    cutoff = order_df["OrderRevenue"].quantile(0.99)
    fig, ax = plt.subplots()
    sns.histplot(order_df.loc[order_df["OrderRevenue"] <= cutoff, "OrderRevenue"], bins=40, color="#f0ad4e", ax=ax)
    ax.set_title("Распределение чека (до 99-го перцентиля)")
    ax.set_xlabel("Сумма заказа")
    ax.set_ylabel("Количество заказов")
    save_figure(fig, output_path)
    return fig


def plot_top_countries(clean_df, output_path=None, top_n=10):
    country_revenue = clean_df.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots()
    sns.barplot(x=country_revenue.values, y=country_revenue.index, color="#4e79a7", ax=ax)
    ax.set_title(f"Топ-{top_n} стран по выручке")
    ax.set_xlabel("Выручка")
    ax.set_ylabel("Страна")
    save_figure(fig, output_path)
    return fig


def compute_customer_concentration(clean_df):
    revenue_by_customer = clean_df.groupby("CustomerID")["Revenue"].sum().sort_values(ascending=False).reset_index(drop=True)
    n = len(revenue_by_customer)
    concentration = pd.DataFrame({
        "Доля клиентов": np.arange(1, n + 1) / n,
        "Накопленная доля выручки": revenue_by_customer.cumsum() / revenue_by_customer.sum(),
    })
    return concentration


def plot_customer_concentration(concentration_df, output_path=None):
    fig, ax = plt.subplots()
    sns.lineplot(
        data=concentration_df,
        x="Доля клиентов",
        y="Накопленная доля выручки",
        linewidth=2.5,
        color="#d9534f",
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.8)
    ax.set_title("Концентрация выручки по клиентской базе")
    ax.set_xlabel("Доля клиентов")
    ax.set_ylabel("Накопленная доля выручки")
    save_figure(fig, output_path)
    return fig


def _quintile_score(series, higher_is_better=True):
    ranked = series.rank(method="first", ascending=True)
    if len(series) == 1:
        return pd.Series(3, index=series.index, dtype=int)
    if len(series) < 5:
        score = (1 + ((ranked - 1) / (len(series) - 1) * 4).round()).astype(int)
        return score if higher_is_better else 6 - score
    labels = [1, 2, 3, 4, 5] if higher_is_better else [5, 4, 3, 2, 1]
    return pd.qcut(ranked, 5, labels=labels).astype(int)


def _segment_from_scores(row):
    r = int(row["R_Score"])
    f = int(row["F_Score"])

    if r >= 4 and f >= 4:
        return "Чемпионы"
    if r >= 3 and f >= 4:
        return "Лояльные"
    if r >= 4 and f >= 2:
        return "Потенциально лояльные"
    if r == 5 and f == 1:
        return "Новые"
    if r == 4 and f == 1:
        return "Обещающие"
    if r == 3 and f <= 2:
        return "Требуют внимания"
    if r <= 2 and f >= 3:
        return "В зоне риска"
    return "Потерянные"


def compute_rfm_table(clean_df, analysis_date=None):
    analysis_date = pd.Timestamp(analysis_date) if analysis_date is not None else clean_df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = clean_df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda s: (analysis_date - s.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
    )

    # для Recency: чем меньше дней — тем лучше, поэтому higher_is_better=False
    rfm["R_Score"] = _quintile_score(rfm["Recency"], higher_is_better=False)
    rfm["F_Score"] = _quintile_score(rfm["Frequency"], higher_is_better=True)
    rfm["M_Score"] = _quintile_score(rfm["Monetary"], higher_is_better=True)
    rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    rfm["Segment"] = rfm.apply(_segment_from_scores, axis=1)
    rfm = rfm.reset_index()

    summary = (
        rfm.groupby("Segment", as_index=False)
        .agg(
            Клиенты=("CustomerID", "nunique"),
            Выручка=("Monetary", "sum"),
            Средняя_выручка_на_клиента=("Monetary", "mean"),
            Средняя_частота=("Frequency", "mean"),
            Средняя_давность_дней=("Recency", "mean"),
        )
        .sort_values("Выручка", ascending=False)
        .reset_index(drop=True)
    )
    summary["Доля клиентов, %"] = (summary["Клиенты"] / summary["Клиенты"].sum() * 100).round(2)
    summary["Доля выручки, %"] = (summary["Выручка"] / summary["Выручка"].sum() * 100).round(2)
    return rfm, summary


def plot_rfm_segments(rfm_summary, output_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_summary = rfm_summary.sort_values("Выручка", ascending=True)
    sns.barplot(data=sorted_summary, x="Выручка", y="Segment", color="#e15759", ax=ax)
    ax.set_title("Выручка по RFM-сегментам")
    ax.set_xlabel("Выручка")
    ax.set_ylabel("Сегмент")
    save_figure(fig, output_path)
    return fig


def build_cohort_retention(clean_df):
    from .analytics import cohort_retention
    return cohort_retention(clean_df)


def plot_cohort_heatmap(retention, output_path=None):
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(
        retention,
        mask=retention.isna(),
        annot=True,
        fmt=".0%",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Retention"},
        ax=ax,
    )
    ax.set_title("Когортный retention по месяцам")
    ax.set_xlabel("Месяц жизни когорты")
    ax.set_ylabel("Когорта первой покупки")
    save_figure(fig, output_path)
    return fig


def build_ml_snapshot_dataset(clean_df, order_df, horizon_days=90):
    from .modeling import build_snapshot_dataset
    return build_snapshot_dataset(clean_df, order_df, horizon_days=horizon_days)


def build_ml_artifacts(clean_df, order_df):
    from .modeling import build_ml_artifacts as build_validated
    return build_validated(clean_df, order_df)


def plot_model_roc_curves(roc_curves, output_path=None):
    fig, ax = plt.subplots()
    for model_name, (fpr, tpr, auc) in roc_curves.items():
        ax.plot(fpr, tpr, linewidth=2.2, label=f"{model_name} (ROC-AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC-кривые моделей")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    save_figure(fig, output_path)
    return fig


def plot_feature_importance(feature_importance, output_path=None, top_n=10):
    top_features = feature_importance.head(top_n).sort_values("Важность", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_features, x="Важность", y="Признак", color="#76b7b2", ax=ax)
    ax.set_title(f"Топ-{top_n} признаков лучшей модели")
    ax.set_xlabel("Важность")
    ax.set_ylabel("Признак")
    save_figure(fig, output_path)
    return fig
