import json
import sys
from pathlib import Path

import nbformat as nbf
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
NOTEBOOK_PATH = ROOT_DIR / "notebooks" / "online_retail_repeat_purchase_analysis.ipynb"


def _load_context():
    model_metrics = pd.read_csv(PROCESSED_DIR / "model_metrics.csv")
    top_countries = pd.read_csv(PROCESSED_DIR / "top_countries.csv")
    retention = pd.read_csv(PROCESSED_DIR / "cohort_retention.csv", index_col=0)
    key_metrics = json.loads((PROCESSED_DIR / "key_metrics.json").read_text(encoding="utf-8"))

    first_month_col = retention.columns[1]
    first_month_ret = retention[first_month_col].dropna()
    first_month_ret = first_month_ret[first_month_ret > 0]
    best_model = model_metrics.sort_values("ROC-AUC", ascending=False).iloc[0]

    return {
        "top_countries": top_countries,
        "key_metrics": key_metrics,
        "avg_first_month_retention": round(float(first_month_ret.mean()), 3),
        "best_cohort_retention": round(float(first_month_ret.max()), 3),
        "best_cohort_name": str(first_month_ret.idxmax()),
        "best_model_name": str(best_model["Модель"]),
        "best_model_auc": float(best_model["ROC-AUC"]),
    }


def build_notebook():
    ctx = _load_context()
    nb = nbf.v4.new_notebook()

    intro_md = f"""
# Анализ повторных покупок и сегментация клиентов интернет-магазина

## Введение

Цель проекта — на реальных транзакционных данных интернет-магазина показать полный аналитический цикл: от очистки данных до бизнес-выводов и baseline ML-модели, которая прогнозирует вероятность повторной покупки.

В качестве источника использован датасет **UCI Online Retail**: история заказов британского онлайн-ритейлера за период с декабря 2010 года по декабрь 2011 года.

После очистки в работе осталось **{ctx["key_metrics"]["clean_rows"]:,}** строк, **{ctx["key_metrics"]["customers"]:,}** клиентов и **{ctx["key_metrics"]["orders"]:,}** заказов. Совокупная выручка составила **{ctx["key_metrics"]["revenue"]:,.2f}**.
""".strip()

    data_description_md = """
## Описание данных

Основные поля датасета:

- `InvoiceNo` — номер транзакции; если начинается с `C`, это отмена заказа.
- `StockCode` — идентификатор товара.
- `Description` — название товара.
- `Quantity` — количество единиц товара в строке заказа.
- `InvoiceDate` — дата и время покупки.
- `UnitPrice` — цена за единицу товара.
- `CustomerID` — идентификатор клиента.
- `Country` — страна клиента.

Для клиентской аналитики нам нужны только строки, которые можно надёжно связать с конкретным клиентом и реальной покупкой. Поэтому качество очистки здесь напрямую влияет на результаты анализа.
""".strip()

    cleaning_md = """
## Загрузка и первичный осмотр данных

Загрузим сырые данные, посмотрим на структуру и оценим качество: пропуски, дубликаты и аномальные значения.
""".strip()

    cleaning_explain_md = """
## Очистка данных

Логика очистки:

1. Удалить строки без `CustomerID` — их нельзя использовать в клиентской аналитике.
2. Исключить отмененные заказы (InvoiceNo начинается с C).
3. Удалить строки с неположительным количеством товара и ценой.
4. Удалить полные дубликаты.

Каждый шаг нужен для сохранения корректной логики транзакций и предотвращения искажений в расчётах выручки, частоты и retention.
""".strip()

    eda_intro_md = """
## Исследовательский анализ данных

Ниже собраны ключевые бизнес-метрики: динамика выручки, активность клиентской базы, распределение чеков, география продаж и концентрация выручки по клиентам.
""".strip()

    revenue_interp_md = """
**Интерпретация:** выручка растёт к осени и достигает пика в ноябре. Для e-commerce это выглядит как сочетание сезонного эффекта и усиления активности перед праздничным периодом.
""".strip()

    active_interp_md = """
**Интерпретация:** вместе с выручкой растёт и число активных клиентов. Значит, рост объясняется не только более дорогими корзинами, но и расширением активной базы.
""".strip()

    order_value_interp_md = """
**Интерпретация:** распределение чеков сильно скошено вправо: большинство заказов имеют умеренный размер, но есть небольшой хвост крупных покупок. Поэтому для описания клиентского поведения полезно смотреть и на средние, и на медианные значения.
""".strip()

    country_interp_md = f"""
**Интерпретация:** продажи сильно концентрируются в Великобритании — это главный рынок. Зарубежные страны работают как заметное, но существенно меньшее дополнение к основной выручке.

Крупнейшая зарубежная страна: **{ctx["top_countries"].iloc[1]["Country"]}**.
""".strip()

    concentration_interp_md = """
**Интерпретация:** выручка распределена неравномерно. Верхняя часть клиентской базы формирует большую долю оборота, поэтому CRM-стратегия должна быть дифференцированной: для ценных клиентов важнее удержание, а для широкой базы — реактивация и повторная конверсия.
""".strip()

    rfm_md = """
## RFM-сегментация клиентов

RFM-модель описывает клиента через три измерения:

- `Recency` — как давно была последняя покупка;
- `Frequency` — как часто клиент покупает;
- `Monetary` — сколько денег клиент приносит.

Для каждой метрики рассчитаны квантильные оценки от 1 до 5, после чего клиенты отнесены к бизнес-сегментам: чемпионы, лояльные, потенциально лояльные, новые, обещающие, требующие внимания, в зоне риска и потерянные.
""".strip()

    rfm_interp_md = """
**Интерпретация:** RFM-сегментация помогает отделить действительно ценных клиентов от клиентов с единичными покупками. Это удобно не только для аналитики, но и для практического CRM: разные сегменты требуют разных коммуникаций и разных бюджетов.
""".strip()

    cohort_md = f"""
## Когортный анализ и retention

Когорта — это группа клиентов, объединённая по месяцу первой покупки. Retention показывает, какая доля клиентов из когорты возвращается в следующие месяцы.

Средний retention в первый месяц после первой покупки составил примерно **{ctx["avg_first_month_retention"]:.1%}**. Лучшая когорта — **{ctx["best_cohort_name"]}** с уровнем около **{ctx["best_cohort_retention"]:.1%}**.

Последние когорты нужно интерпретировать аккуратно: у них меньше времени на "дожитие", поэтому правый край матрицы неполный.
""".strip()

    cohort_interp_md = """
**Интерпретация:** retention подтверждает, что повторные покупки есть, но они не одинаковы для всех когорт. Это важный сигнал: качество привлечения клиентов и опыт первой покупки заметно влияют на возврат.
""".strip()

    ml_md = f"""
## Построение модели прогноза повторной покупки

ML-задача: по истории клиента на момент среза предсказать, совершит ли он хотя бы одну покупку в следующие 90 дней.

Чтобы избежать утечки данных:

- признаки считаются только по прошлой истории на дату среза;
- выборка строится из нескольких ежемесячных снапшотов;
- тест — самый поздний доступный снапшот.

В качестве baseline-моделей сравнивались логистическая регрессия и Random Forest.

Лучшая модель по ROC-AUC: **{ctx["best_model_name"]}** — **{ctx["best_model_auc"]:.3f}**.
""".strip()

    ml_interp_md = """
**Интерпретация:** модель показывает рабочий baseline-уровень качества. Этого достаточно, чтобы ранжировать клиентскую базу по вероятности возврата и использовать прогноз в CRM, но требует дальнейшей настройки перед production-применением.
""".strip()

    conclusions_md = """
## Выводы и рекомендации

1. Клиентская база сильно неоднородна: небольшая доля клиентов приносит основную часть выручки.
2. Сегменты `Чемпионы` и `Лояльные` — приоритет для удержания и персональных предложений.
3. Сегмент `В зоне риска` подходит для реактивационных кампаний.
4. Когортный анализ показывает, что качество привлечения и первых заказов влияет на последующий retention.
5. Baseline ML-модель уже может использоваться для скоринга клиентов перед триггерными коммуникациями.

Практические шаги:

- запустить отдельные сценарии удержания для high-value сегментов;
- тестировать welcome-цепочки для новых когорт;
- использовать прогноз повторной покупки как дополнительный приоритет в CRM;
- регулярно пересчитывать RFM и отслеживать переходы клиентов между сегментами.
""".strip()

    setup_code = """
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from IPython.display import display
import pandas as pd

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

configure_plotting()
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail.csv"
""".strip()

    analysis_code = """
raw_df = load_transactions(RAW_DATA_PATH)
display(raw_df.head())

raw_quality = pd.DataFrame({
    "Признак": raw_df.columns,
    "Тип данных": raw_df.dtypes.astype(str).values,
    "Пропуски": raw_df.isna().sum().values,
    "Доля пропусков, %": (raw_df.isna().mean() * 100).round(2).values,
})
display(raw_quality)

pd.DataFrame({
    "Метрика": [
        "Строк в сыром датасете",
        "Дубликаты",
        "Строк без CustomerID",
        "Отмененные заказы",
        "Строк с Quantity <= 0",
        "Строк с UnitPrice <= 0",
    ],
    "Значение": [
        len(raw_df),
        int(raw_df.duplicated().sum()),
        int(raw_df["CustomerID"].isna().sum()),
        int(raw_df["InvoiceNo"].astype(str).str.startswith("C").sum()),
        int((raw_df["Quantity"] <= 0).sum()),
        int((raw_df["UnitPrice"] <= 0).sum()),
    ],
})
""".strip()

    cleaning_code = """
clean_df, cleaning_summary = clean_transactions(raw_df)
order_df = build_order_table(clean_df)
overview_metrics = calculate_overview_metrics(clean_df, order_df)

display(cleaning_summary)
display(overview_metrics)
clean_df.head()
""".strip()

    eda_code = """
top_products = (
    clean_df.groupby("Description", as_index=False)["Revenue"]
    .sum()
    .sort_values("Revenue", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

top_countries = (
    clean_df.groupby("Country", as_index=False)
    .agg(Выручка=("Revenue", "sum"), Заказы=("InvoiceNo", "nunique"), Клиенты=("CustomerID", "nunique"))
    .sort_values("Выручка", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

display(top_countries)
display(top_products)
""".strip()

    rfm_code = """
rfm_df, rfm_summary = compute_rfm_table(clean_df)
display(rfm_df.head())
display(rfm_summary)
""".strip()

    cohort_code = """
cohort_counts, retention = build_cohort_retention(clean_df)
display(retention.round(3))
""".strip()

    ml_code = """
ml_artifacts = build_ml_artifacts(clean_df, order_df)
display(ml_artifacts["metrics_df"])
display(ml_artifacts["feature_importance"].head(10))
""".strip()

    cells = [
        nbf.v4.new_markdown_cell(intro_md),
        nbf.v4.new_markdown_cell(data_description_md),
        nbf.v4.new_markdown_cell(cleaning_md),
        nbf.v4.new_code_cell(setup_code),
        nbf.v4.new_code_cell(analysis_code),
        nbf.v4.new_markdown_cell(cleaning_explain_md),
        nbf.v4.new_code_cell(cleaning_code),
        nbf.v4.new_markdown_cell(eda_intro_md),
        nbf.v4.new_code_cell(eda_code),
        nbf.v4.new_code_cell("plot_monthly_revenue(clean_df);"),
        nbf.v4.new_markdown_cell(revenue_interp_md),
        nbf.v4.new_code_cell("plot_monthly_active_customers(clean_df);"),
        nbf.v4.new_markdown_cell(active_interp_md),
        nbf.v4.new_code_cell("plot_order_value_distribution(order_df);"),
        nbf.v4.new_markdown_cell(order_value_interp_md),
        nbf.v4.new_code_cell("plot_top_countries(clean_df);"),
        nbf.v4.new_markdown_cell(country_interp_md),
        nbf.v4.new_code_cell("concentration_df = compute_customer_concentration(clean_df)\nplot_customer_concentration(concentration_df);"),
        nbf.v4.new_markdown_cell(concentration_interp_md),
        nbf.v4.new_markdown_cell(rfm_md),
        nbf.v4.new_code_cell(rfm_code),
        nbf.v4.new_code_cell("plot_rfm_segments(rfm_summary);"),
        nbf.v4.new_markdown_cell(rfm_interp_md),
        nbf.v4.new_markdown_cell(cohort_md),
        nbf.v4.new_code_cell(cohort_code),
        nbf.v4.new_code_cell("plot_cohort_heatmap(retention);"),
        nbf.v4.new_markdown_cell(cohort_interp_md),
        nbf.v4.new_markdown_cell(ml_md),
        nbf.v4.new_code_cell(ml_code),
        nbf.v4.new_code_cell('plot_model_roc_curves(ml_artifacts["roc_curves"]);'),
        nbf.v4.new_code_cell('plot_feature_importance(ml_artifacts["feature_importance"]);'),
        nbf.v4.new_markdown_cell(ml_interp_md),
        nbf.v4.new_markdown_cell(conclusions_md),
    ]

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Ноутбук создан: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()
