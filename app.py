"""Portfolio dashboard. Run with: uv run streamlit run app.py"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from retail_repeat_analysis.analytics import cohort_retention, filter_orders, overview, rfm_as_of
from retail_repeat_analysis.experiments import sample_size, synthetic_demo

ROOT = Path(__file__).resolve().parent
plt.switch_backend("Agg")
st.set_page_config(page_title="Online Retail | Удержание клиентов", page_icon="📊", layout="wide")


@st.cache_data
def load_orders(path, modified):
    frame = pd.read_csv(path, dtype={"CustomerID": str, "InvoiceNo": str})
    frame["InvoiceDate"] = pd.to_datetime(frame.InvoiceDate)
    return frame


orders_path = ROOT / "data/processed/orders.csv"
if not orders_path.exists():
    st.error("Не найдена таблица заказов. В полной копии репозитория она находится в data/processed/orders.csv.")
    st.stop()
orders = load_orders(str(orders_path), orders_path.stat().st_mtime_ns)
first_day, last_day = orders.InvoiceDate.min().date(), orders.InvoiceDate.max().date()

st.caption("УЧЕБНЫЙ ПРОЕКТ · UCI ONLINE RETAIL · 2010–2011")
st.title("Повторные покупки и удержание")
st.write("От транзакций к клиентским сегментам, когортам и проверяемым решениям для CRM.")
with st.sidebar:
    st.header("Срез данных")
    period = st.date_input("Период продаж", value=(first_day, last_day), min_value=first_day,
                           max_value=last_day, key="period")
    country = st.selectbox("Страна", ["Все страны"] + sorted(orders.Country.dropna().unique()), key="country")
    st.caption("Выручка и заказы относятся к выбранному периоду. RFM и когорты используют историю до его конечной даты.")
    st.divider()
    st.caption("Источник: UCI Online Retail, CC BY 4.0. Исключены отмены, неположительные покупки, дубликаты и строки без клиента.")
    st.link_button("Репозиторий проекта", "https://github.com/Shishani7/online-retail-repeat-purchase-analysis")

if len(period) != 2:
    st.info("Выберите начальную и конечную дату периода.")
    st.stop()
start, end = period
countries = sorted(orders.Country.unique()) if country == "Все страны" else [country]
selected = filter_orders(orders, start, end, countries)
values = overview(selected)
metrics = st.columns(4)
metrics[0].metric("Выручка, GBP", f"{values['revenue']:,.0f}".replace(",", " "))
metrics[1].metric("Заказы", f"{values['orders']:,}".replace(",", " "))
metrics[2].metric("Клиенты", f"{values['customers']:,}".replace(",", " "))
metrics[3].metric("Средний заказ, GBP", f"{values['average_order']:,.2f}")

if selected.empty:
    st.info("За выбранный период в этой стране нет заказов. Измените фильтры.")
    st.stop()

as_of = pd.Timestamp(end) + pd.Timedelta(days=1)
history = orders.loc[(orders.InvoiceDate < as_of) & orders.Country.isin(countries)].copy()
tabs = st.tabs(["Продажи", "RFM-сегменты", "Когорты", "A/B-дизайн", "Модель"])
with tabs[0]:
    st.subheader("Динамика продаж")
    monthly = selected.groupby(selected.InvoiceDate.dt.to_period("M")).agg(
        Выручка=("OrderRevenue", "sum"), Заказы=("InvoiceNo", "nunique"), Клиенты=("CustomerID", "nunique"))
    monthly.index = monthly.index.to_timestamp()
    left, right = st.columns([2, 1])
    with left:
        st.line_chart(monthly[["Выручка"]], color="#233F6B")
    with right:
        monthly_display = monthly.round(2).copy()
        monthly_display.index = monthly_display.index.strftime("%Y-%m")
        st.dataframe(monthly_display.rename_axis("Месяц"), width="stretch")
    st.caption("Первый и последний месяц выбранного периода могут быть неполными. Выручка — сумма положительных покупок после очистки, без учёта возвратов.")

with tabs[1]:
    st.subheader("Какие клиенты формируют выручку")
    rfm, summary = rfm_as_of(history, as_of)
    st.caption(f"Срез RFM на {pd.Timestamp(end):%d.%m.%Y}; вся история выбранных стран до этой даты. Начальная дата фильтра продаж не обрезает историю RFM.")
    left, right = st.columns([1, 1.3])
    with left:
        st.bar_chart(summary.set_index("Segment")[["Выручка"]], color="#166A70", horizontal=True)
    with right:
        st.dataframe(summary[["Segment", "Клиенты", "Доля клиентов, %", "Доля выручки, %"]],
                     hide_index=True, width="stretch")
    st.caption("R — давность, F — число заказов, M — сумма покупок. Баллы относительны выбранной базе; правила сегментов используют R и F, денежная ценность показана отдельно.")
    st.download_button("Скачать клиентские сегменты CSV", rfm.to_csv(index=False).encode("utf-8-sig"),
                       "rfm_segments.csv", "text/csv")

with tabs[2]:
    st.subheader("Возвращаются ли когорты покупателей")
    _, retention = cohort_retention(history, observation_end=end)
    fig, ax = plt.subplots(figsize=(12, max(3.5, len(retention) * .42)))
    sns.heatmap(retention, mask=retention.isna(), annot=True, fmt=".0%", cmap="YlGnBu",
                linewidths=.4, ax=ax, cbar_kws={"label": "Доля активных клиентов"})
    ax.set_xlabel("Месяц после первой покупки")
    ax.set_ylabel("Месяц первой покупки")
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption("Когорты определены по первой покупке в истории выбранных стран. Ноль — наблюдённый месяц без возвратов; пустая ячейка — будущее или неполный следующий месяц. Месяц 0 задаёт базу когорты (100%).")

with tabs[3]:
    st.subheader("Спланировать проверку CRM-гипотезы")
    st.write("Гипотеза: персональное предложение увеличивает долю клиентов с повторной покупкой за 30 дней. Единица рандомизации — клиент; контроль и тест 50/50.")
    a, b = st.columns(2)
    baseline = a.slider("Исходная конверсия, %", 1.0, 80.0, 20.0, .5, key="baseline") / 100
    mde = b.slider("Минимальный эффект, п.п.", .5, 10.0, 2.0, .5, key="mde") / 100
    plan = sample_size(baseline, mde)
    x, y, z = st.columns(3)
    x.metric("На группу", f"{plan['per_group']:,}".replace(",", " "))
    y.metric("Всего участников", f"{plan['total']:,}".replace(",", " "))
    z.metric("Относительный эффект", f"{plan['relative_mde']:.1%}")
    st.caption("Двустороннее нормальное приближение; α = 5%, мощность = 80%, независимые клиенты, равные группы. Это планирование новой кампании, а не результат исторического датасета.")
    st.write("Основная метрика: покупатели за 30 дней / все назначенные в группу. Контрольные метрики: маржа на участника, стоимость контактов и скидок, отписки. До вывода проверяются распределение групп и полнота наблюдения.")
    if st.button("Показать синтетическую проверку мощности"):
        simulation = synthetic_demo(baseline, mde)
        st.info(f"Синтетическая симуляция: доля обнаруженных эффектов {simulation['simulated_rejection_rate']:.1%} в {simulation['repetitions']} искусственных экспериментах. Это НЕ измеренный эффект CRM-кампании.")
    st.link_button("Полный дизайн эксперимента", "https://github.com/Shishani7/online-retail-repeat-purchase-analysis/blob/main/docs/AB_TEST_DESIGN.md")

with tabs[4]:
    st.subheader("Прогноз повторной покупки за 90 дней")
    report_path = ROOT / "data/processed/temporal_validation.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        st.write(f"Модель выбрана по валидации: **{report['selected_model']}**. Финальный тест не используется для выбора модели.")
        st.dataframe(pd.DataFrame(report["test_metrics"]).round(4), hide_index=True, width="stretch")
        st.caption("Это метрики последнего полностью наблюдённого среза всего датасета. Фильтры продаж не пересчитывают модель или её тестовую выборку. Порог фиксирован: 0,5.")
        with st.expander("Как исключается информация из будущего"):
            st.code("train_cutoff + 90 days <= next_prediction_cutoff")
            st.json({name: report[name] for name in ["selection_train", "validation", "final_train", "test"]})
        st.caption("Учебный baseline на короткой исторической выборке. Скор не является доказательством эффекта скидки; калибровка вероятностей отдельно не выполнялась.")
    else:
        st.info("Отчёт появится после воспроизводимой сборки артефактов: python src/build_assets.py")
