from datetime import date
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_loads_published_data_and_country_filter_changes_metrics():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=60).run()
    assert not app.exception
    assert app.metric[1].value == "18 532"
    assert app.metric[2].value == "4 338"
    app.selectbox[0].select("France").run()
    assert not app.exception
    assert 0 < int(app.metric[1].value.replace(" ", "")) < 18532


def test_dashboard_empty_period_is_handled():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=60).run()
    app.date_input[0].set_value((date(2010, 12, 25), date(2010, 12, 25))).run()
    assert not app.exception
    assert app.metric[1].value == "0"
    assert len(app.info) >= 1


def test_experiment_controls_and_synthetic_demo():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=60).run()
    original = app.metric[4].value
    app.slider[0].set_value(25.0).run()
    assert not app.exception
    assert app.metric[4].value != original
    app.button[0].click().run()
    assert not app.exception
    assert len(app.info) >= 1
