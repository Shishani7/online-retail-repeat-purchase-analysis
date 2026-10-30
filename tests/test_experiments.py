import pytest

from retail_repeat_analysis.experiments import historical_baseline, sample_size, synthetic_demo


def test_sample_size_matches_documented_example_and_allocation():
    result = sample_size()
    assert 6500 <= result["per_group"] <= 6520
    assert result["total"] == 2 * result["per_group"]
    assert result["relative_mde"] == pytest.approx(.1)


def test_smaller_effect_and_greater_power_need_more_clients():
    base = sample_size()["per_group"]
    assert sample_size(mde=.01)["per_group"] > base
    assert sample_size(power=.9)["per_group"] > base


@pytest.mark.parametrize("kwargs", [{"baseline": 0}, {"mde": -.1}, {"baseline": .99, "mde": .02}, {"power": .5}, {"alpha": 0}])
def test_rejects_invalid_planning_inputs(kwargs):
    with pytest.raises(ValueError):
        sample_size(**kwargs)


def test_simulation_is_reproducible_and_explicitly_not_a_campaign():
    first, second = synthetic_demo(), synthetic_demo()
    assert first == second
    assert first["not_a_real_campaign"] is True
    assert .75 < first["simulated_rejection_rate"] < .85


def test_historical_baseline_requires_complete_followup():
    import pandas as pd
    orders = pd.DataFrame({"InvoiceDate": pd.to_datetime(["2011-11-15"])})
    with pytest.raises(ValueError, match="fully observed"):
        historical_baseline(orders, "2011-11-01")
