"""Fixed-sample, equal-allocation planning and explicitly synthetic demonstrations."""
import math

import numpy as np
import pandas as pd
from scipy.stats import norm


def historical_baseline(orders, cutoff, horizon_days=30, segment="В зоне риска"):
    """Estimate an observational repeat-purchase baseline for a pre-cutoff eligible segment."""
    from .analytics import rfm_as_of
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.Timedelta(days=horizon_days)
    if orders.InvoiceDate.max() < end:
        raise ValueError("The historical outcome window is not fully observed")
    rfm, _ = rfm_as_of(orders, cutoff)
    eligible = set(rfm.loc[rfm.Segment == segment, "CustomerID"])
    if not eligible:
        raise ValueError("No eligible historical customers")
    buyers = set(orders.loc[(orders.InvoiceDate >= cutoff) & (orders.InvoiceDate < end), "CustomerID"])
    count = len(eligible.intersection(buyers))
    return {"type": "observational_baseline", "not_a_treatment_effect": True, "segment": segment,
            "cutoff": cutoff.isoformat(), "outcome_end": end.isoformat(), "horizon_days": horizon_days,
            "eligible_customers": len(eligible), "repeat_buyers": count, "repeat_rate": count / len(eligible)}


def sample_size(baseline=0.20, mde=0.02, alpha=0.05, power=0.80):
    if not (0 < baseline < 1 and 0 < mde < 1 - baseline):
        raise ValueError("Baseline and positive absolute MDE must describe probabilities strictly within (0, 1)")
    if not (0 < alpha < 1 and 0.5 < power < 1):
        raise ValueError("Require 0 < alpha < 1 and 0.5 < power < 1")
    alternative = baseline + mde
    pooled = (baseline + alternative) / 2
    null_sd = math.sqrt(2 * pooled * (1 - pooled))
    alternative_sd = math.sqrt(baseline * (1 - baseline) + alternative * (1 - alternative))
    n = math.ceil(((norm.ppf(1 - alpha / 2) * null_sd + norm.ppf(power) * alternative_sd) / mde) ** 2)
    return {"baseline": baseline, "alternative": alternative, "absolute_mde": mde,
            "relative_mde": mde / baseline, "alpha": alpha, "power": power,
            "per_group": n, "total": 2 * n,
            "method": "Two-sided normal approximation, independent clients, 1:1 allocation"}


def synthetic_demo(baseline=0.20, mde=0.02, alpha=0.05, power=0.80, seed=42, repetitions=2000):
    plan = sample_size(baseline, mde, alpha, power)
    rng = np.random.default_rng(seed)
    n = plan["per_group"]
    a = rng.binomial(n, baseline, repetitions)
    b = rng.binomial(n, baseline + mde, repetitions)
    pooled = (a + b) / (2 * n)
    sd = np.sqrt(2 * pooled * (1 - pooled) / n)
    z = np.divide(b / n - a / n, sd, out=np.zeros_like(sd), where=sd > 0)
    p_values = 2 * norm.sf(np.abs(z))
    return {"type": "synthetic_simulation", "not_a_real_campaign": True,
            "assumptions": plan, "seed": seed, "repetitions": repetitions,
            "simulated_rejection_rate": float((p_values < alpha).mean()),
            "note": "All outcomes are generated from assumed Bernoulli rates, not observed CRM effects."}

