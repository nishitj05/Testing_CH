"""Revenue gap calculation aligned with the validated business logic."""
from __future__ import annotations

import pandas as pd

from .config import FISCAL_YEAR_CLOSE_MONTH, PERFORMANCE_THRESHOLD
from .data_processing import load_dataset


def compute_revenue_gap(df: pd.DataFrame | None = None) -> float:
    """Return the revenue gap for December underperformers.

    The calculation is intentionally aligned with the reference validation script:

    1. Keep only December observations.
    2. Filter to rows where ``Performance`` is less than or equal to 90.
    3. Sum the ``YTD_Quota`` and ``YTD_GP`` columns for the remaining records.
    4. Return the difference ``Σ(YTD_Quota) − Σ(YTD_GP)``.

    Args:
        df: Optional DataFrame. When omitted the function loads the configured dataset.

    Returns:
        Revenue gap as a float.
    """

    if df is None:
        df = load_dataset()

    december = df[df["Month"] == FISCAL_YEAR_CLOSE_MONTH]
    underperformers = december[december["Performance"] <= PERFORMANCE_THRESHOLD]

    total_ytd_quota = underperformers["YTD_Quota"].sum()
    total_ytd_gp = underperformers["YTD_GP"].sum()
    revenue_gap = float(total_ytd_quota - total_ytd_gp)
    return revenue_gap


def summarise_december_underperformers(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Provide representative-level attainment summaries for December underperformers."""

    if df is None:
        df = load_dataset()

    december = df[df["Month"] == FISCAL_YEAR_CLOSE_MONTH]
    rep_level = (
        december.groupby("Rep_ID")
        .agg({
            "YTD_Quota": "sum",
            "YTD_GP": "sum",
            "Performance": "first",
        })
        .reset_index()
    )
    rep_level["Rep_Achievement"] = (rep_level["YTD_GP"] / rep_level["YTD_Quota"]) * 100

    bins = [0, 50, 70, 90, 100, float("inf")]
    labels = ["Severe", "Critical", "At Risk", "Acceptable", "High Performer"]
    rep_level["Performance_Bucket"] = pd.cut(
        rep_level["Rep_Achievement"], bins=bins, labels=labels, right=False
    )
    return rep_level
