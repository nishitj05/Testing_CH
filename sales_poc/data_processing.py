"""Data ingestion and feature engineering utilities for the sales performance PoC."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from .config import (
    CATEGORICAL_COLUMNS,
    DATA_PATH,
    FISCAL_YEAR_CLOSE_MONTH,
    MONTH_COLUMN,
)


@dataclass
class DataBundle:
    """Container for the key DataFrames used throughout the PoC."""

    raw: pd.DataFrame
    rep_month: pd.DataFrame
    december_underperformers: pd.DataFrame


def load_dataset(path: str | None = None) -> pd.DataFrame:
    """Load the synthetic dataset and enforce the expected schema.

    Args:
        path: Optional override for the dataset path. Defaults to the configured path.

    Returns:
        DataFrame with categorical columns cast to ``category`` dtype and
        date-like columns parsed to datetime where possible.
    """

    dataset_path = DATA_PATH if path is None else path
    df = pd.read_csv(dataset_path)

    # Normalise column names and strip whitespace from string columns for consistency.
    df.columns = [col.strip() for col in df.columns]
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("category")

    # Ensure month column is treated as ordered categorical for temporal operations.
    if MONTH_COLUMN in df.columns:
        month_mapping = {
            "Jan": "January",
            "Feb": "February",
            "Mar": "March",
            "Apr": "April",
            "May": "May",
            "Jun": "June",
            "Jul": "July",
            "Aug": "August",
            "Sep": "September",
            "Sept": "September",
            "Oct": "October",
            "Nov": "November",
            "Dec": "December",
        }
        df[MONTH_COLUMN] = df[MONTH_COLUMN].replace(month_mapping)
        df[MONTH_COLUMN] = pd.Categorical(
            df[MONTH_COLUMN],
            categories=[
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ],
            ordered=True,
        )

    # Coerce numeric columns from strings with thousands separators if necessary.
    numeric_columns = df.select_dtypes(include=["object"]).columns
    for column in numeric_columns:
        try:
            df[column] = pd.to_numeric(df[column].str.replace(",", ""))
        except (ValueError, AttributeError):
            # Not a numeric column; leave as-is.
            continue

    return df


def create_rep_month_view(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the raw dataset at the representative-month grain.

    The aggregation computes total quota, sales, and gross profit alongside
    attainment metrics useful for modelling.
    """

    aggregations = {
        "Monthly_Quota": "sum",
        "Monthly_Sales": "sum",
        "Monthly_GP": "sum",
        "Monthly_Achievement_Pct": "mean",
        "YTD_Quota": "max",
        "YTD_Sales": "max",
        "YTD_GP": "max",
        "YTD_Achievement_Pct": "max",
        "Performance": "max",
        "Payout_Rate": "mean",
        "Calculated_GP_Margin_Pct": "mean",
    }

    rep_month = (
        df.groupby(["Rep_ID", MONTH_COLUMN], observed=True)
        .agg(aggregations)
        .reset_index()
        .sort_values(by=["Rep_ID", MONTH_COLUMN])
    )

    # Calculate trailing deltas to capture month-on-month trends.
    rep_month["Monthly_Sales_Delta"] = (
        rep_month.groupby("Rep_ID")["Monthly_Sales"].diff().fillna(0.0)
    )
    rep_month["Monthly_GP_Delta"] = (
        rep_month.groupby("Rep_ID")["Monthly_GP"].diff().fillna(0.0)
    )
    rep_month["Achievement_Delta"] = (
        rep_month.groupby("Rep_ID")["Monthly_Achievement_Pct"].diff().fillna(0.0)
    )

    # Flag risk label (1 = at risk of missing quota next month based on achievement < 100).
    rep_month["At_Risk_Next_Month"] = (
        rep_month.groupby("Rep_ID")["Monthly_Achievement_Pct"].shift(-1).fillna(100)
        < 100
    ).astype(int)

    return rep_month


def extract_december_underperformers(df: pd.DataFrame) -> pd.DataFrame:
    """Return December rows where performance falls below the defined threshold."""

    mask = (df[MONTH_COLUMN] == FISCAL_YEAR_CLOSE_MONTH) & (df["Performance"] <= 90)
    december_underperformers = df.loc[mask].copy()
    return december_underperformers


def build_data_bundle(path: str | None = None) -> DataBundle:
    """Load the dataset and prepare the commonly used derived views."""

    raw = load_dataset(path)
    rep_month = create_rep_month_view(raw)
    december_underperformers = extract_december_underperformers(raw)
    return DataBundle(
        raw=raw,
        rep_month=rep_month,
        december_underperformers=december_underperformers,
    )
