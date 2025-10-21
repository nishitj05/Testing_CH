"""Command-line entry points for the sales performance PoC."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import analyse_attainment_drivers, build_rep_segments, train_risk_classifier
from .config import ARTIFACT_DIR
from .data_processing import build_data_bundle
from .revenue_gap import compute_revenue_gap


def run_pipeline(output_dir: Path | None = None) -> dict:
    """Execute the PoC analytics workflow end-to-end."""

    bundle = build_data_bundle()

    revenue_gap = compute_revenue_gap(bundle.raw)
    segments = build_rep_segments(bundle)
    drivers = analyse_attainment_drivers(bundle)
    risk = train_risk_classifier(bundle)

    if output_dir is None:
        output_dir = ARTIFACT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    segments.assignments.to_csv(output_dir / "rep_segments.csv", index=False)
    driver_summary = _summarise_shap(drivers.feature_columns, drivers.shap_values)
    driver_summary.to_csv(output_dir / "attainment_driver_shap.csv", index=False)
    risk.predictions.to_csv(output_dir / "at_risk_predictions.csv", index=False)

    summary = {
        "revenue_gap": revenue_gap,
        "segment_count": int(segments.assignments["Segment"].nunique()),
        "top_drivers": driver_summary.head(10).to_dict(orient="records"),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def _summarise_shap(feature_columns: list[str], shap_values: pd.Series | pd.DataFrame | np.ndarray) -> pd.DataFrame:
    shap_array = np.array(shap_values)
    mean_abs = np.mean(np.abs(shap_array), axis=0)
    return (
        pd.DataFrame({"feature": feature_columns, "mean_abs_shap": mean_abs})
        .sort_values(by="mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    summary = run_pipeline()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
