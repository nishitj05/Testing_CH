"""Flask UI surfacing the PoC analytics with Gemini-authored narratives."""
from __future__ import annotations

from typing import Any, Dict

from flask import Flask, jsonify, render_template

from ..analytics import analyse_attainment_drivers, build_rep_segments, train_risk_classifier
from ..data_processing import build_data_bundle
from ..revenue_gap import compute_revenue_gap
from .gemini import GeminiClient

app = Flask(__name__, template_folder="templates", static_folder="static")

_bundle_cache = build_data_bundle()
_revenue_gap = compute_revenue_gap(_bundle_cache.raw)
_segments = build_rep_segments(_bundle_cache)
_drivers = analyse_attainment_drivers(_bundle_cache)
_risk = train_risk_classifier(_bundle_cache)
gemini_client = GeminiClient()


def _gemini(section: str, context: Dict[str, Any]) -> str:
    prompt = (
        "You are crafting dashboard copy for sales leadership. "
        "Summarise the following context in 3 short bullet points using simple, positive language. "
        "End with a concise call to action. Context: "
        f"Section = {section}. Data snapshot = {context}."
    )
    return gemini_client.generate(prompt).content


@app.route("/")
def dashboard() -> str:
    driver_summary = _summarise_drivers()
    risk_table = _risk_overview()

    context = {
        "revenue_gap": _revenue_gap,
        "segments": _segments.assignments.groupby("Segment").size().to_dict(),
        "top_drivers": driver_summary.head(5).to_dict(orient="records"),
        "risk_reps": risk_table.head(10).to_dict(orient="records"),
        "gemini_copy": {
            "revenue": _gemini("Revenue Gap", {"revenue_gap": round(_revenue_gap, 2)}),
            "segments": _gemini("Rep Segments", {"segment_counts": context_buckets()}),
            "drivers": _gemini(
                "Key Drivers",
                {
                    "top_features": driver_summary.head(3)["feature"].tolist(),
                },
            ),
            "risk": _gemini(
                "At Risk",
                {
                    "high_risk_count": int((risk_table["At_Risk_Prediction"] == 1).sum()),
                },
            ),
        },
    }
    return render_template("dashboard.html", **context)


@app.route("/api/summary")
def api_summary() -> Any:
    summary = {
        "revenue_gap": _revenue_gap,
        "segments": context_buckets(),
        "driver_importance": _summarise_drivers().to_dict(orient="records"),
        "risk_predictions": _risk_overview().to_dict(orient="records"),
    }
    return jsonify(summary)


def _summarise_drivers():
    from ..cli import _summarise_shap

    return _summarise_shap(_drivers.feature_columns, _drivers.shap_values)


def _risk_overview():
    risk = _risk.predictions.copy()
    risk["Month"] = risk["Month"].astype(str)
    return risk.sort_values(by="At_Risk_Score", ascending=False)


def context_buckets() -> Dict[str, int]:
    return _segments.assignments.groupby("Segment").size().to_dict()


if __name__ == "__main__":  # pragma: no cover
    app.run(debug=True)
