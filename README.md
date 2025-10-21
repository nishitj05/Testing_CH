# Sales Performance Analytics PoC

This proof-of-concept bundles the analytics and UI required to explore synthetic sales performance data. It adheres to the validated revenue gap logic provided during discovery and layers additional insights to support revenue uplift, sales representative segmentation, driver analysis, and at-risk detection.

## Features

- **Revenue Gap Baseline** – December underperformers filtered at ≤90% performance with the canonical `Σ(YTD_Quota) − Σ(YTD_GP)` calculation.
- **Representative Segmentation** – K-means clustering on quota, sales, margin, and attainment traits.
- **Driver Analysis** – Gradient boosting model on month-level metrics with SHAP explanations highlighting key attainment drivers.
- **At-Risk Prediction** – Gradient boosting classifier forecasting next-month risk and SHAP-based transparency.
- **Gemini-Powered UI Copy** – Flask dashboard that requests concise narratives from Gemini 2.5 Flash (with graceful fallback when an API key is not provided).

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the Analytics Pipeline

```bash
python -m sales_poc.cli
```

Artifacts (segments, driver importances, risk scores, summary JSON) will be stored in the `artifacts/` directory.

### Launch the Flask UI

```bash
export GEMINI_API_KEY="<your key>"  # optional but recommended
FLASK_APP=sales_poc.ui.flask_app flask run --reload
```

Navigate to http://127.0.0.1:5000 to view the dashboard.

## Project Structure

- `sales_poc/data_processing.py` – ingestion, cleaning, and derived views.
- `sales_poc/revenue_gap.py` – canonical revenue gap logic and December underperformer summaries.
- `sales_poc/analytics.py` – segmentation, driver analysis, and risk modelling utilities.
- `sales_poc/cli.py` – orchestrates the analytics workflow and persists artifacts.
- `sales_poc/ui/` – Gemini client wrapper, Flask app, templates, and static assets.

## Testing

The pipeline relies on pandas, scikit-learn, and shap. For unit testing, you can add tests under a `tests/` directory and run `pytest`. The current scope focuses on exploratory analytics, so no automated tests ship with the PoC.
