"""Sales Performance Analytics Proof-of-Concept."""
from .cli import run_pipeline
from .data_processing import build_data_bundle
from .revenue_gap import compute_revenue_gap

__all__ = ["run_pipeline", "build_data_bundle", "compute_revenue_gap"]
