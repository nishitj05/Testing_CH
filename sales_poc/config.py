"""Configuration constants for the sales performance PoC."""
from __future__ import annotations

from pathlib import Path

# Root directory of the repository
ROOT_DIR = Path(__file__).resolve().parent.parent

# Path to the synthetic dataset shipped with the PoC
DATA_PATH = ROOT_DIR / "Synthetic_Sales_Performance_Data_GPBased_Final.csv"

# Default artifacts directory where trained models and SHAP values are persisted
ARTIFACT_DIR = ROOT_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

# Columns that should be treated as categorical identifiers when loading the dataset
CATEGORICAL_COLUMNS = [
    "Rep_ID",
    "First_Name",
    "Last_Name",
    "Territory",
    "Region",
    "Position",
    "Material_Number",
    "Product_Category_Group",
    "Product_Category",
    "Comp_Catg",
    "Comp_Bucket_Title",
]

# Month column used for temporal filtering.
MONTH_COLUMN = "Month"

# Column indicating attainment performance used to identify under-performers.
PERFORMANCE_COLUMN = "Performance"

# Cut-off used in the revenue gap logic.
PERFORMANCE_THRESHOLD = 90

# Month label designating the fiscal year close.
FISCAL_YEAR_CLOSE_MONTH = "December"
