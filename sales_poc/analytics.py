"""Analytical workloads supporting the sales performance PoC."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_processing import DataBundle


@dataclass
class SegmentationResult:
    assignments: pd.DataFrame
    model: KMeans
    feature_columns: List[str]


@dataclass
class DriverAnalysisResult:
    model: GradientBoostingRegressor
    shap_values: np.ndarray
    expected_value: float
    feature_matrix: pd.DataFrame
    feature_columns: List[str]


@dataclass
class RiskModelResult:
    model: GradientBoostingClassifier
    shap_values: np.ndarray
    expected_value: float
    feature_matrix: pd.DataFrame
    predictions: pd.DataFrame
    feature_columns: List[str]
    classification_report: str


NUMERIC_FEATURES = [
    "Monthly_Quota",
    "Monthly_Sales",
    "Monthly_GP",
    "Monthly_Achievement_Pct",
    "Monthly_Sales_Delta",
    "Monthly_GP_Delta",
    "Achievement_Delta",
    "YTD_Quota",
    "YTD_Sales",
    "YTD_GP",
    "YTD_Achievement_Pct",
    "Payout_Rate",
    "Calculated_GP_Margin_Pct",
]

CATEGORICAL_FEATURES = ["Month"]


def _prepare_features(rep_month: pd.DataFrame) -> pd.DataFrame:
    features = rep_month[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features


def build_rep_segments(bundle: DataBundle, n_segments: int = 5) -> SegmentationResult:
    """Cluster representatives into peer segments using KMeans."""

    rep_level = (
        bundle.rep_month.groupby("Rep_ID")[NUMERIC_FEATURES]
        .agg(["mean", "std"])
        .fillna(0)
    )
    rep_level.columns = ["_".join(col).strip("_") for col in rep_level.columns.to_flat_index()]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rep_level)

    model = KMeans(n_clusters=n_segments, n_init=10, random_state=42)
    segment_labels = model.fit_predict(scaled)

    assignments = rep_level.reset_index()
    assignments["Segment"] = segment_labels

    return SegmentationResult(
        assignments=assignments,
        model=model,
        feature_columns=list(rep_level.columns),
    )


def analyse_attainment_drivers(bundle: DataBundle) -> DriverAnalysisResult:
    """Fit a gradient boosting regressor to explain attainment percentage."""

    features = _prepare_features(bundle.rep_month)
    y = features.pop("Monthly_Achievement_Pct")

    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, [col for col in features.columns if col not in CATEGORICAL_FEATURES]),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = GradientBoostingRegressor(random_state=42)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(features, y)

    # SHAP explanations use the trained booster directly on encoded data.
    transformed = pipeline.named_steps["preprocessor"].transform(features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    sample_size = min(1000, transformed.shape[0])
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(transformed.shape[0], size=sample_size, replace=False)
    transformed_sample = transformed[sample_indices]

    explainer = shap.Explainer(pipeline.named_steps["model"])
    shap_values = explainer(transformed_sample)

    feature_names_numeric = [
        col for col in features.columns if col not in CATEGORICAL_FEATURES
    ]
    feature_names_categorical = pipeline.named_steps["preprocessor"].named_transformers_[
        "categorical"
    ].get_feature_names_out(CATEGORICAL_FEATURES)
    feature_columns = list(feature_names_numeric) + list(feature_names_categorical)

    feature_matrix = pd.DataFrame(
        transformed_sample, columns=feature_columns, index=features.index[sample_indices]
    )

    return DriverAnalysisResult(
        model=pipeline.named_steps["model"],
        shap_values=shap_values.values,
        expected_value=float(shap_values.base_values.mean()),
        feature_matrix=feature_matrix,
        feature_columns=feature_columns,
    )


def train_risk_classifier(bundle: DataBundle) -> RiskModelResult:
    """Train a classifier that predicts whether a rep is at risk next month."""

    features = _prepare_features(bundle.rep_month)
    y = bundle.rep_month.loc[features.index, "At_Risk_Next_Month"]

    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, [col for col in features.columns if col not in CATEGORICAL_FEATURES]),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = GradientBoostingClassifier(random_state=42)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(features, y)

    transformed = pipeline.named_steps["preprocessor"].transform(features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    sample_size = min(1000, transformed.shape[0])
    rng = np.random.default_rng(21)
    sample_indices = rng.choice(transformed.shape[0], size=sample_size, replace=False)
    transformed_sample = transformed[sample_indices]

    explainer = shap.Explainer(pipeline.named_steps["model"])
    shap_values = explainer(transformed_sample)

    feature_names_numeric = [
        col for col in features.columns if col not in CATEGORICAL_FEATURES
    ]
    feature_names_categorical = pipeline.named_steps["preprocessor"].named_transformers_[
        "categorical"
    ].get_feature_names_out(CATEGORICAL_FEATURES)
    feature_columns = list(feature_names_numeric) + list(feature_names_categorical)

    feature_matrix = pd.DataFrame(
        transformed_sample, columns=feature_columns, index=features.index[sample_indices]
    )

    proba = pipeline.predict_proba(features)[:, 1]
    predictions = pd.DataFrame(
        {
            "Rep_ID": bundle.rep_month.loc[features.index, "Rep_ID"].values,
            "Month": bundle.rep_month.loc[features.index, "Month"].values,
            "At_Risk_Score": proba,
            "At_Risk_Prediction": (proba >= 0.5).astype(int),
            "Actual": y.values,
        },
        index=features.index,
    )

    report = classification_report(y, predictions["At_Risk_Prediction"], digits=3)

    return RiskModelResult(
        model=pipeline.named_steps["model"],
        shap_values=shap_values.values,
        expected_value=float(np.mean(shap_values.base_values)),
        feature_matrix=feature_matrix,
        predictions=predictions,
        feature_columns=feature_columns,
        classification_report=report,
    )
