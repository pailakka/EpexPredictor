from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import lightgbm as lgb
import pandas as pd

from .forecastartifactstore import ForecastArtifactStore
from .priceregion import PriceRegion


FEATURE_GROUP_LABELS = {
    "weather": "Weather",
    "time_calendar": "Time & Calendar",
    "demand_load": "Demand & Load",
    "renewables": "Renewables",
    "cross_border": "Cross-Border Capacity & Flows",
    "imbalance_state": "Imbalance & Balancing State",
    "coupled_market": "Coupled-Market Shadow Prices",
    "persistence": "Persistence & Price History",
    "gas_other": "Gas & Other",
}

FEATURE_GROUP_ORDER = [
    "weather",
    "time_calendar",
    "demand_load",
    "renewables",
    "cross_border",
    "imbalance_state",
    "coupled_market",
    "persistence",
    "gas_other",
]

SCENARIO_KNOBS = {
    "load_forecast_pct": "Load forecast delta %",
    "wind_forecast_pct": "Wind forecast delta %",
    "solar_forecast_pct": "Solar forecast delta %",
    "import_headroom_mw": "Import headroom delta MW",
    "cross_border_flow_mw": "Cross-border flow pressure delta MW",
    "coupled_market_spread_delta": "Coupled-market spread delta",
}


def to_utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


@lru_cache(maxsize=64)
def load_booster(model_path: str) -> lgb.Booster:
    return lgb.Booster(model_file=model_path)


@dataclass(frozen=True)
class ModelBundle:
    version: str
    point: lgb.Booster
    q50: lgb.Booster | None
    spike_classifier: lgb.Booster | None
    spike_uplift: lgb.Booster | None
    feature_names: tuple[str, ...]


class ForecastExplainer:
    def __init__(self, region: PriceRegion, artifact_store: ForecastArtifactStore):
        self.region = region
        self.artifact_store = artifact_store

    def _normalize_snapshot_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        required_columns: dict[str, Any] = {
            "generated_at_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "target_time_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "region": pd.Series(dtype="string"),
            "model_version": pd.Series(dtype="string"),
            "predicted_price": pd.Series(dtype=float),
            "actual_price": pd.Series(dtype=float),
            "yesterday_known_price": pd.Series(dtype=float),
            "explainable": pd.Series(dtype=bool),
            "explainable_reason": pd.Series(dtype="object"),
        }
        for column, empty_series in required_columns.items():
            if column not in normalized.columns:
                normalized[column] = empty_series.reindex(normalized.index)
        return normalized

    def build_snapshot_catalog(
        self,
        predictions: pd.DataFrame,
        features: pd.DataFrame,
        start_ts: Any | None = None,
        end_ts: Any | None = None,
    ) -> list[dict[str, Any]]:
        frame = predictions.copy()
        if frame.empty:
            return []
        frame["generated_at_utc"] = pd.to_datetime(frame["generated_at_utc"], utc=True)
        frame["target_time_utc"] = pd.to_datetime(frame["target_time_utc"], utc=True)

        if start_ts is not None:
            frame = frame[frame["target_time_utc"] >= to_utc_timestamp(start_ts)]
        if end_ts is not None:
            frame = frame[frame["target_time_utc"] <= to_utc_timestamp(end_ts)]
        if frame.empty:
            return []

        explainable_keys = pd.DataFrame(columns=["generated_at_utc", "target_time_utc"])
        if not features.empty:
            explainable_keys = features.copy()
            explainable_keys["generated_at_utc"] = pd.to_datetime(explainable_keys["generated_at_utc"], utc=True)
            explainable_keys["target_time_utc"] = pd.to_datetime(explainable_keys["target_time_utc"], utc=True)
            if "model_version" in explainable_keys.columns:
                explainable_keys = explainable_keys[explainable_keys["model_version"].notna()]
            explainable_keys = explainable_keys[["generated_at_utc", "target_time_utc"]].drop_duplicates()

        explainable_frame = frame.merge(
            explainable_keys.assign(explainable=True),
            on=["generated_at_utc", "target_time_utc"],
            how="left",
        )
        explainable_frame["explainable"] = explainable_frame["explainable"].fillna(False)

        runs: list[dict[str, Any]] = []
        for generated_at, group in explainable_frame.groupby("generated_at_utc"):
            model_versions = []
            if "model_version" in group.columns:
                model_versions = sorted(str(value) for value in group["model_version"].dropna().unique())
            runs.append(
                {
                    "generated_at_utc": generated_at.isoformat(),
                    "target_start_utc": group["target_time_utc"].min().isoformat(),
                    "target_end_utc": group["target_time_utc"].max().isoformat(),
                    "row_count": int(len(group)),
                    "explainable_row_count": int(group["explainable"].sum()),
                    "explainable": bool(group["explainable"].any()),
                    "model_versions": model_versions,
                }
            )
        runs.sort(key=lambda item: item["generated_at_utc"], reverse=True)
        return runs

    def merge_snapshot_frames(
        self,
        predictions: pd.DataFrame,
        features: pd.DataFrame,
        actual_prices: pd.DataFrame,
    ) -> pd.DataFrame:
        prediction_frame = self._normalize_snapshot_frame(predictions)
        feature_frame = self._normalize_snapshot_frame(features)
        if prediction_frame.empty:
            return prediction_frame.sort_values(["target_time_utc", "generated_at_utc"]).reset_index(drop=True)

        prediction_frame["generated_at_utc"] = pd.to_datetime(prediction_frame["generated_at_utc"], utc=True)
        prediction_frame["target_time_utc"] = pd.to_datetime(prediction_frame["target_time_utc"], utc=True)
        feature_frame["generated_at_utc"] = pd.to_datetime(feature_frame["generated_at_utc"], utc=True)
        feature_frame["target_time_utc"] = pd.to_datetime(feature_frame["target_time_utc"], utc=True)

        merged = prediction_frame.merge(
            feature_frame,
            on=["generated_at_utc", "target_time_utc", "region"],
            how="left",
            suffixes=("", "_feature"),
        )

        if "model_version_feature" in merged.columns:
            prediction_version = merged.get("model_version")
            feature_version = merged["model_version_feature"]
            merged["model_version"] = prediction_version.where(prediction_version.notna(), feature_version)
            mismatch = prediction_version.notna() & feature_version.notna() & (prediction_version != feature_version)
            merged.loc[mismatch, "model_version"] = pd.NA
            merged.drop(columns=["model_version_feature"], inplace=True)

        actual_series = pd.Series(dtype=float)
        if not actual_prices.empty and "price" in actual_prices.columns:
            actual_series = actual_prices["price"].sort_index()

        merged["actual_price"] = actual_series.reindex(merged["target_time_utc"]).to_numpy()
        merged["yesterday_known_price"] = actual_series.reindex(
            pd.DatetimeIndex(merged["target_time_utc"]) - pd.Timedelta(days=1)
        ).to_numpy()
        merged["explainable"] = merged["model_version"].notna()
        merged["explainable_reason"] = None
        merged.loc[~merged["explainable"], "explainable_reason"] = (
            "Snapshot was recorded before model-version tracking or its feature row is missing."
        )
        return merged.sort_values(["target_time_utc", "generated_at_utc"]).reset_index(drop=True)

    def select_rows(
        self,
        merged: pd.DataFrame,
        selection: str = "latest",
        generated_at: Any | None = None,
    ) -> pd.DataFrame:
        if merged.empty:
            return merged

        frame = merged.copy()
        if generated_at is not None:
            selected_generated_at = to_utc_timestamp(generated_at)
            return frame[frame["generated_at_utc"] == selected_generated_at].sort_values("target_time_utc").reset_index(drop=True)

        if selection == "earliest":
            return (
                frame.sort_values(["target_time_utc", "generated_at_utc"])
                .groupby("target_time_utc", as_index=False, sort=True)
                .head(1)
                .sort_values("target_time_utc")
                .reset_index(drop=True)
            )

        return (
            frame.sort_values(["target_time_utc", "generated_at_utc"])
            .groupby("target_time_utc", as_index=False, sort=True)
            .tail(1)
            .sort_values("target_time_utc")
            .reset_index(drop=True)
        )

    def explain_row(self, row: pd.Series) -> dict[str, Any]:
        frame = pd.DataFrame([row.to_dict()])
        result = self.explain_frame(frame)
        if not result["row_details"]:
            return {
                "explainable": False,
                "reason": "Snapshot row is not explainable.",
                "generated_at_utc": row.get("generated_at_utc"),
                "target_time_utc": row.get("target_time_utc"),
            }
        return result["row_details"][0]

    def explain_frame(self, frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {
                "feature_contributions": pd.DataFrame(index=frame.index),
                "base_value": pd.Series(dtype=float),
                "adjustments": pd.DataFrame(index=frame.index),
                "reconstructed_prediction": pd.Series(dtype=float),
                "row_details": [],
            }

        working = frame.copy()
        explainable = working[working["model_version"].notna()].copy()
        if explainable.empty:
            return {
                "feature_contributions": pd.DataFrame(index=frame.index),
                "base_value": pd.Series(dtype=float),
                "adjustments": pd.DataFrame(index=frame.index),
                "reconstructed_prediction": pd.Series(dtype=float),
                "row_details": [
                    {
                        "explainable": False,
                        "reason": row.get("explainable_reason")
                        or "Snapshot was recorded before model-version tracking or its feature row is missing.",
                        "generated_at_utc": self._iso(row["generated_at_utc"]),
                        "target_time_utc": self._iso(row["target_time_utc"]),
                    }
                    for _, row in working.iterrows()
                ],
            }

        feature_contributions = pd.DataFrame(index=explainable.index)
        base_values = pd.Series(index=explainable.index, dtype=float)
        adjustments = pd.DataFrame(
            0.0,
            index=explainable.index,
            columns=["spike_adjustment", "yesterday_blend_component"],
        )
        reconstructed = pd.Series(index=explainable.index, dtype=float)

        for model_version, version_frame in explainable.groupby("model_version"):
            try:
                bundle = self._load_bundle(str(model_version))
            except FileNotFoundError:
                working.loc[version_frame.index, "explainable_reason"] = (
                    f"Saved point model for version {model_version} is unavailable."
                )
                continue
            version_features = self._prepare_feature_matrix(version_frame, bundle.feature_names)

            point_contrib, point_bias = self._predict_contributions(bundle.point, version_features, bundle.feature_names)
            feature_frame = point_contrib
            base_frame = point_bias
            preblend_scale = 1.0

            if bundle.q50 is not None:
                q50_contrib, q50_bias = self._predict_contributions(bundle.q50, version_features, bundle.feature_names)
                feature_frame = point_contrib.mul(0.85).add(q50_contrib.mul(0.15), fill_value=0.0)
                base_frame = point_bias * 0.85 + q50_bias * 0.15
                preblend_scale = 0.85

            spike_probability = pd.Series(0.0, index=version_frame.index)
            if bundle.spike_classifier is not None:
                spike_probability = pd.Series(
                    bundle.spike_classifier.predict(version_features),
                    index=version_frame.index,
                    dtype=float,
                )

            if bundle.spike_uplift is not None and bundle.spike_classifier is not None:
                uplift = pd.Series(
                    bundle.spike_uplift.predict(version_features),
                    index=version_frame.index,
                    dtype=float,
                )
                adjustments.loc[version_frame.index, "spike_adjustment"] = (
                    uplift * spike_probability.clip(0.0, 1.0) * preblend_scale
                )

            if self.region.yesterday_blend_weight > 0.0:
                yesterday_source = version_frame.get("yesterday_known_price")
                if yesterday_source is None:
                    yesterday = pd.Series(float("nan"), index=version_frame.index, dtype=float)
                else:
                    yesterday = pd.to_numeric(yesterday_source, errors="coerce")
                if yesterday.isna().all():
                    if "price_lag_1d" in version_features.columns:
                        yesterday = pd.to_numeric(version_features["price_lag_1d"], errors="coerce")
                    else:
                        yesterday = pd.Series(float("nan"), index=version_frame.index, dtype=float)
                blend_weight = self.region.yesterday_blend_weight * (1.0 - spike_probability.clip(0.0, 1.0))
                blend_weight = blend_weight.where(yesterday.notna(), 0.0)
                scale = 1.0 - blend_weight
                feature_frame = feature_frame.mul(scale, axis=0)
                base_frame = base_frame * scale
                adjustments.loc[version_frame.index, "spike_adjustment"] = (
                    adjustments.loc[version_frame.index, "spike_adjustment"] * scale
                )
                adjustments.loc[version_frame.index, "yesterday_blend_component"] = yesterday.fillna(0.0) * blend_weight

            feature_contributions = feature_contributions.combine_first(feature_frame)
            base_values.loc[version_frame.index] = base_frame
            reconstructed.loc[version_frame.index] = (
                feature_frame.sum(axis=1)
                + base_frame
                + adjustments.loc[version_frame.index].sum(axis=1)
            )

        row_details: list[dict[str, Any]] = []
        all_feature_columns = sorted(feature_contributions.columns)
        for index, row in working.iterrows():
            if index not in explainable.index or index not in feature_contributions.index:
                row_details.append(
                    {
                        "explainable": False,
                        "reason": row.get("explainable_reason")
                        or "Snapshot was recorded before model-version tracking or its feature row is missing.",
                        "generated_at_utc": self._iso(row["generated_at_utc"]),
                        "target_time_utc": self._iso(row["target_time_utc"]),
                    }
                )
                continue

            contribution_series = feature_contributions.loc[index].reindex(all_feature_columns).fillna(0.0)
            nonzero = contribution_series[contribution_series.abs() > 1e-12]
            feature_names = (
                nonzero.abs().sort_values(ascending=False).index.tolist()
                if not nonzero.empty
                else contribution_series.abs().sort_values(ascending=False).index.tolist()
            )
            feature_items = []
            for rank, feature_name in enumerate(feature_names, start=1):
                group_id = feature_group_for(feature_name)
                feature_items.append(
                    {
                        "feature_name": feature_name,
                        "group_id": group_id,
                        "group_label": FEATURE_GROUP_LABELS[group_id],
                        "feature_value": self._safe_float(row.get(feature_name)),
                        "signed_contribution": self._safe_float(contribution_series[feature_name]),
                        "absolute_rank": rank,
                    }
                )

            group_items = []
            for group_id in FEATURE_GROUP_ORDER:
                group_value = float(
                    contribution_series[
                        [name for name in contribution_series.index if feature_group_for(name) == group_id]
                    ].sum()
                )
                if math.isclose(group_value, 0.0, abs_tol=1e-12):
                    continue
                group_items.append(
                    {
                        "group_id": group_id,
                        "group_label": FEATURE_GROUP_LABELS[group_id],
                        "signed_contribution": group_value,
                        "absolute_contribution": abs(group_value),
                    }
                )
            group_items.sort(key=lambda item: abs(item["signed_contribution"]), reverse=True)

            adjustment_items = []
            for adjustment_name, label in [
                ("spike_adjustment", "Spike adjustment"),
                ("yesterday_blend_component", "Yesterday blend component"),
            ]:
                value = float(adjustments.loc[index, adjustment_name])
                if math.isclose(value, 0.0, abs_tol=1e-12):
                    continue
                adjustment_items.append(
                    {
                        "adjustment_name": adjustment_name,
                        "label": label,
                        "signed_contribution": value,
                        "absolute_contribution": abs(value),
                    }
                )

            reconstructed_prediction = float(reconstructed.loc[index])
            saved_prediction = self._safe_float(row.get("predicted_price"))
            row_details.append(
                {
                    "explainable": True,
                    "region": row.get("region"),
                    "generated_at_utc": self._iso(row["generated_at_utc"]),
                    "target_time_utc": self._iso(row["target_time_utc"]),
                    "model_version": row.get("model_version"),
                    "predicted_price": saved_prediction,
                    "actual_price": self._safe_float(row.get("actual_price")),
                    "base_value": float(base_values.loc[index]),
                    "reconstructed_prediction": reconstructed_prediction,
                    "reconciliation_error": (
                        None if saved_prediction is None else reconstructed_prediction - saved_prediction
                    ),
                    "feature_contributions": feature_items,
                    "group_contributions": group_items,
                    "adjustments": adjustment_items,
                }
            )

        return {
            "feature_contributions": feature_contributions.reindex(working.index).fillna(0.0),
            "base_value": base_values.reindex(working.index).fillna(0.0),
            "adjustments": adjustments.reindex(working.index).fillna(0.0),
            "reconstructed_prediction": reconstructed.reindex(working.index),
            "row_details": row_details,
        }

    def summarize_rows(self, frame: pd.DataFrame) -> dict[str, Any]:
        frame = self._normalize_snapshot_frame(frame)
        explainable = frame[(frame["model_version"].notna()) & frame["actual_price"].notna()].copy()
        if explainable.empty:
            return {
                "rows_evaluated": 0,
                "explainable_rows": 0,
                "group_summary": [],
                "feature_summary": [],
                "adjustment_summary": [],
                "error_slices": {},
            }

        explanation = self.explain_frame(explainable)
        feature_contributions = explanation["feature_contributions"]
        adjustments = explanation["adjustments"]

        explainable["abs_error"] = (
            pd.to_numeric(explainable["predicted_price"], errors="coerce")
            - pd.to_numeric(explainable["actual_price"], errors="coerce")
        ).abs()
        explainable["error_slice"] = self._assign_error_slices(explainable["abs_error"])

        group_summary = self._summarize_groups(feature_contributions)
        feature_summary = self._summarize_features(feature_contributions)
        adjustment_summary = self._summarize_features(adjustments)

        error_slices: dict[str, Any] = {}
        for slice_name, slice_frame in explainable.groupby("error_slice"):
            slice_features = feature_contributions.loc[slice_frame.index]
            slice_adjustments = adjustments.loc[slice_frame.index]
            error_slices[str(slice_name)] = {
                "rows": int(len(slice_frame)),
                "group_summary": self._summarize_groups(slice_features)[:5],
                "feature_summary": self._summarize_features(slice_features)[:5],
                "adjustment_summary": self._summarize_features(slice_adjustments)[:3],
            }

        return {
            "rows_evaluated": int(len(explainable)),
            "explainable_rows": int(len(explainable)),
            "group_summary": group_summary,
            "feature_summary": feature_summary[:20],
            "adjustment_summary": adjustment_summary,
            "error_slices": error_slices,
        }

    def evaluate_scenario(self, row: pd.Series, scenario_inputs: dict[str, float]) -> dict[str, Any]:
        baseline = self.explain_row(row)
        if not baseline.get("explainable"):
            return baseline

        frame = pd.DataFrame([row.to_dict()])
        explainable = frame[frame["model_version"].notna()].copy()
        bundle = self._load_bundle(str(explainable.iloc[0]["model_version"]))
        feature_frame = self._prepare_feature_matrix(explainable, bundle.feature_names)
        scenario_frame = feature_frame.copy()
        applied_changes, changed_features = self._apply_scenario_inputs(scenario_frame, scenario_inputs)

        scenario_row = explainable.copy()
        for feature_name in scenario_frame.columns:
            scenario_row.loc[scenario_row.index[0], feature_name] = scenario_frame.iloc[0][feature_name]

        scenario = self.explain_row(scenario_row.iloc[0])
        baseline_prediction = baseline.get("reconstructed_prediction")
        scenario_prediction = scenario.get("reconstructed_prediction")
        scenario_group_deltas = self._diff_ranked_items(
            baseline.get("group_contributions", []),
            scenario.get("group_contributions", []),
            key="group_id",
            label_key="group_label",
        )
        scenario_adjustment_deltas = self._diff_ranked_items(
            baseline.get("adjustments", []),
            scenario.get("adjustments", []),
            key="adjustment_name",
            label_key="label",
        )
        return {
            "explainable": True,
            "region": row.get("region"),
            "generated_at_utc": self._iso(row["generated_at_utc"]),
            "target_time_utc": self._iso(row["target_time_utc"]),
            "model_version": row.get("model_version"),
            "baseline_prediction": baseline_prediction,
            "scenario_prediction": scenario_prediction,
            "prediction_delta": (
                None
                if baseline_prediction is None or scenario_prediction is None
                else scenario_prediction - baseline_prediction
            ),
            "baseline": baseline,
            "scenario": scenario,
            "applied_inputs": applied_changes,
            "changed_features": changed_features,
            "group_deltas": scenario_group_deltas,
            "adjustment_deltas": scenario_adjustment_deltas,
        }

    def _load_bundle(self, version: str) -> ModelBundle:
        model_files = self.artifact_store.get_model_files(version)
        point_path = model_files.get("point")
        if point_path is None:
            raise FileNotFoundError(f"Saved point model for version {version} is unavailable")

        point = load_booster(point_path)
        q50 = load_booster(model_files["q50"]) if "q50" in model_files else None
        spike_classifier = (
            load_booster(model_files["spike_classifier"])
            if "spike_classifier" in model_files
            else None
        )
        spike_uplift = load_booster(model_files["spike_uplift"]) if "spike_uplift" in model_files else None
        return ModelBundle(
            version=version,
            point=point,
            q50=q50,
            spike_classifier=spike_classifier,
            spike_uplift=spike_uplift,
            feature_names=tuple(point.feature_name()),
        )

    def _prepare_feature_matrix(
        self,
        frame: pd.DataFrame,
        feature_names: tuple[str, ...],
    ) -> pd.DataFrame:
        matrix = frame.reindex(columns=list(feature_names))
        for column in matrix.columns:
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
        return matrix.astype(float)

    def _predict_contributions(
        self,
        model: lgb.Booster,
        features: pd.DataFrame,
        feature_names: tuple[str, ...],
    ) -> tuple[pd.DataFrame, pd.Series]:
        contrib = model.predict(features, pred_contrib=True)
        contribution_frame = pd.DataFrame(contrib[:, :-1], index=features.index, columns=list(feature_names))
        bias = pd.Series(contrib[:, -1], index=features.index, dtype=float)
        return contribution_frame, bias

    def _summarize_groups(self, feature_contributions: pd.DataFrame) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        if feature_contributions.empty:
            return summaries

        for group_id in FEATURE_GROUP_ORDER:
            group_columns = [column for column in feature_contributions.columns if feature_group_for(column) == group_id]
            if not group_columns:
                continue
            group_series = feature_contributions[group_columns].sum(axis=1)
            summaries.append(
                {
                    "group_id": group_id,
                    "group_label": FEATURE_GROUP_LABELS[group_id],
                    "mean_signed_contribution": float(group_series.mean()),
                    "mean_abs_contribution": float(group_series.abs().mean()),
                }
            )
        summaries.sort(key=lambda item: item["mean_abs_contribution"], reverse=True)
        return summaries

    def _summarize_features(self, contributions: pd.DataFrame) -> list[dict[str, Any]]:
        if contributions.empty:
            return []
        rows: list[dict[str, Any]] = []
        for column in contributions.columns:
            series = pd.to_numeric(contributions[column], errors="coerce").fillna(0.0)
            if series.abs().sum() == 0:
                continue
            item = {
                "name": column,
                "mean_signed_contribution": float(series.mean()),
                "mean_abs_contribution": float(series.abs().mean()),
            }
            if column == "spike_adjustment":
                item["label"] = "Spike adjustment"
            elif column == "yesterday_blend_component":
                item["label"] = "Yesterday blend component"
            else:
                group_id = feature_group_for(column)
                item["group_id"] = group_id
                item["group_label"] = FEATURE_GROUP_LABELS[group_id]
            rows.append(item)
        rows.sort(key=lambda item: item["mean_abs_contribution"], reverse=True)
        return rows

    def _assign_error_slices(self, abs_error: pd.Series) -> pd.Series:
        ranked = abs_error.rank(method="first")
        return pd.qcut(ranked, q=min(3, len(abs_error)), labels=["low", "medium", "high"][: min(3, len(abs_error))])

    def _apply_scenario_inputs(
        self,
        feature_frame: pd.DataFrame,
        scenario_inputs: dict[str, float],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        row = feature_frame.iloc[0]
        changed_features: dict[str, tuple[float, float]] = {}
        applied_inputs: list[dict[str, Any]] = []

        def apply_delta(column: str, delta: float) -> None:
            if column not in row.index or pd.isna(row[column]):
                return
            before = float(row[column])
            after = before + delta
            row[column] = after
            if column in changed_features:
                changed_features[column] = (changed_features[column][0], after)
            else:
                changed_features[column] = (before, after)

        def apply_pct(column: str, pct: float) -> float | None:
            if column not in row.index or pd.isna(row[column]):
                return None
            return float(row[column]) * pct / 100.0

        load_delta = apply_pct("market_load_forecast", float(scenario_inputs.get("load_forecast_pct", 0.0)))
        if load_delta is not None and not math.isclose(load_delta, 0.0, abs_tol=1e-12):
            apply_delta("market_load_forecast", load_delta)
            apply_delta("load", load_delta)
            apply_delta("market_residual_load", load_delta)
            applied_inputs.append({"name": "load_forecast_pct", "label": SCENARIO_KNOBS["load_forecast_pct"], "value": float(scenario_inputs["load_forecast_pct"]), "applied": True})
        elif "load_forecast_pct" in scenario_inputs:
            applied_inputs.append({"name": "load_forecast_pct", "label": SCENARIO_KNOBS["load_forecast_pct"], "value": float(scenario_inputs["load_forecast_pct"]), "applied": load_delta is not None, "reason": None if load_delta is not None else "Load forecast feature is unavailable for this row."})

        wind_delta = apply_pct("market_wind_forecast", float(scenario_inputs.get("wind_forecast_pct", 0.0)))
        if wind_delta is not None and not math.isclose(wind_delta, 0.0, abs_tol=1e-12):
            apply_delta("market_wind_forecast", wind_delta)
            apply_delta("market_residual_load", -wind_delta)
            apply_delta("market_dispatchable_gap", -wind_delta)
            applied_inputs.append({"name": "wind_forecast_pct", "label": SCENARIO_KNOBS["wind_forecast_pct"], "value": float(scenario_inputs["wind_forecast_pct"]), "applied": True})
        elif "wind_forecast_pct" in scenario_inputs:
            applied_inputs.append({"name": "wind_forecast_pct", "label": SCENARIO_KNOBS["wind_forecast_pct"], "value": float(scenario_inputs["wind_forecast_pct"]), "applied": wind_delta is not None, "reason": None if wind_delta is not None else "Wind forecast feature is unavailable for this row."})

        solar_delta = apply_pct("market_solar_forecast", float(scenario_inputs.get("solar_forecast_pct", 0.0)))
        if solar_delta is not None and not math.isclose(solar_delta, 0.0, abs_tol=1e-12):
            apply_delta("market_solar_forecast", solar_delta)
            apply_delta("market_residual_load", -solar_delta)
            apply_delta("market_dispatchable_gap", -solar_delta)
            applied_inputs.append({"name": "solar_forecast_pct", "label": SCENARIO_KNOBS["solar_forecast_pct"], "value": float(scenario_inputs["solar_forecast_pct"]), "applied": True})
        elif "solar_forecast_pct" in scenario_inputs:
            applied_inputs.append({"name": "solar_forecast_pct", "label": SCENARIO_KNOBS["solar_forecast_pct"], "value": float(scenario_inputs["solar_forecast_pct"]), "applied": solar_delta is not None, "reason": None if solar_delta is not None else "Solar forecast feature is unavailable for this row."})

        import_delta = float(scenario_inputs.get("import_headroom_mw", 0.0))
        if not math.isclose(import_delta, 0.0, abs_tol=1e-12):
            if "available_import_headroom" in row.index and pd.notna(row["available_import_headroom"]):
                apply_delta("available_import_headroom", import_delta)
                applied_inputs.append({"name": "import_headroom_mw", "label": SCENARIO_KNOBS["import_headroom_mw"], "value": import_delta, "applied": True})
            else:
                applied_inputs.append({"name": "import_headroom_mw", "label": SCENARIO_KNOBS["import_headroom_mw"], "value": import_delta, "applied": False, "reason": "Import headroom feature is unavailable for this row."})

        flow_delta = float(scenario_inputs.get("cross_border_flow_mw", 0.0))
        if not math.isclose(flow_delta, 0.0, abs_tol=1e-12):
            touched = False
            for column in [
                "flow_ee_to_fi_recent",
                "flow_ee_to_fi_recent_mean_24h",
                "flow_se3_to_fi_recent",
                "flow_se3_to_fi_recent_mean_24h",
                "commercial_flow_fi_ee_recent",
                "commercial_flow_fi_ee_recent_mean_24h",
            ]:
                if column in row.index and pd.notna(row[column]):
                    apply_delta(column, flow_delta)
                    touched = True
            applied_inputs.append({"name": "cross_border_flow_mw", "label": SCENARIO_KNOBS["cross_border_flow_mw"], "value": flow_delta, "applied": touched, "reason": None if touched else "Cross-border flow features are unavailable for this row."})

        spread_delta = float(scenario_inputs.get("coupled_market_spread_delta", 0.0))
        if not math.isclose(spread_delta, 0.0, abs_tol=1e-12):
            touched = False
            for column in row.index:
                if column.startswith("shadow_price_") and "_spread_recent" in column and "_std_" not in column:
                    if pd.notna(row[column]):
                        apply_delta(column, spread_delta)
                        touched = True
            applied_inputs.append({"name": "coupled_market_spread_delta", "label": SCENARIO_KNOBS["coupled_market_spread_delta"], "value": spread_delta, "applied": touched, "reason": None if touched else "Coupled-market spread features are unavailable for this row."})

        changed_feature_rows = []
        for feature_name, (before, after) in changed_features.items():
            group_id = feature_group_for(feature_name)
            changed_feature_rows.append(
                {
                    "feature_name": feature_name,
                    "group_id": group_id,
                    "group_label": FEATURE_GROUP_LABELS[group_id],
                    "before": before,
                    "after": after,
                    "delta": after - before,
                }
            )
        changed_feature_rows.sort(key=lambda item: abs(item["delta"]), reverse=True)
        feature_frame.iloc[0] = row
        return applied_inputs, changed_feature_rows

    def _diff_ranked_items(
        self,
        baseline: list[dict[str, Any]],
        scenario: list[dict[str, Any]],
        key: str,
        label_key: str,
    ) -> list[dict[str, Any]]:
        baseline_map = {str(item[key]): item for item in baseline}
        scenario_map = {str(item[key]): item for item in scenario}
        rows: list[dict[str, Any]] = []
        for item_key in sorted(set(baseline_map) | set(scenario_map)):
            baseline_item = baseline_map.get(item_key, {})
            scenario_item = scenario_map.get(item_key, {})
            before = float(baseline_item.get("signed_contribution", 0.0))
            after = float(scenario_item.get("signed_contribution", 0.0))
            label = scenario_item.get(label_key) or baseline_item.get(label_key) or item_key
            rows.append(
                {
                    "name": item_key,
                    "label": label,
                    "baseline": before,
                    "scenario": after,
                    "delta": after - before,
                }
            )
        rows.sort(key=lambda item: abs(item["delta"]), reverse=True)
        return rows

    def _iso(self, value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        return to_utc_timestamp(value).isoformat()

    def _safe_float(self, value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)


def feature_group_for(feature_name: str) -> str:
    if feature_name in {
        "wind_ramp_24h",
        "market_renewable_penetration",
    }:
        return "renewables"
    if feature_name.startswith(("wind_", "temp_", "irradiance_", "pressure_", "humidity_")):
        return "weather"
    if feature_name in {
        "holiday",
        "weekday",
        "month",
        "day_of_year",
        "hour_of_day",
        "hour_of_week",
        "sunelevation",
        "azimuth",
        "sr_influence",
        "ss_influence",
        "morningpeak",
        "eveningpeak",
        "cold_morning_peak",
        "cold_evening_peak",
    } or feature_name.startswith("day_"):
        return "time_calendar"
    if feature_name in {
        "load",
        "market_load_forecast",
        "market_residual_load",
        "load_deviation_norm",
        "load_ramp_24h",
        "market_residual_load_ramp_3h",
        "market_thermal_burden",
        "boiler_consumption_recent",
        "boiler_consumption_recent_mean_24h",
        "boiler_consumption_recent_std_24h",
    }:
        return "demand_load"
    if feature_name in {
        "market_wind_forecast",
        "market_solar_forecast",
        "market_generation_forecast",
        "market_dispatchable_gap",
    }:
        return "renewables"
    if feature_name.startswith(("flow_", "commercial_flow_", "available_import_", "available_export_")):
        return "cross_border"
    if feature_name.startswith("imbalance_"):
        return "imbalance_state"
    if feature_name.startswith("shadow_price_"):
        return "coupled_market"
    if feature_name.startswith(("price_lag_", "price_roll_", "own_price_")):
        return "persistence"
    if feature_name.startswith("gas"):
        return "gas_other"
    return "gas_other"
