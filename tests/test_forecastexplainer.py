"""Tests for predictor.model.forecastexplainer module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from predictor.model.forecastartifactstore import ForecastArtifactStore
from predictor.model.forecastexplainer import (
    FEATURE_GROUP_LABELS,
    ForecastExplainer,
    feature_group_for,
)
from predictor.model.priceregion import PriceRegionName


def build_training_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = pd.date_range("2026-03-01T00:00:00Z", periods=160, freq="15min", tz="UTC")
    frame = pd.DataFrame(index=index)
    frame["wind_0"] = rng.normal(12.0, 1.8, len(index))
    frame["temp_0"] = rng.normal(3.0, 5.0, len(index))
    frame["irradiance_0"] = rng.uniform(0.0, 300.0, len(index))
    frame["pressure_0"] = rng.normal(1015.0, 4.0, len(index))
    frame["humidity_0"] = rng.uniform(45.0, 90.0, len(index))
    frame["holiday"] = 0.0
    frame["day_0"] = (index.dayofweek == 0).astype(float)
    frame["sunelevation"] = rng.normal(15.0, 8.0, len(index))
    frame["azimuth"] = np.linspace(0.0, 360.0, len(index))
    frame["sr_influence"] = rng.uniform(0.0, 900.0, len(index))
    frame["ss_influence"] = rng.uniform(0.0, 900.0, len(index))
    frame["morningpeak"] = ((index.hour >= 6) & (index.hour < 9)).astype(float) * 3600.0
    frame["eveningpeak"] = ((index.hour >= 16) & (index.hour < 20)).astype(float) * 3600.0
    frame["load"] = rng.normal(9600.0, 250.0, len(index))
    frame["market_load_forecast"] = frame["load"] + rng.normal(0.0, 80.0, len(index))
    frame["market_wind_forecast"] = rng.normal(2200.0, 300.0, len(index))
    frame["market_solar_forecast"] = rng.normal(650.0, 120.0, len(index))
    frame["market_generation_forecast"] = rng.normal(9900.0, 250.0, len(index))
    frame["market_residual_load"] = frame["market_load_forecast"] - frame["market_wind_forecast"] - frame["market_solar_forecast"]
    frame["market_dispatchable_gap"] = frame["market_generation_forecast"] - frame["market_wind_forecast"] - frame["market_solar_forecast"]
    frame["available_import_headroom"] = rng.normal(900.0, 60.0, len(index))
    frame["available_export_headroom"] = rng.normal(850.0, 60.0, len(index))
    frame["flow_se3_to_fi_recent"] = rng.normal(420.0, 35.0, len(index))
    frame["flow_se3_to_fi_recent_mean_24h"] = frame["flow_se3_to_fi_recent"] + rng.normal(0.0, 10.0, len(index))
    frame["flow_ee_to_fi_recent"] = rng.normal(520.0, 30.0, len(index))
    frame["flow_ee_to_fi_recent_mean_24h"] = frame["flow_ee_to_fi_recent"] + rng.normal(0.0, 10.0, len(index))
    frame["commercial_flow_fi_ee_recent"] = rng.normal(150.0, 18.0, len(index))
    frame["commercial_flow_fi_ee_recent_mean_24h"] = frame["commercial_flow_fi_ee_recent"] + rng.normal(0.0, 8.0, len(index))
    frame["imbalance_long_recent"] = rng.normal(4.0, 1.2, len(index))
    frame["imbalance_short_recent"] = rng.normal(-3.0, 1.1, len(index))
    frame["shadow_price_se_1_lag_1d"] = rng.normal(7.5, 0.9, len(index))
    frame["shadow_price_se_1_spread_recent"] = rng.normal(1.8, 0.6, len(index))
    frame["shadow_price_se_1_spread_recent_mean_24h"] = frame["shadow_price_se_1_spread_recent"] + rng.normal(0.0, 0.2, len(index))
    frame["price_lag_1d"] = rng.normal(5.0, 0.7, len(index))
    frame["price_roll_mean_24h"] = frame["price_lag_1d"] + rng.normal(0.0, 0.3, len(index))
    frame["price_roll_std_24h"] = rng.uniform(0.2, 1.4, len(index))
    frame["gasprice"] = rng.normal(34.0, 1.0, len(index))
    return frame


def train_bundle(temp_storage_dir: str) -> tuple[ForecastExplainer, pd.DataFrame]:
    region = PriceRegionName.FI.to_region()
    artifact_store = ForecastArtifactStore(region, temp_storage_dir)
    explainer = ForecastExplainer(region, artifact_store)
    features = build_training_frame()

    target = (
        features["market_residual_load"] * 0.0009
        - features["available_import_headroom"] * 0.0013
        + features["shadow_price_se_1_spread_recent"] * 0.6
        + features["imbalance_long_recent"] * 0.25
        + features["price_lag_1d"] * 0.8
        + features["gasprice"] * 0.04
        - features["temp_0"] * 0.05
    )
    params = {
        "force_col_wise": True,
        "verbosity": -1,
        "seed": 42,
        "min_data_in_leaf": 1,
        "num_leaves": 15,
    }
    point = lgb.train(
        params={**params, "objective": "regression"},
        train_set=lgb.Dataset(features, label=target),
    )
    q50 = lgb.train(
        params={**params, "objective": "quantile", "alpha": 0.5},
        train_set=lgb.Dataset(features, label=target),
    )
    spike_target = (target >= target.quantile(0.75)).astype(int)
    spike_classifier = lgb.train(
        params={**params, "objective": "binary"},
        train_set=lgb.Dataset(features, label=spike_target),
    )
    base_prediction = pd.Series(point.predict(features), index=features.index)
    residual = target - base_prediction
    spike_rows = spike_target.astype(bool)
    spike_uplift = lgb.train(
        params={**params, "objective": "regression"},
        train_set=lgb.Dataset(features.loc[spike_rows], label=residual.loc[spike_rows]),
    )

    version = "20260411T080000Z"
    artifact_store.save_model_bundle(
        version,
        {
            "point": point,
            "q50": q50,
            "spike_classifier": spike_classifier,
            "spike_uplift": spike_uplift,
        },
    )

    snapshot_rows = features.iloc[:6].copy()
    point_pred = pd.Series(point.predict(snapshot_rows), index=snapshot_rows.index)
    spike_probability = pd.Series(spike_classifier.predict(snapshot_rows), index=snapshot_rows.index)
    uplift_prediction = pd.Series(spike_uplift.predict(snapshot_rows), index=snapshot_rows.index)
    q50_prediction = pd.Series(q50.predict(snapshot_rows), index=snapshot_rows.index)
    final_prediction = point_pred + uplift_prediction * spike_probability
    final_prediction = final_prediction * 0.85 + q50_prediction * 0.15
    blend_weight = region.yesterday_blend_weight * (1.0 - spike_probability.clip(0.0, 1.0))
    final_prediction = final_prediction * (1.0 - blend_weight) + snapshot_rows["price_lag_1d"] * blend_weight

    merged = snapshot_rows.reset_index().rename(columns={"index": "target_time_utc"})
    merged["generated_at_utc"] = datetime(2026, 4, 11, 8, 0, tzinfo=timezone.utc)
    merged["region"] = region.bidding_zone_entsoe
    merged["model_version"] = version
    merged["predicted_price"] = final_prediction.to_numpy()
    merged["actual_price"] = (final_prediction + pd.Series([0.3, -0.2, 0.8, -0.4, 0.1, 0.6], index=snapshot_rows.index)).to_numpy()
    return explainer, merged


class TestForecastExplainer:
    def test_explain_row_reconciles_saved_prediction(self, temp_storage_dir):
        explainer, merged = train_bundle(temp_storage_dir)

        explanation = explainer.explain_row(merged.iloc[0])

        assert explanation["explainable"] is True
        assert explanation["feature_contributions"]
        assert explanation["group_contributions"]
        assert explanation["adjustments"]
        assert explanation["reconciliation_error"] == pytest.approx(0.0, abs=1e-6)

    def test_explain_row_uses_yesterday_known_price_for_blend_reconciliation(self, temp_storage_dir):
        explainer, merged = train_bundle(temp_storage_dir)

        row = merged.iloc[0].copy()
        row["yesterday_known_price"] = float(row["price_lag_1d"]) + 3.0

        explanation = explainer.explain_row(row)
        bundle = explainer._load_bundle(str(row["model_version"]))
        features = explainer._prepare_feature_matrix(pd.DataFrame([row.to_dict()]), bundle.feature_names)
        spike_probability = float(bundle.spike_classifier.predict(features)[0])
        expected_blend = row["yesterday_known_price"] * explainer.region.yesterday_blend_weight * (1.0 - spike_probability)
        blend_item = next(
            item for item in explanation["adjustments"]
            if item["adjustment_name"] == "yesterday_blend_component"
        )

        assert explanation["explainable"] is True
        assert blend_item["signed_contribution"] == pytest.approx(expected_blend, abs=1e-6)

    def test_grouped_contributions_match_feature_total(self, temp_storage_dir):
        explainer, merged = train_bundle(temp_storage_dir)

        explanation = explainer.explain_row(merged.iloc[1])
        feature_total = sum(item["signed_contribution"] for item in explanation["feature_contributions"])
        group_total = sum(item["signed_contribution"] for item in explanation["group_contributions"])

        assert group_total == pytest.approx(feature_total, abs=1e-6)

    def test_summarize_rows_returns_group_feature_and_error_slices(self, temp_storage_dir):
        explainer, merged = train_bundle(temp_storage_dir)

        summary = explainer.summarize_rows(merged)

        assert summary["rows_evaluated"] == 6
        assert summary["group_summary"]
        assert summary["feature_summary"]
        assert summary["adjustment_summary"]
        assert summary["error_slices"]

    def test_summarize_rows_handles_missing_actual_price_column(self, temp_storage_dir):
        explainer, merged = train_bundle(temp_storage_dir)

        summary = explainer.summarize_rows(merged.drop(columns=["actual_price"]))

        assert summary["rows_evaluated"] == 0
        assert summary["explainable_rows"] == 0
        assert summary["group_summary"] == []

    def test_merge_snapshot_frames_empty_predictions_still_exposes_summary_columns(self, temp_storage_dir):
        explainer, merged = train_bundle(temp_storage_dir)

        result = explainer.merge_snapshot_frames(
            merged.iloc[0:0][["generated_at_utc", "target_time_utc", "region", "predicted_price", "model_version"]],
            merged.iloc[0:0],
            pd.DataFrame({"price": []}, index=pd.DatetimeIndex([], tz="UTC")),
        )

        assert result.empty
        assert "actual_price" in result.columns
        assert "yesterday_known_price" in result.columns
        assert "explainable" in result.columns
        assert "explainable_reason" in result.columns

    def test_scenario_only_changes_supported_features(self, temp_storage_dir):
        explainer, merged = train_bundle(temp_storage_dir)

        result = explainer.evaluate_scenario(
            merged.iloc[0],
            {
                "wind_forecast_pct": 10.0,
                "cross_border_flow_mw": -50.0,
                "coupled_market_spread_delta": 1.2,
            },
        )

        changed = {item["feature_name"] for item in result["changed_features"]}
        assert result["explainable"] is True
        assert result["prediction_delta"] is not None
        assert changed
        assert "market_wind_forecast" in changed
        assert "temp_0" not in changed

    def test_all_model_features_map_to_known_groups(self, temp_storage_dir):
        explainer, merged = train_bundle(temp_storage_dir)
        explanation = explainer.explain_row(merged.iloc[0])

        for item in explanation["feature_contributions"]:
            assert feature_group_for(item["feature_name"]) in FEATURE_GROUP_LABELS

    def test_feature_group_for_current_fi_feature_families(self):
        expected_groups = {
            "weekday": "time_calendar",
            "month": "time_calendar",
            "day_of_year": "time_calendar",
            "hour_of_day": "time_calendar",
            "hour_of_week": "time_calendar",
            "cold_morning_peak": "time_calendar",
            "cold_evening_peak": "time_calendar",
            "own_price_lag_2d": "persistence",
            "own_price_lag_7d": "persistence",
            "own_price_rolling_mean_24h": "persistence",
            "own_price_rolling_max_24h": "persistence",
            "own_price_rolling_min_24h": "persistence",
            "own_price_rolling_mean_72h": "persistence",
            "load_deviation_norm": "demand_load",
            "load_ramp_24h": "demand_load",
            "market_residual_load_ramp_3h": "demand_load",
            "market_thermal_burden": "demand_load",
            "wind_ramp_24h": "renewables",
            "market_renewable_penetration": "renewables",
            "gasprice": "gas_other",
        }

        for feature_name, expected_group in expected_groups.items():
            assert feature_group_for(feature_name) == expected_group
