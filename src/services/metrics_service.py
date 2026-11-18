"""Read and validate the experiment metric artifacts for API responses."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.api.schemas import (
    MetricsOverviewResponse,
    ViterbiMetricData,
    ViterbiMetricsResponse,
    XGBoostMetricData,
    XGBoostMetricsResponse,
    XGBoostMetricRun,
)
from src.config import VITERBI_METRICS_PATH, XGBOOST_METRICS_PATH

log = logging.getLogger("pathsnap.metrics")


def _read_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Metric artifact root must be a JSON object")
    return data


def _numeric_mapping(value: Any, label: str) -> dict[str, int | float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty object")
    result: dict[str, int | float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label} must contain only numeric values")
        if not math.isfinite(item):
            raise ValueError(f"{label} must contain finite values")
        result[key] = item
    return result


def _classifier_run(value: Any, label: str) -> XGBoostMetricRun:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    parameters = _numeric_mapping(value.get("parameters"), f"{label}.parameters")
    metrics = _numeric_mapping(
        {key: item for key, item in value.items() if key != "parameters"},
        f"{label} metrics",
    )
    return XGBoostMetricRun(parameters=parameters, metrics=metrics)


def _unavailable_xgboost(reason: str) -> XGBoostMetricsResponse:
    return XGBoostMetricsResponse(available=False, metrics=None, reason=reason)


def _unavailable_viterbi(reason: str) -> ViterbiMetricsResponse:
    return ViterbiMetricsResponse(available=False, metrics=None, reason=reason)


class MetricsService:
    def __init__(self, xgboost_path: Path = XGBOOST_METRICS_PATH, viterbi_path: Path = VITERBI_METRICS_PATH):
        self.xgboost_path = Path(xgboost_path)
        self.viterbi_path = Path(viterbi_path)

    @staticmethod
    def _reason(label: str, error: Exception) -> str:
        if isinstance(error, FileNotFoundError):
            return f"{label} metrics file is missing."
        if isinstance(error, json.JSONDecodeError):
            return f"{label} metrics file contains invalid JSON."
        if isinstance(error, OSError):
            return f"{label} metrics file could not be read."
        return f"{label} metrics file does not match the expected schema."

    def xgboost(self) -> XGBoostMetricsResponse:
        try:
            raw = _read_object(self.xgboost_path)
            metrics = XGBoostMetricData(
                selected_model=raw["selected_model"],
                algorithm_family=raw["algorithm_family"],
                rationale=raw["rationale"],
                baseline_metrics=_classifier_run(raw["baseline_metrics"], "baseline_metrics"),
                bayesian_tuned_metrics=_classifier_run(raw["bayesian_tuned_metrics"], "bayesian_tuned_metrics"),
            )
            return XGBoostMetricsResponse(available=True, metrics=metrics)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, ValidationError) as exc:
            log.warning("XGBoost metrics unavailable (%s)", type(exc).__name__)
            return _unavailable_xgboost(self._reason("XGBoost", exc))

    def viterbi(self) -> ViterbiMetricsResponse:
        try:
            raw = _read_object(self.viterbi_path)
            priors = _numeric_mapping(raw.get("transition_log_priors"), "transition_log_priors")
            numeric_metrics = {
                key: value for key, value in raw.items() if key != "transition_log_priors"
            }
            for key, value in numeric_metrics.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{key} must be a finite numeric metric")
            metrics = ViterbiMetricData(**{**numeric_metrics, "transition_log_priors": priors})
            return ViterbiMetricsResponse(available=True, metrics=metrics)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, ValidationError) as exc:
            log.warning("Viterbi metrics unavailable (%s)", type(exc).__name__)
            return _unavailable_viterbi(self._reason("Viterbi", exc))

    def overview(self) -> MetricsOverviewResponse:
        return MetricsOverviewResponse(xgboost=self.xgboost(), viterbi=self.viterbi())


metrics_service = MetricsService()
