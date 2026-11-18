from __future__ import annotations

from pydantic import BaseModel, Field

MetricNumber = int | float


class XGBoostMetricRun(BaseModel):
    parameters: dict[str, MetricNumber] = Field(description="Parameters recorded for this classifier run.")
    metrics: dict[str, MetricNumber] = Field(description="Candidate-classification metrics from the source artifact.")


class XGBoostMetricData(BaseModel):
    selected_model: str
    algorithm_family: str
    rationale: str
    baseline_metrics: XGBoostMetricRun
    bayesian_tuned_metrics: XGBoostMetricRun


class XGBoostMetricsResponse(BaseModel):
    available: bool
    evaluation_stage: str = "candidate_classification"
    evaluation_split: str = "5-fold grouped cross-validation and student_test"
    reference_note: str = "Candidate labels use route-membership heuristics; these scores are not route-reconstruction accuracy."
    metrics: XGBoostMetricData | None = None
    reason: str | None = None


class ViterbiMetricData(BaseModel):
    pointwise_xgb_top1_accuracy: MetricNumber = Field(description="Per-observation candidate selection accuracy before sequence decoding.")
    viterbi_sequence_top1_accuracy: MetricNumber = Field(description="Per-observation accuracy after Viterbi sequence decoding.")
    accuracy_gain: MetricNumber = Field(description="Difference between Viterbi and pointwise top-1 accuracy.")
    pointwise_route_continuity: MetricNumber = Field(description="Mean fraction of valid road transitions before Viterbi decoding.")
    viterbi_route_continuity: MetricNumber = Field(description="Mean fraction of valid road transitions after Viterbi decoding.")
    route_continuity_gain: MetricNumber = Field(description="Difference between decoded and pointwise route continuity.")
    viterbi_decoding_time_seconds: MetricNumber = Field(description="Decoding time recorded for the experiment's ten student-test trips.")
    transition_log_priors: dict[str, MetricNumber] = Field(description="Configured log penalties used by the experiment decoder.")


class ViterbiMetricsResponse(BaseModel):
    available: bool
    evaluation_stage: str = "route_reconstruction"
    evaluation_split: str = "student_test"
    reference_note: str = "The experiment uses heuristic route-membership labels, not official ground truth."
    metrics: ViterbiMetricData | None = None
    reason: str | None = None


class MetricsOverviewResponse(BaseModel):
    xgboost: XGBoostMetricsResponse
    viterbi: ViterbiMetricsResponse
