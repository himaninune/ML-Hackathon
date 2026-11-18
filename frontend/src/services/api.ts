import axios, { type AxiosError } from 'axios';
import type {
  EvaluationResponse,
  GPSObservation,
  HealthResponse,
  LatLon,
  MatchedArc,
  MetricsOverviewResponse,
  PredictionResult,
  TripDetailsResponse,
  TripMapResponse,
  TripSummary,
  ViterbiMetricsResponse,
  XGBoostMetricsResponse,
} from '../types/api';

const baseURL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/+$/, '');
const client = axios.create({ baseURL, timeout: 30_000, headers: { Accept: 'application/json' } });

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isCoordinate(value: unknown): value is LatLon {
  return Array.isArray(value) && value.length >= 2 && Number.isFinite(value[0]) && Number.isFinite(value[1]);
}

function isNumberMap(value: unknown): value is Record<string, number> {
  return record(value) && Object.values(value).every((item) => typeof item === 'number' && Number.isFinite(item));
}

function isMetricRun(value: unknown): boolean {
  return record(value) && isNumberMap(value.parameters) && isNumberMap(value.metrics);
}

function isXGBoostMetrics(value: unknown): value is XGBoostMetricsResponse {
  if (!record(value) || typeof value.available !== 'boolean' || typeof value.evaluation_stage !== 'string'
      || typeof value.evaluation_split !== 'string' || typeof value.reference_note !== 'string') return false;
  if (!value.available) return value.metrics === null && (value.reason === null || typeof value.reason === 'string');
  if (!record(value.metrics)) return false;
  const data = value.metrics;
  return typeof data.selected_model === 'string' && typeof data.algorithm_family === 'string'
    && typeof data.rationale === 'string' && isMetricRun(data.baseline_metrics)
    && isMetricRun(data.bayesian_tuned_metrics);
}

function isViterbiMetrics(value: unknown): value is ViterbiMetricsResponse {
  if (!record(value) || typeof value.available !== 'boolean' || typeof value.evaluation_stage !== 'string'
      || typeof value.evaluation_split !== 'string' || typeof value.reference_note !== 'string') return false;
  if (!value.available) return value.metrics === null && (value.reason === null || typeof value.reason === 'string');
  if (!record(value.metrics) || !isNumberMap(value.metrics.transition_log_priors)) return false;
  return Object.entries(value.metrics).every(([key, item]) => key === 'transition_log_priors'
    || (typeof item === 'number' && Number.isFinite(item)));
}

function isMetricsOverview(value: unknown): value is MetricsOverviewResponse {
  return record(value) && isXGBoostMetrics(value.xgboost) && isViterbiMetrics(value.viterbi);
}

function assertValid<T>(value: unknown, valid: (input: unknown) => boolean, label: string): T {
  if (!valid(value)) throw new Error(`The backend returned an invalid ${label} response.`);
  return value as T;
}

function isHealth(value: unknown): value is HealthResponse {
  return record(value) && typeof value.status === 'string' && typeof value.model_loaded === 'boolean'
    && typeof value.road_network_loaded === 'boolean' && typeof value.backend_version === 'string';
}

function isTripSummary(value: unknown): value is TripSummary {
  return record(value) && typeof value.trip_id === 'string' && Number.isFinite(value.observations)
    && (value.duration_seconds === null || Number.isFinite(value.duration_seconds));
}

function isGPSObservation(value: unknown): value is GPSObservation {
  return record(value) && Number.isFinite(value.longitude) && Number.isFinite(value.latitude)
    && Number.isFinite(value.timestamp);
}

function isTripDetails(value: unknown): value is TripDetailsResponse {
  return record(value) && typeof value.trip_id === 'string' && Array.isArray(value.observations)
    && value.observations.every(isGPSObservation) && Number.isFinite(value.duration_seconds);
}

function isMapData(value: unknown): value is TripMapResponse {
  return record(value) && typeof value.trip_id === 'string' && Array.isArray(value.gps_coordinates)
    && value.gps_coordinates.every(isCoordinate);
}

function isMatchedArc(value: unknown): value is MatchedArc {
  return record(value) && Number.isFinite(value.observation_index) && Number.isFinite(value.arc_id)
    && Number.isFinite(value.probability) && Array.isArray(value.coordinates) && value.coordinates.every(isCoordinate);
}

function isEvaluation(value: unknown): value is EvaluationResponse {
  return record(value) && typeof value.available === 'boolean' && record(value.metrics)
    && Object.values(value.metrics).every(Number.isFinite);
}

function isPrediction(value: unknown): value is PredictionResult {
  if (!record(value) || typeof value.trip_id !== 'string' || typeof value.status !== 'string'
      || !Number.isFinite(value.processing_time_seconds) || !record(value.prediction) || !isEvaluation(value.evaluation)) return false;
  const prediction = value.prediction;
  return Number.isFinite(prediction.matched_observations) && Number.isFinite(prediction.total_observations)
    && Array.isArray(prediction.predicted_route) && prediction.predicted_route.every(Number.isFinite)
    && Array.isArray(prediction.matched_arcs) && prediction.matched_arcs.every(isMatchedArc)
    && Array.isArray(prediction.predicted_coordinates) && prediction.predicted_coordinates.every(isCoordinate)
    && (prediction.predicted_route_segments === undefined || (Array.isArray(prediction.predicted_route_segments)
      && prediction.predicted_route_segments.every((segment) => Array.isArray(segment) && segment.every(isCoordinate))))
    && Array.isArray(prediction.gps_coordinates) && prediction.gps_coordinates.every(isCoordinate)
    && Array.isArray(prediction.unmatched_observation_indices) && prediction.unmatched_observation_indices.every(Number.isFinite);
}

export function apiErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<{ detail?: string }>;
    if (axiosError.code === 'ECONNABORTED' || axiosError.code === 'ETIMEDOUT') {
      return 'The request timed out. Route reconstruction can take a while; try again or select a shorter trip.';
    }
    if (axiosError.response) {
      const detail = axiosError.response.data?.detail;
      return typeof detail === 'string' ? detail : `The backend returned HTTP ${axiosError.response.status}.`;
    }
    return `Cannot reach the PathSnap backend at ${baseURL}. Check the API URL, backend process, and network connection.`;
  }
  return error instanceof Error ? error.message : 'An unexpected request error occurred.';
}

export const api = {
  async health(): Promise<HealthResponse> {
    const response = await client.get<unknown>('/api/health');
    return assertValid<HealthResponse>(response.data, isHealth, 'health');
  },
  async trips(): Promise<TripSummary[]> {
    const response = await client.get<unknown>('/api/trips');
    if (!record(response.data) || !Array.isArray(response.data.trips) || !response.data.trips.every(isTripSummary)) {
      throw new Error('The backend returned an invalid trip list response.');
    }
    return response.data.trips;
  },
  async trip(tripId: string): Promise<TripDetailsResponse> {
    const response = await client.get<unknown>(`/api/trips/${encodeURIComponent(tripId)}`);
    return assertValid<TripDetailsResponse>(response.data, isTripDetails, 'trip details');
  },
  async map(tripId: string): Promise<TripMapResponse> {
    const response = await client.get<unknown>(`/api/trips/${encodeURIComponent(tripId)}/map`);
    return assertValid<TripMapResponse>(response.data, isMapData, 'map');
  },
  async metricsXGBoost(): Promise<XGBoostMetricsResponse> {
    const response = await client.get<unknown>('/api/metrics/xgboost');
    return assertValid<XGBoostMetricsResponse>(response.data, isXGBoostMetrics, 'XGBoost metrics');
  },
  async metricsViterbi(): Promise<ViterbiMetricsResponse> {
    const response = await client.get<unknown>('/api/metrics/viterbi');
    return assertValid<ViterbiMetricsResponse>(response.data, isViterbiMetrics, 'Viterbi metrics');
  },
  async metricsOverview(): Promise<MetricsOverviewResponse> {
    const response = await client.get<unknown>('/api/metrics/overview');
    return assertValid<MetricsOverviewResponse>(response.data, isMetricsOverview, 'metrics overview');
  },
  async predict(tripId: string): Promise<PredictionResult> {
    const response = await client.post<unknown>('/api/predict', { trip_id: tripId }, { timeout: 300_000 });
    return assertValid<PredictionResult>(response.data, isPrediction, 'prediction');
  },
  baseURL,
};
