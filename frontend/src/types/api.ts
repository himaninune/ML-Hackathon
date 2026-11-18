export type LatLon = [latitude: number, longitude: number];

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_error: string | null;
  road_network_loaded: boolean;
  backend_version: string;
}

export interface TripSummary {
  trip_id: string;
  observations: number;
  duration_seconds: number | null;
}

export interface GPSObservation {
  longitude: number;
  latitude: number;
  timestamp: number;
}

export interface TripDetailsResponse {
  trip_id: string;
  observations: GPSObservation[];
  duration_seconds: number;
}

export interface TripMapResponse {
  trip_id: string;
  gps_coordinates: LatLon[];
}

export interface MatchedArc {
  observation_index: number;
  arc_id: number;
  probability: number;
  coordinates: LatLon[];
}

export interface EvaluationResponse {
  available: boolean;
  reference_type?: string;
  metrics: Record<string, number>;
  reason?: string;
}

export interface PredictionResult {
  trip_id: string;
  status: string;
  processing_time_seconds: number;
  prediction: {
    matched_observations: number;
    total_observations: number;
    predicted_route: number[];
    matched_arcs: MatchedArc[];
    predicted_coordinates: LatLon[];
    predicted_route_segments?: LatLon[][];
    gps_coordinates: LatLon[];
    unmatched_observation_indices: number[];
  };
  evaluation: EvaluationResponse;
}

export interface RecentPrediction {
  trip_id: string;
  created_at: string;
  processing_time_seconds: number;
  matched_observations: number;
  total_observations: number;
}
