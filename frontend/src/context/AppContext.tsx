import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { api } from '../services/api';
import type { HealthResponse, PredictionResult, RecentPrediction, TripSummary } from '../types/api';

const RECENTS_KEY = 'pathsnap.recent-predictions.v1';

function readRecents(): RecentPrediction[] {
  try {
    const saved: unknown = JSON.parse(sessionStorage.getItem(RECENTS_KEY) ?? '[]');
    return Array.isArray(saved) ? saved.filter((row): row is RecentPrediction =>
      typeof row?.trip_id === 'string' && typeof row?.created_at === 'string'
      && Number.isFinite(row?.matched_observations) && Number.isFinite(row?.total_observations)
      && Number.isFinite(row?.processing_time_seconds)).slice(0, 8) : [];
  } catch {
    return [];
  }
}

interface AppContextValue {
  health: HealthResponse | null;
  trips: TripSummary[];
  healthError: string | null;
  tripsError: string | null;
  refreshing: boolean;
  selectedTripId: string;
  predictions: Record<string, PredictionResult>;
  recentPredictions: RecentPrediction[];
  setSelectedTripId: (tripId: string) => void;
  refresh: () => Promise<void>;
  savePrediction: (prediction: PredictionResult) => void;
}

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [trips, setTrips] = useState<TripSummary[]>([]);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [tripsError, setTripsError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedTripId, setSelectedTripId] = useState('');
  const [predictions, setPredictions] = useState<Record<string, PredictionResult>>({});
  const [recentPredictions, setRecentPredictions] = useState<RecentPrediction[]>(readRecents);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    const [healthResult, tripsResult] = await Promise.allSettled([api.health(), api.trips()]);
    if (healthResult.status === 'fulfilled') {
      setHealth(healthResult.value);
      setHealthError(null);
    } else {
      setHealthError(healthResult.reason instanceof Error ? healthResult.reason.message : 'Health check failed.');
    }
    if (tripsResult.status === 'fulfilled') {
      setTrips(tripsResult.value);
      setTripsError(null);
      setSelectedTripId((current) => current || tripsResult.value[0]?.trip_id || '');
    } else {
      setTripsError(tripsResult.reason instanceof Error ? tripsResult.reason.message : 'Trip list could not be loaded.');
    }
    setRefreshing(false);
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => {
      api.health().then(setHealth).catch((error: unknown) => {
        setHealthError(error instanceof Error ? error.message : 'Health check failed.');
      });
    }, 30_000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const savePrediction = useCallback((result: PredictionResult) => {
    setPredictions((current) => ({ ...current, [result.trip_id]: result }));
    const activity: RecentPrediction = {
      trip_id: result.trip_id,
      created_at: new Date().toISOString(),
      processing_time_seconds: result.processing_time_seconds,
      matched_observations: result.prediction.matched_observations,
      total_observations: result.prediction.total_observations,
    };
    setRecentPredictions((current) => {
      const next = [activity, ...current.filter((row) => row.trip_id !== activity.trip_id)].slice(0, 8);
      sessionStorage.setItem(RECENTS_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const value = useMemo(() => ({
    health, trips, healthError, tripsError, refreshing, selectedTripId, predictions, recentPredictions,
    setSelectedTripId, refresh, savePrediction,
  }), [health, trips, healthError, tripsError, refreshing, selectedTripId, predictions, recentPredictions, refresh, savePrediction]);

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) throw new Error('useApp must be used inside AppProvider.');
  return context;
}
