import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { AlertCircle, ArrowRight, Check, ChevronLeft, ChevronRight, CircleStop, Cpu, LoaderCircle, MapPinned, Play, Route, Timer, Waypoints } from 'lucide-react';
import { EvaluationPanel } from '../components/evaluation/EvaluationPanel';
import { MapPanel } from '../components/map/MapPanel';
import { PageHeader } from '../components/ui/PageHeader';
import { MetricCard } from '../components/ui/MetricCard';
import { StatusPill } from '../components/ui/StatusPill';
import { useApp } from '../context/AppContext';
import { api, apiErrorMessage } from '../services/api';
import type { LatLon, PredictionResult, TripMapResponse } from '../types/api';

const PAGE_SIZE = 24;

function ResultStatus({ result }: { result: PredictionResult }) {
  const { matched_observations: matched, total_observations: total } = result.prediction;
  const tone = matched === total && total > 0 ? 'good' : matched > 0 ? 'warn' : 'bad';
  const label = matched === total && total > 0 ? 'Complete match' : matched > 0 ? 'Partial match' : 'No observations matched';
  return <StatusPill tone={tone} label={label} />;
}

export function Prediction() {
  const { tripId: routeTripId } = useParams();
  const navigate = useNavigate();
  const { trips, health, selectedTripId, setSelectedTripId, predictions, savePrediction } = useApp();
  const [tripId, setTripId] = useState(routeTripId || selectedTripId);
  const [mapPreview, setMapPreview] = useState<TripMapResponse | null>(null);
  const [mapError, setMapError] = useState<string | null>(null);
  const [loadingMap, setLoadingMap] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [selectedObservation, setSelectedObservation] = useState<number | null>(null);
  const [page, setPage] = useState(0);

  useEffect(() => {
    const initial = routeTripId || selectedTripId || trips[0]?.trip_id || '';
    if (initial && initial !== tripId) setTripId(initial);
  }, [routeTripId, selectedTripId, trips, tripId]);

  useEffect(() => {
    if (tripId) setSelectedTripId(tripId);
  }, [tripId, setSelectedTripId]);

  useEffect(() => {
    if (!tripId) { setMapPreview(null); return; }
    let active = true;
    setLoadingMap(true);
    setMapError(null);
    api.map(tripId).then((data) => { if (active) setMapPreview(data); })
      .catch((error: unknown) => { if (active) { setMapPreview(null); setMapError(apiErrorMessage(error)); } })
      .finally(() => { if (active) setLoadingMap(false); });
    return () => { active = false; };
  }, [tripId]);

  useEffect(() => {
    if (!predicting || startedAt === null) return undefined;
    const timer = window.setInterval(() => setElapsed((Date.now() - startedAt) / 1000), 250);
    return () => window.clearInterval(timer);
  }, [predicting, startedAt]);

  const result = predictions[tripId];
  const prediction = result?.prediction;
  const gpsCoordinates = prediction?.gps_coordinates ?? mapPreview?.gps_coordinates ?? [];
  const routeSegments = useMemo(() => {
    if (!prediction) return [];
    if (prediction.predicted_route_segments?.length) return prediction.predicted_route_segments;
    return prediction.predicted_coordinates.length ? [prediction.predicted_coordinates] : [];
  }, [prediction]);
  const matchedByObservation = useMemo(() => new Map((prediction?.matched_arcs ?? []).map((arc) => [arc.observation_index, arc])), [prediction]);
  const unmatched = useMemo(() => new Set(prediction?.unmatched_observation_indices ?? []), [prediction]);
  const total = prediction?.total_observations ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const visibleRows = Array.from({ length: Math.min(PAGE_SIZE, Math.max(0, total - page * PAGE_SIZE)) }, (_, offset) => page * PAGE_SIZE + offset);

  const chooseTrip = (nextTripId: string) => {
    setTripId(nextTripId);
    setSelectedTripId(nextTripId);
    setSelectedObservation(null);
    setPage(0);
    setRequestError(null);
    navigate(nextTripId ? `/predict/${nextTripId}` : '/predict', { replace: true });
  };

  const runPrediction = async () => {
    if (!tripId || !health?.model_loaded) return;
    setRequestError(null);
    setPredicting(true);
    const started = Date.now();
    setStartedAt(started);
    setElapsed(0);
    setSelectedObservation(null);
    try {
      const response = await api.predict(tripId);
      savePrediction(response);
      setSelectedObservation(null);
      setPage(0);
    } catch (error) {
      setRequestError(apiErrorMessage(error));
    } finally {
      setPredicting(false);
      setElapsed((Date.now() - started) / 1000);
    }
  };

  const displayCoordinates: LatLon[] = gpsCoordinates;
  const matchedCount = prediction?.matched_observations ?? 0;
  const coverage = total > 0 ? matchedCount / total * 100 : 0;

  return <div className="page-stack">
    <PageHeader eyebrow="MAP MATCHING" title="Route reconstruction" description="Choose a GPS trip to see its trace and reconstruct the roads it followed." actions={result ? <Link className="button button-secondary" to={`/evaluation/${tripId}`}>View evaluation <ArrowRight size={15} /></Link> : undefined} />

    <section className="panel prediction-control">
      <div className="prediction-control-copy"><div className="prediction-control-icon"><Route size={19} /></div><div><span className="eyebrow">SELECTED TRIP</span><strong>Choose a trip to reconstruct</strong><small>Select a GPS trip, then run map matching to view the reconstructed route.</small></div></div>
      <div className="prediction-control-actions">
        <label className="select-wrap"><span className="sr-only">Trip ID</span><select value={tripId} onChange={(event) => chooseTrip(event.target.value)} disabled={!trips.length}>
          {!trips.length && <option value="">No trips available</option>}
          {trips.map((trip) => <option key={trip.trip_id} value={trip.trip_id}>{trip.trip_id} · {trip.observations.toLocaleString()} GPS points</option>)}
        </select></label>
        <button className="button button-primary predict-button" onClick={() => void runPrediction()} disabled={!tripId || predicting || !health?.model_loaded}>
          {predicting ? <><LoaderCircle className="spin" size={16} /> Reconstructing</> : <><Play size={15} fill="currentColor" /> Predict route</>}
        </button>
      </div>
    </section>

    {predicting && <section className="progress-panel panel"><div className="progress-spinner"><LoaderCircle size={20} className="spin" /></div><div className="progress-copy"><strong>Decoding trip {tripId}</strong><span>Candidate generation, XGBoost scoring, road transitions, and Viterbi are running on the backend.</span></div><div className="progress-time"><Timer size={14} /> {elapsed.toFixed(1)}s</div><div className="progress-track"><i /></div></section>}

    {requestError && <div className="inline-error panel"><AlertCircle size={18} /><div><strong>Prediction request failed</strong><span>{requestError}</span></div><button className="icon-button" aria-label="Dismiss error" onClick={() => setRequestError(null)}><CircleStop size={16} /></button></div>}
    {!health?.model_loaded && <div className="model-unavailable panel"><Cpu size={17} /><span>The backend reports that the XGBoost model is unavailable. Route prediction is disabled.</span></div>}

    {result && prediction ? <>
      <section className="result-banner panel">
        <div className="result-check"><Check size={17} /></div>
        <div className="result-heading"><span className="eyebrow">PREDICTION RESPONSE · TRIP {result.trip_id}</span><strong>{prediction.predicted_route.length.toLocaleString()} road arcs in reconstructed route</strong><span>{prediction.matched_arcs.length.toLocaleString()} per-observation matches returned by backend</span></div>
        <ResultStatus result={result} />
      </section>
      <section className="metric-grid prediction-metric-grid">
        <MetricCard label="Observation coverage" value={`${coverage.toFixed(1)}%`} detail={`${matchedCount.toLocaleString()} matched · ${prediction.unmatched_observation_indices.length.toLocaleString()} unmatched`} icon={Waypoints} tone={coverage === 100 ? 'green' : coverage > 0 ? 'amber' : 'violet'} />
        <MetricCard label="Total observations" value={prediction.total_observations.toLocaleString()} detail="GPS points processed" icon={MapPinned} tone="blue" />
        <MetricCard label="Predicted route" value={prediction.predicted_route.length.toLocaleString()} detail={`${routeSegments.length} separate geometry segments`} icon={Route} tone="green" />
        <MetricCard label="Processing time" value={`${result.processing_time_seconds.toFixed(2)}s`} detail="Reported by backend" icon={Timer} tone="violet" />
      </section>
      {prediction.matched_observations < prediction.total_observations && <div className="partial-match-note"><AlertCircle size={15} /><span>This prediction matched {prediction.matched_observations} of {prediction.total_observations} observations. Unmatched points are highlighted separately; a successful HTTP response does not imply full coverage.</span></div>}
      {prediction.matched_observations === 0 && <div className="partial-match-note no-match-note"><AlertCircle size={15} /><span>The backend returned a prediction response, but no observations were matched. Review the trip network and backend logs.</span></div>}

      {displayCoordinates.length > 0 ? <MapPanel gpsCoordinates={displayCoordinates} routeSegments={routeSegments} matchedArcs={prediction.matched_arcs}
        unmatchedIndices={prediction.unmatched_observation_indices} selectedObservation={selectedObservation} onSelectObservation={setSelectedObservation} tripId={tripId} />
        : <section className="panel map-data-warning"><MapPinned size={18} /><span>The response contains no GPS coordinates to plot.</span></section>}

      <section className="panel observations-panel">
        <div className="section-title"><div><span className="eyebrow">OBSERVATION-LEVEL OUTPUT</span><h2>Matched observations</h2></div><span className="table-count">{prediction.matched_arcs.length.toLocaleString()} selected arcs</span></div>
        {prediction.matched_arcs.length === 0 ? <div className="catalog-empty"><strong>No per-observation matches returned.</strong><span>The route can still contain fewer or no arcs; inspect backend status.</span></div> : <>
          <div className="table-scroll"><table className="data-table observations-table"><thead><tr><th>OBSERVATION</th><th>SELECTED ARC</th><th>MODEL SCORE</th><th>STATUS</th><th /></tr></thead><tbody>
            {visibleRows.map((index) => {
              const matched = matchedByObservation.get(index);
              const isUnmatched = unmatched.has(index) || !matched;
              return <tr key={index} className={selectedObservation === index ? 'selected-table-row' : ''} onClick={() => setSelectedObservation(index)}>
                <td>#{index.toString().padStart(4, '0')}</td><td>{matched ? `Arc ${matched.arc_id}` : '—'}</td>
                <td>{matched ? `${(matched.probability * 100).toFixed(2)}%` : '—'}</td>
                <td><StatusPill label={isUnmatched ? 'Unmatched' : 'Matched'} tone={isUnmatched ? 'bad' : 'good'} /></td>
                <td><button className="row-select-button" onClick={(event) => { event.stopPropagation(); setSelectedObservation(index); }} aria-label={`Select observation ${index}`}><MapPinned size={14} /></button></td>
              </tr>;
            })}
          </tbody></table></div>
          <div className="pagination"><span>Showing {Math.min(total, page * PAGE_SIZE + 1)}–{Math.min(total, (page + 1) * PAGE_SIZE)} of {total.toLocaleString()}</span><div><button disabled={page === 0} onClick={() => setPage((current) => Math.max(0, current - 1))} aria-label="Previous page"><ChevronLeft size={16} /></button><span>Page {page + 1} / {pages}</span><button disabled={page + 1 >= pages} onClick={() => setPage((current) => Math.min(pages - 1, current + 1))} aria-label="Next page"><ChevronRight size={16} /></button></div></div>
        </>}
      </section>
      <EvaluationPanel evaluation={result.evaluation} />
    </> : !predicting && <section className="panel prediction-empty">
      <div className="prediction-empty-art"><div className="prediction-empty-pin"><MapPinned size={22} /></div><i /><i /><i /></div>
      <span className="eyebrow">READY FOR INFERENCE</span><h2>{tripId ? `Trip ${tripId} is ready` : 'Select a trip to begin'}</h2>
      <p>{loadingMap ? 'Loading the original GPS preview…' : mapError ? mapError : mapPreview?.gps_coordinates.length ? `${mapPreview.gps_coordinates.length.toLocaleString()} original GPS coordinates loaded. Run prediction to decode the road route.` : 'Choose an available trip above to load its original GPS trajectory.'}</p>
      {mapPreview?.gps_coordinates.length ? <MapPanel gpsCoordinates={mapPreview.gps_coordinates} tripId={tripId} compact /> : null}
    </section>}
  </div>;
}
