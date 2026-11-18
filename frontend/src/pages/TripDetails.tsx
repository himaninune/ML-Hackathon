import { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { ArrowLeft, ArrowRight, Clock3, MapPinned, Route, Timer } from 'lucide-react';
import { MapPanel } from '../components/map/MapPanel';
import { PageHeader } from '../components/ui/PageHeader';
import { useApp } from '../context/AppContext';
import { api, apiErrorMessage } from '../services/api';
import type { TripDetailsResponse } from '../types/api';

function formatTimestamp(value: number) {
  const millis = Math.abs(value) < 100_000_000_000 ? value * 1000 : value;
  const date = new Date(millis);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

export function TripDetails() {
  const { tripId = '' } = useParams();
  const navigate = useNavigate();
  const { trips, setSelectedTripId } = useApp();
  const [details, setDetails] = useState<TripDetailsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    setSelectedTripId(tripId);
    setLoading(true);
    setError(null);
    api.trip(tripId).then((result) => { if (active) setDetails(result); })
      .catch((reason: unknown) => { if (active) setError(apiErrorMessage(reason)); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [tripId, setSelectedTripId]);

  const coordinates = details?.observations.map((point) => [point.latitude, point.longitude] as [number, number]) ?? [];
  const timestamps = details?.observations.map((point) => point.timestamp) ?? [];
  const knownTrip = trips.find((trip) => trip.trip_id === tripId);

  return <div className="page-stack">
    <PageHeader eyebrow="TRIP DETAIL" title={details ? `Trip ${tripId}` : 'Trajectory detail'} description="Inspect the original GPS observations and timing metadata. Reference routes are not exposed." actions={<button className="button button-secondary" onClick={() => navigate('/trips')}><ArrowLeft size={15} /> Back to trips</button>} />
    {loading ? <div className="panel loading-panel"><div className="loading-orbit" /><span>Loading trip observations…</span></div>
      : error ? <div className="panel inline-error"><strong>Trip details could not be loaded</strong><span>{error}</span><Link to="/trips" className="button button-secondary">Return to trip list</Link></div>
        : details && <>
          <section className="detail-metrics">
            <div className="detail-stat panel"><div className="metric-icon tone-blue"><MapPinned size={18} /></div><span>GPS observations</span><strong>{details.observations.length.toLocaleString()}</strong><small>{knownTrip ? 'Count reported by backend trip list' : 'Count in returned observations'}</small></div>
            <div className="detail-stat panel"><div className="metric-icon tone-violet"><Timer size={18} /></div><span>Trip duration</span><strong>{details.duration_seconds < 60 ? `${details.duration_seconds.toFixed(0)}s` : `${Math.floor(details.duration_seconds / 60)}m ${Math.round(details.duration_seconds % 60)}s`}</strong><small>Elapsed timestamp range</small></div>
            <div className="detail-stat panel"><div className="metric-icon tone-green"><Clock3 size={18} /></div><span>First observation</span><strong className="timestamp-value">{timestamps.length ? formatTimestamp(timestamps[0]) : '—'}</strong><small>Timestamp from trip detail response</small></div>
            <div className="detail-stat panel"><div className="metric-icon tone-amber"><Route size={18} /></div><span>Last observation</span><strong className="timestamp-value">{timestamps.length ? formatTimestamp(timestamps[timestamps.length - 1]) : '—'}</strong><small>Timestamp from trip detail response</small></div>
          </section>
          <MapPanel gpsCoordinates={coordinates} tripId={tripId} compact />
          <div className="detail-actions"><span><MapPinned size={15} /> Preview contains only original GPS data.</span><Link to={`/predict/${tripId}`} className="button button-primary">Reconstruct this route <ArrowRight size={15} /></Link></div>
        </>}
  </div>;
}
