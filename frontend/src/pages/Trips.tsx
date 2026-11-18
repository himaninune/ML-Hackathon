import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowDownUp, ArrowRight, Clock3, MapPinned, Search, Waypoints } from 'lucide-react';
import { PageHeader } from '../components/ui/PageHeader';
import { StatusPill } from '../components/ui/StatusPill';
import { useApp } from '../context/AppContext';

type SortKey = 'trip_id' | 'observations' | 'duration_seconds';

function formatDuration(seconds: number | null) {
  if (seconds === null) return '—';
  const minutes = Math.floor(seconds / 60);
  return minutes > 0 ? `${minutes}m ${Math.round(seconds % 60)}s` : `${Math.round(seconds)}s`;
}

export function Trips() {
  const { trips, tripsError, refreshing, refresh, setSelectedTripId } = useApp();
  const [query, setQuery] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('trip_id');
  const [descending, setDescending] = useState(false);

  const rows = useMemo(() => trips.filter((trip) => trip.trip_id.toLowerCase().includes(query.toLowerCase()))
    .sort((a, b) => {
      const left = a[sortKey] ?? -1;
      const right = b[sortKey] ?? -1;
      const compared = typeof left === 'string' ? left.localeCompare(String(right)) : Number(left) - Number(right);
      return descending ? -compared : compared;
    }), [trips, query, sortKey, descending]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setDescending((value) => !value);
    else { setSortKey(key); setDescending(false); }
  };

  return <div className="page-stack">
    <PageHeader eyebrow="TRIP CATALOG" title="Available trajectories" description="Browse the trip set exposed by the connected backend. The list is loaded dynamically and can support additional partitions later." actions={<button className="button button-secondary" onClick={() => void refresh()} disabled={refreshing}><ArrowDownUp size={15} className={refreshing ? 'spin' : ''} /> Refresh list</button>} />
    <div className="catalog-summary">
      <div className="catalog-summary-icon"><Waypoints size={18} /></div><div><strong>{trips.length}</strong><span>backend trips</span></div><i />
      <div className="catalog-summary-icon catalog-summary-blue"><MapPinned size={18} /></div><div><strong>{trips.reduce((total, trip) => total + trip.observations, 0).toLocaleString()}</strong><span>GPS observations</span></div>
      <StatusPill label="Live API data" tone="good" />
    </div>

    <section className="panel trip-catalog">
      <div className="catalog-toolbar"><div><span className="eyebrow">TRIP INVENTORY</span><h2>Choose a trajectory</h2></div>
        <label className="search-field"><Search size={15} /><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search trip ID" aria-label="Search trips by ID" /></label>
      </div>
      {tripsError ? <div className="inline-error"><strong>Trip list unavailable</strong><span>{tripsError}</span><button className="button button-secondary" onClick={() => void refresh()}>Try again</button></div>
        : rows.length === 0 ? <div className="catalog-empty"><div className="empty-icon"><Search size={20} /></div><strong>{trips.length ? 'No trips match this search' : 'No trips returned by the backend'}</strong><span>{trips.length ? 'Try another trip ID.' : 'Check the backend connection, then refresh.'}</span></div>
          : <div className="table-scroll"><table className="data-table"><thead><tr>
            <th><button onClick={() => toggleSort('trip_id')}>TRIP ID <ArrowDownUp size={12} /></button></th>
            <th><button onClick={() => toggleSort('observations')}>GPS OBSERVATIONS <ArrowDownUp size={12} /></button></th>
            <th><button onClick={() => toggleSort('duration_seconds')}>DURATION <ArrowDownUp size={12} /></button></th>
            <th>DATA SOURCE</th><th /></tr></thead><tbody>
            {rows.map((trip) => <tr key={trip.trip_id}>
              <td><span className="trip-id-cell"><span className="trip-row-dot" />{trip.trip_id}</span></td>
              <td><span className="observations-cell">{trip.observations.toLocaleString()} <small>points</small></span></td>
              <td><span className="duration-cell"><Clock3 size={13} /> {formatDuration(trip.duration_seconds)}</span></td>
              <td><span className="data-source-label">Backend trip set</span></td>
              <td><Link to={`/trips/${trip.trip_id}`} onClick={() => setSelectedTripId(trip.trip_id)} className="button button-small">View trip <ArrowRight size={14} /></Link></td>
            </tr>)}
          </tbody></table></div>}
      <div className="table-foot"><span>Showing {rows.length} of {trips.length} trips</span><span>Reference routes are not included in this catalog.</span></div>
    </section>
  </div>;
}
