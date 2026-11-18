import { Link } from 'react-router-dom';
import { Activity, ArrowRight, Clock3, Cpu, MapPinned, Radio, Route, Satellite, ShieldCheck } from 'lucide-react';
import { PageHeader } from '../components/ui/PageHeader';
import { MetricCard } from '../components/ui/MetricCard';
import { StatusPill } from '../components/ui/StatusPill';
import { useApp } from '../context/AppContext';

const stages = [
  { label: 'GPS input', detail: 'Noisy trajectory', icon: Satellite },
  { label: 'Candidates', detail: 'Nearby road arcs', icon: MapPinned },
  { label: 'XGBoost', detail: 'Candidate scoring', icon: Cpu },
  { label: 'HMM', detail: 'Road transitions', icon: Radio },
  { label: 'Viterbi', detail: 'Sequence decoding', icon: Activity },
  { label: 'Route', detail: 'Reconstructed path', icon: Route },
];

export function Dashboard() {
  const { health, healthError, trips, tripsError, recentPredictions } = useApp();
  const totalObservations = trips.reduce((sum, trip) => sum + trip.observations, 0);
  const apiOnline = health?.status === 'healthy';

  return <div className="page-stack">
    <PageHeader eyebrow="SYSTEM OVERVIEW" title="Map matching, in motion." description="A live workspace for inspecting GPS trips and reconstructing road-consistent routes." actions={<Link to="/predict" className="button button-primary"><Route size={16} /> Open route workspace <ArrowRight size={15} /></Link>} />

    <section className="hero-panel panel">
      <div className="hero-copy">
        <div className="hero-kicker"><span className="hero-kicker-dot" /> DISTRIBUTED INFERENCE PIPELINE</div>
        <h2>From noisy GPS to a <em>coherent road path.</em></h2>
        <p>Candidate ranking meets network-aware sequence decoding. Select a trip to run the complete PathSnap prediction pipeline.</p>
        <div className="hero-actions"><Link to="/trips" className="button button-secondary">Browse trips <ArrowRight size={15} /></Link><span className="hero-note"><ShieldCheck size={14} /> Reference routes remain server-side</span></div>
      </div>
      <div className="hero-visual" aria-hidden="true">
        <div className="hero-orbit orbit-one" /><div className="hero-orbit orbit-two" />
        <svg viewBox="0 0 300 210" className="hero-route-svg" fill="none">
          <path d="M24 165C51 154 48 112 80 118C110 125 107 71 139 82C168 92 160 135 195 123C226 113 226 53 273 45" stroke="#58A6FF" strokeWidth="2" strokeDasharray="4 7" opacity=".65" />
          <path d="M24 165L58 142L79 118L104 111L139 82L163 103L181 126L212 112L239 76L273 45" stroke="#54D6A1" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" />
          {[ [24,165], [58,142], [80,118], [104,111], [139,82], [163,103], [181,126], [212,112], [239,76], [273,45] ].map(([x, y], i) => <circle key={i} cx={x} cy={y} r={i === 9 ? 5 : 3.3} fill={i === 9 ? '#54D6A1' : '#0E1B2A'} stroke={i === 9 ? '#54D6A1' : '#83BCFF'} strokeWidth="2" />)}
        </svg>
        <div className="hero-visual-caption"><span /> GPS trace <span className="caption-route" /> decoded route</div>
      </div>
    </section>

    <section className="metric-grid">
      <MetricCard label="Backend connection" value={apiOnline ? 'Connected' : 'Unavailable'} detail={healthError || `API ${health?.backend_version ? `v${health.backend_version}` : 'health status'}`} icon={Radio} tone={apiOnline ? 'green' : 'amber'} />
      <MetricCard label="XGBoost model" value={health?.model_loaded ? 'Ready' : 'Unavailable'} detail={health?.model_loaded ? 'Candidate scoring enabled' : health?.model_error || 'Waiting for backend status'} icon={Cpu} tone={health?.model_loaded ? 'violet' : 'amber'} />
      <MetricCard label="Available trips" value={trips.length} detail={tripsError || 'Loaded dynamically from the backend'} icon={MapPinned} tone="blue" />
      <MetricCard label="GPS observations" value={totalObservations.toLocaleString()} detail="Across the trips currently available" icon={Activity} tone="green" />
    </section>

    <section className="pipeline-panel panel">
      <div className="section-title"><div><span className="eyebrow">HOW PATHSNAP WORKS</span><h2>Prediction pipeline</h2></div><span className="pipeline-tag"><span /> 6 stages</span></div>
      <div className="pipeline-flow">
        {stages.map(({ label, detail, icon: Icon }, index) => <div className="pipeline-stage" key={label}>
          <div className={`pipeline-stage-icon pipeline-stage-${index}`}><Icon size={17} strokeWidth={1.8} /></div>
          <div><strong>{label}</strong><span>{detail}</span></div>
          {index < stages.length - 1 && <ArrowRight className="pipeline-arrow" size={15} />}
        </div>)}
      </div>
      <div className="pipeline-footnote"><span>Classifier scores inform emissions; Viterbi selects the route sequence using network transitions.</span><span>REFERENCE ROUTES ARE NOT SENT TO THE MODEL</span></div>
    </section>

    <section className="panel activity-panel">
      <div className="section-title"><div><span className="eyebrow">THIS BROWSER SESSION</span><h2>Recent prediction activity</h2></div><Link className="text-link" to="/predict">Open workspace <ArrowRight size={14} /></Link></div>
      {recentPredictions.length === 0 ? <div className="activity-empty"><div className="activity-empty-icon"><Clock3 size={18} /></div><div><strong>No predictions in this session yet</strong><span>Completed API predictions will appear here. Nothing is prefilled.</span></div></div> : <div className="activity-list">
        {recentPredictions.map((activity) => {
          const coverage = activity.total_observations ? Math.round(activity.matched_observations / activity.total_observations * 100) : 0;
          return <Link key={`${activity.trip_id}-${activity.created_at}`} to={`/predict/${activity.trip_id}`} className="activity-row">
            <div className="activity-trip-icon"><Route size={16} /></div>
            <div className="activity-trip"><strong>Trip {activity.trip_id}</strong><span>{new Date(activity.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} · {activity.processing_time_seconds.toFixed(2)}s</span></div>
            <div className="activity-coverage"><span>{activity.matched_observations.toLocaleString()} / {activity.total_observations.toLocaleString()} matched</span><div className="coverage-track"><i style={{ width: `${coverage}%` }} /></div></div>
            <StatusPill label={coverage === 100 ? 'Complete' : coverage > 0 ? 'Partial' : 'No matches'} tone={coverage === 100 ? 'good' : coverage > 0 ? 'warn' : 'bad'} />
          </Link>;
        })}
      </div>}
    </section>
    <div className="dashboard-bottom"><span>All values shown here come from backend responses or predictions run in this browser session.</span><StatusPill label={health?.road_network_loaded ? 'Road network cached' : 'Road networks load per trip'} tone={health?.road_network_loaded ? 'good' : 'neutral'} /></div>
  </div>;
}
