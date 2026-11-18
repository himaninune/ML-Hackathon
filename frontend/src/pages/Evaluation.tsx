import { useEffect } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ArrowRight, ChartNoAxesColumnIncreasing, Route } from 'lucide-react';
import { EvaluationPanel } from '../components/evaluation/EvaluationPanel';
import { PageHeader } from '../components/ui/PageHeader';
import { MetricCard } from '../components/ui/MetricCard';
import { StatusPill } from '../components/ui/StatusPill';
import { useApp } from '../context/AppContext';

export function Evaluation() {
  const { tripId: routeTripId } = useParams();
  const { selectedTripId, setSelectedTripId, predictions } = useApp();
  const tripId = routeTripId || selectedTripId;
  const result = predictions[tripId];

  useEffect(() => { if (routeTripId) setSelectedTripId(routeTripId); }, [routeTripId, setSelectedTripId]);

  if (!result) return <div className="page-stack">
    <PageHeader eyebrow="SERVER-SIDE METRICS" title="Evaluation" description="Metrics are displayed only when returned by a completed backend prediction." />
    <section className="panel evaluation-empty-state"><div className="empty-icon"><ChartNoAxesColumnIncreasing size={22} /></div><h2>No prediction available in this session</h2><p>Run a route reconstruction first. The frontend does not calculate or invent evaluation values.</p><Link to={tripId ? `/predict/${tripId}` : '/predict'} className="button button-primary">Open route workspace <ArrowRight size={15} /></Link></section>
  </div>;

  const ratio = result.prediction.total_observations ? result.prediction.matched_observations / result.prediction.total_observations : 0;
  return <div className="page-stack">
    <PageHeader eyebrow="SERVER-SIDE METRICS" title="Route evaluation" description="The evaluation response is passed through from FastAPI; the frontend does not compute replacement metrics." actions={<Link to={`/predict/${tripId}`} className="button button-secondary"><Route size={15} /> View prediction</Link>} />
    <section className="evaluation-context panel"><div className="evaluation-context-icon"><ChartNoAxesColumnIncreasing size={19} /></div><div><span className="eyebrow">PREDICTION CONTEXT</span><strong>Trip {result.trip_id}</strong><span>{result.processing_time_seconds.toFixed(2)}s backend processing · {result.prediction.predicted_route.length.toLocaleString()} predicted route arcs</span></div><StatusPill label={ratio === 1 && result.prediction.total_observations > 0 ? 'Full coverage' : ratio > 0 ? 'Partial coverage' : 'No matches'} tone={ratio === 1 && result.prediction.total_observations > 0 ? 'good' : ratio > 0 ? 'warn' : 'bad'} /></section>
    <section className="metric-grid evaluation-coverage-grid">
      <MetricCard label="Matched observations" value={result.prediction.matched_observations.toLocaleString()} detail="Reported by prediction API" icon={Route} tone="green" />
      <MetricCard label="Unmatched observations" value={result.prediction.unmatched_observation_indices.length.toLocaleString()} detail="Reported by prediction API" icon={ChartNoAxesColumnIncreasing} tone="amber" />
      <MetricCard label="Processing time" value={`${result.processing_time_seconds.toFixed(2)}s`} detail="Reported by prediction API" icon={ChartNoAxesColumnIncreasing} tone="violet" />
      <MetricCard label="Coverage" value={`${(ratio * 100).toFixed(1)}%`} detail="Matched / total observations" icon={Route} tone="blue" />
    </section>
    <EvaluationPanel evaluation={result.evaluation} />
  </div>;
}
