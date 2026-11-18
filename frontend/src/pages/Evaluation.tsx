import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { AlertCircle, ArrowRight, ChartNoAxesColumnIncreasing, Route } from 'lucide-react';
import { EvaluationPanel } from '../components/evaluation/EvaluationPanel';
import { ExperimentMetrics } from '../components/evaluation/ExperimentMetrics';
import { PageHeader } from '../components/ui/PageHeader';
import { MetricCard } from '../components/ui/MetricCard';
import { StatusPill } from '../components/ui/StatusPill';
import { useApp } from '../context/AppContext';
import { api, apiErrorMessage } from '../services/api';
import type { MetricsOverviewResponse } from '../types/api';

export function Evaluation() {
  const { tripId: routeTripId } = useParams();
  const { selectedTripId, setSelectedTripId, predictions } = useApp();
  const tripId = routeTripId || selectedTripId;
  const result = predictions[tripId];
  const [overview, setOverview] = useState<MetricsOverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { if (routeTripId) setSelectedTripId(routeTripId); }, [routeTripId, setSelectedTripId]);
  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    api.metricsOverview().then((data) => { if (active) setOverview(data); })
      .catch((requestError: unknown) => { if (active) { setOverview(null); setError(apiErrorMessage(requestError)); } })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);

  const ratio = result?.prediction.total_observations
    ? result.prediction.matched_observations / result.prediction.total_observations : 0;

  return <div className="page-stack">
    <PageHeader eyebrow="EXPERIMENT RESULTS FROM FASTAPI" title="Evaluation dashboard" description="Candidate-classification results and sequence-based route reconstruction results are reported in separate sections from their server-side metric artifacts." actions={result ? <Link to={`/predict/${tripId}`} className="button button-secondary"><Route size={15} /> View prediction</Link> : <Link to="/predict" className="button button-primary">Open route workspace <ArrowRight size={15} /></Link>} />

    <section className="panel pipeline-panel evaluation-pipeline-panel">
      <div className="section-title"><div><span className="eyebrow">END-TO-END WORKFLOW</span><h2>From GPS observations to a road sequence</h2></div><span className="pipeline-tag"><span /> Pipeline stages</span></div>
      <div className="evaluation-pipeline-flow">
        {['GPS observations', 'Candidate generation', 'XGBoost classification', 'HMM emissions + transitions', 'Viterbi decoding', 'Reconstructed route'].map((stage, index) => <div className="evaluation-pipeline-step" key={stage}><i>{index + 1}</i><span>{stage}</span></div>)}
      </div>
      <p className="evaluation-pipeline-note">XGBoost measures candidate classification. HMM + Viterbi measures sequence-level route output and continuity. Their scores are not interchangeable.</p>
    </section>

    {loading && <section className="panel metrics-loading"><span className="loading-indicator" /> Loading experiment metrics from the backend…</section>}
    {error && <section className="panel metrics-fetch-error"><AlertCircle size={17} /><div><strong>Could not load experiment metrics</strong><span>{error}</span></div></section>}
    {overview && <ExperimentMetrics overview={overview} />}

    {result ? <>
      <section className="section-title trip-result-title"><div><span className="eyebrow">CURRENT BROWSER SESSION</span><h2>Prediction for trip {result.trip_id}</h2></div></section>
      <section className="evaluation-context panel"><div className="evaluation-context-icon"><ChartNoAxesColumnIncreasing size={19} /></div><div><span className="eyebrow">PREDICTION RESPONSE</span><strong>{result.trip_id}</strong><span>{result.processing_time_seconds.toFixed(2)}s backend processing · {result.prediction.predicted_route.length.toLocaleString()} predicted route arcs</span></div><StatusPill label={ratio === 1 && result.prediction.total_observations > 0 ? 'Full coverage' : ratio > 0 ? 'Partial coverage' : 'No matches'} tone={ratio === 1 && result.prediction.total_observations > 0 ? 'good' : ratio > 0 ? 'warn' : 'bad'} /></section>
      <div className="metric-grid evaluation-coverage-grid">
        <MetricCard label="Matched observations" value={result.prediction.matched_observations.toLocaleString()} detail="Reported by prediction API" icon={Route} tone="green" />
        <MetricCard label="Unmatched observations" value={result.prediction.unmatched_observation_indices.length.toLocaleString()} detail="Reported by prediction API" icon={ChartNoAxesColumnIncreasing} tone="amber" />
        <MetricCard label="Processing time" value={`${result.processing_time_seconds.toFixed(2)}s`} detail="Reported by prediction API" icon={ChartNoAxesColumnIncreasing} tone="violet" />
        <MetricCard label="Coverage" value={`${(ratio * 100).toFixed(1)}%`} detail="Matched / total observations" icon={Route} tone="blue" />
      </div>
      <EvaluationPanel evaluation={result.evaluation} />
    </> : <section className="panel evaluation-empty-state compact-evaluation-empty"><div className="empty-icon"><ChartNoAxesColumnIncreasing size={22} /></div><h2>No per-trip prediction in this session</h2><p>Experiment metrics above are loaded from the backend. Run a trip prediction to see its response-specific coverage and server-side comparison here.</p><Link to={tripId ? `/predict/${tripId}` : '/predict'} className="button button-primary">Open route workspace <ArrowRight size={15} /></Link></section>}
  </div>;
}
