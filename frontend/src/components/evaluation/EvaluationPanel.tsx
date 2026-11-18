import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { ChartNoAxesColumnIncreasing, Info } from 'lucide-react';
import type { EvaluationResponse } from '../../types/api';

const labels: Record<string, string> = {
  sequence_similarity: 'Sequence similarity',
  arc_set_precision: 'Arc-set precision',
  arc_set_recall: 'Arc-set recall',
  arc_set_f1: 'Arc-set F1',
};

function displayValue(value: number) {
  return value >= 0 && value <= 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
}

export function EvaluationPanel({ evaluation }: { evaluation: EvaluationResponse }) {
  if (!evaluation.available) {
    return <section className="panel evaluation-panel">
      <div className="section-title"><div><span className="eyebrow">Evaluation</span><h2>Route comparison</h2></div><Info size={18} /></div>
      <div className="empty-evaluation">
        <div className="empty-icon"><ChartNoAxesColumnIncreasing size={20} /></div>
        <div><strong>Evaluation metrics unavailable for this trip.</strong><p>{evaluation.reason || 'The backend did not return reference-based metrics.'}</p></div>
      </div>
    </section>;
  }

  const metrics = Object.entries(evaluation.metrics).filter(([, value]) => Number.isFinite(value));
  const heuristic = evaluation.reference_type === 'route_membership_heuristic';
  return <section className="panel evaluation-panel">
    <div className="section-title">
      <div><span className="eyebrow">Evaluation</span><h2>Route comparison</h2></div>
      <span className="metric-count">{metrics.length} metrics</span>
    </div>
    {heuristic && <div className="heuristic-banner"><Info size={15} /><strong>Heuristic Evaluation — Not Official Ground Truth</strong></div>}
    {metrics.length === 0 ? <div className="empty-evaluation"><p>Evaluation metrics unavailable for this trip.</p></div> : <>
      <div className="evaluation-metrics">
        {metrics.map(([key, value]) => <div className="evaluation-metric" key={key}>
          <span>{labels[key] || key.replaceAll('_', ' ')}</span><strong>{displayValue(value)}</strong>
        </div>)}
      </div>
      <div className="chart-wrap" role="img" aria-label="Route evaluation metrics">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={metrics.map(([key, value]) => ({ name: labels[key] || key.replaceAll('_', ' '), value: value >= 0 && value <= 1 ? value * 100 : value }))} margin={{ top: 8, right: 12, bottom: 20, left: -16 }}>
            <CartesianGrid stroke="#213044" vertical={false} />
            <XAxis dataKey="name" tick={{ fill: '#8291a5', fontSize: 10 }} axisLine={false} tickLine={false} interval={0} angle={-9} textAnchor="end" />
            <YAxis domain={[0, 100]} tick={{ fill: '#65758b', fontSize: 10 }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ background: '#101c2c', border: '1px solid #26384d', borderRadius: 10, color: '#e6edf6' }} formatter={(value) => [`${Number(value).toFixed(1)}%`, 'Score']} />
            <Bar dataKey="value" fill="#68d6ab" radius={[5, 5, 0, 0]} maxBarSize={42} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </>}
  </section>;
}
