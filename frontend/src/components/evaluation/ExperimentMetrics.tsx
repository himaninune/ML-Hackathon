import { useMemo } from 'react';
import { AlertCircle, Info, Sparkles } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { MetricsOverviewResponse, ViterbiMetricsResponse, XGBoostMetricsResponse } from '../../types/api';

const metricNames: Record<string, string> = {
  cv_5fold_top1_accuracy: '5-fold CV candidate top-1 accuracy',
  cv_5fold_top1_accuracy_std: '5-fold CV top-1 accuracy standard deviation',
  cv_5fold_macro_f1: '5-fold CV candidate macro F1',
  cv_5fold_macro_f1_std: '5-fold CV macro F1 standard deviation',
  cv_5fold_log_loss: '5-fold CV candidate log loss',
  student_test_top1_accuracy: 'Student-test candidate top-1 accuracy',
  student_test_macro_f1: 'Student-test candidate macro F1',
  student_test_precision_macro: 'Student-test candidate macro precision',
  student_test_recall_macro: 'Student-test candidate macro recall',
  student_test_log_loss: 'Student-test candidate log loss',
  training_time_seconds: 'Training time',
  inference_time_seconds: 'Inference time',
  optimisation_time_seconds: 'Bayesian optimisation time',
  pointwise_xgb_top1_accuracy: 'Pointwise XGBoost observation accuracy',
  viterbi_sequence_top1_accuracy: 'Viterbi sequence observation accuracy',
  accuracy_gain: 'Sequence accuracy gain',
  pointwise_route_continuity: 'Pointwise XGBoost route continuity',
  viterbi_route_continuity: 'Viterbi route continuity',
  route_continuity_gain: 'Route continuity gain',
  viterbi_decoding_time_seconds: 'Viterbi decoding time',
};

function labelFor(key: string) {
  return metricNames[key] || key.replaceAll('_', ' ');
}

function formatMetric(key: string, value: number) {
  if (key.endsWith('_seconds')) return `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })} s`;
  if (/accuracy|macro_f1|precision|recall|continuity/.test(key)) {
    const percentage = value * 100;
    return `${key.endsWith('_gain') ? `${percentage >= 0 ? '+' : ''}${percentage.toFixed(2)}` : percentage.toFixed(2)}%`;
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function MetricRows({ values }: { values: Record<string, number> }) {
  return <div className="experiment-metric-list">
    {Object.entries(values).map(([key, value]) => <div className="experiment-metric-row" key={key}>
      <span>{labelFor(key)}</span><strong>{formatMetric(key, value)}</strong>
    </div>)}
  </div>;
}

function ParameterRows({ values }: { values: Record<string, number> }) {
  return <details className="parameter-details">
    <summary>Recorded model parameters</summary>
    <div className="parameter-list">
      {Object.entries(values).map(([key, value]) => <div key={key}><span>{key.replaceAll('_', ' ')}</span><strong>{value}</strong></div>)}
    </div>
  </details>;
}

function MetricUnavailable({ reason }: { reason: string | null }) {
  return <div className="metrics-unavailable"><AlertCircle size={17} /><div><strong>Metrics unavailable</strong><span>{reason || 'The backend did not return metrics for this experiment.'}</span></div></div>;
}

function ChartTooltip({ value, name }: { value?: unknown; name?: string }) {
  const numeric = typeof value === 'number' ? value : Number(value);
  return <span>{name}: {numeric.toFixed(2)}%</span>;
}

function XGBoostSection({ response }: { response: XGBoostMetricsResponse }) {
  const data = response.metrics;
  const chartData = useMemo(() => {
    if (!data) return [];
    const baseline = data.baseline_metrics.metrics;
    const tuned = data.bayesian_tuned_metrics.metrics;
    const definitions = [
      ['student_test_top1_accuracy', 'Top-1'],
      ['student_test_macro_f1', 'Macro F1'],
      ['student_test_precision_macro', 'Precision'],
      ['student_test_recall_macro', 'Recall'],
    ] as const;
    return definitions.flatMap(([key, name]) => typeof baseline[key] === 'number' && typeof tuned[key] === 'number'
      ? [{ name, baseline: baseline[key] * 100, tuned: tuned[key] * 100 }]
      : []);
  }, [data]);

  return <section className="panel experiment-section">
    <div className="section-title"><div><span className="eyebrow">CANDIDATE CLASSIFICATION</span><h2>XGBoost model evaluation</h2></div><span className="experiment-stage-tag">{response.evaluation_split}</span></div>
    {!response.available || !data ? <MetricUnavailable reason={response.reason} /> : <>
      <div className="experiment-context"><Sparkles size={16} /><div><strong>Selected for deployment: {data.selected_model}</strong><span>{data.rationale}</span></div></div>
      <div className="experiment-note"><Info size={15} /><span>{response.reference_note} Candidate-classification scores measure candidate labels, not complete route reconstruction.</span></div>
      <div className="experiment-run-grid">
        <article className="experiment-run-card">
          <div className="experiment-run-heading"><div><span className="eyebrow">DEPLOYED CLASSIFIER</span><h3>Untuned baseline</h3></div><span className="metric-count">Selected model</span></div>
          <MetricRows values={data.baseline_metrics.metrics} />
          <ParameterRows values={data.baseline_metrics.parameters} />
        </article>
        <article className="experiment-run-card">
          <div className="experiment-run-heading"><div><span className="eyebrow">EXPERIMENTAL COMPARISON</span><h3>Bayesian-tuned candidate</h3></div></div>
          <MetricRows values={data.bayesian_tuned_metrics.metrics} />
          <ParameterRows values={data.bayesian_tuned_metrics.parameters} />
        </article>
      </div>
      {chartData.length > 0 && <div className="experiment-chart-block">
        <div className="section-title"><div><span className="eyebrow">STUDENT-TEST COMPARISON</span><h3>Candidate classification scores</h3></div></div>
        <div className="experiment-chart" role="img" aria-label="Student-test candidate classification comparison between baseline and Bayesian-tuned XGBoost">
          <ResponsiveContainer width="100%" height="100%"><BarChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: -14 }}>
            <CartesianGrid stroke="#213044" vertical={false} />
            <XAxis dataKey="name" tick={{ fill: '#8291a5', fontSize: 10 }} axisLine={false} tickLine={false} />
            <YAxis domain={[0, 100]} tick={{ fill: '#65758b', fontSize: 10 }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ background: '#101c2c', border: '1px solid #26384d', borderRadius: 10, color: '#e6edf6' }} formatter={(value, name) => <ChartTooltip value={value} name={String(name)} />} />
            <Legend />
            <Bar dataKey="baseline" name="Untuned baseline" fill="#58a6ff" radius={[4, 4, 0, 0]} />
            <Bar dataKey="tuned" name="Bayesian-tuned" fill="#9f87ff" radius={[4, 4, 0, 0]} />
          </BarChart></ResponsiveContainer>
        </div>
      </div>}
    </>}
  </section>;
}

function ViterbiSection({ response }: { response: ViterbiMetricsResponse }) {
  const metrics = response.metrics;
  const chartData = metrics ? [
    { name: 'Observation accuracy', pointwise: metrics.pointwise_xgb_top1_accuracy * 100, sequence: metrics.viterbi_sequence_top1_accuracy * 100 },
    { name: 'Route continuity', pointwise: metrics.pointwise_route_continuity * 100, sequence: metrics.viterbi_route_continuity * 100 },
  ] : [];
  const values = metrics ? Object.fromEntries(Object.entries(metrics).filter(([, value]) => typeof value === 'number')) : {};

  return <section className="panel experiment-section">
    <div className="section-title"><div><span className="eyebrow">SEQUENCE-BASED ROUTE RECONSTRUCTION</span><h2>HMM + Viterbi evaluation</h2></div><span className="experiment-stage-tag">{response.evaluation_split}</span></div>
    {!response.available || !metrics ? <MetricUnavailable reason={response.reason} /> : <>
      <div className="experiment-note heuristic-note"><Info size={15} /><span>{response.reference_note} These experiment results are not official evaluation or verified GPS ground truth.</span></div>
      <div className="experiment-chart-block">
        <div className="section-title"><div><span className="eyebrow">POINTWISE VS SEQUENCE</span><h3>Observed route metrics</h3></div></div>
        <div className="experiment-chart" role="img" aria-label="Student-test pointwise XGBoost versus Viterbi observation accuracy and route continuity">
          <ResponsiveContainer width="100%" height="100%"><BarChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: -14 }}>
            <CartesianGrid stroke="#213044" vertical={false} />
            <XAxis dataKey="name" tick={{ fill: '#8291a5', fontSize: 10 }} axisLine={false} tickLine={false} />
            <YAxis domain={[0, 100]} tick={{ fill: '#65758b', fontSize: 10 }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ background: '#101c2c', border: '1px solid #26384d', borderRadius: 10, color: '#e6edf6' }} formatter={(value, name) => <ChartTooltip value={value} name={String(name)} />} />
            <Legend />
            <Bar dataKey="pointwise" name="Pointwise XGBoost" fill="#58a6ff" radius={[4, 4, 0, 0]} />
            <Bar dataKey="sequence" name="HMM + Viterbi" fill="#54d6a1" radius={[4, 4, 0, 0]} />
          </BarChart></ResponsiveContainer>
        </div>
      </div>
      <div className="experiment-viterbi-lower">
        <article className="experiment-run-card"><div className="experiment-run-heading"><div><span className="eyebrow">ALL RECORDED VALUES</span><h3>Route evaluation metrics</h3></div></div><MetricRows values={values} /></article>
        <article className="experiment-run-card"><div className="experiment-run-heading"><div><span className="eyebrow">DECODER CONFIGURATION</span><h3>Transition log priors</h3></div></div><MetricRows values={metrics.transition_log_priors} /></article>
      </div>
    </>}
  </section>;
}

export function ExperimentMetrics({ overview }: { overview: MetricsOverviewResponse }) {
  return <div className="experiment-metrics-stack">
    <XGBoostSection response={overview.xgboost} />
    <ViterbiSection response={overview.viterbi} />
  </div>;
}
