import type { LucideIcon } from 'lucide-react';

export function MetricCard({ label, value, detail, icon: Icon, tone = 'blue' }: {
  label: string;
  value: string | number;
  detail: string;
  icon: LucideIcon;
  tone?: 'blue' | 'green' | 'amber' | 'violet';
}) {
  return <article className="metric-card panel">
    <div className={`metric-icon tone-${tone}`}><Icon size={18} strokeWidth={1.8} /></div>
    <div className="metric-copy">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  </article>;
}
