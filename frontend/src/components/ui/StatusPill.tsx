import { Circle } from 'lucide-react';

export function StatusPill({ label, tone = 'neutral', pulse = false }: {
  label: string;
  tone?: 'good' | 'warn' | 'bad' | 'neutral';
  pulse?: boolean;
}) {
  return <span className={`status-pill status-${tone}`}>
    <Circle size={7} fill="currentColor" className={pulse ? 'status-pulse' : ''} aria-hidden="true" />
    {label}
  </span>;
}
