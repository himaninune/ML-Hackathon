import { Link } from 'react-router-dom';
import { Compass, MoveLeft } from 'lucide-react';

export function NotFound() {
  return <div className="not-found panel"><div className="not-found-icon"><Compass size={24} /></div><span className="eyebrow">404 · OFF ROUTE</span><h1>This page isn’t on the map.</h1><p>The PathSnap workspace could not find that route.</p><Link to="/" className="button button-primary"><MoveLeft size={16} /> Return to overview</Link></div>;
}
