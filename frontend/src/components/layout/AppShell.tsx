import { NavLink, Outlet } from 'react-router-dom';
import { Activity, ArrowUpRight, ClipboardList, Compass, Gauge, Map, MapPinned, RefreshCw, Route, Satellite, Settings2 } from 'lucide-react';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import { StatusPill } from '../ui/StatusPill';

const links = [
  { to: '/', label: 'Overview', icon: Gauge, end: true },
  { to: '/trips', label: 'Trips', icon: ClipboardList },
  { to: '/predict', label: 'Route workspace', icon: Route },
  { to: '/evaluation', label: 'Evaluation', icon: Activity },
];

export function AppShell() {
  const { health, healthError, refreshing, refresh, trips } = useApp();
  const apiOnline = health?.status === 'healthy';
  return <div className="app-frame">
    <aside className="sidebar">
      <NavLink to="/" className="brand-lockup">
        <div className="brand-mark"><Compass size={22} strokeWidth={1.7} /><span /></div>
        <div><strong>PathSnap</strong><small>ROUTE INTELLIGENCE</small></div>
      </NavLink>
      <div className="sidebar-caption">WORKSPACE</div>
      <nav className="primary-nav" aria-label="Main navigation">
        {links.map(({ to, label, icon: Icon, end }) => <NavLink key={to} to={to} end={end} className={({ isActive }) => `nav-link ${isActive ? 'nav-link-active' : ''}`}>
          <Icon size={17} strokeWidth={1.8} /><span>{label}</span><ArrowUpRight className="nav-arrow" size={14} />
        </NavLink>)}
      </nav>
      <div className="sidebar-spacer" />
      <div className="sidebar-network-card">
        <div className="network-card-icon"><Satellite size={16} /></div>
        <div><strong>Distributed setup</strong><span>API connected over LAN</span></div>
        <span className={`network-led ${apiOnline ? 'led-online' : ''}`} />
      </div>
      <div className="sidebar-foot"><span>PATHSNAP PLATFORM</span><span>v1.0</span></div>
    </aside>

    <main className="main-shell">
      <header className="topbar">
        <div className="breadcrumb"><MapPinned size={15} /><span>Research console</span><span className="breadcrumb-slash">/</span><span className="breadcrumb-current">Map matching</span></div>
        <div className="topbar-right">
          <div className="topbar-health">
            <StatusPill label={apiOnline ? 'API online' : healthError ? 'API offline' : 'Connecting'} tone={apiOnline ? 'good' : healthError ? 'bad' : 'warn'} pulse={apiOnline} />
            <span className="model-health">{health?.model_loaded ? 'XGBoost ready' : 'Model unavailable'}</span>
          </div>
          <button className="icon-button" onClick={() => void refresh()} disabled={refreshing} title="Refresh backend status and trips" aria-label="Refresh backend status and trips">
            <RefreshCw size={16} className={refreshing ? 'spin' : ''} />
          </button>
          <div className="topbar-divider" />
          <div className="topbar-system"><span className="system-dot" /><span>{trips.length} trips indexed</span></div>
        </div>
      </header>
      <div className="mobile-nav">
        {links.map(({ to, label, icon: Icon, end }) => <NavLink key={to} to={to} end={end} className={({ isActive }) => `mobile-nav-link ${isActive ? 'mobile-nav-active' : ''}`}><Icon size={16} /><span>{label}</span></NavLink>)}
      </div>
      <div className="page-canvas"><Outlet /></div>
      <footer className="app-footer"><span><Map size={13} /> PathSnap · Distributed Map-Matching &amp; Route Reconstruction</span><span><Settings2 size={13} /> Backend <code>{api.baseURL}</code></span></footer>
    </main>
  </div>;
}
