import { lazy, Suspense } from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import { AppShell } from './components/layout/AppShell';

const Dashboard = lazy(() => import('./pages/Dashboard').then((module) => ({ default: module.Dashboard })));
const Trips = lazy(() => import('./pages/Trips').then((module) => ({ default: module.Trips })));
const TripDetails = lazy(() => import('./pages/TripDetails').then((module) => ({ default: module.TripDetails })));
const Prediction = lazy(() => import('./pages/Prediction').then((module) => ({ default: module.Prediction })));
const Evaluation = lazy(() => import('./pages/Evaluation').then((module) => ({ default: module.Evaluation })));
const NotFound = lazy(() => import('./pages/NotFound').then((module) => ({ default: module.NotFound })));

export default function App() {
  return <AppProvider><BrowserRouter><Suspense fallback={<div className="route-loading"><span className="loading-indicator" />Loading workspace</div>}><Routes>
    <Route element={<AppShell />}>
      <Route index element={<Dashboard />} />
      <Route path="trips" element={<Trips />} />
      <Route path="trips/:tripId" element={<TripDetails />} />
      <Route path="predict" element={<Prediction />} />
      <Route path="predict/:tripId" element={<Prediction />} />
      <Route path="evaluation" element={<Evaluation />} />
      <Route path="evaluation/:tripId" element={<Evaluation />} />
      <Route path="*" element={<NotFound />} />
    </Route>
  </Routes></Suspense></BrowserRouter></AppProvider>;
}
