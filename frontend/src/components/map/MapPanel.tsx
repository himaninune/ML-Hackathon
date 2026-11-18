import { MapPinned } from 'lucide-react';
import type { LatLon, MatchedArc } from '../../types/api';
import { RouteMap } from './RouteMap';

export function MapPanel({
  gpsCoordinates,
  routeSegments,
  matchedArcs,
  unmatchedIndices,
  selectedObservation,
  onSelectObservation,
  tripId,
  compact = false,
}: {
  gpsCoordinates: LatLon[];
  routeSegments?: LatLon[][];
  matchedArcs?: MatchedArc[];
  unmatchedIndices?: number[];
  selectedObservation?: number | null;
  onSelectObservation?: (index: number) => void;
  tripId?: string;
  compact?: boolean;
}) {
  return <section className={`panel map-panel ${compact ? 'map-panel-compact' : ''}`}>
    {!compact && <div className="section-title map-heading">
      <div><span className="eyebrow">Geospatial view</span><h2><MapPinned size={19} /> Trajectory map</h2></div>
      {tripId && <span className="map-trip-chip">TRIP {tripId}</span>}
    </div>}
    <RouteMap gpsCoordinates={gpsCoordinates} routeSegments={routeSegments} matchedArcs={matchedArcs}
      unmatchedIndices={unmatchedIndices} selectedObservation={selectedObservation}
      onSelectObservation={onSelectObservation} tripId={tripId} compact={compact} />
  </section>;
}
