import { useMemo } from 'react';
import { CircleMarker, FeatureGroup, LayersControl, MapContainer, Polyline, Popup, TileLayer, Tooltip, useMap } from 'react-leaflet';
import type { LatLngBoundsExpression } from 'leaflet';
import type { LatLon, MatchedArc } from '../../types/api';

function FitRoute({ points }: { points: LatLon[] }) {
  const map = useMap();
  const fit = () => {
    if (points.length === 1) map.setView(points[0], 16);
    else if (points.length > 1) map.fitBounds(points as LatLngBoundsExpression, { padding: [30, 30], maxZoom: 16 });
  };
  return <button type="button" className="fit-route-control" onClick={fit} title="Fit map to trajectory">Fit route</button>;
}

export function RouteMap({
  gpsCoordinates,
  routeSegments = [],
  matchedArcs = [],
  unmatchedIndices = [],
  selectedObservation = null,
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
  const validGps = gpsCoordinates.filter((point) => Number.isFinite(point[0]) && Number.isFinite(point[1]));
  const segments = routeSegments.filter((segment) => segment.length > 0);
  const boundsPoints = useMemo(() => [
    ...validGps,
    ...segments.flat(),
    ...matchedArcs.flatMap((arc) => arc.coordinates),
  ], [validGps, segments, matchedArcs]);
  const center = boundsPoints[0];
  const unmatched = new Set(unmatchedIndices);
  const matches = new Map(matchedArcs.map((arc) => [arc.observation_index, arc]));

  if (!center) return <div className={`map-empty ${compact ? 'map-empty-compact' : ''}`}><span>No GPS coordinates are available for this trip.</span></div>;

  return <div className={`route-map ${compact ? 'route-map-compact' : ''}`}>
    <MapContainer key={`${tripId || 'trip'}-${gpsCoordinates.length}-${segments.length}`} center={center} zoom={13} scrollWheelZoom className="leaflet-map">
      <TileLayer
        attribution='&copy; OpenStreetMap contributors &copy; CARTO'
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
      />
      <LayersControl position="topright">
        <LayersControl.Overlay checked name="Original GPS trajectory">
          <FeatureGroup>
            {validGps.length > 1 && <Polyline positions={validGps} pathOptions={{ color: '#58a6ff', weight: 2.5, opacity: 0.78 }} />}
            {validGps.map((point, index) => {
              const match = matches.get(index);
              const isUnmatched = unmatched.has(index);
              const isSelected = selectedObservation === index;
              return <CircleMarker
                key={`gps-${index}`}
                center={point}
                radius={isSelected ? 6 : isUnmatched ? 5 : 2.8}
                pathOptions={{
                  color: isSelected ? '#f2c96d' : isUnmatched ? '#ff6b72' : '#78b7ff',
                  fillColor: isSelected ? '#f2c96d' : isUnmatched ? '#ff6b72' : '#58a6ff',
                  fillOpacity: isSelected || isUnmatched ? 0.95 : 0.6,
                  weight: isSelected ? 2 : 1,
                }}
                eventHandlers={{ click: () => onSelectObservation?.(index) }}
              >
                <Tooltip direction="top">Observation {index}{isUnmatched ? ' · unmatched' : match ? ` · arc ${match.arc_id}` : ''}</Tooltip>
                <Popup>
                  <div className="map-popup">
                    <strong>GPS observation {index}</strong>
                    <span>{point[0].toFixed(6)}, {point[1].toFixed(6)}</span>
                    {match ? <span>Matched arc {match.arc_id} · {(match.probability * 100).toFixed(1)}% score</span> : isUnmatched ? <span className="popup-unmatched">No selected road arc</span> : <span>GPS trajectory point</span>}
                  </div>
                </Popup>
              </CircleMarker>;
            })}
          </FeatureGroup>
        </LayersControl.Overlay>
        {segments.length > 0 && <LayersControl.Overlay checked name="Predicted road route">
          <FeatureGroup>{segments.map((segment, index) => <Polyline key={`route-${index}`} positions={segment} pathOptions={{ color: '#54d6a1', weight: 5, opacity: 0.9 }} />)}</FeatureGroup>
        </LayersControl.Overlay>}
        {matchedArcs.length > 0 && <LayersControl.Overlay name="Matched road arcs">
          <FeatureGroup>{matchedArcs.map((arc) => <Polyline key={`arc-${arc.observation_index}`} positions={arc.coordinates} pathOptions={{ color: '#4ed6c4', weight: 2, opacity: 0.48 }} />)}</FeatureGroup>
        </LayersControl.Overlay>}
        {unmatchedIndices.length > 0 && <LayersControl.Overlay checked name="Unmatched observations">
          <FeatureGroup>{unmatchedIndices.filter((index) => validGps[index]).map((index) => <CircleMarker key={`unmatched-${index}`} center={validGps[index]} radius={6} pathOptions={{ color: '#ff6b72', fillColor: '#ff6b72', fillOpacity: 0.92 }}>
            <Tooltip>Observation {index} · unmatched</Tooltip>
          </CircleMarker>)}</FeatureGroup>
        </LayersControl.Overlay>}
      </LayersControl>
      <FitRoute points={boundsPoints} />
    </MapContainer>
    {!compact && <div className="map-legend">
      <span><i className="legend-blue" /> GPS trajectory</span>
      {segments.length > 0 && <span><i className="legend-green" /> Predicted route</span>}
      {matchedArcs.length > 0 && <span><i className="legend-teal" /> Matched arcs</span>}
      {unmatchedIndices.length > 0 && <span><i className="legend-red" /> Unmatched</span>}
      {selectedObservation !== null && <span><i className="legend-yellow" /> Selected</span>}
    </div>}
  </div>;
}
