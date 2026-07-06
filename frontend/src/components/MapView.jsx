import { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, ImageOverlay, GeoJSON } from "react-leaflet";
import L from "leaflet";
import { getDetections, maskBlobUrl } from "../api.js";

// Compute a lat/lng bounds box that encloses all polygon coordinates in the
// FeatureCollection. GeoJSON coords are [lng, lat]; Leaflet bounds want [lat, lng].
function boundsFromFeatures(fc) {
  if (!fc || !Array.isArray(fc.features) || fc.features.length === 0) return null;
  let minLat = Infinity;
  let minLng = Infinity;
  let maxLat = -Infinity;
  let maxLng = -Infinity;
  const visit = (coords) => {
    for (const c of coords) {
      if (Array.isArray(c[0])) {
        visit(c);
      } else {
        const [lng, lat] = c;
        if (lat < minLat) minLat = lat;
        if (lat > maxLat) maxLat = lat;
        if (lng < minLng) minLng = lng;
        if (lng > maxLng) maxLng = lng;
      }
    }
  };
  for (const f of fc.features) {
    if (f?.geometry?.coordinates) visit(f.geometry.coordinates);
  }
  if (!isFinite(minLat) || !isFinite(minLng)) return null;
  return L.latLngBounds([minLat, minLng], [maxLat, maxLng]);
}

function onEachFeature(feature, layer) {
  const p = feature?.properties || {};
  const area = typeof p.area_m2 === "number" ? p.area_m2.toFixed(1) : p.area_m2 ?? "—";
  const conf =
    typeof p.confidence === "number" ? p.confidence.toFixed(3) : p.confidence ?? "—";
  layer.bindPopup(
    `<strong>Detection #${p.detection_id ?? "?"}</strong><br/>area: ${area} m²<br/>confidence: ${conf}`
  );
}

export default function MapView({ run }) {
  const [maskObjectUrl, setMaskObjectUrl] = useState(null);
  const [features, setFeatures] = useState(null);
  const [error, setError] = useState("");

  const runId = run?.run_id;

  useEffect(() => {
    let cancelled = false;
    let createdUrl = null;
    setError("");
    setFeatures(null);
    setMaskObjectUrl(null);

    if (!runId) return undefined;

    (async () => {
      try {
        const [fc, url] = await Promise.all([
          getDetections(runId).catch(() => null),
          maskBlobUrl(runId).catch(() => null)
        ]);
        if (cancelled) {
          if (url) URL.revokeObjectURL(url);
          return;
        }
        if (fc) setFeatures(fc);
        if (url) {
          createdUrl = url;
          setMaskObjectUrl(url);
        }
      } catch (err) {
        if (!cancelled) setError("failed to load run overlay");
      }
    })();

    return () => {
      cancelled = true;
      if (createdUrl) URL.revokeObjectURL(createdUrl);
    };
  }, [runId]);

  const bounds = useMemo(() => boundsFromFeatures(features), [features]);

  // GeoJSON layer needs a key so it re-renders when features change.
  const geoKey = runId ? `geo-${runId}-${features?.features?.length ?? 0}` : "geo-none";

  if (!run) {
    return (
      <div className="card map-card empty">
        <p className="muted">Run a detection or select a past run to see the change mask.</p>
      </div>
    );
  }

  const hasGeo = !!bounds;
  const center = hasGeo ? bounds.getCenter() : { lat: 0, lng: 0 };

  return (
    <div className="card map-card">
      <div className="map-head">
        <h2>Change overlay</h2>
        <span className="muted">
          run #{run.run_id} · {run.num_detections ?? 0} detections ·{" "}
          {run.total_area_m2 != null ? `${Number(run.total_area_m2).toFixed(1)} m²` : "—"}
        </span>
      </div>
      {error && <div className="error">{error}</div>}
      <div className="map-holder">
        {hasGeo ? (
          <MapContainer
            key={geoKey}
            center={[center.lat, center.lng]}
            zoom={16}
            bounds={bounds}
            scrollWheelZoom
            style={{ height: "100%", width: "100%" }}
          >
            <TileLayer
              attribution='&copy; OpenStreetMap contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {maskObjectUrl && bounds && (
              <ImageOverlay url={maskObjectUrl} bounds={bounds} opacity={0.55} />
            )}
            {features && <GeoJSON key={geoKey} data={features} onEachFeature={onEachFeature} />}
          </MapContainer>
        ) : (
          // Fallback for pixel-coordinate masks (no geotransform): show the raw mask.
          <div className="pixel-fallback">
            {maskObjectUrl ? (
              <img src={maskObjectUrl} alt="change mask" />
            ) : (
              <p className="muted">No georeferenced polygons; mask unavailable.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
