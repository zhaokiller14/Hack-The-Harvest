/**
 * Creates a Leaflet map with a polygon draw control.
 * @param {string} containerId
 * @param {[number,number]} center
 * @param {number} zoom
 * @param {(geom: object) => void} [onDraw]  - called whenever a polygon is drawn or loaded
 * @returns {{ map, resultLayer, getPolygon, clearDraw, showPolygon, loadPolygon }}
 */
export function createMap(containerId, center = [34.0, 9.0], zoom = 7, onDraw = null) {
  const map = L.map(containerId).setView(center, zoom);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap',
    maxZoom: 18,
  }).addTo(map);

  const drawnItems = new L.FeatureGroup().addTo(map);
  const resultLayer = new L.FeatureGroup().addTo(map);

  new L.Control.Draw({
    edit: { featureGroup: drawnItems },
    draw: {
      polygon:      { allowIntersection: false, showArea: true },
      rectangle:    false,
      circle:       false,
      marker:       false,
      polyline:     false,
      circlemarker: false,
    },
  }).addTo(map);

  let _polygon = null;

  function _setAndNotify(geom) {
    _polygon = geom;
    if (onDraw) onDraw(geom);
  }

  map.on(L.Draw.Event.CREATED, (e) => {
    drawnItems.clearLayers();
    drawnItems.addLayer(e.layer);
    _setAndNotify(e.layer.toGeoJSON().geometry);
  });

  /**
   * Load a GeoJSON geometry onto the draw layer programmatically
   * (e.g. from a pasted textarea). Fires onDraw callback.
   */
  function loadPolygon(geom) {
    drawnItems.clearLayers();
    const layer = L.geoJSON(geom, {
      style: { color: '#FF5F03', fillColor: '#FF5F03', fillOpacity: .15, weight: 2 },
    });
    layer.eachLayer(l => drawnItems.addLayer(l));
    map.fitBounds(drawnItems.getBounds(), { padding: [40, 40] });
    _setAndNotify(geom);
  }

  return {
    map,
    resultLayer,
    getPolygon:  () => _polygon,
    clearDraw: () => {
      drawnItems.clearLayers();
      resultLayer.clearLayers();
      _polygon = null;
    },
    loadPolygon,
    showPolygon: (geom, style = {}) => {
      const defaults = { color: '#FF5F03', fillColor: '#FF5F03', fillOpacity: .2, weight: 2 };
      L.geoJSON(geom, { style: { ...defaults, ...style } }).addTo(resultLayer);
    },
  };
}
