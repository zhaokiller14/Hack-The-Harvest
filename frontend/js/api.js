async function _request(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  cartographier:      (body) => _request('POST', '/api/cartographier', body),
  classifierMetrics:  ()     => _request('GET',  '/api/cartographier/classifier-metrics'),
  predireRendement:   (body) => _request('POST', '/api/predire-rendement', body),
  diagnosticAnomalie: (body) => _request('POST', '/api/diagnostic-anomalie', body),
  listParcels:        ()     => _request('GET',  '/api/parcels'),
};
