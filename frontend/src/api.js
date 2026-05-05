import { API_BASE } from './config.js';
import { getSessionId, setSessionId } from './session.js';

const SESSION_HEADER = 'X-Session-Id';

const buildHeaders = (extra = {}, { json = true } = {}) => {
  const headers = { [SESSION_HEADER]: getSessionId(), ...extra };
  if (json) headers['Content-Type'] = 'application/json';
  return headers;
};

const syncSessionFromResponse = (res) => {
  const sid = res.headers.get(SESSION_HEADER);
  if (sid) setSessionId(sid);
};

const fetchJson = async (path, options = {}) => {
  const { headers: extraHeaders, _json = true, ...rest } = options;
  const res = await fetch(`${API_BASE}${path}`, {
    headers: buildHeaders(extraHeaders, { json: _json }),
    ...rest,
  });
  syncSessionFromResponse(res);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
};

export const api = {
  publicConfig: () => fetchJson('/api/config'),
  generate: (body) =>
    fetchJson('/api/generate', { method: 'POST', body: JSON.stringify(body) }),
  compare: (body) =>
    fetchJson('/api/compare', { method: 'POST', body: JSON.stringify(body) }),
  tokenize: (body) =>
    fetchJson('/api/tokenize', { method: 'POST', body: JSON.stringify(body) }),
  verifyClaims: (body) =>
    fetchJson('/api/verify-claims', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  uploadText: (body) =>
    fetchJson('/api/upload-text', { method: 'POST', body: JSON.stringify(body) }),
  uploadUrl: (body) =>
    fetchJson('/api/upload-url', { method: 'POST', body: JSON.stringify(body) }),
  uploadPdf: async (file) => {
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch(`${API_BASE}/api/upload-pdf`, {
      method: 'POST',
      headers: { [SESSION_HEADER]: getSessionId() },
      body: fd,
    });
    syncSessionFromResponse(res);
    if (!res.ok) {
      let detail = 'Error al subir PDF';
      try {
        const data = await res.json();
        detail = data.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }
    return res.json();
  },
  ragStatus: () => fetchJson('/api/rag-status'),
  clearRag: () => fetchJson('/api/clear-rag', { method: 'DELETE', _json: false }),
};
