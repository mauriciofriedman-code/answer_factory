import { API_BASE } from './config.js';

const fetchJson = async (path, options = {}) => {
  const res = await fetch(`${API_BASE}${path}`, {
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
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
      credentials: 'include',
      body: fd,
    });
    if (!res.ok) throw new Error('Error al subir PDF');
    return res.json();
  },
  ragStatus: () => fetchJson('/api/rag-status'),
  clearRag: () => fetchJson('/api/clear-rag', { method: 'DELETE' }),
};
