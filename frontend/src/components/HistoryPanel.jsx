import { useEffect, useState } from 'react';
import {
  loadHistory,
  deleteEntry,
  clearHistory,
  exportEntriesToPdf,
} from '../history.js';

export default function HistoryPanel({ refreshKey, onReuse }) {
  const [entries, setEntries] = useState([]);
  const [openId, setOpenId] = useState(null);
  const [notes, setNotes] = useState({});

  useEffect(() => {
    setEntries(loadHistory());
  }, [refreshKey]);

  const handleDelete = (id) => {
    if (!confirm('¿Borrar esta iteración?')) return;
    setEntries(deleteEntry(id));
  };

  const handleClear = () => {
    if (!confirm('¿Borrar TODO el historial?')) return;
    setEntries(clearHistory());
  };

  const handleExportOne = (entry) => {
    exportEntriesToPdf(
      `iteracion_${slug(entry.created_at)}.pdf`,
      [{ ...entry, notes: notes[entry.id] }]
    );
  };

  const handleExportAll = () => {
    if (!entries.length) return;
    exportEntriesToPdf(
      `compromiso_lunes_${slug(new Date().toISOString())}.pdf`,
      entries.map((e) => ({ ...e, notes: notes[e.id] }))
    );
  };

  return (
    <div className="card">
      <div className="card-header-row">
        <h2 className="card-title">Historial de iteraciones</h2>
        <div className="row-gap">
          <button
            className="button"
            onClick={handleExportAll}
            disabled={!entries.length}
          >
            ⬇ Exportar todo (PDF)
          </button>
          <button
            className="button danger small"
            onClick={handleClear}
            disabled={!entries.length}
          >
            🗑 Limpiar
          </button>
        </div>
      </div>

      <p className="hint-block">
        Cada generación se guarda automáticamente en este navegador. Exporta a PDF
        para subir tu compromiso del lunes al foro de Moodle. En el diálogo de
        impresión elige <strong>Guardar como PDF</strong> (o <em>Save as PDF</em>) como destino.
      </p>

      {entries.length === 0 && (
        <div className="response-empty">Aún no has generado nada.</div>
      )}

      <ul className="history-list">
        {entries.map((e) => {
          const open = openId === e.id;
          return (
            <li key={e.id} className={`history-item ${open ? 'open' : ''}`}>
              <div className="history-row">
                <button
                  className="history-toggle"
                  onClick={() => setOpenId(open ? null : e.id)}
                >
                  <span className="history-when">{formatWhen(e.created_at)}</span>
                  <span className="history-kind">{e.kind || 'Lab'}</span>
                  <span className="history-prompt">{truncate(e.prompt, 80)}</span>
                </button>
                <div className="row-gap small">
                  <button
                    className="button small"
                    onClick={() => onReuse?.(e)}
                    title="Reusar este prompt en el laboratorio"
                  >
                    ↺ Reusar
                  </button>
                  <button className="button small" onClick={() => handleExportOne(e)}>
                    ⬇ PDF
                  </button>
                  <button
                    className="button danger small"
                    onClick={() => handleDelete(e.id)}
                  >
                    ✕
                  </button>
                </div>
              </div>

              {open && (
                <div className="history-body">
                  <div className="meta-row">
                    <span className="badge">{e.model}</span>
                    <span className="badge muted">estilo: {e.style}</span>
                    {e.params && (
                      <span className="badge muted">
                        temp {e.params.temperature} · top_p {e.params.top_p}
                      </span>
                    )}
                    {e.use_rag && (
                      <span className="badge ok">RAG · {e.chunks_used || 0}</span>
                    )}
                  </div>

                  <h4 className="subheading">Prompt</h4>
                  <pre className="prompt-preview">{e.prompt}</pre>

                  <h4 className="subheading">Respuesta</h4>
                  <div className="response-display compact">{e.response}</div>

                  <h4 className="subheading">Mi reflexión (opcional)</h4>
                  <textarea
                    className="upload-input"
                    rows={3}
                    placeholder="¿Qué cambió respecto a la iteración anterior? ¿Qué ajustaría la próxima vez?"
                    value={notes[e.id] || ''}
                    onChange={(ev) =>
                      setNotes((n) => ({ ...n, [e.id]: ev.target.value }))
                    }
                  />
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function truncate(s, n) {
  if (!s) return '';
  return s.length > n ? `${s.slice(0, n)}…` : s;
}

function slug(iso) {
  return (iso || '').replace(/[^0-9]/g, '').slice(0, 14);
}

function formatWhen(iso) {
  try {
    return new Date(iso).toLocaleString('es-MX', {
      day: '2-digit',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}
