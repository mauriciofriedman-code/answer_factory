import { useState } from 'react';
import { api } from '../api.js';

export default function RagPanel({ status, onChange }) {
  const [textTitle, setTextTitle] = useState('');
  const [textAuthor, setTextAuthor] = useState('');
  const [textBody, setTextBody] = useState('');
  const [url, setUrl] = useState('');
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState({});

  const reportError = (key, err) =>
    setFeedback((f) => ({ ...f, [key]: { type: 'error', text: err.message || String(err) } }));

  const reportOk = (key, text) =>
    setFeedback((f) => ({ ...f, [key]: { type: 'ok', text } }));

  const handleText = async () => {
    if (!textBody.trim()) {
      alert('Pega algún texto antes de subirlo.');
      return;
    }
    setBusy(true);
    try {
      const data = await api.uploadText({
        text: textBody,
        title: textTitle || 'Documento sin título',
        author: textAuthor || 'Autor desconocido',
        chunk_size: 1000,
        chunk_overlap: 150,
      });
      reportOk('text', `✓ "${data.metadata.title}" — ${data.chunks_created} fragmentos.`);
      setTextBody('');
      setTextTitle('');
      setTextAuthor('');
      onChange?.();
    } catch (e) {
      reportError('text', e);
    } finally {
      setBusy(false);
    }
  };

  const handleUrl = async () => {
    if (!url.trim()) return;
    setBusy(true);
    try {
      const data = await api.uploadUrl({ url, chunk_size: 1000, chunk_overlap: 150 });
      reportOk('url', `✓ ${data.metadata.title} — ${data.chunks_created} fragmentos.`);
      setUrl('');
      onChange?.();
    } catch (e) {
      reportError('url', e);
    } finally {
      setBusy(false);
    }
  };

  const handlePdf = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setBusy(true);
    try {
      const data = await api.uploadPdf(file);
      reportOk('pdf', `✓ ${file.name} — ${data.chunks_created} fragmentos.`);
      onChange?.();
    } catch (err) {
      reportError('pdf', err);
    } finally {
      setBusy(false);
      e.target.value = '';
    }
  };

  const handleClear = async () => {
    if (!confirm('¿Borrar todas las fuentes que has subido en esta sesión?')) return;
    try {
      await api.clearRag();
      setFeedback({});
      onChange?.();
    } catch (err) {
      alert(err.message);
    }
  };

  return (
    <div className="card">
      <div className="card-header-row">
        <h2 className="card-title">Carga de fuentes (RAG)</h2>
        <button className="button danger small" onClick={handleClear} disabled={busy}>
          🗑️ Limpiar mis fuentes
        </button>
      </div>

      {status?.total_chunks > 0 && (
        <div className="status-pill">
          📊 {status.total_chunks} fragmentos · {status.sources?.length || 0} fuente
          {(status.sources?.length || 0) === 1 ? '' : 's'} en tu sesión
        </div>
      )}

      <div className="upload-grid">
        <div className="upload-box">
          <div className="upload-title">📝 Texto plano</div>
          <input
            className="upload-input"
            placeholder="Título"
            value={textTitle}
            onChange={(e) => setTextTitle(e.target.value)}
          />
          <input
            className="upload-input"
            placeholder="Autor"
            value={textAuthor}
            onChange={(e) => setTextAuthor(e.target.value)}
          />
          <textarea
            className="upload-input"
            placeholder="Pega aquí el contenido…"
            rows={4}
            value={textBody}
            onChange={(e) => setTextBody(e.target.value)}
          />
          <button className="button block" onClick={handleText} disabled={busy}>
            Subir texto
          </button>
          {feedback.text && <FeedbackLine info={feedback.text} />}
        </div>

        <div className="upload-box">
          <div className="upload-title">📄 PDF</div>
          <input type="file" accept=".pdf" onChange={handlePdf} disabled={busy} />
          {feedback.pdf && <FeedbackLine info={feedback.pdf} />}
        </div>

        <div className="upload-box">
          <div className="upload-title">🔗 URL</div>
          <input
            className="upload-input"
            placeholder="https://ejemplo.com/articulo"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
          <button className="button block" onClick={handleUrl} disabled={busy}>
            Obtener contenido
          </button>
          {feedback.url && <FeedbackLine info={feedback.url} />}
        </div>
      </div>
    </div>
  );
}

function FeedbackLine({ info }) {
  return (
    <div className={`feedback-line ${info.type}`}>{info.text}</div>
  );
}
