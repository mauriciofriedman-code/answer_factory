import LogprobsPanel from './LogprobsPanel.jsx';
import VerifyPanel from './VerifyPanel.jsx';

export default function ResponsePanel({ result, useRag, loading, ragHasContent, judgeModel }) {
  return (
    <div className="card">
      <h2 className="card-title">Respuesta</h2>

      {loading && <div className="loading-row"><span className="spinner" /> Generando…</div>}

      {result && (
        <>
          <div className="meta-row">
            <span className="badge">{result.model}</span>
            <span className={`badge ${useRag ? 'ok' : 'muted'}`}>
              {useRag ? `RAG · ${result.chunks_used} fragmentos` : 'RAG desactivado'}
            </span>
            {result.usage?.input_tokens != null && (
              <span className="badge muted">
                in {result.usage.input_tokens} · out {result.usage.output_tokens}
              </span>
            )}
            {result.finish_reason && (
              <span className="badge muted">finish: {result.finish_reason}</span>
            )}
          </div>

          {result.sources?.length > 0 && (
            <div className="sources-list">
              <strong>Fuentes recuperadas</strong>
              {result.sources.map((s, i) => (
                <div key={i} className="source-item">
                  <div className="source-head">
                    <span className="source-title">📄 {s.title}</span>
                    {s.author && (
                      <span className="source-author"> — {s.author}</span>
                    )}
                    <span className="source-loc"> · {s.location}</span>
                    {s.similarity != null && (
                      <span className="similarity"> · sim {s.similarity}</span>
                    )}
                  </div>
                  {s.snippet && (
                    <blockquote className="source-snippet">«{s.snippet}»</blockquote>
                  )}
                </div>
              ))}
            </div>
          )}

          <div className="response-display">{result.response || '—'}</div>

          {result.logprobs?.length > 0 && <LogprobsPanel logprobs={result.logprobs} />}

          {result.response && (
            <VerifyPanel
              responseText={result.response}
              ragHasContent={ragHasContent}
              judgeModel={judgeModel}
            />
          )}
        </>
      )}

      {!result && !loading && (
        <div className="response-empty">La respuesta de la IA aparecerá aquí…</div>
      )}
    </div>
  );
}
