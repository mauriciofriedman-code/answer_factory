import { useState } from 'react';
import { api } from '../api.js';

const VERDICT_META = {
  SUSTENTADA: { label: '✓ Sustentada', tone: 'ok' },
  CONTRADICHA: { label: '✗ Contradicha', tone: 'error' },
  NO_HAY_EVIDENCIA: { label: '? Sin evidencia', tone: 'warn' },
};

export default function VerifyPanel({ responseText, ragHasContent, judgeModel }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  if (!responseText) return null;

  const handleVerify = async () => {
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      const data = await api.verifyClaims({
        response_text: responseText,
        judge_model: judgeModel || 'gpt-4o-mini',
      });
      setReport(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="verify-block">
      <div className="verify-header">
        <h3 className="subheading" style={{ margin: 0 }}>
          Verificar contra mis fuentes
        </h3>
        <button
          className="button"
          onClick={handleVerify}
          disabled={loading || !ragHasContent}
          title={
            ragHasContent
              ? 'Extrae las afirmaciones de la respuesta y compáralas contra tu RAG'
              : 'Sube fuentes al RAG primero'
          }
        >
          {loading ? 'Verificando…' : 'Verificar afirmaciones'}
        </button>
      </div>

      {!ragHasContent && (
        <p className="hint-block">
          Sube fuentes al RAG primero (texto, PDF o URL) y luego podrás contrastar la
          respuesta contra ellas.
        </p>
      )}

      {error && <div className="error-text">Error: {error}</div>}

      {report && (
        <>
          <div className="meta-row">
            <span className="badge">{report.summary.total} afirmaciones</span>
            <span className="badge ok">✓ {report.summary.sustentadas} sustentadas</span>
            <span className="badge error">
              ✗ {report.summary.contradichas} contradichas
            </span>
            <span className="badge warn">
              ? {report.summary.sin_evidencia} sin evidencia
            </span>
          </div>

          <ul className="claim-list">
            {report.claims.map((c, i) => (
              <li key={i} className={`claim-item verdict-${c.verdict}`}>
                <div className="claim-row">
                  <span className={`verdict-pill ${VERDICT_META[c.verdict].tone}`}>
                    {VERDICT_META[c.verdict].label}
                  </span>
                  <span className="claim-text">{c.claim}</span>
                </div>
                {c.explanation && <div className="claim-explanation">{c.explanation}</div>}
                {c.context_chunks?.length > 0 && (
                  <details className="claim-context">
                    <summary>Ver fragmentos consultados</summary>
                    {c.context_chunks.map((ch, k) => (
                      <div key={k} className="claim-context-chunk">
                        <strong>{ch.title}</strong> · frag {ch.page}
                        <div className="claim-snippet">{ch.snippet}…</div>
                      </div>
                    ))}
                  </details>
                )}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
