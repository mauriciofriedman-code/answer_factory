import { useState } from 'react';
import { api } from '../api.js';

const PALETTE = [
  '#fef3c7', '#dbeafe', '#fce7f3', '#dcfce7', '#fed7aa',
  '#e9d5ff', '#fee2e2', '#cffafe', '#fef9c3', '#e0e7ff',
];

export default function TokenVisualizer({ models, defaultModel }) {
  const [text, setText] = useState(
    'La fotosíntesis convierte azúcar en oxígeno usando luz solar.'
  );
  const [model, setModel] = useState(defaultModel || 'gpt-4o-mini');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleTokenize = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const data = await api.tokenize({ text, model });
      setResult(data);
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2 className="card-title">Visualizador de tokens</h2>
      <p className="hint-block">
        Un modelo de lenguaje no ve palabras: ve <strong>tokens</strong>. Pega cualquier
        texto y mira cómo se parte. Verás que palabras como <em>«fotosíntesis»</em> o
        <em>«azúcar»</em> pueden costar más tokens que palabras inglesas comunes —
        eso explica por qué el español, hebreo y otros idiomas pesan más en tokens
        (y en costo).
      </p>

      <label className="field-label">Texto a tokenizar</label>
      <textarea
        className="prompt-input"
        rows={3}
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <label className="field-label">Modelo (define el tokenizer)</label>
      <select
        className="style-selector"
        value={model}
        onChange={(e) => setModel(e.target.value)}
      >
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.label}
          </option>
        ))}
      </select>

      <div className="row-gap">
        <button className="button primary" onClick={handleTokenize} disabled={loading}>
          {loading ? 'Tokenizando…' : 'Tokenizar'}
        </button>
      </div>

      {result && (
        <>
          <div className="meta-row">
            <span className="badge">{result.count} tokens</span>
            <span className="badge muted">encoding · {result.encoding}</span>
            <span className="badge muted">
              ratio · {(result.count / Math.max(text.length, 1)).toFixed(2)} tok/char
            </span>
          </div>

          <div className="token-canvas">
            {result.tokens.map((t, i) => (
              <span
                key={i}
                className="token-chip"
                style={{ background: PALETTE[i % PALETTE.length] }}
                title={`id ${t.id}`}
              >
                {visibleText(t.text)}
              </span>
            ))}
          </div>

          <details className="token-ids">
            <summary>Ver IDs numéricos</summary>
            <pre className="prompt-preview">{JSON.stringify(result.tokens.map((t) => t.id))}</pre>
          </details>
        </>
      )}
    </div>
  );
}

function visibleText(text) {
  if (text === '\n') return '↵';
  if (text === ' ') return '·';
  return text.replace(/ /g, '·').replace(/\n/g, '↵');
}
