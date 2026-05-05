import { useState } from 'react';
import { api } from '../api.js';
import { saveEntry } from '../history.js';

const PRESETS = [
  { label: 'Conservador', temperature: 0.2, top_p: 0.9, frequency_penalty: 0, presence_penalty: 0, max_tokens: 1800, style: 'natural' },
  { label: 'Balanceado',  temperature: 0.5, top_p: 0.9, frequency_penalty: 0, presence_penalty: 0, max_tokens: 2000, style: 'friendly_teacher' },
  { label: 'Creativo',    temperature: 0.9, top_p: 1.0, frequency_penalty: 0, presence_penalty: 0.3, max_tokens: 2200, style: 'natural' },
];

const STYLE_LABELS = {
  natural: '🎯 Natural',
  scientific: '🔬 Científico',
  friendly_teacher: '👨‍🏫 Profesor amigable',
  craftd: '🧱 CRAFT-D',
};

export default function CompareMode({ models, defaultModel, onHistoryChange }) {
  const [prompt, setPrompt] = useState('');
  const [useRag, setUseRag] = useState(false);
  const [variants, setVariants] = useState(
    PRESETS.map((p) => ({ ...p, model: defaultModel || 'claude-sonnet-4-6' }))
  );
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const updateVariant = (i, patch) =>
    setVariants((vs) => vs.map((v, idx) => (idx === i ? { ...v, ...patch } : v)));

  const addVariant = () => {
    if (variants.length >= 4) return;
    setVariants((vs) => [
      ...vs,
      { ...PRESETS[1], label: `Variante ${vs.length + 1}`, model: defaultModel || 'claude-sonnet-4-6' },
    ]);
  };

  const removeVariant = (i) => {
    if (variants.length <= 2) return;
    setVariants((vs) => vs.filter((_, idx) => idx !== i));
  };

  const handleRun = async () => {
    if (!prompt.trim()) {
      alert('Escribe un prompt antes de comparar.');
      return;
    }
    setLoading(true);
    setResults([]);
    try {
      const data = await api.compare({
        prompt,
        use_rag: useRag,
        variants: variants.map((v) => ({
          label: v.label,
          model: v.model,
          style: v.style,
          temperature: v.temperature,
          top_p: v.top_p,
          frequency_penalty: v.frequency_penalty,
          presence_penalty: v.presence_penalty,
          max_tokens: v.max_tokens,
          stop_sequences: [],
        })),
      });
      setResults(data.results || []);
      data.results?.forEach((r, idx) => {
        if (r.response) {
          saveEntry({
            kind: `Comparar · ${r.label}`,
            prompt,
            response: r.response,
            model: r.model,
            style: variants[idx]?.style,
            params: variants[idx],
            use_rag: useRag,
            chunks_used: r.chunks_used,
            sources: r.sources,
          });
        }
      });
      onHistoryChange?.();
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="card">
        <h2 className="card-title">Comparar lado a lado</h2>
        <p className="hint-block">
          Mismo prompt, diferentes parámetros. Ideal para la <strong>Demo bajo el capó</strong>{' '}
          de la Sesión 1: muestra cómo cambia la salida al mover temperature y system prompt.
        </p>

        <label className="field-label">Prompt común</label>
        <textarea
          className="prompt-input"
          rows={3}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Ejemplo: Diseña una rúbrica de 4 niveles para una exposición oral de 5° de primaria sobre ecosistemas."
        />

        <div className="toggle-row">
          <label>Usar contexto RAG</label>
          <button
            className={`toggle ${useRag ? 'on' : ''}`}
            onClick={() => setUseRag((v) => !v)}
            aria-pressed={useRag}
          >
            <span />
          </button>
          <span className="toggle-state">{useRag ? 'Activado' : 'Desactivado'}</span>
        </div>

        <div className="row-gap">
          <button className="button primary" onClick={handleRun} disabled={loading}>
            {loading ? 'Generando…' : `Correr ${variants.length} variantes`}
          </button>
          <button className="button" onClick={addVariant} disabled={variants.length >= 4}>
            + Agregar variante
          </button>
        </div>
      </div>

      <div className={`compare-grid cols-${variants.length}`}>
        {variants.map((v, i) => (
          <VariantCard
            key={i}
            index={i}
            variant={v}
            models={models}
            result={results[i]}
            loading={loading}
            canRemove={variants.length > 2}
            onChange={(patch) => updateVariant(i, patch)}
            onRemove={() => removeVariant(i)}
          />
        ))}
      </div>
    </div>
  );
}

function VariantCard({ index, variant, models, result, loading, canRemove, onChange, onRemove }) {
  return (
    <div className="card variant-card">
      <div className="card-header-row">
        <input
          className="variant-title-input"
          value={variant.label}
          onChange={(e) => onChange({ label: e.target.value })}
        />
        {canRemove && (
          <button className="button danger small" onClick={onRemove}>
            ✕
          </button>
        )}
      </div>

      <label className="field-label">Modelo</label>
      <select
        className="style-selector"
        value={variant.model}
        onChange={(e) => onChange({ model: e.target.value })}
      >
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.label}
          </option>
        ))}
      </select>

      <label className="field-label">Estilo</label>
      <select
        className="style-selector"
        value={variant.style}
        onChange={(e) => onChange({ style: e.target.value })}
      >
        {Object.entries(STYLE_LABELS).map(([v, l]) => (
          <option key={v} value={v}>
            {l}
          </option>
        ))}
      </select>

      <MiniSlider
        label="Temp"
        min={0}
        max={2}
        step={0.1}
        value={variant.temperature}
        onChange={(v) => onChange({ temperature: v })}
        format={(v) => v.toFixed(1)}
      />
      <MiniSlider
        label="Top-P"
        min={0}
        max={1}
        step={0.01}
        value={variant.top_p}
        onChange={(v) => onChange({ top_p: v })}
        format={(v) => v.toFixed(2)}
      />
      <MiniSlider
        label="Freq pen"
        min={-2}
        max={2}
        step={0.1}
        value={variant.frequency_penalty}
        onChange={(v) => onChange({ frequency_penalty: v })}
        format={(v) => v.toFixed(1)}
      />
      <MiniSlider
        label="Pres pen"
        min={-2}
        max={2}
        step={0.1}
        value={variant.presence_penalty}
        onChange={(v) => onChange({ presence_penalty: v })}
        format={(v) => v.toFixed(1)}
      />
      <MiniSlider
        label="Max tok"
        min={50}
        max={2000}
        step={50}
        value={variant.max_tokens}
        onChange={(v) => onChange({ max_tokens: v })}
      />

      <div className="variant-output">
        {loading && !result && <span className="muted-italic">Esperando…</span>}
        {result?.error && <span className="error-text">Error: {result.error}</span>}
        {result?.response && (
          <>
            <div className="meta-row">
              <span className="badge">{result.model}</span>
              {result.usage?.input_tokens != null && (
                <span className="badge muted">
                  in {result.usage.input_tokens} · out {result.usage.output_tokens}
                </span>
              )}
            </div>
            <div className="response-display compact">{result.response}</div>
          </>
        )}
        {!loading && !result && (
          <span className="muted-italic">— corre las variantes para ver salida —</span>
        )}
      </div>
    </div>
  );
}

function MiniSlider({ label, min, max, step, value, onChange, format }) {
  const display = format ? format(value) : value;
  return (
    <div className="mini-slider">
      <div className="mini-slider-label">
        <span>{label}</span>
        <span className="slider-value">{display}</span>
      </div>
      <input
        type="range"
        className="slider"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}
