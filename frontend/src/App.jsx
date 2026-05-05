import { useEffect, useState } from 'react';
import Mascot from './components/Mascot.jsx';
import StyleSelector from './components/StyleSelector.jsx';
import ParametersCard from './components/ParametersCard.jsx';
import RagPanel from './components/RagPanel.jsx';
import ResponsePanel from './components/ResponsePanel.jsx';
import CraftDWizard from './components/CraftDWizard.jsx';
import CompareMode from './components/CompareMode.jsx';
import HistoryPanel from './components/HistoryPanel.jsx';
import TokenVisualizer from './components/TokenVisualizer.jsx';
import PresetBar from './components/PresetBar.jsx';
import { api } from './api.js';
import { saveEntry } from './history.js';

const DEFAULT_PARAMS = {
  temperature: 0.7,
  top_p: 0.9,
  frequency_penalty: 0.0,
  presence_penalty: 0.0,
  max_tokens: 2000,
};

const TABS = [
  { id: 'lab', label: 'Laboratorio' },
  { id: 'craftd', label: 'Wizard CRAFT-D' },
  { id: 'compare', label: 'Comparar' },
  { id: 'tokens', label: 'Tokens' },
  { id: 'history', label: 'Historial' },
];

export default function App() {
  const [tab, setTab] = useState('lab');

  const [prompt, setPrompt] = useState('');
  const [stopSequences, setStopSequences] = useState('');
  const [style, setStyle] = useState('natural');
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [useRag, setUseRag] = useState(false);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState('claude-sonnet-4-6');
  const [defaultModel, setDefaultModel] = useState('claude-sonnet-4-6');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [ragStatus, setRagStatus] = useState({ total_chunks: 0, sources: [] });
  const [mood, setMood] = useState('normal');
  const [historyTick, setHistoryTick] = useState(0);
  const [activePresetId, setActivePresetId] = useState(null);
  const [returnLogprobs, setReturnLogprobs] = useState(false);

  useEffect(() => {
    api
      .publicConfig()
      .then((cfg) => {
        setModels(cfg.models || []);
        setModel(cfg.default_model || 'claude-sonnet-4-6');
        setDefaultModel(cfg.default_model || 'claude-sonnet-4-6');
      })
      .catch(() => {});
    refreshRag();
  }, []);

  const refreshRag = () => {
    api
      .ragStatus()
      .then(setRagStatus)
      .catch(() => {});
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      alert('Escribe una pregunta o prompt.');
      return;
    }
    setLoading(true);
    setMood('happy');
    try {
      const stopArr = stopSequences
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);
      const data = await api.generate({
        prompt,
        model,
        style,
        ...params,
        stop_sequences: stopArr,
        use_rag: useRag,
        return_logprobs: returnLogprobs,
        top_logprobs: 3,
      });
      setResult(data);
      saveEntry({
        kind: 'Laboratorio',
        prompt,
        response: data.response,
        model: data.model,
        style,
        params,
        use_rag: useRag,
        chunks_used: data.chunks_used,
        sources: data.sources,
      });
      setHistoryTick((n) => n + 1);
    } catch (err) {
      setResult({
        response: `Error: ${err.message}`,
        chunks_used: 0,
        sources: [],
        model,
      });
    } finally {
      setLoading(false);
      setTimeout(() => setMood('normal'), 1500);
    }
  };

  const handleUsePromptFromCraftD = (built) => {
    setPrompt(built);
    setStyle('craftd');
    setTab('lab');
  };

  const handleReuseFromHistory = (entry) => {
    setPrompt(entry.prompt || '');
    if (entry.style) setStyle(entry.style);
    if (entry.params) setParams({ ...DEFAULT_PARAMS, ...entry.params });
    if (entry.model) setModel(entry.model);
    setTab('lab');
  };

  const handleApplyPreset = (preset) => {
    setStyle(preset.style);
    setParams({ ...DEFAULT_PARAMS, ...preset.params });
    setActivePresetId(preset.id);
  };

  const handleResetToBase = () => {
    setStyle('natural');
    setParams(DEFAULT_PARAMS);
    setActivePresetId(null);
  };

  return (
    <div className="page">
      <div className="container">
        <Mascot mood={mood} />

        <header className="page-header">
          <h1>The Answer Factory</h1>
          <p>
            Laboratorio de prompts y parámetros para docentes — Sesión 1 del
            crash course de IA aplicada a la práctica docente.
          </p>
        </header>

        <nav className="tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`tab ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>

        {tab === 'lab' && (
          <>
            <PresetBar
              onApply={handleApplyPreset}
              onReset={handleResetToBase}
              activeId={activePresetId}
            />

            <div className="main-grid">
              <div className="card">
                <h2 className="card-title">Entrada y controles</h2>

                <label className="field-label">Pregunta o prompt</label>
                <textarea
                  className="prompt-input"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Escribe aquí lo que quieres pedirle al modelo…"
                />

                <StyleSelector
                  value={style}
                  onChange={(v) => {
                    setStyle(v);
                    setActivePresetId(null);
                  }}
                />

                <label className="field-label">Modelo</label>
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

                <label className="field-label">
                  Secuencias de parada (separadas por coma)
                </label>
                <input
                  className="upload-input"
                  value={stopSequences}
                  onChange={(e) => setStopSequences(e.target.value)}
                  placeholder="ej: FIN, ###, ALTO"
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
                  <span className="toggle-state">
                    {useRag ? 'Activado' : 'Desactivado'}
                  </span>
                </div>

                <div className="toggle-row">
                  <label title="Solo OpenAI. Devuelve las palabras alternativas que el modelo consideró.">
                    Mostrar logprobs (alternativas que la IA consideró)
                  </label>
                  <button
                    className={`toggle ${returnLogprobs ? 'on' : ''}`}
                    onClick={() => setReturnLogprobs((v) => !v)}
                    aria-pressed={returnLogprobs}
                  >
                    <span />
                  </button>
                  <span className="toggle-state">
                    {returnLogprobs ? 'Activado' : 'Desactivado'}
                  </span>
                </div>

                <button
                  className="button primary block"
                  onClick={handleGenerate}
                  disabled={loading}
                >
                  {loading ? 'Generando…' : 'Generar respuesta'}
                </button>
              </div>

              <ParametersCard
                params={params}
                onChange={(p) => {
                  setParams(p);
                  setActivePresetId(null);
                }}
              />
            </div>

            <RagPanel status={ragStatus} onChange={refreshRag} />

            <ResponsePanel
              result={result}
              useRag={useRag}
              loading={loading}
              ragHasContent={ragStatus?.total_chunks > 0}
              judgeModel={model}
            />
          </>
        )}

        {tab === 'craftd' && (
          <CraftDWizard onUsePrompt={handleUsePromptFromCraftD} />
        )}

        {tab === 'compare' && (
          <CompareMode
            models={models}
            defaultModel={defaultModel}
            onHistoryChange={() => setHistoryTick((n) => n + 1)}
          />
        )}

        {tab === 'tokens' && (
          <TokenVisualizer models={models} defaultModel={defaultModel} />
        )}

        {tab === 'history' && (
          <HistoryPanel
            refreshKey={historyTick}
            onReuse={handleReuseFromHistory}
          />
        )}

        <footer className="page-footer">
          Autoría · Mauricio Friedman ·{' '}
          <em>«La IA no es el fin, sino un medio. El ser humano es el fin.»</em>
        </footer>
      </div>
    </div>
  );
}
