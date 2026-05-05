import { PRESETS } from '../config.js';

export default function PresetBar({ onApply, onReset, activeId }) {
  return (
    <div className="preset-bar">
      <span className="preset-label">Presets pedagógicos:</span>
      <div className="preset-buttons">
        <button
          className={`preset-btn ${activeId === null ? 'active' : ''}`}
          title="Volver al estilo natural y a los parámetros por defecto del laboratorio."
          onClick={onReset}
        >
          ↺ Calibración base
        </button>
        {PRESETS.map((p) => (
          <button
            key={p.id}
            className={`preset-btn ${activeId === p.id ? 'active' : ''}`}
            title={p.description}
            onClick={() => onApply(p)}
          >
            {p.label}
          </button>
        ))}
      </div>
    </div>
  );
}
