import { PRESETS } from '../config.js';

export default function PresetBar({ onApply, activeId }) {
  return (
    <div className="preset-bar">
      <span className="preset-label">Presets pedagógicos:</span>
      <div className="preset-buttons">
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
