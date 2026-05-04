import { STYLE_GROUPS } from '../config.js';

export default function StyleSelector({ value, onChange }) {
  return (
    <div className="style-selector-wrap">
      <label className="field-label">Estilo de respuesta</label>
      <select
        className="style-selector"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {STYLE_GROUPS.map((group) => (
          <optgroup key={group.id} label={group.label}>
            {group.options.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
      <div className="style-hint">
        {STYLE_GROUPS.find((g) =>
          g.options.some((o) => o.value === value)
        )?.hint}
      </div>
    </div>
  );
}
