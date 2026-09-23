export const LOCKED_NOTE = 'Este modelo no deja mover esta perilla.';

export default function Slider({ label, value, min, max, step, onChange, description, format, locked }) {
  const display = format ? format(value) : value;
  return (
    <div className={`slider-container ${locked ? 'locked' : ''}`}>
      <div className="slider-label">
        <span>{label}</span>
        <span className="slider-value">{locked ? 'bloqueada' : display}</span>
      </div>
      <input
        type="range"
        className="slider"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={locked}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      {locked && <div className="slider-description locked-note">{LOCKED_NOTE}</div>}
      {description && <div className="slider-description">{description}</div>}
    </div>
  );
}
