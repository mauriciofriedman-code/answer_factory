export default function Slider({ label, value, min, max, step, onChange, description, format }) {
  const display = format ? format(value) : value;
  return (
    <div className="slider-container">
      <div className="slider-label">
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
      {description && <div className="slider-description">{description}</div>}
    </div>
  );
}
