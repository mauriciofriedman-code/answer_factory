import Slider from './Slider.jsx';
import { PARAM_DESCRIPTIONS } from '../config.js';

export default function ParametersCard({ params, onChange }) {
  const update = (k) => (v) => onChange({ ...params, [k]: v });

  return (
    <div className="card">
      <h2 className="card-title">Parámetros del modelo</h2>

      <Slider
        label="Temperature"
        min={0}
        max={2}
        step={0.1}
        value={params.temperature}
        onChange={update('temperature')}
        description={PARAM_DESCRIPTIONS.temperature}
        format={(v) => v.toFixed(1)}
      />
      <Slider
        label="Top-P"
        min={0}
        max={1}
        step={0.01}
        value={params.top_p}
        onChange={update('top_p')}
        description={PARAM_DESCRIPTIONS.top_p}
        format={(v) => v.toFixed(2)}
      />
      <Slider
        label="Frequency penalty"
        min={-2}
        max={2}
        step={0.1}
        value={params.frequency_penalty}
        onChange={update('frequency_penalty')}
        description={PARAM_DESCRIPTIONS.frequency_penalty}
        format={(v) => v.toFixed(1)}
      />
      <Slider
        label="Presence penalty"
        min={-2}
        max={2}
        step={0.1}
        value={params.presence_penalty}
        onChange={update('presence_penalty')}
        description={PARAM_DESCRIPTIONS.presence_penalty}
        format={(v) => v.toFixed(1)}
      />
      <Slider
        label="Máximo de tokens"
        min={50}
        max={2000}
        step={50}
        value={params.max_tokens}
        onChange={update('max_tokens')}
        description={PARAM_DESCRIPTIONS.max_tokens}
      />
    </div>
  );
}
