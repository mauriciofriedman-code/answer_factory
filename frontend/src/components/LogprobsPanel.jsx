export default function LogprobsPanel({ logprobs }) {
  if (!logprobs?.length) return null;

  return (
    <div className="logprobs-block">
      <h3 className="subheading">
        Distribución de probabilidades (predice, no sabe)
      </h3>
      <p className="hint-block">
        Para cada token elegido, ves las {Math.max(...logprobs.map(l => l.top_alternatives?.length || 0))}{' '}
        palabras alternativas que el modelo consideró y su probabilidad. Pasa el cursor
        sobre cada token para ver el ranking completo.
      </p>

      <div className="logprob-line">
        {logprobs.map((step, i) => (
          <TokenWithAlts key={i} step={step} />
        ))}
      </div>
    </div>
  );
}

function TokenWithAlts({ step }) {
  const chosenProb = Math.exp(step.logprob);
  const alts = step.top_alternatives || [];
  const tooltip = alts
    .map((a) => {
      const p = Math.exp(a.logprob);
      const marker = a.token === step.token ? '●' : ' ';
      return `${marker} ${displayToken(a.token)}  ${(p * 100).toFixed(1)}%`;
    })
    .join('\n');

  // Color saturation by chosen probability: very probable → vivid, ambiguous → pale
  const saturation = Math.max(0.1, Math.min(1, chosenProb));
  const bg = `rgba(102, 126, 234, ${0.15 + saturation * 0.5})`;

  return (
    <span
      className="logprob-chip"
      title={tooltip}
      style={{ background: bg }}
    >
      {displayToken(step.token)}
      <span className="logprob-pct">{(chosenProb * 100).toFixed(0)}%</span>
    </span>
  );
}

function displayToken(t) {
  if (!t) return '';
  if (t === '\n') return '↵';
  return t.replace(/ /g, '·').replace(/\n/g, '↵');
}
