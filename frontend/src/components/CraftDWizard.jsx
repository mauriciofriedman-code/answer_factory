import { useMemo, useState } from 'react';

const FIELDS = [
  {
    id: 'context',
    label: 'C — Contexto',
    placeholder: 'Estoy planeando una clase de fracciones para 4° de primaria, en un grupo de 28 alumnos…',
    hint: 'Dónde, cuándo, con quién, qué precede esta tarea.',
  },
  {
    id: 'role',
    label: 'R — Rol',
    placeholder: 'Actúa como diseñador instruccional con experiencia en primaria…',
    hint: 'Qué expertise debe asumir el modelo.',
  },
  {
    id: 'audience',
    label: 'A — Audiencia',
    placeholder: 'La salida la voy a leer yo (docente). Su nivel: secundaria pública, lectoescritura básica…',
    hint: 'Quién leerá la salida y qué nivel cognitivo tiene.',
  },
  {
    id: 'format',
    label: 'F — Formato',
    placeholder: 'Devuélvelo como tabla con columnas: Etapa, Tiempo (min), Acción del docente, Acción del alumno…',
    hint: 'Estructura concreta esperada (lista, tabla, JSON, prosa, etc.).',
  },
  {
    id: 'tone',
    label: 'T — Tono',
    placeholder: 'Tono profesional, claro, no condescendiente con el docente.',
    hint: 'Registro y voz.',
  },
  {
    id: 'data',
    label: 'D — Datos / Restricciones',
    placeholder: 'Tiempo total: 50 min. Sin proyector. Material disponible: gises de colores, hojas blancas. No usar tarea casa.',
    hint: 'Datos duros, restricciones, prohibiciones explícitas.',
  },
];

const HEADERS = {
  context: 'CONTEXTO',
  role: 'ROL',
  audience: 'AUDIENCIA',
  format: 'FORMATO',
  tone: 'TONO',
  data: 'DATOS / RESTRICCIONES',
};

export function buildCraftDPrompt(values, userRequest) {
  const blocks = [];
  for (const f of FIELDS) {
    const v = (values[f.id] || '').trim();
    if (v) blocks.push(`## ${HEADERS[f.id]}\n${v}`);
  }
  if (userRequest && userRequest.trim()) {
    blocks.push(`## TAREA CONCRETA\n${userRequest.trim()}`);
  }
  return blocks.join('\n\n');
}

export default function CraftDWizard({ onUsePrompt }) {
  const [values, setValues] = useState({});
  const [task, setTask] = useState('');

  const update = (id) => (e) => setValues((v) => ({ ...v, [id]: e.target.value }));

  const built = useMemo(() => buildCraftDPrompt(values, task), [values, task]);
  const filledCount = FIELDS.filter((f) => (values[f.id] || '').trim()).length;

  return (
    <div className="card">
      <div className="card-header-row">
        <h2 className="card-title">Wizard CRAFT-D</h2>
        <span className="badge muted">{filledCount}/6 campos completados</span>
      </div>
      <p className="hint-block">
        Completa los seis campos. El prompt resultante se construye en vivo abajo
        y puedes enviarlo al laboratorio principal con un clic.
      </p>

      <div className="craftd-grid">
        {FIELDS.map((f) => (
          <div key={f.id} className="craftd-field">
            <label className="field-label">{f.label}</label>
            <textarea
              className="upload-input"
              rows={3}
              value={values[f.id] || ''}
              onChange={update(f.id)}
              placeholder={f.placeholder}
            />
            <div className="field-hint">{f.hint}</div>
          </div>
        ))}
      </div>

      <label className="field-label" style={{ marginTop: 16 }}>
        Tarea concreta (qué quieres que produzca el modelo)
      </label>
      <textarea
        className="upload-input"
        rows={2}
        value={task}
        onChange={(e) => setTask(e.target.value)}
        placeholder="Diséñame una secuencia de inicio (10 min) que detone la pregunta esencial."
      />

      <h3 className="subheading">Prompt construido</h3>
      <pre className="prompt-preview">{built || '— completa los campos arriba —'}</pre>

      <div className="row-gap">
        <button
          className="button primary"
          onClick={() => onUsePrompt?.(built)}
          disabled={!built}
        >
          Enviar al laboratorio
        </button>
        <button
          className="button"
          onClick={() => {
            navigator.clipboard?.writeText(built);
          }}
          disabled={!built}
        >
          Copiar al portapapeles
        </button>
      </div>
    </div>
  );
}
