// Historial de iteraciones — persiste en localStorage por navegador.
const KEY = 'answer_factory_history_v1';
const MAX = 50;

export function loadHistory() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveEntry(entry) {
  const list = loadHistory();
  const next = [
    {
      id: crypto.randomUUID
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      created_at: new Date().toISOString(),
      ...entry,
    },
    ...list,
  ].slice(0, MAX);
  localStorage.setItem(KEY, JSON.stringify(next));
  return next;
}

export function deleteEntry(id) {
  const next = loadHistory().filter((e) => e.id !== id);
  localStorage.setItem(KEY, JSON.stringify(next));
  return next;
}

export function clearHistory() {
  localStorage.removeItem(KEY);
  return [];
}

export function entryToMarkdown(entry) {
  const lines = [];
  lines.push(`# Iteración — ${formatDate(entry.created_at)}`);
  lines.push('');
  if (entry.kind) lines.push(`**Modo:** ${entry.kind}  `);
  lines.push(`**Modelo:** ${entry.model || '—'}  `);
  lines.push(`**Estilo:** ${entry.style || '—'}  `);
  if (entry.params) {
    lines.push(
      `**Parámetros:** temp ${entry.params.temperature} · top_p ${entry.params.top_p} · ` +
        `freq ${entry.params.frequency_penalty} · pres ${entry.params.presence_penalty} · ` +
        `max ${entry.params.max_tokens}  `
    );
  }
  if (entry.use_rag) lines.push(`**RAG:** activado · ${entry.chunks_used || 0} fragmentos  `);

  lines.push('');
  lines.push('## Prompt');
  lines.push('');
  lines.push('```');
  lines.push(entry.prompt || '');
  lines.push('```');
  lines.push('');
  lines.push('## Respuesta');
  lines.push('');
  lines.push(entry.response || '');

  if (entry.sources?.length) {
    lines.push('');
    lines.push('## Fuentes citadas');
    for (const s of entry.sources) {
      lines.push(
        `- ${s.title} — ${s.author} · fragmento ${s.page}` +
          (s.similarity != null ? ` · sim ${s.similarity}` : '')
      );
    }
  }

  if (entry.notes) {
    lines.push('');
    lines.push('## Mi reflexión');
    lines.push('');
    lines.push(entry.notes);
  }

  return lines.join('\n');
}

export function downloadMarkdown(filename, markdown) {
  const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function formatDate(iso) {
  try {
    return new Date(iso).toLocaleString('es-MX', {
      dateStyle: 'medium',
      timeStyle: 'short',
    });
  } catch {
    return iso;
  }
}
