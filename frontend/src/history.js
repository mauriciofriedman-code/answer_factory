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

// ──────────────────────────────────────────────────────────────────────
// Renderizado a HTML/PDF
// ──────────────────────────────────────────────────────────────────────

const escapeHtml = (s) =>
  String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

// Mini-parser de Markdown: cubre lo que la IA suele devolver
// (headings, bold, italic, listas, tablas pipe). No pretende ser CommonMark.
function markdownToHtml(src) {
  if (!src) return '';
  const text = String(src);
  const lines = text.split(/\r?\n/);
  const out = [];
  let i = 0;

  const flushParagraph = (buf) => {
    if (!buf.length) return;
    const joined = buf.join(' ');
    out.push(`<p>${inlineFormat(joined)}</p>`);
    buf.length = 0;
  };

  const buf = [];

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    // Línea vacía → cierra párrafo
    if (!trimmed) {
      flushParagraph(buf);
      i++;
      continue;
    }

    // Tabla pipe: detectar header + separador
    if (
      trimmed.includes('|') &&
      i + 1 < lines.length &&
      /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])
    ) {
      flushParagraph(buf);
      const headerCells = splitPipeRow(trimmed);
      i += 2; // saltamos header y separador
      const bodyRows = [];
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
        bodyRows.push(splitPipeRow(lines[i].trim()));
        i++;
      }
      out.push(renderTable(headerCells, bodyRows));
      continue;
    }

    // Headings
    const h = /^(#{1,6})\s+(.*)$/.exec(trimmed);
    if (h) {
      flushParagraph(buf);
      const level = Math.min(h[1].length, 4); // limitamos a h4 para que el PDF no salte mucho
      out.push(`<h${level}>${inlineFormat(h[2])}</h${level}>`);
      i++;
      continue;
    }

    // Bullet list
    if (/^[-*•]\s+/.test(trimmed)) {
      flushParagraph(buf);
      const items = [];
      while (i < lines.length && /^[-*•]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*•]\s+/, ''));
        i++;
      }
      out.push(
        '<ul>' + items.map((it) => `<li>${inlineFormat(it)}</li>`).join('') + '</ul>'
      );
      continue;
    }

    // Numbered list
    if (/^\d+\.\s+/.test(trimmed)) {
      flushParagraph(buf);
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ''));
        i++;
      }
      out.push(
        '<ol>' + items.map((it) => `<li>${inlineFormat(it)}</li>`).join('') + '</ol>'
      );
      continue;
    }

    // Línea normal → acumula
    buf.push(trimmed);
    i++;
  }

  flushParagraph(buf);
  return out.join('\n');
}

function splitPipeRow(row) {
  let s = row.trim();
  if (s.startsWith('|')) s = s.slice(1);
  if (s.endsWith('|')) s = s.slice(0, -1);
  return s.split('|').map((c) => c.trim());
}

function renderTable(header, rows) {
  const ths = header.map((c) => `<th>${inlineFormat(c)}</th>`).join('');
  const trs = rows
    .map(
      (r) =>
        '<tr>' +
        r.map((c) => `<td>${inlineFormat(c)}</td>`).join('') +
        '</tr>'
    )
    .join('');
  return `<table><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;
}

function inlineFormat(s) {
  // Escapamos primero, luego aplicamos formato — los marcadores ya son seguros.
  let t = escapeHtml(s);
  t = t.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  t = t.replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
  return t;
}

// ──────────────────────────────────────────────────────────────────────
// Plantilla del PDF
// ──────────────────────────────────────────────────────────────────────

const PRINT_CSS = `
  @page { size: A4; margin: 22mm 18mm; }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    color: #1f2933;
    line-height: 1.55;
    font-size: 11.5pt;
    margin: 0;
  }
  h1 { font-size: 20pt; margin: 0 0 4pt; color: #0f172a; }
  h2 { font-size: 14pt; margin: 18pt 0 6pt; color: #1e3a8a; border-bottom: 1px solid #cbd5e1; padding-bottom: 3pt; }
  h3 { font-size: 12pt; margin: 12pt 0 4pt; color: #1e3a8a; }
  h4 { font-size: 11pt; margin: 10pt 0 4pt; color: #334155; }
  p { margin: 0 0 8pt; }
  ul, ol { margin: 0 0 8pt 18pt; padding: 0; }
  li { margin: 0 0 3pt; }
  strong { color: #0f172a; }
  code {
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 0.92em;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
  }
  pre.prompt {
    background: #f8fafc;
    border-left: 3px solid #6366f1;
    padding: 8pt 10pt;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-size: 9.5pt;
    color: #334155;
    margin: 0 0 12pt;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 6pt 0 12pt;
    font-size: 10pt;
    page-break-inside: avoid;
  }
  th, td {
    border: 1px solid #cbd5e1;
    padding: 4pt 6pt;
    text-align: left;
    vertical-align: top;
  }
  th { background: #eef2ff; color: #1e3a8a; }
  .meta {
    font-size: 9.5pt;
    color: #64748b;
    margin: 0 0 14pt;
  }
  .meta .pill {
    display: inline-block;
    background: #f1f5f9;
    padding: 1pt 6pt;
    border-radius: 10pt;
    margin: 0 4pt 4pt 0;
  }
  .meta .pill.rag { background: #dcfce7; color: #166534; }
  .sources li { font-size: 10pt; color: #475569; }
  .sources blockquote {
    margin: 3pt 0 6pt 0;
    padding: 4pt 8pt;
    border-left: 2px solid #6366f1;
    background: #f8fafc;
    font-style: italic;
    font-size: 9.5pt;
    color: #334155;
  }
  .reflection {
    background: #fefce8;
    border-left: 3px solid #ca8a04;
    padding: 8pt 10pt;
    margin: 6pt 0 0;
    white-space: pre-wrap;
  }
  .footer {
    margin-top: 18pt;
    padding-top: 6pt;
    border-top: 1px solid #e2e8f0;
    font-size: 9pt;
    color: #94a3b8;
    font-style: italic;
  }
  .entry { page-break-after: always; }
  .entry:last-child { page-break-after: auto; }
`;

export function entryToHtml(entry) {
  const dateStr = formatDate(entry.created_at);
  const pills = [];
  if (entry.kind) pills.push(`<span class="pill">${escapeHtml(entry.kind)}</span>`);
  if (entry.model) pills.push(`<span class="pill">${escapeHtml(entry.model)}</span>`);
  if (entry.style) pills.push(`<span class="pill">estilo: ${escapeHtml(entry.style)}</span>`);
  if (entry.params) {
    pills.push(
      `<span class="pill">temp ${entry.params.temperature} · top_p ${entry.params.top_p} · max ${entry.params.max_tokens}</span>`
    );
  }
  if (entry.use_rag) {
    pills.push(`<span class="pill rag">RAG · ${entry.chunks_used || 0} fragmentos</span>`);
  }

  const sourcesHtml =
    entry.sources?.length
      ? `<h2>Fuentes citadas</h2>
         <ul class="sources">
           ${entry.sources
             .map(
               (s) => `
                 <li>
                   <strong>${escapeHtml(s.title || 'Sin título')}</strong>${
                 s.author ? ` — ${escapeHtml(s.author)}` : ''
               } · ${escapeHtml(s.location || `fragmento ${s.page ?? '?'}`)}${
                 s.similarity != null ? ` · sim ${s.similarity}` : ''
               }
                   ${
                     s.snippet
                       ? `<blockquote>«${escapeHtml(s.snippet)}»</blockquote>`
                       : ''
                   }
                 </li>`
             )
             .join('')}
         </ul>`
      : '';

  const notesHtml = entry.notes
    ? `<h2>Mi reflexión</h2><div class="reflection">${escapeHtml(entry.notes)}</div>`
    : '';

  return `
    <article class="entry">
      <h1>Iteración — ${escapeHtml(dateStr)}</h1>
      <div class="meta">${pills.join('')}</div>

      <h2>Prompt</h2>
      <pre class="prompt">${escapeHtml(entry.prompt || '')}</pre>

      <h2>Respuesta</h2>
      ${markdownToHtml(entry.response || '')}

      ${sourcesHtml}
      ${notesHtml}

      <div class="footer">The Answer Factory · Mauricio Friedman</div>
    </article>
  `;
}

function buildPrintDocument(title, bodyHtml) {
  return `<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>${escapeHtml(title)}</title>
  <style>${PRINT_CSS}</style>
</head>
<body>${bodyHtml}</body>
</html>`;
}

// Abre un iframe oculto, escribe el HTML, dispara print().
// El navegador ofrece "Guardar como PDF" en el diálogo de impresión.
export function exportEntriesToPdf(filename, entries) {
  const list = Array.isArray(entries) ? entries : [entries];
  const body = list.map(entryToHtml).join('\n');
  const html = buildPrintDocument(filename, body);

  const iframe = document.createElement('iframe');
  iframe.style.position = 'fixed';
  iframe.style.right = '0';
  iframe.style.bottom = '0';
  iframe.style.width = '0';
  iframe.style.height = '0';
  iframe.style.border = '0';
  iframe.setAttribute('aria-hidden', 'true');

  document.body.appendChild(iframe);

  // Sugerimos al diálogo el nombre de archivo.
  try {
    iframe.contentWindow.document.title = filename;
  } catch (_) {}

  const cleanup = () => {
    setTimeout(() => {
      if (iframe.parentNode) iframe.parentNode.removeChild(iframe);
    }, 1000);
  };

  iframe.onload = () => {
    try {
      iframe.contentWindow.document.title = filename;
      iframe.contentWindow.focus();
      iframe.contentWindow.print();
    } catch (err) {
      console.error('Error al imprimir', err);
    }
    // El usuario cierra el diálogo; limpiamos al rato.
    iframe.contentWindow.onafterprint = cleanup;
    setTimeout(cleanup, 60_000);
  };

  const doc = iframe.contentWindow.document;
  doc.open();
  doc.write(html);
  doc.close();
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
