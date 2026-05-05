// Identificador de sesión persistido en localStorage. Reemplaza la cookie
// cross-site que el browser bloqueaba en incógnito y bajo ITP/ETP.
const STORAGE_KEY = 'answer_factory_session_id';

const generate = () => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID().replace(/-/g, '');
  }
  // Fallback: 22 chars base36 sobre Math.random.
  return Array.from({ length: 4 }, () =>
    Math.random().toString(36).slice(2, 10)
  ).join('');
};

export const getSessionId = () => {
  try {
    let sid = localStorage.getItem(STORAGE_KEY);
    if (!sid) {
      sid = generate();
      localStorage.setItem(STORAGE_KEY, sid);
    }
    return sid;
  } catch {
    // Modo súper restrictivo (storage bloqueado): mantenemos un id en memoria.
    if (!globalThis.__afSidMemory) globalThis.__afSidMemory = generate();
    return globalThis.__afSidMemory;
  }
};

export const setSessionId = (sid) => {
  if (!sid || typeof sid !== 'string') return;
  try {
    localStorage.setItem(STORAGE_KEY, sid);
  } catch {
    globalThis.__afSidMemory = sid;
  }
};

export const resetSessionId = () => {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    globalThis.__afSidMemory = undefined;
  }
};
