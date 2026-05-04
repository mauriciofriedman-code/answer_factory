// API base configurable: prod via VITE_API_BASE, dev cae a localhost.
const fromEnv = import.meta.env.VITE_API_BASE;
const isLocal =
  typeof window !== 'undefined' &&
  (window.location.hostname === 'localhost' ||
    window.location.hostname === '127.0.0.1');

export const API_BASE = fromEnv || (isLocal ? 'http://localhost:8000' : '');

export const STYLE_GROUPS = [
  {
    id: 'academic',
    label: 'Académicos',
    hint: 'Configuraciones serias para trabajo docente.',
    options: [
      { value: 'natural', label: '🎯 Natural — sin ingeniería de prompt' },
      { value: 'scientific', label: '🔬 Científico — rigor y terminología precisa' },
      { value: 'friendly_teacher', label: '👨‍🏫 Profesor amigable — claro y cálido' },
      { value: 'craftd', label: '🧱 CRAFT-D — Contexto · Rol · Audiencia · Formato · Tono · Datos' },
    ],
  },
  {
    id: 'fun',
    label: 'Para pasar un buen rato',
    hint: 'Complemento lúdico — sirve para experimentar el efecto del system prompt.',
    options: [
      { value: 'child_5yo', label: '👶 Niño de 5 años' },
      { value: 'shakespeare', label: '🎭 Shakespeare dramático' },
      { value: 'wise_grandmother', label: '👵 Abuela sabia' },
      { value: 'confused_robot', label: '🤖 Robot confundido' },
      { value: 'philosopher', label: '🤔 Filósofo existencial' },
      { value: 'salesperson', label: '💼 Vendedor entusiasta' },
      { value: 'storyteller', label: '📚 Narrador de cuentos' },
      { value: 'comedian', label: '😄 Comediante' },
      { value: 'pirate', label: '🏴‍☠️ Pirata' },
      { value: 'chef', label: '👨‍🍳 Chef explicando recetas' },
    ],
  },
];

export const PRESETS = [
  {
    id: 'craftd_balanceado',
    label: '🧱 CRAFT-D · balanceado',
    description: 'Combinación pedagógica recomendada para iterar planeación y rúbricas.',
    style: 'craftd',
    params: {
      temperature: 0.5,
      top_p: 0.9,
      frequency_penalty: 0.0,
      presence_penalty: 0.0,
      max_tokens: 700,
    },
  },
  {
    id: 'conservador',
    label: '🎯 Conservador',
    description: 'Salida muy enfocada y predecible. Ideal para tareas factuales.',
    style: 'natural',
    params: {
      temperature: 0.2,
      top_p: 0.9,
      frequency_penalty: 0.0,
      presence_penalty: 0.0,
      max_tokens: 500,
    },
  },
  {
    id: 'profesor',
    label: '👨‍🏫 Profesor amigable',
    description: 'Tono cálido, ejemplos concretos, pregunta de cierre.',
    style: 'friendly_teacher',
    params: {
      temperature: 0.5,
      top_p: 0.9,
      frequency_penalty: 0.0,
      presence_penalty: 0.0,
      max_tokens: 600,
    },
  },
  {
    id: 'creativo',
    label: '✨ Creativo',
    description: 'Más variedad léxica y de enfoque. Ideal para lluvia de ideas.',
    style: 'natural',
    params: {
      temperature: 0.9,
      top_p: 1.0,
      frequency_penalty: 0.0,
      presence_penalty: 0.3,
      max_tokens: 600,
    },
  },
];

export const PARAM_DESCRIPTIONS = {
  temperature:
    'Controla la creatividad. 0 = enfocado y predecible. 2 = más creativo y aleatorio.',
  top_p:
    'Restringe el universo de palabras consideradas. 0.1 = solo las muy probables. 1.0 = todas las opciones.',
  frequency_penalty:
    'Penaliza repetir palabras. Negativo permite repetición; positivo la evita.',
  presence_penalty:
    'Empuja al modelo a hablar de temas nuevos. Positivo introduce más variedad de contenido.',
  max_tokens: 'Límite máximo de longitud de la respuesta (≈ palabras).',
};
