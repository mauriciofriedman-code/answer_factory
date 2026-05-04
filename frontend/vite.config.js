import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
  },
  preview: {
    host: true,
    // Permite cualquier subdominio en Render y dominios personalizados.
    // Para restringir a un solo host, reemplaza `true` por la lista exacta:
    // allowedHosts: ['answer-factory-frontend.onrender.com']
    allowedHosts: true,
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
});
