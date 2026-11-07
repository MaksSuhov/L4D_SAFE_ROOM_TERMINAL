import { fileURLToPath, URL } from 'node:url'

import tailwindcss from '@tailwindcss/vite'
import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

/**
 * The dev server proxies the API and its documentation to the FastAPI backend.
 *
 * Everything the browser talks to goes through this origin — `/api` for the
 * endpoints, `/docs` and `/openapi.json` for the reference — so the session
 * cookie behaves exactly as it will in production, where both are served from
 * one host. `src/api/config.js` therefore defaults to same-origin requests.
 */
const BACKEND = 'http://localhost:8000'

export default defineConfig({
  plugins: [vue(), tailwindcss()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true },
      '/docs': { target: BACKEND, changeOrigin: true },
      '/openapi.json': { target: BACKEND, changeOrigin: true },
    },
  },
})
