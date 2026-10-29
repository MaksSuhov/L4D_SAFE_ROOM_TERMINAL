/**
 * Runtime configuration of the API client.
 *
 * `VITE_API_BASE_URL` defaults to the backend documented in the repository
 * README; setting it to an empty string switches the app to same-origin
 * requests, which the Vite dev proxy forwards to the backend.
 */
const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL

export const apiBaseUrl =
  configuredBaseUrl === undefined ? 'http://localhost:8000' : configuredBaseUrl.trim()

export const apiPrefix = (import.meta.env.VITE_API_PREFIX ?? '/api/v1').trim()

export const apiDocsUrl = `${apiBaseUrl}/docs`
