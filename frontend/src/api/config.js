/**
 * Runtime configuration of the API client.
 *
 * `VITE_API_BASE_URL` defaults to an empty string, which makes every request
 * same-origin. That is what the cookie session wants: the Vite dev proxy (see
 * vite.config.js) forwards `/api`, `/docs` and `/openapi.json` to the backend,
 * so the browser treats the session exactly as it would in production. Pointing
 * the client straight at `http://localhost:8000` also works, but only while the
 * frontend is served from the same *site*: a page on `127.0.0.1` talking to
 * `localhost` is a different site, and the browser would drop the cookies.
 */
const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL

export const apiBaseUrl =
  configuredBaseUrl === undefined ? '' : configuredBaseUrl.trim().replace(/\/+$/, '')

export const apiPrefix = (import.meta.env.VITE_API_PREFIX ?? '/api/v1').trim()

export const apiDocsUrl = `${apiBaseUrl}/docs`

/** Names the client must agree on with the backend (`AUTH_CSRF_*` in .env). */
export const csrfCookieName = (import.meta.env.VITE_CSRF_COOKIE_NAME ?? 'l4d_csrf').trim()

export const csrfHeaderName = (import.meta.env.VITE_CSRF_HEADER_NAME ?? 'X-CSRF-Token').trim()
