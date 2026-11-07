import { readCsrfToken } from './csrf'
import { ApiError } from './errors'

const DEFAULT_TIMEOUT_MS = 60000

/** Methods whose requests change something and therefore need the CSRF mark. */
const STATE_CHANGING = new Set(['POST', 'PUT', 'PATCH', 'DELETE'])

/** Pull a readable message out of a FastAPI error body. */
function extractDetail(body) {
  if (!body || typeof body !== 'object') return null
  if (typeof body.detail === 'string') return body.detail
  if (Array.isArray(body.detail)) {
    const [first] = body.detail
    return typeof first?.msg === 'string' ? first.msg : null
  }
  return null
}

/** The machine-readable reason the backend attaches to refusals of access. */
function extractCode(body) {
  if (!body || typeof body !== 'object') return null
  return typeof body.code === 'string' ? body.code : null
}

async function readBody(response) {
  const text = await response.text()
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return text
  }
}

/**
 * Minimal JSON client over `fetch`.
 *
 * It owns the concerns every call shares: URL composition, serialisation, the
 * cookies that carry the session, the CSRF mark, the timeout, and the
 * translation of transport failures into {@link ApiError}. Renewing the session
 * is deliberately not part of it: the client asks the injected handler to renew
 * and retries once if the handler says it worked.
 */
export class HttpClient {
  #baseUrl
  #timeoutMs
  #csrf

  onUnauthorized = null

  constructor({ baseUrl = '', timeoutMs = DEFAULT_TIMEOUT_MS, csrf = {} } = {}) {
    this.#baseUrl = baseUrl.replace(/\/+$/, '')
    this.#timeoutMs = timeoutMs
    this.#csrf = csrf
  }

  get(path, options) {
    return this.#request('GET', path, options)
  }

  post(path, payload, options) {
    return this.#request('POST', path, { ...options, payload })
  }

  /** POST with form fields, as OAuth 2 asks for: `username` and `password`. */
  postForm(path, fields, options) {
    return this.#request('POST', path, { ...options, form: fields })
  }

  patch(path, payload, options) {
    return this.#request('PATCH', path, { ...options, payload })
  }

  delete(path, options) {
    return this.#request('DELETE', path, options)
  }

  async #request(method, path, { payload, form, retry = true } = {}) {
    let attempt = await this.#send(method, path, { payload, form })

    if (attempt.response.status === 401 && retry && this.onUnauthorized) {
      // One retry, and only if the handler managed to renew: otherwise an expired
      // cookie would turn every call into an endless loop of refreshes.
      const renewed = await this.onUnauthorized()
      if (renewed) {
        attempt = await this.#send(method, path, { payload, form })
      }
    }

    const { response, body } = attempt
    if (!response.ok) {
      throw new ApiError(`HTTP ${response.status}`, {
        status: response.status,
        detail: extractDetail(body),
        code: extractCode(body),
      })
    }

    return body
  }

  async #send(method, path, { payload, form }) {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), this.#timeoutMs)
    const hasBody = payload !== undefined || form !== undefined

    try {
      const response = await fetch(`${this.#baseUrl}${path}`, {
        method,
        // The session lives in httpOnly cookies, so they must travel with every
        // call — including the cross-origin one to the host that serves the API.
        credentials: 'include',
        headers: this.#headers(method, hasBody, form !== undefined),
        body: form !== undefined ? form : hasBody ? JSON.stringify(payload) : undefined,
        signal: controller.signal,
      })
      return { response, body: await readBody(response) }
    } catch (cause) {
      if (cause instanceof ApiError) throw cause
      if (cause?.name === 'AbortError') {
        throw new ApiError('Request timed out', { status: 408, cause })
      }
      throw new ApiError('Network request failed', { status: 0, cause })
    } finally {
      clearTimeout(timer)
    }
  }

  #headers(method, hasBody, isForm) {
    const headers = { Accept: 'application/json' }

    if (isForm) {
      headers['Content-Type'] = 'application/x-www-form-urlencoded'
    } else if (hasBody) {
      headers['Content-Type'] = 'application/json'
    }

    if (STATE_CHANGING.has(method) && this.#csrf.headerName) {
      const token = readCsrfToken(this.#csrf.cookieName)
      if (token) headers[this.#csrf.headerName] = token
    }

    return headers
  }
}
