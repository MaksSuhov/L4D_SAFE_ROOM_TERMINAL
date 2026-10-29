import { ApiError } from './errors'

const DEFAULT_TIMEOUT_MS = 60000

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

/**
 * Minimal JSON client over `fetch`.
 *
 * It owns the concerns every call shares: URL composition, serialisation, the
 * timeout, and the translation of transport failures into {@link ApiError}.
 */
export class HttpClient {
  #baseUrl
  #timeoutMs

  constructor({ baseUrl = '', timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
    this.#baseUrl = baseUrl.replace(/\/+$/, '')
    this.#timeoutMs = timeoutMs
  }

  get(path) {
    return this.#request('GET', path)
  }

  post(path, payload) {
    return this.#request('POST', path, payload)
  }

  async #request(method, path, payload) {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), this.#timeoutMs)
    const hasBody = payload !== undefined

    try {
      const response = await fetch(`${this.#baseUrl}${path}`, {
        method,
        headers: {
          Accept: 'application/json',
          ...(hasBody ? { 'Content-Type': 'application/json' } : {}),
        },
        body: hasBody ? JSON.stringify(payload) : undefined,
        signal: controller.signal,
      })

      const body = await this.#readBody(response)

      if (!response.ok) {
        throw new ApiError(`HTTP ${response.status}`, {
          status: response.status,
          detail: extractDetail(body),
        })
      }

      return body
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

  async #readBody(response) {
    const text = await response.text()
    if (!text) return null
    try {
      return JSON.parse(text)
    } catch {
      return text
    }
  }
}
