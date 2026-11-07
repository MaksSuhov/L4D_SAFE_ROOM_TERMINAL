import { apiBaseUrl, apiPrefix, csrfCookieName, csrfHeaderName } from './config'
import { HttpClient } from './HttpClient'

/**
 * Typed facade over the safe-room API.
 *
 * Components never build URLs: they ask for a domain operation, which keeps the
 * HTTP contract in one place and the components free of transport details. Every
 * call rides on the session cookie; calls that only establish a session say so
 * by asking not to be retried after a refusal.
 */
export class TerminalApi {
  #http
  #prefix

  constructor({
    baseUrl = apiBaseUrl,
    apiPrefix: prefix = apiPrefix,
    csrfCookie = csrfCookieName,
    csrfHeader = csrfHeaderName,
  } = {}) {
    this.#http = new HttpClient({
      baseUrl,
      csrf: { cookieName: csrfCookie, headerName: csrfHeader },
    })
    this.#prefix = prefix.replace(/\/+$/, '')
  }

  /** Give the client the one thing it must not know: how a session is renewed. */
  onUnauthorized(handler) {
    this.#http.onUnauthorized = handler
  }

  // --- Session --------------------------------------------------------------

  /** Who the current session belongs to; refuses when there is no session. */
  currentUser() {
    return this.#http.get(`${this.#prefix}/auth/me`, { retry: false })
  }

  /** OAuth 2 password grant: the email travels in the `username` field. */
  logIn({ email, password }) {
    return this.#http.postForm(
      `${this.#prefix}/auth/token`,
      new URLSearchParams({ username: email, password }),
      { retry: false },
    )
  }

  register({ email, password, displayName }) {
    const body = { email, password }
    if (displayName) body.display_name = displayName
    return this.#http.post(`${this.#prefix}/auth/register`, body, { retry: false })
  }

  logOut() {
    return this.#http.post(`${this.#prefix}/auth/logout`)
  }

  logOutEverywhere() {
    return this.#http.post(`${this.#prefix}/auth/logout-all`)
  }

  /** OAuth 2 refresh grant: the token itself stays in its httpOnly cookie. */
  renewSession() {
    return this.#http.post(`${this.#prefix}/auth/refresh`, undefined, { retry: false })
  }

  // --- Archive --------------------------------------------------------------

  /** Every survivor the terminal can impersonate. */
  listCharacters() {
    return this.#http.get(`${this.#prefix}/characters`)
  }

  /** Ask one survivor a question; resolves with the answer and its evidence. */
  sendMessage({ characterId, message }) {
    return this.#http.post(`${this.#prefix}/chat`, {
      character_id: characterId,
      message,
    })
  }

  // --- Administration -------------------------------------------------------

  listUsers() {
    return this.#http.get(`${this.#prefix}/admin/users`)
  }

  updateUser(userId, changes) {
    return this.#http.patch(`${this.#prefix}/admin/users/${userId}`, changes)
  }

  deleteUser(userId) {
    return this.#http.delete(`${this.#prefix}/admin/users/${userId}`)
  }

  revokeUserSessions(userId) {
    return this.#http.post(`${this.#prefix}/admin/users/${userId}/sessions/revoke`)
  }
}

export const terminalApi = new TerminalApi()
