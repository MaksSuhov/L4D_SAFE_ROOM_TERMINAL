import { apiBaseUrl, apiPrefix } from './config'
import { HttpClient } from './HttpClient'

/**
 * Typed facade over the safe-room API.
 *
 * Components never build URLs: they ask for a domain operation, which keeps the
 * HTTP contract in one place and the components free of transport details.
 */
export class TerminalApi {
  #http
  #prefix

  constructor({ baseUrl = apiBaseUrl, apiPrefix: prefix = apiPrefix } = {}) {
    this.#http = new HttpClient({ baseUrl })
    this.#prefix = prefix.replace(/\/+$/, '')
  }

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
}

export const terminalApi = new TerminalApi()
