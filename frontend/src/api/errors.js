/**
 * Transport-level failure of the API client.
 *
 * `status` follows HTTP semantics, with `0` reserved for "the request never
 * reached the server" and `408` for a timeout, so callers can branch on the
 * kind of failure without inspecting the original error.
 */
export class ApiError extends Error {
  constructor(message, { status = 0, detail = null, cause } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
    this.cause = cause
  }
}

const STATUS_MESSAGES = {
  404: 'Такого выжившего нет в реестре убежища.',
  408: 'Терминал не дождался ответа от шлюза ИИ.',
  422: 'Сообщение отклонено: оно пустое или длиннее 2000 символов.',
  500: 'Внутренняя ошибка терминала.',
  502: 'Шлюз ИИ недоступен: провайдер не ответил.',
  504: 'Провайдер ИИ не ответил вовремя.',
  0: 'Нет связи с терминалом. Проверь, что сервер запущен на localhost:8000.',
}

/**
 * Turn any thrown value into a message the operator can act on. The status map
 * wins over the raw backend `detail`, which is written for developers.
 */
export function describeError(error, fallback) {
  if (error instanceof ApiError) {
    return STATUS_MESSAGES[error.status] ?? error.detail ?? fallback
  }
  return error?.message || fallback
}
