/**
 * Transport-level failure of the API client.
 *
 * `status` follows HTTP semantics, with `0` reserved for "the request never
 * reached the server" and `408` for a timeout, so callers can branch on the
 * kind of failure without inspecting the original error. `code` is the reason
 * the backend attaches to a refusal — the client branches on it where two
 * failures share a status, such as an expired session and a wrong password.
 */
export class ApiError extends Error {
  constructor(message, { status = 0, detail = null, code = null, cause } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
    this.code = code
    this.cause = cause
  }

  /** True when the session is unusable and the visitor has to sign in again. */
  get isSessionGone() {
    return this.status === 401 && this.code !== 'invalid_credentials'
  }
}

const CODE_MESSAGES = {
  invalid_credentials: 'Неверная почта или пароль.',
  account_inactive: 'Учётная запись заблокирована.',
  email_taken: 'На эту почту уже заведена учётная запись.',
  weak_password: 'Пароль слишком простой: нужно не меньше 10 символов.',
  expired_token: 'Сессия истекла — войдите заново.',
  refresh_token_reused: 'Сессия отозвана: токен обновления предъявили повторно.',
  invalid_token: 'Сессия не подтверждена — войдите заново.',
  forbidden: 'Недостаточно прав для этого действия.',
}

const STATUS_MESSAGES = {
  401: 'Сессия не подтверждена — войдите заново.',
  403: 'Недостаточно прав для этого действия.',
  404: 'Такого выжившего нет в реестре убежища.',
  408: 'Терминал не дождался ответа от шлюза ИИ.',
  409: 'На эту почту уже заведена учётная запись.',
  422: 'Сообщение отклонено: оно пустое или длиннее 2000 символов.',
  500: 'Внутренняя ошибка терминала.',
  502: 'Шлюз ИИ недоступен: провайдер не ответил.',
  504: 'Провайдер ИИ не ответил вовремя.',
  0: 'Нет связи с терминалом. Проверь, что сервер запущен на localhost:8000.',
}

/**
 * Turn any thrown value into a message the operator can act on. The reason the
 * backend named wins, then the status, then its developer-facing detail.
 */
export function describeError(error, fallback) {
  if (error instanceof ApiError) {
    return (
      CODE_MESSAGES[error.code] ?? STATUS_MESSAGES[error.status] ?? error.detail ?? fallback
    )
  }
  return error?.message || fallback
}
