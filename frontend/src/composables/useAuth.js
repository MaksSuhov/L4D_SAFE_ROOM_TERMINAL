import { computed, ref } from 'vue'

import { describeError } from '@/api/errors'

/**
 * The session of the visitor at the terminal.
 *
 * The tokens themselves are never here: they live in httpOnly cookies, which no
 * script can read. What the screen needs is narrower — who is signed in, whether
 * they are an operator, and how to open and close a session. Renewal is wired
 * into the API client rather than called from components, so an expired access
 * token is invisible to the chat screen: the client asks for a new pair, retries
 * once, and reopens the session only if that failed.
 */
export function useAuth(api) {
  const user = ref(null)
  const status = ref('unknown')
  const isBusy = ref(false)
  const error = ref('')

  const isAuthenticated = computed(() => status.value === 'authenticated')
  const isAdmin = computed(() => user.value?.role === 'admin')

  function applySession(session) {
    user.value = session.user
    status.value = 'authenticated'
    error.value = ''
  }

  function forget() {
    user.value = null
    status.value = 'anonymous'
  }

  async function run(action, fallback) {
    isBusy.value = true
    error.value = ''
    try {
      applySession(await action())
      return true
    } catch (cause) {
      error.value = describeError(cause, fallback)
      return false
    } finally {
      isBusy.value = false
    }
  }

  /** Ask the backend who the session belongs to; a cookie is the whole answer. */
  async function bootstrap() {
    try {
      user.value = await api.currentUser()
      status.value = 'authenticated'
    } catch {
      forget()
    }
  }

  function logIn({ email, password }) {
    return run(
      () => api.logIn({ email, password }),
      'Терминал не пустил: проверьте почту и пароль.',
    )
  }

  function register({ email, password, displayName }) {
    return run(
      () => api.register({ email, password, displayName }),
      'Учётную запись не удалось завести.',
    )
  }

  /** Exchange the refresh cookie for a new pair; called by the API client. */
  async function renew() {
    try {
      applySession(await api.renewSession())
      return true
    } catch {
      forget()
      return false
    }
  }

  async function logOut() {
    try {
      await api.logOut()
    } finally {
      forget()
      error.value = ''
    }
  }

  async function logOutEverywhere() {
    try {
      await api.logOutEverywhere()
    } finally {
      forget()
      error.value = ''
    }
  }

  function clearError() {
    error.value = ''
  }

  api.onUnauthorized(renew)

  return {
    user,
    status,
    isBusy,
    error,
    isAuthenticated,
    isAdmin,
    bootstrap,
    logIn,
    register,
    logOut,
    logOutEverywhere,
    clearError,
  }
}
