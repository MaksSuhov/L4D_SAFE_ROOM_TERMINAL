import { ref } from 'vue'

import { describeError } from '@/api/errors'

/**
 * The account list an operator works with.
 *
 * Every rule this screen obeys is enforced by the backend as well — the panel
 * only spares the operator from making mistakes it would have refused anyway.
 * The one exception is an operator's own record: the backend forbids touching
 * it, and the screen greys those controls out instead of offering an error.
 */
export function useUserAdmin(api) {
  const users = ref([])
  const isLoading = ref(false)
  const isBusy = ref(false)
  const error = ref('')
  const notice = ref('')

  async function load() {
    isLoading.value = true
    try {
      users.value = await api.listUsers()
      error.value = ''
    } catch (cause) {
      error.value = describeError(cause, 'Не удалось прочитать список учётных записей.')
    } finally {
      isLoading.value = false
    }
  }

  async function changeRole(user, role) {
    if (user.role === role) return
    await apply(() => api.updateUser(user.id, { role }), 'Роль не изменилась.')
  }

  async function setActive(user, isActive) {
    if (user.is_active === isActive) return
    await apply(
      () => api.updateUser(user.id, { is_active: isActive }),
      'Доступ не изменился.',
    )
  }

  async function revokeSessions(user) {
    await apply(async () => {
      const report = await api.revokeUserSessions(user.id)
      notice.value = `Сессии погашены: ${report.revoked}.`
    }, 'Сессии не удалось погасить.')
  }

  async function remove(user) {
    await apply(() => api.deleteUser(user.id), 'Учётную запись не удалось удалить.')
  }

  async function apply(action, fallback) {
    isBusy.value = true
    notice.value = ''
    try {
      await action()
      await load()
    } catch (cause) {
      error.value = describeError(cause, fallback)
    } finally {
      isBusy.value = false
    }
  }

  return { users, isLoading, isBusy, error, notice, load, changeRole, setActive, revokeSessions, remove }
}
