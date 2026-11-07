<script setup>
/**
 * The operator's list of accounts.
 *
 * It is a panel rather than a route: the terminal has one screen, and looking at
 * accounts is something an operator does without leaving the conversation.
 */
import { KeyRound, RefreshCw, ShieldCheck, ShieldOff, Trash2, UserRoundX, X } from 'lucide-vue-next'
import { onMounted } from 'vue'

import { useUserAdmin } from '@/composables/useUserAdmin'

const props = defineProps({
  api: { type: Object, required: true },
  currentUserId: { type: String, default: '' },
})

const emit = defineEmits(['close'])

const { users, isLoading, isBusy, error, notice, load, changeRole, setActive, revokeSessions, remove } =
  useUserAdmin(props.api)

const ROLES = [
  { value: 'member', label: 'Выживший' },
  { value: 'admin', label: 'Оператор' },
]

/** An operator may not touch their own record: the backend refuses it too. */
function isSelf(user) {
  return user.id === props.currentUserId
}

onMounted(load)
</script>

<template>
  <div class="fixed inset-0 z-30 flex items-start justify-center overflow-y-auto p-4 sm:p-8">
    <div class="absolute inset-0 bg-bunker-950/80 backdrop-blur-sm" @click="emit('close')" />

    <section
      class="relative w-full max-w-2xl border border-term-amber/20 bg-bunker-900/95 p-5 shadow-2xl"
      role="dialog"
      aria-modal="true"
      aria-label="Учётные записи терминала"
    >
      <header class="mb-4 flex items-center gap-3">
        <KeyRound class="size-4 text-term-amber" aria-hidden="true" />
        <h2 class="text-xs tracking-[0.25em] text-term-amber text-glow-amber uppercase">
          Учётные записи
        </h2>
        <span class="text-[0.6rem] tracking-[0.2em] text-zinc-600 uppercase">
          {{ users.length }} на терминале
        </span>

        <div class="ml-auto flex items-center gap-2">
          <button
            type="button"
            class="border border-zinc-800 p-1.5 text-zinc-500 transition-colors hover:border-term-green/40 hover:text-term-green"
            aria-label="Обновить список"
            :disabled="isLoading"
            @click="load"
          >
            <RefreshCw class="size-3.5" :class="isLoading && 'animate-spin'" aria-hidden="true" />
          </button>
          <button
            type="button"
            class="border border-zinc-800 p-1.5 text-zinc-500 transition-colors hover:border-term-blood/40 hover:text-term-blood"
            aria-label="Закрыть"
            @click="emit('close')"
          >
            <X class="size-3.5" aria-hidden="true" />
          </button>
        </div>
      </header>

      <p
        v-if="error"
        class="mb-3 border border-term-blood/40 bg-term-blood/5 px-3 py-2 text-xs text-term-blood"
        role="alert"
      >
        {{ error }}
      </p>
      <p
        v-else-if="notice"
        class="mb-3 border border-term-green/30 bg-term-green/5 px-3 py-2 text-xs text-term-green"
      >
        {{ notice }}
      </p>

      <ul class="flex flex-col divide-y divide-zinc-900">
        <li
          v-for="account in users"
          :key="account.id"
          class="flex flex-wrap items-center gap-3 py-3"
          :class="!account.is_active && 'opacity-50'"
        >
          <div class="min-w-0 flex-1">
            <p class="truncate text-sm text-zinc-200">
              {{ account.display_name }}
              <span v-if="isSelf(account)" class="ml-2 text-[0.6rem] text-term-amber/70 uppercase">
                это вы
              </span>
            </p>
            <p class="truncate text-[0.65rem] text-zinc-600">{{ account.email }}</p>
          </div>

          <select
            :value="account.role"
            :disabled="isBusy || isSelf(account)"
            class="border border-zinc-800 bg-bunker-950/70 px-2 py-1 text-[0.65rem] tracking-[0.15em] text-zinc-300 uppercase disabled:cursor-not-allowed disabled:text-zinc-600"
            :aria-label="`Роль ${account.email}`"
            @change="changeRole(account, $event.target.value)"
          >
            <option v-for="role in ROLES" :key="role.value" :value="role.value">
              {{ role.label }}
            </option>
          </select>

          <button
            type="button"
            :disabled="isBusy || isSelf(account)"
            class="flex items-center gap-1 border px-2 py-1 text-[0.6rem] tracking-[0.15em] uppercase transition-colors disabled:cursor-not-allowed disabled:border-zinc-900 disabled:text-zinc-700"
            :class="
              account.is_active
                ? 'border-term-green/30 text-term-green hover:bg-term-green/10'
                : 'border-term-amber/30 text-term-amber hover:bg-term-amber/10'
            "
            @click="setActive(account, !account.is_active)"
          >
            <component
              :is="account.is_active ? ShieldCheck : ShieldOff"
              class="size-3.5"
              aria-hidden="true"
            />
            {{ account.is_active ? 'Доступен' : 'Заблокирован' }}
          </button>

          <button
            type="button"
            :disabled="isBusy"
            class="flex items-center gap-1 border border-zinc-800 px-2 py-1 text-[0.6rem] tracking-[0.15em] text-zinc-500 uppercase transition-colors hover:border-term-amber/40 hover:text-term-amber disabled:cursor-not-allowed"
            :aria-label="`Погасить сессии ${account.email}`"
            @click="revokeSessions(account)"
          >
            <UserRoundX class="size-3.5" aria-hidden="true" />
            Разлогинить
          </button>

          <button
            type="button"
            :disabled="isBusy || isSelf(account)"
            class="flex items-center gap-1 border border-zinc-800 px-2 py-1 text-[0.6rem] tracking-[0.15em] text-zinc-500 uppercase transition-colors hover:border-term-blood/50 hover:text-term-blood disabled:cursor-not-allowed disabled:hover:border-zinc-800 disabled:hover:text-zinc-600"
            :aria-label="`Удалить ${account.email}`"
            @click="remove(account)"
          >
            <Trash2 class="size-3.5" aria-hidden="true" />
          </button>
        </li>
      </ul>

      <p class="mt-4 text-[0.65rem] leading-relaxed text-zinc-600">
        Блокировка действует сразу: права читаются из учётной записи, а не из выданного токена.
        Собственную запись оператор изменить не может — иначе терминал остался бы без оператора.
      </p>
    </section>
  </div>
</template>
