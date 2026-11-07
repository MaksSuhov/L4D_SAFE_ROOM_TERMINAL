<script setup>
/**
 * The way in: one form for signing in and for opening an access of one's own.
 *
 * The screen holds no session state — it reports what the visitor typed and
 * shows what the terminal answered. Whether the session opened is decided by
 * the composable behind it, because a browser cannot read the tokens it gets.
 */
import { KeyRound, LogIn, ShieldPlus, TriangleAlert } from 'lucide-vue-next'
import { computed, ref } from 'vue'

const MIN_PASSWORD_LENGTH = 10
/** Matches `MIN_PASSWORD_LENGTH` in `app/domain/password_policy.py`. */

const props = defineProps({
  busy: { type: Boolean, default: false },
  error: { type: String, default: '' },
})

const emit = defineEmits(['log-in', 'register'])

const mode = ref('log-in')
const email = ref('')
const password = ref('')
const displayName = ref('')

const isRegistering = computed(() => mode.value === 'register')
const canSubmit = computed(
  () => email.value.trim().length > 0 && password.value.length >= MIN_PASSWORD_LENGTH,
)

function submit() {
  if (!canSubmit.value || props.busy) return

  if (isRegistering.value) {
    emit('register', {
      email: email.value.trim(),
      password: password.value,
      displayName: displayName.value.trim() || undefined,
    })
    return
  }
  emit('log-in', { email: email.value.trim(), password: password.value })
}

function switchTo(next) {
  mode.value = next
}
</script>

<template>
  <div class="relative flex h-dvh items-center justify-center overflow-hidden px-4">
    <div class="terminal-backdrop" aria-hidden="true" />
    <div class="crt-overlay" aria-hidden="true" />

    <section
      class="relative w-full max-w-sm border border-term-amber/20 bg-bunker-900/80 p-6 backdrop-blur"
    >
      <header class="mb-5">
        <p class="text-[0.6rem] tracking-[0.3em] text-zinc-500 uppercase">Бункер · пост охраны</p>
        <h1
          class="mt-1 flex items-center gap-2 text-sm tracking-[0.18em] text-term-amber text-glow-amber uppercase"
        >
          <KeyRound class="size-4" aria-hidden="true" />
          {{ isRegistering ? 'Заведение доступа' : 'Вход на терминал' }}
          <span class="cursor-blink text-term-green" aria-hidden="true">▍</span>
        </h1>
        <p class="mt-2 text-xs leading-relaxed text-zinc-500">
          Архив убежища открыт только выжившим с учётной записью. Сессия живёт в защищённых
          cookie: страница токенов не видит, а обновляет их сама.
        </p>
      </header>

      <div class="mb-4 flex gap-2" role="tablist">
        <button
          type="button"
          role="tab"
          :aria-selected="!isRegistering"
          class="flex-1 rounded-sm border px-3 py-1.5 text-[0.65rem] tracking-[0.2em] uppercase transition-colors"
          :class="
            !isRegistering
              ? 'border-term-amber/50 text-term-amber'
              : 'border-zinc-800 text-zinc-500 hover:text-zinc-300'
          "
          @click="switchTo('log-in')"
        >
          Вход
        </button>
        <button
          type="button"
          role="tab"
          :aria-selected="isRegistering"
          class="flex-1 rounded-sm border px-3 py-1.5 text-[0.65rem] tracking-[0.2em] uppercase transition-colors"
          :class="
            isRegistering
              ? 'border-term-amber/50 text-term-amber'
              : 'border-zinc-800 text-zinc-500 hover:text-zinc-300'
          "
          @click="switchTo('register')"
        >
          Регистрация
        </button>
      </div>

      <form class="flex flex-col gap-3" @submit.prevent="submit">
        <label class="flex flex-col gap-1">
          <span class="text-[0.6rem] tracking-[0.2em] text-zinc-500 uppercase">Почта</span>
          <input
            v-model="email"
            type="email"
            name="email"
            autocomplete="username"
            required
            placeholder="ellis@l4d.dev"
            class="border border-zinc-800 bg-bunker-950/70 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-700 focus:border-term-amber/50 focus:outline-none"
          />
        </label>

        <label v-if="isRegistering" class="flex flex-col gap-1">
          <span class="text-[0.6rem] tracking-[0.2em] text-zinc-500 uppercase">
            Подпись на терминале
          </span>
          <input
            v-model="displayName"
            type="text"
            name="display_name"
            maxlength="64"
            placeholder="Эллис"
            class="border border-zinc-800 bg-bunker-950/70 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-700 focus:border-term-amber/50 focus:outline-none"
          />
        </label>

        <label class="flex flex-col gap-1">
          <span class="text-[0.6rem] tracking-[0.2em] text-zinc-500 uppercase">Пароль</span>
          <input
            v-model="password"
            type="password"
            name="password"
            :autocomplete="isRegistering ? 'new-password' : 'current-password'"
            :minlength="MIN_PASSWORD_LENGTH"
            required
            placeholder="не меньше 10 символов"
            class="border border-zinc-800 bg-bunker-950/70 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-700 focus:border-term-amber/50 focus:outline-none"
          />
        </label>

        <button
          type="submit"
          :disabled="!canSubmit || busy"
          class="mt-1 flex items-center justify-center gap-2 border border-term-amber/50 px-4 py-2 text-[0.65rem] tracking-[0.25em] text-term-amber uppercase transition-colors hover:bg-term-amber/10 disabled:border-zinc-800 disabled:text-zinc-600 disabled:hover:bg-transparent"
        >
          <component :is="isRegistering ? ShieldPlus : LogIn" class="size-4" aria-hidden="true" />
          {{ busy ? 'Проверяю…' : isRegistering ? 'Завести доступ' : 'Войти' }}
        </button>
      </form>

      <p
        v-if="error"
        class="mt-4 flex items-start gap-2 border border-term-blood/40 bg-term-blood/5 px-3 py-2 text-xs text-term-blood"
        role="alert"
      >
        <TriangleAlert class="mt-0.5 size-3.5 shrink-0" aria-hidden="true" />
        <span>{{ error }}</span>
      </p>
    </section>
  </div>
</template>
