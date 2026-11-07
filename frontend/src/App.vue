<script setup>
/**
 * Unified chat screen of the safe-room terminal.
 *
 * The screen is a composition of three zones — the transcript, the composer and
 * the responder selector — glued together by three composables that own the
 * session state (`useAuth`, `useCharacters`, `useChat`). No component talks to
 * `fetch`.
 *
 * The archive is open to signed-in visitors only, so the screen is gated twice:
 * the login form replaces it while there is no session, and the transcript is
 * cleared when a session closes, so the next person at the terminal never reads
 * the previous one's conversation.
 */
import { computed, onMounted, ref, watch } from 'vue'

import AuthScreen from '@/components/AuthScreen.vue'
import ChatComposer from '@/components/ChatComposer.vue'
import ChatWindow from '@/components/ChatWindow.vue'
import CharacterSelector from '@/components/CharacterSelector.vue'
import TerminalHeader from '@/components/TerminalHeader.vue'
import UserAdminPanel from '@/components/UserAdminPanel.vue'
import { terminalApi } from '@/api/terminalApi'
import { useAuth } from '@/composables/useAuth'
import { useCharacters } from '@/composables/useCharacters'
import { useChat } from '@/composables/useChat'

const draft = ref('')
const isAdminPanelOpen = ref(false)

/** Mirrors the backend validation of `ChatRequest.message`. */
const MAX_MESSAGE_LENGTH = 2000

const {
  user,
  status,
  isBusy,
  error: authError,
  isAuthenticated,
  isAdmin,
  bootstrap,
  logIn,
  register,
  logOut,
} = useAuth(terminalApi)

const {
  characters,
  selectedId,
  selectedCharacter,
  isLoading: isLoadingCharacters,
  error: rosterError,
  load: loadCharacters,
  selectCharacter,
} = useCharacters(terminalApi)

const { messages, isSending, send, reset: resetSession } = useChat(terminalApi)

const connectionStatus = computed(() => {
  if (isLoadingCharacters.value) return 'connecting'
  return rosterError.value ? 'offline' : 'online'
})

onMounted(() => {
  // The session cookie is the only proof of identity a page can check, and the
  // backend is the one that decides whether it is still good.
  bootstrap()
})

watch(isAuthenticated, (isOpen) => {
  if (isOpen) {
    loadCharacters()
    return
  }
  resetSession()
  isAdminPanelOpen.value = false
  draft.value = ''
})

async function submitCredentials(credentials) {
  if (await logIn(credentials)) draft.value = ''
}

async function submitRegistration(credentials) {
  if (await register(credentials)) draft.value = ''
}

function closeSession() {
  logOut()
}

async function submitMessage() {
  const character = selectedCharacter.value
  const text = draft.value.trim()
  if (!character || !text || isSending.value) return
  if (text.length > MAX_MESSAGE_LENGTH) return

  draft.value = ''
  await send({ characterId: character.id, characterName: character.name, text })
}

function startNewSession() {
  resetSession()
  draft.value = ''
}

function applySuggestion(prompt) {
  draft.value = prompt
}
</script>

<template>
  <div class="relative flex h-dvh flex-col overflow-hidden">
    <div class="terminal-backdrop" aria-hidden="true" />
    <div class="crt-overlay" aria-hidden="true" />

    <p
      v-if="status === 'unknown'"
      class="relative z-10 m-auto text-[0.65rem] tracking-[0.3em] text-term-amber/70 uppercase"
      role="status"
    >
      Проверка пропуска<span class="cursor-blink">▍</span>
    </p>

    <AuthScreen
      v-else-if="!isAuthenticated"
      :busy="isBusy"
      :error="authError"
      @log-in="submitCredentials"
      @register="submitRegistration"
    />

    <template v-else>
      <TerminalHeader
        :status="connectionStatus"
        :user-name="user?.display_name ?? ''"
        :is-admin="isAdmin"
        @new-session="startNewSession"
        @retry="loadCharacters"
        @log-out="closeSession"
        @open-admin="isAdminPanelOpen = true"
      />

      <main class="relative min-h-0 flex-1">
        <ChatWindow
          :messages="messages"
          :is-sending="isSending"
          :character="selectedCharacter"
          :character-count="characters.length"
          @suggestion="applySuggestion"
        />
      </main>

      <footer class="relative shrink-0 border-t border-term-amber/15 bg-bunker-900/70 backdrop-blur">
        <div class="mx-auto flex w-full max-w-3xl flex-col gap-3 px-4 py-3 sm:px-6">
          <p
            v-if="rosterError"
            class="border border-term-blood/40 bg-term-blood/5 px-3 py-2 text-xs text-term-blood"
            role="alert"
          >
            {{ rosterError }}
          </p>

          <div class="flex items-end justify-between gap-4">
            <CharacterSelector
              :characters="characters"
              :model-value="selectedId"
              :loading="isLoadingCharacters"
              :disabled="isSending"
              @update:model-value="selectCharacter"
            />

            <p
              v-if="selectedCharacter"
              class="hidden text-right text-[0.6rem] tracking-[0.2em] text-zinc-600 uppercase sm:block"
            >
              Ответит: <span class="text-term-amber/80">{{ selectedCharacter.name }}</span>
            </p>
          </div>

          <ChatComposer
            v-model="draft"
            :disabled="!selectedCharacter"
            :loading="isSending"
            :max-length="MAX_MESSAGE_LENGTH"
            @submit="submitMessage"
          />
        </div>
      </footer>

      <UserAdminPanel
        v-if="isAdminPanelOpen"
        :api="terminalApi"
        :current-user-id="user?.id ?? ''"
        @close="isAdminPanelOpen = false"
      />
    </template>
  </div>
</template>
