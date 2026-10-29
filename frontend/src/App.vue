<script setup>
/**
 * Unified chat screen of the safe-room terminal.
 *
 * The screen is a composition of three zones — the transcript, the composer and
 * the responder selector — glued together by two composables that own the
 * session state (`useCharacters`, `useChat`). No component talks to `fetch`.
 */
import { computed, onMounted, ref } from 'vue'

import ChatComposer from '@/components/ChatComposer.vue'
import ChatWindow from '@/components/ChatWindow.vue'
import CharacterSelector from '@/components/CharacterSelector.vue'
import TerminalHeader from '@/components/TerminalHeader.vue'
import { terminalApi } from '@/api/terminalApi'
import { useCharacters } from '@/composables/useCharacters'
import { useChat } from '@/composables/useChat'

const draft = ref('')

/** Mirrors the backend validation of `ChatRequest.message`. */
const MAX_MESSAGE_LENGTH = 2000

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

onMounted(loadCharacters)

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
    <div class="crt-overlay" aria-hidden="true" />

    <TerminalHeader
      :status="connectionStatus"
      :character-count="characters.length"
      @new-session="startNewSession"
      @retry="loadCharacters"
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
  </div>
</template>
