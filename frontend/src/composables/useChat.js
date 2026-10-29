import { ref } from 'vue'

import { describeError } from '@/api/errors'

let messageCounter = 0

function createMessageId() {
  messageCounter += 1
  return `msg-${messageCounter}`
}

/**
 * The transcript of the current session.
 *
 * Messages are append-only and grouped by role, so a component can render the
 * whole log without knowing how the answer was produced. A failed exchange
 * becomes a visible system entry instead of a silent rejection, which is what
 * an operator of a terminal would expect.
 */
export function useChat(api) {
  const messages = ref([])
  const isSending = ref(false)

  function append(message) {
    messages.value = [
      ...messages.value,
      { id: createMessageId(), createdAt: Date.now(), sources: [], ...message },
    ]
  }

  async function send({ characterId, characterName, text }) {
    if (isSending.value) return

    append({ role: 'user', text })
    isSending.value = true

    try {
      const reply = await api.sendMessage({ characterId, message: text })
      append({
        role: 'character',
        text: reply.answer,
        characterId: reply.character_id,
        characterName: reply.character_name || characterName,
        model: reply.model,
        sources: reply.sources ?? [],
      })
    } catch (cause) {
      append({
        role: 'system',
        variant: 'error',
        text: describeError(cause, 'Терминал не смог получить ответ.'),
      })
    } finally {
      isSending.value = false
    }
  }

  function reset() {
    messages.value = []
  }

  return { messages, isSending, send, reset }
}
