<script setup>
import { ArrowDown } from 'lucide-vue-next'
import { nextTick, ref, watch } from 'vue'

import ChatMessage from './ChatMessage.vue'
import EmptyTerminal from './EmptyTerminal.vue'
import ThinkingBubble from './ThinkingBubble.vue'

const props = defineProps({
  messages: { type: Array, required: true },
  isSending: { type: Boolean, default: false },
  character: { type: Object, default: null },
  characterCount: { type: Number, default: 0 },
})

defineEmits(['suggestion'])

const PIN_THRESHOLD_PX = 120

const scroller = ref(null)
const isPinnedToBottom = ref(true)

/**
 * Keep the transcript glued to the newest line, but never yank the view away
 * from an operator who scrolled up to read the archive sources.
 */
function updatePinState() {
  const element = scroller.value
  if (!element) return
  const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight
  isPinnedToBottom.value = distanceFromBottom < PIN_THRESHOLD_PX
}

async function scrollToBottom() {
  await nextTick()
  const element = scroller.value
  if (element) element.scrollTop = element.scrollHeight
}

watch(
  () => [props.messages.length, props.isSending],
  async () => {
    if (isPinnedToBottom.value) await scrollToBottom()
  },
)

defineExpose({ scrollToBottom })
</script>

<template>
  <div class="relative h-full">
    <div
      ref="scroller"
      class="scrollbar-terminal h-full overflow-y-auto"
      @scroll.passive="updatePinState"
    >
      <div class="mx-auto w-full max-w-3xl px-4 py-6 sm:px-6">
        <EmptyTerminal
          v-if="messages.length === 0 && !isSending"
          :character-count="characterCount"
          @suggestion="$emit('suggestion', $event)"
        />

        <div v-else class="space-y-6" role="log" aria-live="polite">
          <ChatMessage v-for="message in messages" :key="message.id" :message="message" />
          <ThinkingBubble v-if="isSending" :character="character" />
        </div>
      </div>
    </div>

    <Transition name="pop">
      <button
        v-if="!isPinnedToBottom"
        type="button"
        class="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-sm border border-term-amber/40 bg-bunker-900/95 px-3 py-1.5 text-[0.65rem] tracking-[0.2em] text-term-amber uppercase transition-colors hover:bg-term-amber/10"
        @click="scrollToBottom"
      >
        <ArrowDown class="mr-1 inline size-3.5" aria-hidden="true" />К последней передаче
      </button>
    </Transition>
  </div>
</template>
