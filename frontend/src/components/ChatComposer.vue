<script setup>
import { ArrowUp } from 'lucide-vue-next'
import { computed, nextTick, onMounted, ref, useTemplateRef, watch } from 'vue'

const props = defineProps({
  modelValue: { type: String, default: '' },
  /** Set when there is no responder to talk to (roster still loading). */
  disabled: { type: Boolean, default: false },
  /** A request is in flight; the field is locked until the answer lands. */
  loading: { type: Boolean, default: false },
  maxLength: { type: Number, default: 2000 },
  placeholder: { type: String, default: 'Сообщение выжившему…' },
})

const emit = defineEmits(['update:modelValue', 'submit'])

const MAX_TEXTAREA_HEIGHT_PX = 200

const textarea = useTemplateRef('textarea')
const isOverLimit = computed(() => props.modelValue.length > props.maxLength)
const canSubmit = computed(
  () => !props.disabled && !props.loading && !isOverLimit.value && props.modelValue.trim() !== '',
)

function resize() {
  const element = textarea.value
  if (!element) return
  element.style.height = 'auto'
  element.style.height = `${Math.min(element.scrollHeight, MAX_TEXTAREA_HEIGHT_PX)}px`
}

watch(
  () => props.modelValue,
  () => nextTick(resize),
)

watch(
  () => props.loading,
  async (loading) => {
    if (!loading) {
      await nextTick()
      textarea.value?.focus()
    }
  },
)

onMounted(() => {
  resize()
  textarea.value?.focus()
})

function submit() {
  if (!canSubmit.value) return
  emit('submit')
}

/**
 * Enter sends; Shift+Enter and IME composition keep the newline behaviour.
 *
 * The keydown handler deliberately does not re-check the draft: `modelValue` is
 * only refreshed on the next render, so a keystroke that arrives right after a
 * paste can still see the previous prop. The parent owns the draft ref and is
 * the single place that decides whether a message is sendable.
 */
function handleKeydown(event) {
  if (event.key !== 'Enter' || event.shiftKey || event.isComposing) return
  if (props.disabled || props.loading) return

  event.preventDefault()
  emit('submit')
}
</script>

<template>
  <div
    class="rounded-sm border bg-bunker-900/80 transition-colors focus-within:border-term-amber/50"
    :class="isOverLimit ? 'border-term-blood/60' : 'border-zinc-700/70'"
  >
    <div class="flex items-end gap-2 px-3 py-2">
      <span class="pb-2 pl-0.5 text-sm text-term-amber/70 select-none" aria-hidden="true">&gt;</span>

      <textarea
        ref="textarea"
        :value="modelValue"
        :placeholder="loading ? 'Ожидание ответа…' : placeholder"
        :disabled="disabled || loading"
        :maxlength="maxLength + 200"
        rows="1"
        class="scrollbar-terminal max-h-[200px] w-full resize-none bg-transparent py-2 text-sm leading-relaxed text-zinc-200 placeholder:text-zinc-600 focus:outline-none disabled:cursor-not-allowed disabled:text-zinc-500"
        aria-label="Сообщение выжившему"
        @input="emit('update:modelValue', $event.target.value)"
        @keydown="handleKeydown"
      />

      <button
        type="button"
        class="mb-0.5 flex size-9 shrink-0 items-center justify-center rounded-sm border transition-colors"
        :class="
          canSubmit
            ? 'border-term-amber/60 bg-term-amber/15 text-term-amber hover:bg-term-amber/25'
            : 'border-zinc-800 bg-bunker-800 text-zinc-600'
        "
        :disabled="!canSubmit"
        :aria-label="loading ? 'Ожидание ответа' : 'Отправить сообщение'"
        @click="submit"
      >
        <ArrowUp class="size-4" aria-hidden="true" />
      </button>
    </div>

    <div
      class="flex items-center justify-between gap-3 border-t border-zinc-800/80 px-3 py-1.5 text-[0.6rem] tracking-[0.2em] uppercase"
    >
      <p v-if="loading" class="flex items-center gap-2 text-term-green">
        <span>Thinking</span>
        <span class="flex gap-1" aria-hidden="true">
          <span
            v-for="dot in 3"
            :key="dot"
            class="size-1 animate-pulse rounded-full bg-term-green"
            :style="{ animationDelay: `${dot * 0.15}s` }"
          />
        </span>
      </p>
      <p v-else-if="disabled" class="text-zinc-600">Терминал не готов к передаче</p>
      <p v-else class="text-zinc-600">Enter — отправить · Shift+Enter — новая строка</p>

      <p class="tabular-nums" :class="isOverLimit ? 'text-term-blood' : 'text-zinc-600'">
        {{ modelValue.length }} / {{ maxLength }}
      </p>
    </div>
  </div>
</template>
