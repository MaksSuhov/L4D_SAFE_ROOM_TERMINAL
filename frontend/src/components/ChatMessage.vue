<script setup>
import { computed } from 'vue'

import CharacterAvatar from './CharacterAvatar.vue'
import SourceSpoiler from './SourceSpoiler.vue'
import { formatClock } from '@/lib/time'

const props = defineProps({
  message: { type: Object, required: true },
})

const isUser = computed(() => props.message.role === 'user')
const isSystem = computed(() => props.message.role === 'system')
const isError = computed(() => props.message.variant === 'error')
const hasSources = computed(() => (props.message.sources?.length ?? 0) > 0)
const clock = computed(() => formatClock(props.message.createdAt))
</script>

<template>
  <!-- Failure of a transmission: printed by the terminal itself. -->
  <div
    v-if="isSystem"
    class="message-entry flex items-start gap-3 border border-term-blood/40 bg-term-blood/5 px-3.5 py-2.5"
    :class="isError ? 'text-term-blood' : 'text-zinc-400'"
    role="status"
  >
    <span class="mt-0.5 shrink-0 text-xs">!</span>
    <p class="text-xs leading-relaxed">
      <span class="mr-2 tracking-[0.2em] uppercase opacity-70">Сбой канала</span>
      <span class="text-zinc-300">{{ message.text }}</span>
    </p>
    <span v-if="clock" class="ml-auto shrink-0 text-[0.6rem] text-zinc-600 tabular-nums">
      {{ clock }}
    </span>
  </div>

  <!-- The operator: a plain block flushed to the right edge. -->
  <div v-else-if="isUser" class="message-entry flex justify-end">
    <div class="max-w-[85%] space-y-1">
      <div
        class="flex items-center justify-end gap-3 text-[0.6rem] tracking-[0.25em] text-zinc-600 uppercase"
      >
        <span>Оператор</span>
        <span class="tabular-nums">{{ clock }}</span>
      </div>
      <p
        class="rounded-sm border border-zinc-700/70 bg-zinc-900/70 px-3.5 py-2.5 text-sm leading-relaxed whitespace-pre-wrap text-zinc-200"
      >
        {{ message.text }}
      </p>
    </div>
  </div>

  <!-- A survivor answering: avatar, callsign, answer, evidence. -->
  <div v-else class="message-entry flex gap-3">
    <CharacterAvatar
      :name="message.characterName"
      :avatar-url="message.avatarUrl ?? null"
      size="md"
    />
    <div class="min-w-0 flex-1 space-y-2">
      <div
        class="flex flex-wrap items-center gap-x-3 gap-y-1 text-[0.6rem] tracking-[0.25em] uppercase"
      >
        <span class="font-semibold text-term-amber text-glow-amber">{{ message.characterName }}</span>
        <span class="text-zinc-600">канал активен</span>
        <span class="text-zinc-600 tabular-nums">{{ clock }}</span>
        <span v-if="message.model" class="text-zinc-700">модель: {{ message.model }}</span>
      </div>

      <p
        class="border-l-2 border-term-amber/35 bg-zinc-900/40 px-3.5 py-2.5 text-sm leading-relaxed whitespace-pre-wrap text-zinc-200"
      >
        {{ message.text }}
      </p>

      <SourceSpoiler v-if="hasSources" :sources="message.sources" />
    </div>
  </div>
</template>
