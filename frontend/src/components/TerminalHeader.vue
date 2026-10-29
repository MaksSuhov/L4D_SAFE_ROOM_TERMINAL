<script setup>
import { BookOpen, RadioTower, RefreshCw, RotateCcw } from 'lucide-vue-next'
import { computed } from 'vue'

import { apiDocsUrl } from '@/api/config'

const props = defineProps({
  /** 'connecting' | 'online' | 'offline' — the state of the roster request. */
  status: { type: String, default: 'connecting' },
  characterCount: { type: Number, default: 0 },
})

defineEmits(['new-session', 'retry'])

const STATUS = {
  connecting: { label: 'Поиск сигнала', dot: 'bg-term-amber animate-pulse', text: 'text-term-amber' },
  online: { label: 'Канал открыт', dot: 'bg-term-green', text: 'text-term-green' },
  offline: { label: 'Нет связи', dot: 'bg-term-blood', text: 'text-term-blood' },
}

const state = computed(() => STATUS[props.status] ?? STATUS.connecting)
</script>

<template>
  <header class="relative shrink-0 border-b border-term-amber/15 bg-bunker-900/70 backdrop-blur">
    <div class="mx-auto flex w-full max-w-3xl items-center gap-3 px-4 py-3 sm:px-6">
      <div class="min-w-0">
        <h1 class="flex items-center gap-2 text-xs tracking-[0.25em] text-term-amber text-glow-amber uppercase">
          Safe-room terminal
          <span class="cursor-blink text-term-green" aria-hidden="true">▍</span>
        </h1>
        <p class="mt-0.5 truncate text-[0.6rem] tracking-[0.2em] text-zinc-500 uppercase">
          L4D · архив лора убежища · выживших: {{ characterCount }}
        </p>
      </div>

      <div class="ml-auto flex items-center gap-2">
        <span
          class="hidden items-center gap-2 rounded-sm border border-zinc-800 bg-bunker-950/60 px-2.5 py-1 text-[0.6rem] tracking-[0.2em] uppercase sm:flex"
          :class="state.text"
          role="status"
        >
          <RadioTower class="size-3.5" aria-hidden="true" />
          {{ state.label }}
          <span class="size-1.5 rounded-full" :class="state.dot" aria-hidden="true" />
        </span>

        <button
          v-if="status === 'offline'"
          type="button"
          class="flex shrink-0 items-center gap-1.5 rounded-sm border border-term-blood/50 px-2.5 py-1 text-[0.6rem] tracking-[0.2em] whitespace-nowrap text-term-blood uppercase transition-colors hover:bg-term-blood/10"
          @click="$emit('retry')"
        >
          <RefreshCw class="size-3.5" aria-hidden="true" />
          Переподключиться
        </button>

        <button
          type="button"
          class="flex shrink-0 items-center gap-1.5 rounded-sm border border-zinc-800 px-2.5 py-1 text-[0.6rem] tracking-[0.2em] whitespace-nowrap text-zinc-500 uppercase transition-colors hover:border-term-amber/40 hover:text-term-amber"
          @click="$emit('new-session')"
        >
          <RotateCcw class="size-3.5" aria-hidden="true" />
          Новая сессия
        </button>

        <a
          :href="apiDocsUrl"
          target="_blank"
          rel="noopener"
          class="hidden shrink-0 items-center gap-1.5 rounded-sm border border-zinc-800 px-2.5 py-1 text-[0.6rem] tracking-[0.2em] whitespace-nowrap text-zinc-500 uppercase transition-colors hover:border-term-green/40 hover:text-term-green md:flex"
        >
          <BookOpen class="size-3.5" aria-hidden="true" />
          API
        </a>
      </div>
    </div>
  </header>
</template>
