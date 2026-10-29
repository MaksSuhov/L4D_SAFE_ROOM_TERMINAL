<script setup>
import { BookOpen, RadioTower, RefreshCw, RotateCcw } from 'lucide-vue-next'
import { computed } from 'vue'

import { apiDocsUrl } from '@/api/config'

/** The blood-splattered "4" cut from the official logo by the asset pipeline. */
const MARK_URL = '/brand/l4d-mark.png'

const props = defineProps({
  /** 'connecting' | 'online' | 'offline' — the state of the roster request. */
  status: { type: String, default: 'connecting' },
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
      <img
        :src="MARK_URL"
        alt=""
        class="size-8 shrink-0 drop-shadow-[0_0_8px_rgba(194,30,37,0.45)]"
        aria-hidden="true"
      />

      <div class="min-w-0">
        <h1
          class="flex items-center gap-2 text-[0.65rem] tracking-[0.12em] text-term-amber text-glow-amber uppercase sm:text-xs sm:tracking-[0.25em]"
        >
          <span class="truncate">L4D SAFE-ROOM TERMINAL</span>
          <span class="cursor-blink shrink-0 text-term-green" aria-hidden="true">▍</span>
        </h1>
        <p class="mt-0.5 truncate text-[0.6rem] tracking-[0.2em] text-zinc-500 uppercase">
          Архив лора убежища<span class="hidden sm:inline"> · канал выживших</span>
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
          aria-label="Новая сессия"
          @click="$emit('new-session')"
        >
          <RotateCcw class="size-3.5" aria-hidden="true" />
          <!-- On a narrow screen the icon alone keeps the wordmark on one line. -->
          <span class="hidden sm:inline">Новая сессия</span>
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
