<script setup>
import { ChevronRight, MessageSquareQuote, Scroll, SprayCan } from 'lucide-vue-next'
import { computed } from 'vue'

import { formatSimilarity, similarityRatio, sourceTypeAccent, sourceTypeLabel } from '@/lib/lore'

const props = defineProps({
  sources: { type: Array, required: true },
})

/** One icon per evidence tier, so the kind of a record reads at a glance. */
const TIER_ICONS = {
  official: Scroll,
  graffiti: SprayCan,
  voice: MessageSquareQuote,
}

const entries = computed(() =>
  props.sources.map((source) => ({
    ...source,
    label: sourceTypeLabel(source.source_type),
    accent: sourceTypeAccent(source.source_type),
    icon: TIER_ICONS[source.source_type] ?? Scroll,
    match: formatSimilarity(source.similarity),
    ratio: similarityRatio(source.similarity),
  })),
)
</script>

<template>
  <div class="space-y-1.5">
    <p class="flex items-center gap-2 text-[0.6rem] tracking-[0.25em] text-term-green/70 uppercase">
      <span>Источники из архива</span>
      <span class="text-zinc-600">[{{ entries.length }}]</span>
    </p>

    <details
      v-for="entry in entries"
      :key="entry.id"
      class="group/src rounded-sm border bg-bunker-900/70 transition-colors"
      :class="entry.accent"
    >
      <summary
        class="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-[0.7rem] transition-colors hover:bg-white/5 [&::-webkit-details-marker]:hidden"
      >
        <ChevronRight
          class="size-3.5 shrink-0 transition-transform group-open/src:rotate-90"
          aria-hidden="true"
        />
        <component :is="entry.icon" class="size-3.5 shrink-0 opacity-70" aria-hidden="true" />
        <span class="shrink-0 tracking-[0.15em] uppercase">{{ entry.label }}</span>
        <span class="truncate text-zinc-500">{{ entry.title }}</span>
        <span class="ml-auto flex shrink-0 items-center gap-2">
          <span class="h-1 w-10 overflow-hidden bg-bunker-700">
            <span
              class="block h-full bg-current opacity-60"
              :style="{ width: `${entry.ratio * 100}%` }"
            />
          </span>
          <span class="text-[0.6rem] text-zinc-500 tabular-nums">{{ entry.match }}</span>
        </span>
      </summary>
      <p
        class="border-t px-3 py-2 text-xs leading-relaxed whitespace-pre-wrap text-zinc-400"
        :class="entry.accent"
      >
        {{ entry.content }}
      </p>
    </details>
  </div>
</template>
